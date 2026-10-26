import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from app import create_app
from services.pdf_service import extract_pdf_text, PDFProcessingError
from services.rag_service import (
    chunk_document_pages,
    split_text_into_chunks,
    get_embedding_model,
    create_vector_store,
    load_vector_store,
    search_similar_chunks
)
from services.ai_service import build_rag_prompt, generate_rag_answer, get_gemini_chat_model

from langchain_core.embeddings import Embeddings

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")


class SemanticMockEmbeddings(Embeddings):
    """
    Mock embeddings based on keyword presence to test semantic retrieval
    and page ranking deterministically without requiring an active Gemini API key.
    """
    def __init__(self):
        self.keywords = [
            'process', 'pcb', 'state', 'scheduler',
            'round', 'robin', 'quantum', 'fcfs', 'sjf', 'priority', 'starvation', 'convoy',
            'deadlock', 'coffman', 'exclusion', 'preemption', 'circular', 'banker',
            'paging', 'frame', 'page', 'table', 'fault', 'virtual', 'memory'
        ]
        self.dim = 768

    def _embed(self, text: str):
        text_lower = text.lower()
        vec = [0.0] * self.dim
        for idx, kw in enumerate(self.keywords):
            if kw in text_lower:
                vec[idx % self.dim] += 1.0
                vec[(idx * 7) % self.dim] += 0.5
        norm = sum(x**2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


from models.database import init_db, db_session, User, Document, Chat, Message

class TestCompleteRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.client = cls.app.test_client()
        cls.embed_mock = SemanticMockEmbeddings()

        # Create a test user and test document in DB for API tests
        db_session.query(Message).delete()
        db_session.query(Chat).delete()
        db_session.query(Document).delete()
        db_session.query(User).delete()
        db_session.commit()

        user = User(username="rag_tester", email="rag@college.edu")
        user.set_password("password123")
        db_session.add(user)
        db_session.commit()

        doc = Document(
            id="doc_test",
            user_id=user.id,
            filename="operating_systems_lecture.pdf",
            filepath=os.path.join(UPLOADS_DIR, "operating_systems_lecture.pdf"),
            processed=True,
            chunk_count=4
        )
        db_session.add(doc)
        db_session.commit()

        # Log in the client
        cls.client.post('/api/auth/login', json={'username_or_email': 'rag@college.edu', 'password': 'password123'})

    # -------------------------------------------------------------
    # 1. PDF Processing Tests
    # -------------------------------------------------------------
    def test_01_pdf_extraction_valid_files(self):
        valid_files = ['db_notes.pdf', 'lecture_notes.pdf', 'networks_ch1.pdf', 'operating_systems_lecture.pdf']
        for fname in valid_files:
            fpath = os.path.join(UPLOADS_DIR, fname)
            res = extract_pdf_text(fpath)
            self.assertGreater(res['total_pages'], 0, f'Failed on {fname}')
            self.assertTrue(res['has_extractable_text'], f'No text in {fname}')
            self.assertGreater(res['total_characters'], 0, f'0 chars in {fname}')
            self.assertEqual(len(res['pages']), res['total_pages'])
            for i, p in enumerate(res['pages'], 1):
                self.assertEqual(p['page_number'], i)
                self.assertIsInstance(p['text'], str)

    def test_02_pdf_corrupted_and_empty_text_files(self):
        # Corrupted PDF
        with self.assertRaises(PDFProcessingError):
            extract_pdf_text(os.path.join(UPLOADS_DIR, 'corrupted.pdf'))

        # Non-existent file
        with self.assertRaises(PDFProcessingError):
            extract_pdf_text(os.path.join(UPLOADS_DIR, 'does_not_exist.pdf'))

        # PDF with no extractable text layer (scanned)
        res = extract_pdf_text(os.path.join(UPLOADS_DIR, 'multipage_notes.pdf'))
        self.assertEqual(res['total_pages'], 3)
        self.assertFalse(res['has_extractable_text'])

    def test_03_chunking_preserves_metadata(self):
        fpath = os.path.join(UPLOADS_DIR, 'operating_systems_lecture.pdf')
        res = extract_pdf_text(fpath)
        chunks = chunk_document_pages(res['pages'], filename='operating_systems_lecture.pdf', chunk_size=300, chunk_overlap=50)
        self.assertGreaterEqual(len(chunks), 4)
        pages_found = set()
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk['chunk_index'], i)
            self.assertIn('page', chunk)
            self.assertGreaterEqual(chunk['page'], 1)
            pages_found.add(chunk['page'])
            self.assertEqual(chunk['filename'], 'operating_systems_lecture.pdf')
            self.assertGreater(chunk['char_count'], 0)
            self.assertGreater(chunk['word_count'], 0)
        self.assertEqual(pages_found, {1, 2, 3, 4})

    # -------------------------------------------------------------
    # 2. Embedding Model & Dimensionality Tests
    # -------------------------------------------------------------
    def test_04_embedding_config_and_dimensions(self):
        with self.assertRaises(ValueError):
            get_embedding_model(api_key='')
        vecs = self.embed_mock.embed_documents(['Process scheduling', 'Memory management'])
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 768)

    # -------------------------------------------------------------
    # 3. FAISS Vector Store Lifecycle & Retrieval Accuracy
    # -------------------------------------------------------------
    def test_05_faiss_vector_store_create_load_and_search(self):
        fpath = os.path.join(UPLOADS_DIR, 'operating_systems_lecture.pdf')
        pdf_res = extract_pdf_text(fpath)
        chunks = chunk_document_pages(pdf_res['pages'], filename='operating_systems_lecture.pdf', chunk_size=300, chunk_overlap=50)

        with patch('services.rag_service.get_embedding_model', return_value=self.embed_mock):
            doc_id = 'test_lifecycle_os_doc'
            create_res = create_vector_store(chunks=chunks, document_id=doc_id)
            self.assertTrue(create_res['created'])
            self.assertGreater(create_res['vectors'], 0)
            self.assertTrue(os.path.exists(create_res['store_path']))
            self.assertTrue(os.path.exists(os.path.join(create_res['store_path'], 'index.faiss')))
            self.assertTrue(os.path.exists(os.path.join(create_res['store_path'], 'index.pkl')))

            # Load vector store back from disk
            loaded_vs = load_vector_store(doc_id)
            self.assertIsNotNone(loaded_vs)

            # RAG Retrieval Quality Checks:
            # 1. Deadlock question -> Page 3
            deadlock_res = search_similar_chunks(doc_id, query='What are the four Coffman conditions for deadlock?', k=2)
            self.assertGreater(len(deadlock_res), 0)
            self.assertEqual(deadlock_res[0]['page'], 3)
            self.assertIn('Coffman', deadlock_res[0]['text'])

            # 2. Round Robin question -> Page 2
            rr_res = search_similar_chunks(doc_id, query='How does Round Robin scheduling work with time quantum?', k=2)
            self.assertGreater(len(rr_res), 0)
            self.assertEqual(rr_res[0]['page'], 2)
            self.assertIn('Round Robin', rr_res[0]['text'])

            # 3. Memory Paging question -> Page 4
            mem_res = search_similar_chunks(doc_id, query='What is the difference between pages and frames in memory?', k=2)
            self.assertGreater(len(mem_res), 0)
            self.assertEqual(mem_res[0]['page'], 4)
            self.assertIn('Frames', mem_res[0]['text'])

            # 4. PCB question -> Page 1
            pcb_res = search_similar_chunks(doc_id, query='What information is stored inside the PCB?', k=2)
            self.assertGreater(len(pcb_res), 0)
            self.assertEqual(pcb_res[0]['page'], 1)
            self.assertIn('PCB', pcb_res[0]['text'])

    # -------------------------------------------------------------
    # 4. RAG Quality, Prompt Grounding, and AI Chat Tests
    # -------------------------------------------------------------
    def test_06_rag_prompt_construction(self):
        chunks = [
            {'text': 'First chunk about Deadlocks.', 'page': 3, 'filename': 'os.pdf'},
            {'text': 'Second chunk about Semaphores.', 'page': 4, 'filename': 'os.pdf'}
        ]
        prompt = build_rag_prompt('What is a deadlock?', chunks)
        self.assertIn('STUDY MATERIAL CONTEXT', prompt)
        self.assertIn('[Source #1 | Document: os.pdf | Page: 3]', prompt)
        self.assertIn('First chunk about Deadlocks.', prompt)
        self.assertIn('[Source #2 | Document: os.pdf | Page: 4]', prompt)
        self.assertIn('STUDENT QUESTION:\nWhat is a deadlock?', prompt)
        self.assertIn('The uploaded material does not provide enough information', prompt)

    def test_07_rag_chat_endpoint_grounded_answer(self):
        self.app.config['GEMINI_API_KEY'] = 'fake-gemini-key'
        mock_chunks = [
            {'text': 'Round Robin uses time quantum for CPU scheduling.', 'page': 2, 'chunk_index': 2, 'filename': 'operating_systems_lecture.pdf', 'document_id': 'doc_test'},
            {'text': 'FCFS is non-preemptive scheduling.', 'page': 2, 'chunk_index': 3, 'filename': 'operating_systems_lecture.pdf', 'document_id': 'doc_test'}
        ]
        with patch('routes.chat.search_similar_chunks', return_value=mock_chunks):
            with patch('services.ai_service.ChatGoogleGenerativeAI') as MockChat:
                mock_inst = MagicMock()
                mock_inst.invoke.return_value = MagicMock(content='Round Robin scheduling allocates a fixed time quantum to each process.')
                MockChat.return_value = mock_inst

                res = self.client.post('/api/chat', json={
                    'question': 'What is Round Robin scheduling?',
                    'document_id': 'doc_test'
                })
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertTrue(data['success'])
                self.assertIn('Round Robin scheduling allocates', data['answer'])
                self.assertEqual(data['sources'], [{'document': 'operating_systems_lecture.pdf', 'page': 2}])

    # -------------------------------------------------------------
    # 5. Hallucination Protection Tests
    # -------------------------------------------------------------
    def test_08_hallucination_guard_empty_retrieval(self):
        self.app.config['GEMINI_API_KEY'] = 'fake-gemini-key'
        with patch('routes.chat.search_similar_chunks', return_value=[]):
            res = self.client.post('/api/chat', json={
                'question': 'How many moons does Jupiter have?',
                'document_id': 'doc_test'
            })
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data['success'])
            self.assertEqual(data['answer'], 'The uploaded material does not provide enough information to answer this question.')
            self.assertEqual(data['sources'], [])

    # -------------------------------------------------------------
    # 6. Edge Cases & Error Handling
    # -------------------------------------------------------------
    def test_09_edge_cases_and_error_handling(self):
        self.app.config['GEMINI_API_KEY'] = 'fake-gemini-key'

        # Empty question -> 400
        res = self.client.post('/api/chat', json={'question': ''})
        self.assertEqual(res.status_code, 400)

        # Whitespace-only question -> 400
        res = self.client.post('/api/chat', json={'question': '   \n\t  '})
        self.assertEqual(res.status_code, 400)

        # Missing payload -> 400
        res = self.client.post('/api/chat', json={})
        self.assertEqual(res.status_code, 400)

        # Extremely short question (should reach pipeline safely)
        with patch('routes.chat.search_similar_chunks', return_value=[]):
            res = self.client.post('/api/chat', json={'question': '?'})
            self.assertEqual(res.status_code, 200)

        # Extremely long question (5000 chars)
        with patch('routes.chat.search_similar_chunks', return_value=[]):
            res = self.client.post('/api/chat', json={'question': 'Explain OS scheduling ' * 300})
            self.assertEqual(res.status_code, 200)

        # Invalid document ID (path traversal)
        res = self.client.post('/api/chat', json={'question': 'Test', 'document_id': '../etc/secret'})
        self.assertEqual(res.status_code, 400)

        # Non-existent document ID
        res = self.client.post('/api/chat', json={'question': 'Test', 'document_id': 'missing_doc_999'})
        self.assertEqual(res.status_code, 404)

        # Missing API Key
        self.app.config['GEMINI_API_KEY'] = ''
        res = self.client.post('/api/chat', json={'question': 'Test', 'document_id': 'doc_test'})
        self.assertEqual(res.status_code, 500)
        self.assertIn('GEMINI_API_KEY is not configured', res.get_json()['error'])

        # Gemini API Runtime Error -> 500 sanitized
        self.app.config['GEMINI_API_KEY'] = 'fake-gemini-key'
        with patch('routes.chat.search_similar_chunks', return_value=[{'text': 'Sample text', 'page': 1, 'filename': 'os.pdf'}]):
            with patch('services.ai_service.ChatGoogleGenerativeAI') as MockChat:
                mock_inst = MagicMock()
                mock_inst.invoke.side_effect = RuntimeError('Gemini connection timed out')
                MockChat.return_value = mock_inst

                res = self.client.post('/api/chat', json={'question': 'What is deadlock?', 'document_id': 'doc_test'})
                self.assertEqual(res.status_code, 500)
                self.assertFalse(res.get_json()['success'])

    # -------------------------------------------------------------
    # 7. Detailed Step-by-Step RAG Trace Inspection
    # -------------------------------------------------------------
    def test_10_rag_trace_quality_inspection(self):
        doc_id = 'test_lifecycle_os_doc'
        question = 'What are the four Coffman conditions for deadlock?'

        with patch('services.rag_service.get_embedding_model', return_value=self.embed_mock):
            chunks = search_similar_chunks(doc_id, question, k=3)
            self.assertTrue(len(chunks) > 0)

            # Verify retrieval accuracy
            top_chunk = chunks[0]
            self.assertEqual(top_chunk['page'], 3)
            self.assertIn('Coffman', top_chunk['text'])

            # Verify prompt construction
            prompt = build_rag_prompt(question, chunks)
            self.assertIn('[Source #1 | Document: operating_systems_lecture.pdf | Page: 3]', prompt)
            self.assertIn('Mutual Exclusion', prompt)
            self.assertIn('Circular Wait', prompt)


if __name__ == '__main__':
    unittest.main()
