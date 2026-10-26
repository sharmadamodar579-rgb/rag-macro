"""
Automated Test Suite for Database, Authentication, Document Management, and User Isolation.
"""
import os
import sys
import unittest
import json
import io
import shutil
from unittest.mock import patch

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from models.database import init_db, db_session, User, Document, Chat, Message
from config import Config
from langchain_core.embeddings import Embeddings


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


class TestDatabaseAndAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.app.config['GEMINI_API_KEY'] = 'test-gemini-key'
        cls.mock_embeddings = SemanticMockEmbeddings()

    def setUp(self):
        # Create a fresh client for each test method to isolate cookies/sessions
        self.client = self.app.test_client()
        # Clean database tables
        db_session.query(Message).delete()
        db_session.query(Chat).delete()
        db_session.query(Document).delete()
        db_session.query(User).delete()
        db_session.commit()

    def tearDown(self):
        db_session.remove()

    def register_user(self, username, email, password, client=None):
        c = client or self.client
        return c.post('/api/auth/register', 
            data=json.dumps({'username': username, 'email': email, 'password': password}),
            content_type='application/json'
        )

    def login_user(self, identifier, password, client=None):
        c = client or self.client
        return c.post('/api/auth/login',
            data=json.dumps({'username_or_email': identifier, 'password': password}),
            content_type='application/json'
        )

    # 1. Registration Test
    def test_01_registration(self):
        # Valid registration
        res = self.register_user('alice', 'alice@college.edu', 'secret123')
        self.assertEqual(res.status_code, 201)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertIn('user', data)
        self.assertEqual(data['user']['username'], 'alice')
        self.assertEqual(data['user']['email'], 'alice@college.edu')
        self.assertNotIn('password', data['user'])
        self.assertNotIn('password_hash', data['user'])

        # Duplicate email
        res_dup_email = self.register_user('alice2', 'alice@college.edu', 'secret123')
        self.assertEqual(res_dup_email.status_code, 400)
        self.assertIn('already exists', res_dup_email.get_json()['error'])

        # Duplicate username
        res_dup_user = self.register_user('alice', 'alice2@college.edu', 'secret123')
        self.assertEqual(res_dup_user.status_code, 400)
        self.assertIn('already taken', res_dup_user.get_json()['error'])

        # Short password (< 6 chars)
        res_short = self.register_user('bob', 'bob@college.edu', '123')
        self.assertEqual(res_short.status_code, 400)
        self.assertIn('at least 6 characters', res_short.get_json()['error'])

        # Short username (< 3 chars)
        res_short_user = self.register_user('al', 'al@college.edu', 'secret123')
        self.assertEqual(res_short_user.status_code, 400)
        self.assertIn('at least 3 characters', res_short_user.get_json()['error'])

    # 2. Login & Logout Test
    def test_02_login_and_logout(self):
        self.register_user('bob', 'bob@college.edu', 'mypassword')
        
        # Logout to clear registration session
        self.client.post('/api/auth/logout')

        # Login with email
        res_email = self.login_user('bob@college.edu', 'mypassword')
        self.assertEqual(res_email.status_code, 200)
        data = res_email.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['user']['username'], 'bob')

        # Check me endpoint while logged in
        res_me = self.client.get('/api/auth/me')
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.get_json()['user']['email'], 'bob@college.edu')

        # Logout
        res_logout = self.client.post('/api/auth/logout')
        self.assertEqual(res_logout.status_code, 200)

        # Check me endpoint after logout -> 401
        res_me_after = self.client.get('/api/auth/me')
        self.assertEqual(res_me_after.status_code, 401)

        # Login with username
        res_uname = self.login_user('bob', 'mypassword')
        self.assertEqual(res_uname.status_code, 200)

        # Invalid password
        res_invalid = self.login_user('bob@college.edu', 'wrongpassword')
        self.assertEqual(res_invalid.status_code, 401)

        # Non-existent user
        res_no_user = self.login_user('nobody@college.edu', 'wrongpassword')
        self.assertEqual(res_no_user.status_code, 401)

    # 3. Password Hashing Test
    def test_03_password_hashing(self):
        plain_pw = 'super_secret_password_123'
        self.register_user('charlie', 'charlie@college.edu', plain_pw)

        user = db_session.query(User).filter_by(username='charlie').first()
        self.assertIsNotNone(user)
        # Verify password is not stored in plaintext
        self.assertNotEqual(user.password_hash, plain_pw)
        self.assertTrue(user.password_hash.startswith('scrypt:') or user.password_hash.startswith('pbkdf2:'))
        # Verify check_password logic
        self.assertTrue(user.check_password(plain_pw))
        self.assertFalse(user.check_password('wrong_password'))

    # 4. PDF Upload & Document Database Record & Processing Test
    @patch('services.rag_service.get_embedding_model')
    def test_04_document_upload_and_processing(self, mock_get_embed):
        mock_get_embed.return_value = self.mock_embeddings

        # Unauthenticated upload attempt should fail with 401
        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_unauth = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'os_notes.pdf')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res_unauth.status_code, 401)

        # Register and login user
        self.register_user('david', 'david@college.edu', 'password123')

        # Authenticated upload
        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'os_notes.pdf')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res_upload.status_code, 200)
        data = res_upload.get_json()
        self.assertTrue(data['success'])
        doc_id = data['document']['id']
        self.assertTrue(data['document']['processed'])
        self.assertGreater(data['document']['chunk_count'], 0)

        # Verify record in SQLite database
        doc_db = db_session.query(Document).filter_by(id=doc_id).first()
        self.assertIsNotNone(doc_db)
        self.assertEqual(doc_db.filename, 'os_notes.pdf')
        self.assertTrue(doc_db.processed)
        self.assertGreater(doc_db.chunk_count, 0)
        self.assertEqual(doc_db.user.username, 'david')
        self.assertTrue(os.path.exists(doc_db.filepath))

        # Verify Vector Store exists on disk
        vector_dir = os.path.join(Config.VECTOR_STORE_FOLDER, doc_id)
        self.assertTrue(os.path.exists(vector_dir))
        self.assertTrue(os.path.exists(os.path.join(vector_dir, 'index.faiss')))

        # Test GET /api/documents
        res_docs = self.client.get('/api/documents')
        self.assertEqual(res_docs.status_code, 200)
        docs_list = res_docs.get_json()['documents']
        self.assertEqual(len(docs_list), 1)
        self.assertEqual(docs_list[0]['id'], doc_id)

        # Test GET /api/documents/<id>
        res_single = self.client.get(f'/api/documents/{doc_id}')
        self.assertEqual(res_single.status_code, 200)
        self.assertEqual(res_single.get_json()['document']['id'], doc_id)

    # 5. RAG Chat & Chat Persistence & Chat History Test
    @patch('services.rag_service.get_embedding_model')
    @patch('routes.chat.generate_rag_answer')
    def test_05_rag_chat_and_history(self, mock_gen_answer, mock_get_embed):
        mock_get_embed.return_value = self.mock_embeddings
        mock_gen_answer.return_value = "A deadlock occurs when processes are waiting for resources held by each other. The four Coffman conditions are Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait."

        self.register_user('eve', 'eve@college.edu', 'password123')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'os_notes.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        # Query RAG Chat endpoint
        chat_payload = {
            'document_id': doc_id,
            'question': 'What is a deadlock and what are its four conditions?'
        }
        res_chat = self.client.post('/api/chat',
            data=json.dumps(chat_payload),
            content_type='application/json'
        )
        self.assertEqual(res_chat.status_code, 200)
        chat_data = res_chat.get_json()
        self.assertTrue(chat_data['success'])
        self.assertIn('answer', chat_data)
        self.assertIn('Coffman', chat_data['answer'])
        self.assertIn('sources', chat_data)
        self.assertGreater(len(chat_data['sources']), 0)
        self.assertIn('chat_id', chat_data)
        chat_id = chat_data['chat_id']

        # Verify chat and messages persisted in SQLite DB
        chat_db = db_session.query(Chat).filter_by(id=chat_id).first()
        self.assertIsNotNone(chat_db)
        self.assertEqual(chat_db.document_id, doc_id)
        self.assertEqual(len(chat_db.messages), 2)
        self.assertEqual(chat_db.messages[0].role, 'user')
        self.assertEqual(chat_db.messages[0].content, chat_payload['question'])
        self.assertEqual(chat_db.messages[1].role, 'assistant')
        self.assertEqual(chat_db.messages[1].content, chat_data['answer'])

        # Verify GET /api/chat/history
        res_hist = self.client.get(f'/api/chat/history?document_id={doc_id}')
        self.assertEqual(res_hist.status_code, 200)
        hist_data = res_hist.get_json()
        self.assertTrue(hist_data['success'])
        self.assertEqual(len(hist_data['history']), 1)
        self.assertEqual(hist_data['history'][0]['id'], chat_id)
        self.assertEqual(len(hist_data['history'][0]['messages']), 2)

    # 6. User Isolation & Security Test
    @patch('services.rag_service.get_embedding_model')
    @patch('routes.chat.generate_rag_answer')
    def test_06_user_isolation(self, mock_gen_answer, mock_get_embed):
        mock_get_embed.return_value = self.mock_embeddings
        mock_gen_answer.return_value = "CPU scheduling chooses processes from the ready queue."

        # Setup Client 1 (User 1)
        client1 = self.app.test_client()
        self.register_user('user1', 'user1@college.edu', 'pw123456', client=client1)

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = client1.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'user1_private_doc.pdf')},
            content_type='multipart/form-data'
        )
        user1_doc_id = res_upload.get_json()['document']['id']

        # User 1 asks a question
        res_chat = client1.post('/api/chat',
            data=json.dumps({'document_id': user1_doc_id, 'question': 'What is CPU scheduling?'}),
            content_type='application/json'
        )
        self.assertEqual(res_chat.status_code, 200)

        # Setup Client 2 (User 2)
        client2 = self.app.test_client()
        self.register_user('user2', 'user2@college.edu', 'pw123456', client=client2)

        # 1. User 2 should NOT see User 1's document in list
        res_list = client2.get('/api/documents')
        self.assertEqual(res_list.status_code, 200)
        self.assertEqual(len(res_list.get_json()['documents']), 0)

        # 2. User 2 should NOT be able to view User 1's document by ID
        res_get = client2.get(f'/api/documents/{user1_doc_id}')
        self.assertEqual(res_get.status_code, 404)

        # 3. User 2 should NOT be able to search User 1's document
        res_search = client2.post(f'/api/documents/{user1_doc_id}/search',
            data=json.dumps({'query': 'CPU'}),
            content_type='application/json'
        )
        self.assertEqual(res_search.status_code, 404)

        # 4. User 2 should NOT be able to chat on User 1's document
        res_chat_hack = client2.post('/api/chat',
            data=json.dumps({'document_id': user1_doc_id, 'question': 'Tell me everything'}),
            content_type='application/json'
        )
        self.assertEqual(res_chat_hack.status_code, 404)

        # 5. User 2 should NOT see User 1's chat history
        res_hist_hack = client2.get(f'/api/chat/history?document_id={user1_doc_id}')
        self.assertEqual(res_hist_hack.status_code, 404)
        res_all_hist = client2.get('/api/chat/history')
        self.assertEqual(len(res_all_hist.get_json()['history']), 0)

        # 6. User 2 should NOT be able to delete User 1's document
        res_del_hack = client2.delete(f'/api/documents/{user1_doc_id}')
        self.assertEqual(res_del_hack.status_code, 404)

        # Verify User 1's document is still safe in DB
        doc_db = db_session.query(Document).filter_by(id=user1_doc_id).first()
        self.assertIsNotNone(doc_db)

    # 7. Document Deletion Test
    @patch('services.rag_service.get_embedding_model')
    def test_07_document_deletion(self, mock_get_embed):
        mock_get_embed.return_value = self.mock_embeddings

        self.register_user('frank', 'frank@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'doc_to_delete.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']
        
        doc_db = db_session.query(Document).filter_by(id=doc_id).first()
        self.assertIsNotNone(doc_db)
        doc_filepath = doc_db.filepath
        vector_dir = os.path.join(Config.VECTOR_STORE_FOLDER, doc_id)

        self.assertTrue(os.path.exists(doc_filepath))
        self.assertTrue(os.path.exists(vector_dir))

        # Delete document
        res_del = self.client.delete(f'/api/documents/{doc_id}')
        self.assertEqual(res_del.status_code, 200)
        self.assertTrue(res_del.get_json()['success'])

        # Verify deleted from SQLite DB
        doc_db_after = db_session.query(Document).filter_by(id=doc_id).first()
        self.assertIsNone(doc_db_after)

        # Verify deleted from file system and FAISS vector store
        self.assertFalse(os.path.exists(doc_filepath))
        self.assertFalse(os.path.exists(vector_dir))

        # List should now be empty
        res_list = self.client.get('/api/documents')
        self.assertEqual(len(res_list.get_json()['documents']), 0)


if __name__ == '__main__':
    unittest.main()
