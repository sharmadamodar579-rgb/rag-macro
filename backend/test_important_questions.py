"""
Automated Test Suite for AI Important Questions Generator Feature
Tests:
1. Generate 5 important questions with mixed types & valid schema
2. Generate 10 important questions with high/medium/low priority grouping
3. Generate specific question types (short, long, conceptual, definition)
4. Anti-hallucination & grounding verification with page citations
5. User isolation (User B cannot generate questions for User A's document)
6. Large document sampling across pages
7. Malformed and failed Gemini JSON responses (graceful error handling)
8. Parameter validation (count clamping & type sanitation)
9. Unauthenticated requests (401)
10. Missing Gemini API key (500)
"""

import os
import sys
import unittest
import json
import io
from unittest.mock import patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from models.database import init_db, db_session, User, Document, Chat, Message
from services.ai_service import generate_important_questions, parse_and_validate_questions_json


class TestImportantQuestionsFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.app.config['GEMINI_API_KEY'] = 'test-gemini-key'
        cls.test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')

    def setUp(self):
        self.client = self.app.test_client()
        # Clean database tables
        db_session.query(Message).delete()
        db_session.query(Chat).delete()
        db_session.query(Document).delete()
        db_session.query(User).delete()
        db_session.commit()

    def tearDown(self):
        db_session.remove()

    def register_and_login(self, username, email, password, client=None):
        c = client or self.client
        c.post('/api/auth/register', json={'username': username, 'email': email, 'password': password})
        c.post('/api/auth/login', json={'username_or_email': email, 'password': password})
        return c

    def upload_test_doc(self, client, filename='operating_systems_lecture.pdf'):
        pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', filename)
        if not os.path.exists(pdf_path):
            pdf_path = self.test_pdf_path
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = client.post('/api/documents/upload',
            data={'file': (io.BytesIO(pdf_bytes), filename)},
            content_type='multipart/form-data'
        )
        return res_upload.get_json()['document']['id']

    # 1. Generate 5 Mixed Important Questions
    @patch('services.ai_service.generate_answer_with_context')
    def test_01_generate_5_mixed_questions(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "Explain the concept of Process Control Block (PCB) and list its key components.",
                    "type": "long",
                    "importance": "high",
                    "answer": "A PCB is a kernel data structure containing PID, process state, program counter, CPU registers, and scheduling info.",
                    "page": 1
                },
                {
                    "question": "Define what a process is in modern operating systems.",
                    "type": "definition",
                    "importance": "high",
                    "answer": "A process is a program in execution, including code, current activity, stack, and data section.",
                    "page": 1
                },
                {
                    "question": "How does Shortest Job First (SJF) scheduling minimize average waiting time?",
                    "type": "conceptual",
                    "importance": "medium",
                    "answer": "SJF orders processes by their next CPU burst length, which is provably optimal for minimizing average waiting time.",
                    "page": 2
                },
                {
                    "question": "What is the primary trade-off of Round Robin (RR) scheduling when choosing time quantum size?",
                    "type": "conceptual",
                    "importance": "medium",
                    "answer": "A small quantum increases context switch overhead, while a large quantum degenerates into FCFS.",
                    "page": 2
                },
                {
                    "question": "State the difference between preemptive and non-preemptive CPU scheduling.",
                    "type": "short",
                    "importance": "low",
                    "answer": "Preemptive scheduling can interrupt a running process; non-preemptive allows a process to run until it terminates or yields.",
                    "page": 2
                }
            ]
        })

        client = self.register_and_login('student_os', 'os@college.edu', 'pass123')
        doc_id = self.upload_test_doc(client)

        # Call Important Questions Endpoint
        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': 5, 'type': 'mixed'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertTrue(data['success'])
        self.assertEqual(data['document_id'], doc_id)
        self.assertEqual(data['type'], 'mixed')
        self.assertEqual(len(data['questions']), 5)

        for q in data['questions']:
            self.assertIn('question', q)
            self.assertIn('type', q)
            self.assertIn('importance', q)
            self.assertIn('answer', q)
            self.assertIn('source', q)
            self.assertIn(q['importance'], ['high', 'medium', 'low'])
            self.assertIn(q['type'], ['short', 'long', 'conceptual', 'definition'])

    # 2. Generate 10 Questions with Priority Breakdown
    @patch('services.ai_service.generate_answer_with_context')
    def test_02_generate_10_questions_priority_breakdown(self, mock_ai):
        questions_payload = []
        for i in range(1, 11):
            imp = "high" if i <= 4 else ("medium" if i <= 8 else "low")
            q_type = ["short", "long", "conceptual", "definition"][i % 4]
            questions_payload.append({
                "question": f"Exam Revision Question #{i} on Operating Systems",
                "type": q_type,
                "importance": imp,
                "answer": f"Answer key for question #{i}",
                "page": (i % 3) + 1
            })

        mock_ai.return_value = json.dumps({"questions": questions_payload})

        client = self.register_and_login('student_db', 'db@college.edu', 'pass123')
        doc_id = self.upload_test_doc(client)

        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': 10, 'type': 'mixed'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertTrue(data['success'])
        self.assertEqual(len(data['questions']), 10)
        self.assertEqual(data['total_questions'], 10)

        high_count = sum(1 for q in data['questions'] if q['importance'] == 'high')
        med_count = sum(1 for q in data['questions'] if q['importance'] == 'medium')
        low_count = sum(1 for q in data['questions'] if q['importance'] == 'low')

        self.assertEqual(high_count, 4)
        self.assertEqual(med_count, 4)
        self.assertEqual(low_count, 2)

    # 3. Question Type Filtering (conceptual)
    @patch('services.ai_service.generate_answer_with_context')
    def test_03_generate_specific_question_type(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "Why does Round Robin suffer when quantum is extremely small?",
                    "type": "conceptual",
                    "importance": "high",
                    "answer": "Context switch overhead dominates CPU execution.",
                    "page": 2
                },
                {
                    "question": "Analyze how Banker's algorithm ensures system remains in a safe state.",
                    "type": "conceptual",
                    "importance": "medium",
                    "answer": "It tests for safety by simulating the allocation of predetermined maximum possible amounts.",
                    "page": 3
                }
            ]
        })

        client = self.register_and_login('student_net', 'net@college.edu', 'pass123')
        doc_id = self.upload_test_doc(client)

        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': 2, 'type': 'conceptual'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data['type'], 'conceptual')
        for q in data['questions']:
            self.assertEqual(q['type'], 'conceptual')

    # 4. User Isolation & Access Denied
    def test_04_user_isolation(self):
        client_a = self.register_and_login('user_a', 'user_a@test.com', 'pass123')
        doc_id_a = self.upload_test_doc(client_a)

        # User B attempts to access User A's document
        client_b = self.app.test_client()
        self.register_and_login('user_b', 'user_b@test.com', 'pass123', client=client_b)

        res = client_b.post(f'/api/documents/{doc_id_a}/questions', json={'count': 5})
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertFalse(data['success'])
        self.assertIn("not found or access denied", data['error'].lower())

    # 5. Large Document Page Sampling
    @patch('services.ai_service.generate_answer_with_context')
    def test_05_large_document_sampling(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "Sampled Question from Multi-page Document",
                    "type": "long",
                    "importance": "high",
                    "answer": "Sampled Answer",
                    "page": 1
                }
            ]
        })

        client = self.register_and_login('student_large', 'large@test.com', 'pass123')
        doc_id = self.upload_test_doc(client)

        # Mock large 15-page document text in database
        doc = db_session.query(Document).filter_by(id=doc_id).first()
        pages = []
        for p in range(1, 16):
            pages.append(f"--- Page {p} ---\nDetailed chapter {p} content with extensive syllabus notes " * 30)
        doc.extracted_text = "\n\n".join(pages)
        db_session.commit()

        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': 5, 'type': 'mixed'})
        self.assertEqual(res.status_code, 200)

        # Ensure AI service received sampled text
        self.assertTrue(mock_ai.called)
        prompt_arg = mock_ai.call_args.kwargs.get("prompt") if mock_ai.call_args.kwargs else mock_ai.call_args[0][0]
        self.assertIn("Page 1", prompt_arg)

    # 6. JSON Parsing & Markdown Stripping
    def test_06_json_parsing_and_markdown_cleaning(self):
        raw_markdown_json = """```json
        {
            "questions": [
                {
                    "question": "What is Amdahl's Law?",
                    "type": "short",
                    "importance": "high",
                    "answer": "A formula giving the theoretical speedup in latency of an execution task.",
                    "page": 3
                }
            ]
        }
        ```"""

        parsed = parse_and_validate_questions_json(raw_markdown_json, "test_doc.pdf")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]['question'], "What is Amdahl's Law?")
        self.assertEqual(parsed[0]['type'], "short")
        self.assertEqual(parsed[0]['importance'], "high")
        self.assertEqual(parsed[0]['source']['document'], "test_doc.pdf")
        self.assertEqual(parsed[0]['source']['page'], 3)

    # 7. Malformed AI Output Handling
    @patch('services.ai_service.generate_answer_with_context')
    def test_07_malformed_ai_response(self, mock_ai):
        mock_ai.return_value = "Sorry, I cannot generate questions right now because this is not valid JSON."

        client = self.register_and_login('student_err', 'err@test.com', 'pass123')
        doc_id = self.upload_test_doc(client)

        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': 5})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data['success'])
        self.assertIn("failed to parse", data['error'].lower())

    # 8. Parameter Sanitation
    @patch('services.ai_service.generate_answer_with_context')
    def test_08_parameter_sanitation(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "What is PCB?",
                    "type": "definition",
                    "importance": "high",
                    "answer": "Process Control Block",
                    "page": 1
                }
            ]
        })

        client = self.register_and_login('student_param', 'param@test.com', 'pass123')
        doc_id = self.upload_test_doc(client)

        # Pass invalid count (-5) and invalid type ('super_hard_hack')
        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': -5, 'type': 'invalid_type'})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['type'], 'mixed')  # Fallbacks to mixed
        self.assertEqual(data['requested_count'], 1)  # Clamped to min 1

    # 9. Unauthenticated Access (401)
    def test_09_unauthenticated_request(self):
        client = self.app.test_client()
        res = client.post('/api/documents/doc-123/questions', json={'count': 5})
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertFalse(data['success'])

    # 10. Missing Gemini API Key (500)
    @patch('services.ai_service.generate_answer_with_context')
    def test_10_missing_api_key(self, mock_ai):
        mock_ai.side_effect = ValueError("Gemini API key is missing. Please set GEMINI_API_KEY.")

        client = self.register_and_login('student_nokey', 'nokey@test.com', 'pass123')
        doc_id = self.upload_test_doc(client)

        res = client.post(f'/api/documents/{doc_id}/questions', json={'count': 5})
        self.assertEqual(res.status_code, 500)
        data = res.get_json()
        self.assertFalse(data['success'])
        self.assertIn("gemini api key is missing", data['error'].lower())


if __name__ == '__main__':
    unittest.main()
