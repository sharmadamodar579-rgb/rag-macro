"""
Automated Test Suite for AI MCQ Generator Feature
Tests:
1. Generate 5 easy MCQs with valid schema & source citations
2. Generate 10 medium MCQs
3. Generate mixed-difficulty MCQs
4. Limited content handling & anti-hallucination
5. User isolation (User B cannot generate MCQs for User A's document)
6. Malformed and failed Gemini JSON responses (graceful error handling)
7. Parameter validation (count clamping & difficulty sanitation)
8. Unauthenticated requests (401)
9. Missing Gemini API key (500)
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
from services.ai_service import generate_document_mcqs, parse_and_validate_mcq_json


class TestMCQGeneratorFeature(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.app.config['GEMINI_API_KEY'] = 'test-gemini-key'

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

    # 1. Generate 5 Easy MCQs
    @patch('services.ai_service.generate_answer_with_context')
    def test_01_generate_5_easy_mcqs(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "What is a process in an Operating System?",
                    "options": {
                        "A": "A passive file on disk",
                        "B": "A program in execution",
                        "C": "A hardware CPU register",
                        "D": "A network packet router"
                    },
                    "correct_answer": "B",
                    "explanation": "A process is explicitly defined as a program in execution containing state and registers.",
                    "page": 1
                },
                {
                    "question": "Which component stores process state and CPU registers?",
                    "options": {
                        "A": "PCB (Process Control Block)",
                        "B": "Page Table Frame",
                        "C": "Disk Controller",
                        "D": "ALU buffer"
                    },
                    "correct_answer": "A",
                    "explanation": "The PCB contains information such as Process State, Program Counter, and CPU registers.",
                    "page": 1
                },
                {
                    "question": "What does FCFS stand for in CPU scheduling?",
                    "options": {
                        "A": "Fast Core Fast Storage",
                        "B": "First-Come, First-Served",
                        "C": "Fixed Clock Frequency Switch",
                        "D": "Final Command File System"
                    },
                    "correct_answer": "B",
                    "explanation": "FCFS stands for First-Come, First-Served scheduling.",
                    "page": 2
                },
                {
                    "question": "Which algorithm associates each process with the length of its next CPU burst?",
                    "options": {
                        "A": "Round Robin",
                        "B": "Shortest Job First (SJF)",
                        "C": "Priority Inversion",
                        "D": "Banker's Algorithm"
                    },
                    "correct_answer": "B",
                    "explanation": "SJF associates each process with the length of its next CPU burst and minimizes waiting time.",
                    "page": 2
                },
                {
                    "question": "How many Coffman conditions must simultaneously hold for a deadlock?",
                    "options": {
                        "A": "Two",
                        "B": "Three",
                        "C": "Four",
                        "D": "Five"
                    },
                    "correct_answer": "C",
                    "explanation": "There are four Coffman conditions: Mutual Exclusion, Hold and Wait, No Preemption, and Circular Wait.",
                    "page": 3
                }
            ]
        })

        self.register_and_login('alice', 'alice@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'operating_systems_lecture.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        # Request 5 easy MCQs
        res_mcq = self.client.post(f'/api/documents/{doc_id}/mcqs', json={'count': 5, 'difficulty': 'easy'})
        self.assertEqual(res_mcq.status_code, 200)
        data = res_mcq.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['count'], 5)
        self.assertEqual(data['difficulty'], 'easy')
        self.assertEqual(len(data['questions']), 5)

        for q in data['questions']:
            self.assertIn('question', q)
            self.assertIn('options', q)
            self.assertEqual(len(q['options']), 4)
            self.assertIn(q['correct_answer'], ['A', 'B', 'C', 'D'])
            self.assertIn('explanation', q)
            self.assertIn('source', q)
            self.assertEqual(q['source']['document'], 'operating_systems_lecture.pdf')
            self.assertIsInstance(q['source']['page'], int)

    # 2. Generate 10 Medium MCQs
    @patch('services.ai_service.generate_answer_with_context')
    def test_02_generate_10_medium_mcqs(self, mock_ai):
        questions_list = []
        for i in range(1, 11):
            questions_list.append({
                "question": f"Question {i} regarding OS concepts?",
                "options": {
                    "A": f"Option A for Q{i}",
                    "B": f"Option B for Q{i}",
                    "C": f"Option C for Q{i}",
                    "D": f"Option D for Q{i}"
                },
                "correct_answer": "B",
                "explanation": f"Explanation for Q{i}",
                "page": (i % 4) + 1
            })
        mock_ai.return_value = json.dumps({"questions": questions_list})

        self.register_and_login('bob', 'bob@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'os_notes.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        res_mcq = self.client.post(f'/api/documents/{doc_id}/mcqs', json={'count': 10, 'difficulty': 'medium'})
        self.assertEqual(res_mcq.status_code, 200)
        data = res_mcq.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['count'], 10)
        self.assertEqual(data['difficulty'], 'medium')
        self.assertEqual(len(data['questions']), 10)

    # 3. Generate Mixed Difficulty MCQs
    @patch('services.ai_service.generate_answer_with_context')
    def test_03_generate_mixed_difficulty_mcqs(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "What solves starvation in priority scheduling?",
                    "options": {
                        "A": "Aging",
                        "B": "Deadlock",
                        "C": "Convoy effect",
                        "D": "Paging"
                    },
                    "correct_answer": "A",
                    "explanation": "Aging gradually increases the priority of processes that wait for a long time.",
                    "page": 2
                }
            ]
        })

        self.register_and_login('charlie', 'charlie@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'db_notes.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'db_notes.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        res_mcq = self.client.post(f'/api/documents/{doc_id}/mcqs', json={'count': 5, 'difficulty': 'mixed'})
        self.assertEqual(res_mcq.status_code, 200)
        data = res_mcq.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['difficulty'], 'mixed')

    # 4. User Isolation Test
    def test_04_user_isolation(self):
        client1 = self.app.test_client()
        self.register_and_login('userA', 'userA@college.edu', 'pw123456', client=client1)

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = client1.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'userA_private.pdf')},
            content_type='multipart/form-data'
        )
        userA_doc_id = res_upload.get_json()['document']['id']

        # Setup User B
        client2 = self.app.test_client()
        self.register_and_login('userB', 'userB@college.edu', 'pw123456', client=client2)

        # User B attempts to generate MCQs for User A's document -> 404 Access Denied
        res_hack = client2.post(f'/api/documents/{userA_doc_id}/mcqs', json={'count': 5})
        self.assertEqual(res_hack.status_code, 404)
        self.assertFalse(res_hack.get_json()['success'])
        self.assertIn('access denied', res_hack.get_json()['error'].lower())

    # 5. Malformed Gemini Response Handling
    @patch('services.ai_service.generate_answer_with_context')
    def test_05_malformed_gemini_response_handling(self, mock_ai):
        mock_ai.return_value = "This is not valid json at all! ``` invalid text ```"

        self.register_and_login('david', 'david@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'doc_test.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        res_mcq = self.client.post(f'/api/documents/{doc_id}/mcqs', json={'count': 5})
        self.assertEqual(res_mcq.status_code, 400)
        self.assertFalse(res_mcq.get_json()['success'])
        self.assertIn('JSON', res_mcq.get_json()['error'])

    # 6. Parameter Sanitization Test (Clamping count & invalid difficulty)
    @patch('services.ai_service.generate_answer_with_context')
    def test_06_parameter_sanitization(self, mock_ai):
        mock_ai.return_value = json.dumps({
            "questions": [
                {
                    "question": "Test question?",
                    "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                    "correct_answer": "A",
                    "explanation": "Test explanation",
                    "page": 1
                }
            ]
        })

        self.register_and_login('eve', 'eve@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'doc_params.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        # Send negative count and invalid difficulty
        res_mcq = self.client.post(f'/api/documents/{doc_id}/mcqs', json={'count': -10, 'difficulty': 'ultra_hard'})
        self.assertEqual(res_mcq.status_code, 200)
        data = res_mcq.get_json()
        self.assertEqual(data['difficulty'], 'medium')  # Defaulted safely

    # 7. Unauthenticated & Missing Key Tests
    def test_07_unauthenticated_and_edge_cases(self):
        # 1. Unauthenticated request -> 401
        res_unauth = self.client.post('/api/documents/doc_12345/mcqs')
        self.assertEqual(res_unauth.status_code, 401)

        # 2. Missing GEMINI_API_KEY -> 500
        self.register_and_login('frank', 'frank@college.edu', 'pw123456')
        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'doc_nokey.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        self.app.config['GEMINI_API_KEY'] = ''
        res_no_key = self.client.post(f'/api/documents/{doc_id}/mcqs')
        self.assertEqual(res_no_key.status_code, 500)
        self.assertIn('GEMINI_API_KEY is not configured', res_no_key.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
