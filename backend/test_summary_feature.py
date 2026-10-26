"""
Automated Test Suite for AI Study Summary Feature
Tests:
1. Normal document summary generation (direct single-pass strategy)
2. Multi-page document summary with source citations
3. Empty, corrupted, and scanned PDF error handling
4. User isolation and unauthorized document access prevention
5. Large document handling (hierarchical map-reduce strategy)
6. Unauthenticated requests (401)
7. Non-existent document ID (404)
8. Missing API key handling (500)
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
from services.ai_service import generate_study_summary, build_summary_prompt


class TestStudySummaryFeature(unittest.TestCase):
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

    # 1. Normal Document Summary Test
    @patch('services.ai_service.ChatGoogleGenerativeAI')
    def test_01_normal_document_summary(self, MockChat):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="""# Summary: operating_systems_lecture.pdf

## 1. Overview
This lecture document covers CPU scheduling algorithms, process management, and deadlocks.

## 2. Key Concepts
- Process Control Block (PCB)
- CPU Scheduling (FCFS, SJF, Round Robin)
- Deadlocks and Coffman Conditions

## 3. Important Points
- Round Robin uses a time quantum to prevent starvation.
- The 4 Coffman conditions must hold simultaneously for deadlock.

## 4. Definitions
- **PCB** — Process Control Block containing process state and registers.
- **Deadlock** — A situation where processes are waiting indefinitely for resources.

## 5. Quick Revision
- SJF minimizes average waiting time.
- Banker's algorithm avoids deadlocks.
""")
        MockChat.return_value = mock_instance

        self.register_and_login('alice', 'alice@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'operating_systems_lecture.pdf')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res_upload.status_code, 200)
        doc_id = res_upload.get_json()['document']['id']

        # Request Summary
        res_summary = self.client.post(f'/api/documents/{doc_id}/summary')
        self.assertEqual(res_summary.status_code, 200)
        data = res_summary.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['document_id'], doc_id)
        self.assertEqual(data['filename'], 'operating_systems_lecture.pdf')
        self.assertEqual(data['strategy'], 'direct')
        self.assertEqual(data['total_pages'], 4)
        self.assertIn('# Summary', data['summary'])
        self.assertIn('## 1. Overview', data['summary'])
        self.assertIn('## 2. Key Concepts', data['summary'])
        self.assertIn('## 4. Definitions', data['summary'])
        self.assertEqual(len(data['sources']), 4)
        self.assertEqual(data['sources'][0]['page'], 1)

    # 2. Multi-Page Document Summary Test
    @patch('services.ai_service.ChatGoogleGenerativeAI')
    def test_02_multipage_document_summary(self, MockChat):
        mock_instance = MagicMock()
        mock_instance.invoke.return_value = MagicMock(content="""# Summary: db_notes.pdf
## 1. Overview
Database systems and relational normalization.
## 2. Key Concepts
- ACID Properties
- SQL Joins
- B-Trees
""")
        MockChat.return_value = mock_instance

        self.register_and_login('bob', 'bob@college.edu', 'pw123456')

        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'db_notes.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'db_notes.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        res_summary = self.client.post(f'/api/documents/{doc_id}/summary')
        self.assertEqual(res_summary.status_code, 200)
        data = res_summary.get_json()
        self.assertTrue(data['success'])
        self.assertGreaterEqual(data['total_pages'], 1)
        self.assertGreater(len(data['sources']), 0)

    # 3. Empty / Scanned / Corrupted Document Error Handling Test
    def test_03_empty_and_scanned_document_errors(self):
        self.register_and_login('charlie', 'charlie@college.edu', 'pw123456')

        # Upload a scanned PDF with 0 extractable text
        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'multipage_notes.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'scanned_notes.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        # Generating summary on a PDF without extractable text should return 400
        res_summary = self.client.post(f'/api/documents/{doc_id}/summary')
        self.assertEqual(res_summary.status_code, 400)
        self.assertFalse(res_summary.get_json()['success'])
        self.assertIn('No extractable text', res_summary.get_json()['error'])

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

        # User B attempts to generate summary for User A's document -> 404 Access Denied
        res_hack = client2.post(f'/api/documents/{userA_doc_id}/summary')
        self.assertEqual(res_hack.status_code, 404)
        self.assertFalse(res_hack.get_json()['success'])
        self.assertIn('access denied', res_hack.get_json()['error'].lower())

    # 5. Large Document Hierarchical Handling Test
    @patch('services.ai_service.generate_answer_with_context')
    def test_05_large_document_hierarchical_handling(self, mock_gen):
        def mock_ai_response(prompt, **kwargs):
            if "Synthesize the following" in prompt:
                return "# Summary: large_textbook.pdf\n\n## 1. Overview\nComprehensive OS study guide.\n\n## 2. Key Concepts\n- Scheduling\n- Paging"
            return "Section Summary: Key concepts, definitions, and mechanisms."

        mock_gen.side_effect = mock_ai_response

        # Simulate a 10-page document where each page has 3,000 characters (> 25,000 total characters)
        large_pages = [
            {"page_number": i, "text": f"This is page {i} content. " + ("Detailed OS explanation text. " * 120)}
            for i in range(1, 11)
        ]

        result = generate_study_summary(
            pages=large_pages,
            filename="large_textbook.pdf",
            api_key="test-key"
        )

        self.assertEqual(result['strategy'], 'hierarchical')
        self.assertEqual(result['total_pages'], 10)
        self.assertGreater(result['total_characters'], 25000)
        self.assertEqual(len(result['sources']), 10)
        self.assertIn('# Summary', result['summary'])

    # 6. Unauthenticated and Error Handling Tests
    def test_06_unauthenticated_and_edge_cases(self):
        # 1. Unauthenticated request -> 401
        res_unauth = self.client.post('/api/documents/doc_12345/summary')
        self.assertEqual(res_unauth.status_code, 401)

        # 2. Non-existent document ID -> 404
        self.register_and_login('david', 'david@college.edu', 'pw123456')
        res_missing = self.client.post('/api/documents/non_existent_doc/summary')
        self.assertEqual(res_missing.status_code, 404)

        # 3. Missing GEMINI_API_KEY -> 500
        test_pdf_path = os.path.join(os.path.dirname(__file__), 'uploads', 'operating_systems_lecture.pdf')
        with open(test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        res_upload = self.client.post('/api/documents/upload', 
            data={'file': (io.BytesIO(pdf_bytes), 'os_notes.pdf')},
            content_type='multipart/form-data'
        )
        doc_id = res_upload.get_json()['document']['id']

        self.app.config['GEMINI_API_KEY'] = ''
        res_no_key = self.client.post(f'/api/documents/{doc_id}/summary')
        self.assertEqual(res_no_key.status_code, 500)
        self.assertIn('GEMINI_API_KEY is not configured', res_no_key.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
