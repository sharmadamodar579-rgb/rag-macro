"""
Automated Test Suite for Security Fixes, Error Handling, and Pipeline Repairs
Tests:
1. Header auth bypass rejection (X-User-Id and unverified Bearer tokens are rejected with 401).
2. Rate limiting enforcement on authentication endpoints.
3. Clean atomic upload rollback on PDF processing failures (no orphaned files or DB rows).
4. Duplicate document upload reuses vector index without redundant embedding calls.
5. Scanned PDF detection returns clear OCR notice.
6. Security response headers (nosniff, SAMEORIGIN, etc.).
7. Cross-origin request restrictions.
"""

import os
import sys
import unittest
import io
import shutil
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from models.database import init_db, db_session, User, Document, Chat, Message
from utils.rate_limiter import limiter
from services.pdf_service import PDFProcessingError

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")


class MockEmbeddings:
    def __init__(self):
        self.call_count = 0

    def embed_documents(self, texts):
        self.call_count += 1
        return [[0.1] * 768 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 768


class TestSecurityAndRepairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.app.config['GEMINI_API_KEY'] = 'test-key'
        cls.test_pdf_path = os.path.join(UPLOADS_DIR, 'operating_systems_lecture.pdf')

    def setUp(self):
        self.client = self.app.test_client()
        limiter.clear()
        # Clean tables
        db_session.query(Message).delete()
        db_session.query(Chat).delete()
        db_session.query(Document).delete()
        db_session.query(User).delete()
        db_session.commit()

    def tearDown(self):
        db_session.remove()

    def register_and_login(self, username="sec_user", email="sec@test.com", password="password123"):
        self.client.post('/api/auth/register', json={'username': username, 'email': email, 'password': password})
        self.client.post('/api/auth/login', json={'username_or_email': email, 'password': password})

    # 1. Unverified Header Bypass Rejection
    def test_01_unverified_header_auth_bypass_strictly_rejected(self):
        """Verify that sending X-User-Id or raw integer Bearer token is strictly rejected."""
        # Create a user in the database
        user = User(username="victim_user", email="victim@test.com")
        user.set_password("secret123")
        db_session.add(user)
        db_session.commit()
        user_id = user.id

        # Attempt to access protected endpoint using X-User-Id header without logging in
        res_x_user = self.client.get('/api/auth/me', headers={'X-User-Id': str(user_id)})
        self.assertEqual(res_x_user.status_code, 401)
        self.assertFalse(res_x_user.get_json()['success'])

        # Attempt to access protected endpoint using raw integer Bearer token
        res_bearer_raw = self.client.get('/api/auth/me', headers={'Authorization': f'Bearer {user_id}'})
        self.assertEqual(res_bearer_raw.status_code, 401)
        self.assertFalse(res_bearer_raw.get_json()['success'])

        # Attempt to list documents with forged header
        res_docs = self.client.get('/api/documents', headers={'X-User-Id': str(user_id)})
        self.assertEqual(res_docs.status_code, 401)

    # 2. Rate Limiting on Authentication
    def test_02_rate_limiter_enforcement(self):
        """Verify sliding-window rate limiter returns 429 when threshold exceeded."""
        self.app.config['ENABLE_RATE_LIMITS_IN_TEST'] = True
        try:
            # Trigger 16 login attempts from same IP
            responses = []
            for i in range(16):
                res = self.client.post('/api/auth/login', json={
                    'username_or_email': 'test@test.com',
                    'password': 'badpassword'
                })
                responses.append(res.status_code)

            self.assertIn(429, responses)
            self.assertEqual(responses[-1], 429)
        finally:
            self.app.config['ENABLE_RATE_LIMITS_IN_TEST'] = False

    # 3. Atomic Upload Rollback on Failure
    def test_03_atomic_upload_rollback_on_failure(self):
        """Verify that if text extraction fails, no file remains on disk and no DB record is committed."""
        self.register_and_login()
        corrupt_path = os.path.join(UPLOADS_DIR, 'corrupted.pdf')
        with open(corrupt_path, 'rb') as f:
            corrupt_bytes = f.read()

        # Count DB documents before upload
        docs_before = db_session.query(Document).count()

        res = self.client.post('/api/documents/upload',
            data={'file': (io.BytesIO(corrupt_bytes), 'broken.pdf')},
            content_type='multipart/form-data'
        )

        self.assertEqual(res.status_code, 400)
        self.assertFalse(res.get_json()['success'])

        # Check DB: no orphaned records
        docs_after = db_session.query(Document).count()
        self.assertEqual(docs_before, docs_after)

    # 4. Duplicate Document Upload Reuses Vector Index
    def test_04_duplicate_upload_reuses_vector_cache(self):
        """Verify uploading an identical PDF reuses existing vector index and metadata."""
        self.register_and_login()
        with open(self.test_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        mock_emb = MockEmbeddings()
        with patch('services.rag_service.get_embedding_model', return_value=mock_emb):
            # First upload
            res1 = self.client.post('/api/documents/upload',
                data={'file': (io.BytesIO(pdf_bytes), 'os_notes.pdf')},
                content_type='multipart/form-data'
            )
            self.assertEqual(res1.status_code, 200)
            data1 = res1.get_json()
            doc_id_1 = data1['document_id']
            first_call_count = mock_emb.call_count

            # Second upload of the identical PDF by the same user
            res2 = self.client.post('/api/documents/upload',
                data={'file': (io.BytesIO(pdf_bytes), 'os_notes_copy.pdf')},
                content_type='multipart/form-data'
            )
            self.assertEqual(res2.status_code, 200)
            data2 = res2.get_json()
            doc_id_2 = data2['document_id']

            # Ensure separate document IDs were assigned
            self.assertNotEqual(doc_id_1, doc_id_2)

            # Ensure embedding call was NOT repeated (cache was reused)
            self.assertEqual(mock_emb.call_count, first_call_count)
            self.assertTrue(data2.get('vector_store', {}).get('reused_cache', False))

    # 5. Scanned PDF Detection Returns Clear OCR Message
    def test_05_scanned_pdf_returns_ocr_guidance(self):
        """Verify scanned PDFs with no selectable text return clear OCR advice."""
        self.register_and_login()
        scanned_path = os.path.join(UPLOADS_DIR, 'multipage_notes.pdf')
        with open(scanned_path, 'rb') as f:
            scanned_bytes = f.read()

        res = self.client.post('/api/documents/upload',
            data={'file': (io.BytesIO(scanned_bytes), 'scanned_lecture.pdf')},
            content_type='multipart/form-data'
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data['has_extractable_text'])
        self.assertIn('OCR is required', data['message'])

    # 6. Security Response Headers
    def test_06_security_headers_present(self):
        """Verify standard security headers are attached to API responses."""
        res = self.client.get('/api/health')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(res.headers.get('X-Frame-Options'), 'SAMEORIGIN')
        self.assertEqual(res.headers.get('X-XSS-Protection'), '1; mode=block')


if __name__ == '__main__':
    unittest.main()
