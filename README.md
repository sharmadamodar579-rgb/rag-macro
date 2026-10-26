# 🎓 AI-Powered College Study Assistant (RAG)

An enterprise-grade, secure, and beginner-friendly College Study Assistant web application built with **Retrieval-Augmented Generation (RAG)**. Students can upload study notes and textbooks in PDF format, index them semantically using FAISS vector search, and interact with their materials via Google Gemini AI to ask questions, generate chapter summaries, practice with interactive MCQs, and generate prioritized revision exam questions.

---

## 🌟 Key Features

1. **Secure Multi-User Workspace**:
   - Secure student registration, password hashing (PBKDF2/Werkzeug), and cryptographically signed session management.
   - Strict cross-user data isolation: students can only access, search, and delete their own documents and chat histories.
   - In-memory sliding-window rate limiting on authentication and AI endpoints.
   - Restrictive CORS configuration and standard HTTP security headers (`nosniff`, `SAMEORIGIN`, `X-XSS-Protection`).

2. **Full RAG Pipeline**:
   - **PyPDF Document Parsing**: Robust page-by-page text extraction with support for multi-page documents.
   - **Encrypted & Scanned PDF Handling**: Clear detection of password-protected documents and scanned image-only PDFs with informative OCR guidance.
   - **Text Chunking**: Configurable chunk size and overlap using LangChain's `RecursiveCharacterTextSplitter`.
   - **Vector Embeddings**: Google Generative AI embeddings (`models/text-embedding-004`).
   - **FAISS Vector Store**: On-disk vector indexes saved with path-traversal protection.
   - **Smart Deduplication**: Automatic SHA-256 fingerprinting reuses existing vector indexes for identical files, eliminating duplicate embedding costs.

3. **Intelligent AI Study Tools (Google Gemini)**:
   - **Document-Based Q&A**: Answers strictly grounded in retrieved chunks with citations (document name + page number). Anti-hallucination prompts prevent fabricated answers.
   - **Study Summaries**: Single-pass summary for brief documents and hierarchical map-reduce synthesis for multi-page lecture notes.
   - **Interactive Practice MCQs**: Generates multiple-choice quizzes (5 to 20 questions) with difficulty levels (easy, medium, hard, mixed), instant answer checking, explanations, page references, and score tracking.
   - **Important Exam Questions**: Generates high/medium/low priority exam questions categorised by type (short, long, conceptual, definitions) with collapsible answer keys and clipboard copy.

4. **Modern Dark-Themed Frontend**:
   - Unified responsive UI served directly by Flask or any web server.
   - Dynamic environment API URL resolution without hardcoded localhost endpoints.
   - Interactive tabbed study viewer, document library management, real-time status banners, and quiz player.

---

## 📁 Project Architecture

```text
rag-macro/
├── backend/
│   ├── app.py                      # Flask application factory, routes & error handlers
│   ├── config.py                   # Centralized configuration & environment loader
│   ├── wsgi.py                     # Production WSGI entry point (Waitress / Gunicorn)
│   ├── requirements.txt            # Python dependencies
│   │
│   ├── models/
│   │   ├── database.py             # SQLite models (User, Document, Chat, Message), auth helpers & migrations
│   │   └── app.db                  # Local SQLite database (auto-created on startup)
│   │
│   ├── routes/
│   │   ├── auth.py                 # User registration, login, logout, profile endpoints
│   │   ├── documents.py            # PDF upload, listing, deletion, search, summary, MCQs & questions
│   │   └── chat.py                 # RAG Q&A and conversation history endpoints
│   │
│   ├── services/
│   │   ├── pdf_service.py          # PyPDF extraction, page parsing, encryption & scanned detection
│   │   ├── rag_service.py          # Text chunking, Gemini embeddings & FAISS vector search
│   │   └── ai_service.py           # Gemini LLM prompts, structured JSON validation & synthesis
│   │
│   ├── utils/
│   │   ├── helpers.py              # Shared formatters & response utilities
│   │   └── rate_limiter.py         # Thread-safe in-memory sliding-window rate limiter
│   │
│   ├── uploads/                    # Uploaded study materials and test fixtures
│   └── vector_store/               # FAISS vector indexes (namespaced by document ID)
│
├── frontend/
│   ├── index.html                  # Landing page & backend API health checker
│   ├── login.html                  # Secure student sign-in & registration portal
│   ├── dashboard.html              # Interactive study workspace (Uploads, Chunks, Q&A, Quizzes)
│   ├── css/
│   │   └── style.css               # Modern dark theme styles
│   └── js/
│       ├── auth.js                 # Authentication client logic
│       └── main.js                 # Study workspace, document manager, RAG chat & quiz engine
│
├── .env.example                    # Template environment variables
├── .env                            # Local configuration (never committed to Git)
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10 to 3.14** installed.
- (Optional) **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/) for live AI features.

---

### 2. Installation

#### Windows (PowerShell):
```powershell
# Navigate into the project folder
cd c:\Users\HP\Desktop\rag\rag-macro

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r backend/requirements.txt
```

#### Linux / macOS:
```bash
# Navigate into the project folder
cd /path/to/rag-macro

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required dependencies
pip install -r backend/requirements.txt
```

---

### 3. Environment Configuration

Copy the example configuration to `.env`:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# Linux / macOS
cp .env.example .env
```

Open `.env` and set your configuration:
```ini
# Google Gemini API Key
GEMINI_API_KEY=your_actual_gemini_api_key_here

# AI Models
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004

# Environment Mode
FLASK_ENV=development
FLASK_DEBUG=False

# Cryptographic Secret Key
SECRET_KEY=dev-secret-key-study-assistant-2025

# Allowed CORS Origins
CORS_ORIGINS=http://127.0.0.1:5000,http://localhost:5000
```

> **Note**: For production, generate a secure random secret key with:
> `python -c "import secrets; print(secrets.token_hex(32))"`

---

### 4. Running the Application

#### Option A: Development Server (Flask)
```powershell
python backend/app.py
```
The server will start at `http://127.0.0.1:5000/`.

#### Option B: Production Server (Waitress — Recommended on Windows & Linux)
```powershell
python backend/wsgi.py
```
or via the Waitress CLI:
```powershell
waitress-serve --listen=127.0.0.1:5000 wsgi:app
```

#### Option C: Production Server (Gunicorn — Linux/macOS)
```bash
gunicorn -w 4 -b 127.0.0.1:5000 wsgi:app
```

Once running, navigate to:
- **Landing Page**: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **Login / Register**: [http://127.0.0.1:5000/login.html](http://127.0.0.1:5000/login.html)
- **Study Dashboard**: [http://127.0.0.1:5000/dashboard.html](http://127.0.0.1:5000/dashboard.html)
- **API Health Check**: [http://127.0.0.1:5000/api/health](http://127.0.0.1:5000/api/health)

---

## 🧪 Automated Testing

The project includes an automated test suite with **46 test cases** verifying all components with mocked Gemini calls (zero API costs during testing).

Run the full test suite with `pytest`:
```powershell
python -m pytest backend/ -v
```

Test coverage includes:
- **`test_database_auth.py` (7 tests)**: Registration, login/logout, password hashing, document processing, cross-user isolation, document deletion.
- **`test_rag_pipeline.py` (10 tests)**: PDF text extraction, corrupt/empty PDF handling, text chunking with metadata, FAISS vector indexing, similarity retrieval, grounded prompt construction, hallucination guards.
- **`test_summary_feature.py` (6 tests)**: Single-pass summary, multi-page hierarchical summary, scanned PDF error detection, user isolation.
- **`test_mcq_feature.py` (7 tests)**: MCQ schema verification, difficulty modes, count clamping, malformed JSON recovery, access control.
- **`test_important_questions.py` (10 tests)**: Mixed question generation, priority ranking (high/medium/low), page citations, input sanitization.
- **`test_security_and_repairs.py` (6 tests)**: Rejection of unverified `X-User-Id` and `Bearer` headers, rate limiter enforcement, atomic upload rollbacks, document deduplication, scanned PDF advice, security headers.

---

## 🔒 Security Architecture

1. **Authentication**:
   - Strictly validates cryptographically signed session cookies or signed Bearer tokens.
   - Forged or unverified user ID headers (`X-User-Id` or raw integer tokens) are rejected with HTTP 401.
2. **Access Control**:
   - Database queries explicitly filter by `user_id = current_user.id`.
   - Modifying a `document_id` in a request returns HTTP 404/403.
3. **Storage Security & Path Traversal Prevention**:
   - All uploaded filenames are sanitized with `werkzeug.utils.secure_filename`.
   - Document IDs are strictly validated with `^[a-zA-Z0-9_\-]+$`.
   - Upload and vector store paths verify canonical common prefixes to prevent traversal outside server directories.
4. **Rate Limiting**:
   - Thread-safe sliding-window rate limiting prevents brute force on login/registration and protects expensive AI endpoints.
5. **Data Protection**:
   - Passwords are never stored in plain text (hashed with PBKDF2-SHA256).
   - Deleting a document completely deletes the database record, its chat history, the PDF from disk, and its FAISS index directory.

---

## ❓ Troubleshooting & FAQ

- **"No extractable text found in this PDF"**:
  The document contains scanned images without an embedded OCR text layer. Use an OCR utility to convert scanned documents to searchable PDFs before uploading.
- **"The uploaded PDF is password protected / encrypted"**:
  Remove password encryption before uploading lecture notes.
- **"GEMINI_API_KEY is missing or not configured"**:
  Add your Google Gemini API key to `.env`. The UI and endpoints return clear, helpful guidance when the key is not configured.
- **"Rate limit exceeded. Too many requests"**:
  Wait for the cooldown window (default: 60 seconds) or increase thresholds in `backend/config.py`.
