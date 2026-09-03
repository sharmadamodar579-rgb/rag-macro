# 🎓 AI-Powered College Study Assistant (RAG)

An intelligent study assistant that allows students to upload PDF lecture notes and textbooks, process and index them with FAISS vector search, and interact with the material using Google Gemini AI to ask questions, generate summaries, and practice with MCQs.

---

## 📁 Project Architecture

```
AI-College-Study-Assistant/
│
├── backend/
│   ├── app.py              # Main Flask application entry point & factory
│   ├── config.py           # Configuration management (environment variables)
│   │
│   ├── routes/             # API route controllers (Blueprints)
│   │   ├── auth.py         # Authentication endpoints (future)
│   │   ├── documents.py    # PDF upload & processing endpoints (future)
│   │   └── chat.py         # Q&A / Chat interaction endpoints (future)
│   │
│   ├── services/           # Core business & processing logic
│   │   ├── pdf_service.py  # PyPDF text extraction service (future)
│   │   ├── rag_service.py  # LangChain & FAISS indexing & retrieval (future)
│   │   └── ai_service.py   # Gemini API LLM integration (future)
│   │
│   ├── models/             # Database models & schemas
│   │   └── database.py     # SQLite & SQLAlchemy models (future)
│   │
│   ├── utils/              # Shared helper functions & formatters
│   │   └── helpers.py
│   │
│   ├── uploads/            # Temporary storage for uploaded study PDFs
│   ├── vector_store/       # Local FAISS vector index files
│   └── requirements.txt    # Python package dependencies
│
├── frontend/
│   ├── index.html          # Landing page & API health indicator
│   ├── login.html          # User login & registration interface
│   ├── dashboard.html      # Main study dashboard (Uploads + Q&A workspace)
│   │
│   ├── css/
│   │   └── style.css       # Unified modern dark theme styling
│   │
│   └── js/
│       ├── main.js         # API connectivity & global UI scripts
│       ├── auth.js         # Authentication UI handler
│       └── chat.js         # Study chat & document interaction handler
│
├── .env.example            # Template for environment configuration
├── .env                    # Local environment variables (do NOT commit)
├── .gitignore              # Files and folders excluded from Git
└── README.md               # Project documentation & setup guide
```

---

## 🚀 Quick Start Guide

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

### 3. Configure Environment Variables

Copy `.env.example` to `.env` if not already present:

```bash
cp .env.example .env
```

---

### 4. Run the Flask Development Server

```bash
python backend/app.py
```

The Flask server will start on `http://127.0.0.1:5000/`.

---

### 5. Verify the API

Open your browser or run:
```bash
curl http://127.0.0.1:5000/
```

Response:
```json
{
  "message": "AI College Study Assistant API is running!"
}
```

---

### 6. View the Frontend

Simply open `frontend/index.html` in your web browser. Click the **"Check Backend Status"** button to verify communication with the Flask backend.
