# AI College Study Assistant — Brain

## 1. Project Overview

We are building an **AI-Powered College Study Assistant using RAG**.

The application allows college students to upload their study material, mainly PDFs such as notes, books, and study material, and ask questions about that content using normal language.

The system retrieves relevant information from the uploaded documents and then uses an AI model to generate the final answer.

The core approach is:

**Retrieval-Augmented Generation (RAG)**

---

## 2. Main Goal

Build a simple, reliable and beginner-friendly web application where a student can:

1. Upload study PDFs.
2. Process the PDF content.
3. Ask questions about the uploaded material.
4. Get answers based on the uploaded material.
5. See the source/reference used for the answer.
6. Generate summaries.
7. Generate important questions.
8. Generate MCQs.
9. Manage their account and study history.

---

## 3. Technology Stack

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- Flask

### Database

- SQLite
- SQLAlchemy / Flask-SQLAlchemy

### RAG

- LangChain
- FAISS

### AI

- Gemini API

### PDF Processing

- PyPDF

---

## 4. Core RAG Pipeline

The most important part of the project is the RAG pipeline.

```text
PDF Upload
    ↓
Read PDF
    ↓
Extract Text
    ↓
Split Text into Chunks
    ↓
Create Embeddings
    ↓
Store Embeddings in FAISS
    ↓
Student Asks Question
    ↓
Convert Question to Embedding
    ↓
Search FAISS
    ↓
Retrieve Relevant Content
    ↓
Send Relevant Content + Question to Gemini
    ↓
Generate Answer
    ↓
Return Answer + Source Reference
```

Do not bypass the retrieval step when answering questions about uploaded documents.

---

## 5. Project Architecture

Use this general structure:

```text
AI-College-Study-Assistant/
│
├── backend/
│   ├── app.py
│   ├── config.py
│   │
│   ├── routes/
│   │   ├── auth.py
│   │   ├── documents.py
│   │   └── chat.py
│   │
│   ├── services/
│   │   ├── pdf_service.py
│   │   ├── rag_service.py
│   │   └── ai_service.py
│   │
│   ├── models/
│   │   └── database.py
│   │
│   ├── utils/
│   │   └── helpers.py
│   │
│   ├── uploads/
│   ├── vector_store/
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── login.html
│   ├── dashboard.html
│   │
│   ├── css/
│   │   └── style.css
│   │
│   └── js/
│       ├── main.js
│       ├── auth.js
│       └── chat.js
│
├── .env
├── .env.example
├── .gitignore
├── README.md
└── brain.md
```

---

## 6. Responsibilities of Important Files

### `backend/app.py`

Main Flask application.

Responsible for:

* Creating the Flask app.
* Registering routes.
* Enabling CORS.
* Loading configuration.
* Starting the server.

Do not put all application logic inside `app.py`.

---

### `backend/config.py`

Contains configuration values.

Possible configuration:

```text
SECRET_KEY
GEMINI_API_KEY
DATABASE_URL
UPLOAD_FOLDER
VECTOR_STORE_FOLDER
```

Secrets must come from environment variables.

Never hard-code API keys.

---

### `routes/auth.py`

Future authentication functionality:

* Registration
* Login
* Logout
* User-related endpoints

---

### `routes/documents.py`

Handles document-related operations:

* PDF upload
* Listing documents
* Document deletion
* Document processing

---

### `routes/chat.py`

Handles questions from the student:

```text
Question
   ↓
RAG service
   ↓
Relevant document content
   ↓
AI service
   ↓
Answer
```

---

### `services/pdf_service.py`

Responsible for PDF processing.

Expected responsibilities:

* Open PDF
* Extract text using PyPDF
* Handle multiple pages
* Clean extracted text
* Prepare text for chunking

Do not put Gemini or FAISS logic here.

---

### `services/rag_service.py`

Responsible for the RAG pipeline.

Expected responsibilities:

* Text splitting
* Embedding creation
* FAISS storage
* Similarity search
* Relevant chunk retrieval

This is the core retrieval layer.

---

### `services/ai_service.py`

Responsible for communication with Gemini.

Expected responsibilities:

* Build AI prompts
* Send retrieved context to Gemini
* Generate answers
* Generate summaries
* Generate important questions
* Generate MCQs

Gemini should receive relevant retrieved content rather than the entire PDF whenever possible.

---

### `models/database.py`

Responsible for database models.

Possible future models:

```text
User
Document
Chat
Message
```

Keep database logic separate from routes.

---

## 7. Current Development Stage

We are building the project **step by step**.

Do NOT implement the entire project at once.

Current priority:

```text
1. Project setup
2. Flask server
3. PDF upload
4. PDF text extraction
5. Text chunking
6. Embeddings
7. FAISS
8. Gemini
9. Complete RAG chat
10. Database/authentication
11. Summary
12. MCQs
13. Important questions
14. Source references
15. Frontend improvements
16. Testing
17. Deployment
```

Only work on the requested/current step unless explicitly asked to continue.

---

## 8. Development Rules

### Rule 1 — Do not rewrite the entire project

When fixing or adding a feature:

* Inspect only the relevant files.
* Change only what is necessary.
* Preserve working functionality.
* Do not randomly restructure the project.

### Rule 2 — Keep code beginner-friendly

The project is also intended for learning.

Prefer:

* Clear variable names
* Simple functions
* Comments for important logic
* Modular code
* Straightforward Flask routes

Avoid unnecessary abstraction.

### Rule 3 — Separate responsibilities

Do not mix:

```text
PDF processing
RAG
Gemini
Database
Authentication
Frontend
```

Keep each responsibility in its appropriate file/service.

### Rule 4 — No hard-coded secrets

Never write:

```python
GEMINI_API_KEY = "actual-key"
```

Use:

```text
.env
```

and environment variables.

### Rule 5 — Test every major step

After implementing a feature:

1. Run the application.
2. Test the relevant endpoint.
3. Check the browser/response.
4. Fix errors before moving to the next step.

---

## 9. PDF Processing Rules

The system should eventually support:

```text
PDF
 ↓
PyPDF
 ↓
Extract text
 ↓
Clean text
 ↓
Split into meaningful chunks
```

Do not send an entire large PDF directly to Gemini for every question.

The purpose of RAG is to retrieve only the relevant information.

---

## 10. RAG Rules

When a student asks:

> What is process scheduling?

The system should:

```text
Question
 ↓
Embedding
 ↓
FAISS similarity search
 ↓
Relevant chunks
 ↓
Gemini
 ↓
Answer
```

The AI should primarily answer using the retrieved document context.

If the uploaded material does not contain enough information, the system should clearly indicate that rather than confidently inventing information.

---

## 11. Source References

Answers should eventually show where the information came from.

For example:

```text
Answer:
Process scheduling is ...

Sources:
- Operating Systems Notes.pdf
- Page 12
```

Source metadata should be preserved during document processing so it can be returned with the retrieved chunks.

---

## 12. Summary Feature

The system should eventually allow:

```text
Uploaded PDF
      ↓
Relevant/full document content
      ↓
Gemini
      ↓
Clear Study Summary
```

The summary should be useful for students and structured into understandable sections.

---

## 13. MCQ Feature

The system should eventually generate:

* Question
* Options
* Correct answer
* Optional explanation

Example:

```text
Q1. Which algorithm is used for ...?

A. ...
B. ...
C. ...
D. ...

Correct Answer: B
```

MCQs should be based on the student's uploaded study material.

---

## 14. Important Questions

The system should eventually generate important/revision questions based on the uploaded material.

Example:

```text
1. Explain process scheduling.
2. What is a PCB?
3. Explain the different scheduling algorithms.
```

These should be generated from the document content.

---

## 15. Frontend

The frontend should remain simple and clean.

Expected pages:

```text
Landing/Login
     ↓
Dashboard
     ↓
Upload Documents
     ↓
Document List
     ↓
AI Chat
```

The dashboard can later include:

* Uploaded documents
* Chat
* Summary
* MCQs
* Important questions
* History

---

## 16. API Design

Keep the backend API organized.

Possible endpoints:

```text
GET  /
POST /api/auth/register
POST /api/auth/login

POST /api/documents/upload
GET  /api/documents
DELETE /api/documents/<id>

POST /api/chat
GET  /api/chat/history

POST /api/documents/<id>/summary
POST /api/documents/<id>/mcqs
POST /api/documents/<id>/questions
```

These are planned endpoints, not requirements to implement immediately.

---

## 17. Error Handling

The application should handle:

* Invalid PDF
* Empty PDF
* PDF with no extractable text
* Large files
* Missing API key
* Gemini API errors
* FAISS errors
* Database errors
* Invalid requests

Return useful error messages.

Do not expose secrets or internal stack traces to normal users.

---

## 18. Important Instruction for Coding AI

Before modifying code:

1. Understand the requested task.
2. Inspect only the relevant files.
3. Identify the existing implementation.
4. Make the smallest necessary change.
5. Do not replace working code unnecessarily.
6. Explain what was changed.
7. Explain how to test it.

Do NOT automatically scan, rewrite, or rebuild the entire project for a small feature or bug fix.

---

## 19. Current Goal

The immediate goal is to create the foundation of the application and then incrementally build the RAG system.

The project should eventually demonstrate how:

```text
AI
+
Natural Language Processing
+
Vector Search
+
RAG
+
Database
+
Web Development
```

can be combined to create a practical college study assistant.

---

## 20. Important Principle

**Build → Test → Fix → Continue**

Never implement five major systems simultaneously.

Each major component should work independently before connecting it to the next component.
