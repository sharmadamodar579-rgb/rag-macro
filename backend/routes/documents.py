import os
import re
import uuid
import shutil
import hashlib
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from models.database import db_session, Document, User, login_required
from services.pdf_service import extract_pdf_text, PDFProcessingError
from services.rag_service import (
    chunk_document_pages,
    create_vector_store,
    search_similar_chunks,
    get_document_vector_store_path
)
from services.ai_service import (
    generate_study_summary,
    generate_document_mcqs,
    generate_important_questions
)
from utils.rate_limiter import rate_limit

# Create a Blueprint for document-related routes
documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


def is_allowed_file(filename: str) -> bool:
    """Validate that the file extension is strictly .pdf."""
    return bool(filename and "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"})


@documents_bp.route("/status", methods=["GET"])
def documents_status():
    """Endpoint to verify the documents blueprint status."""
    return jsonify({
        "status": "success",
        "message": "Documents route is active and ready."
    }), 200


@documents_bp.route("", methods=["GET"])
@login_required
def list_documents(current_user: User):
    """
    List all documents owned by the currently authenticated user.
    """
    docs = (
        db_session.query(Document)
        .filter_by(user_id=current_user.id)
        .order_by(Document.uploaded_at.desc())
        .all()
    )
    return jsonify({
        "success": True,
        "count": len(docs),
        "documents": [d.to_dict() for d in docs]
    }), 200


@documents_bp.route("/<document_id>", methods=["GET"])
@login_required
def get_document(current_user: User, document_id: str):
    """
    Retrieve metadata for a specific document owned by the current user.
    """
    if not document_id or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        return jsonify({
            "success": False,
            "error": "Invalid document ID format."
        }), 400

    doc = (
        db_session.query(Document)
        .filter_by(id=document_id, user_id=current_user.id)
        .first()
    )
    if not doc:
        return jsonify({
            "success": False,
            "error": "Document not found or access denied."
        }), 404

    return jsonify({
        "success": True,
        "document": doc.to_dict()
    }), 200


@documents_bp.route("/upload", methods=["POST"])
@login_required
@rate_limit(max_requests=20, window_seconds=60, key_type="user")
def upload_document(current_user: User):
    """
    Handle PDF file upload, text extraction, chunking, and FAISS vector indexing.
    Guarantees atomic rollback on processing errors (no orphaned files or records).
    Avoids duplicate embedding calls when an identical file was already indexed by this user.
    """
    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file part found in the request. Please provide a file with field name 'file'."
        }), 400

    file = request.files["file"]

    if not file or not file.filename:
        return jsonify({
            "success": False,
            "error": "No file selected. Please choose a PDF file to upload."
        }), 400

    if not is_allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Only PDF documents (.pdf) are supported."
        }), 400

    # Sanitize filename
    raw_filename = file.filename
    filename = secure_filename(raw_filename)
    if not filename:
        filename = "document.pdf"

    upload_folder = current_app.config.get("UPLOAD_FOLDER")
    if not upload_folder:
        return jsonify({
            "success": False,
            "error": "Upload folder is not configured on the server."
        }), 500

    os.makedirs(upload_folder, exist_ok=True)

    document_id = f"doc_{uuid.uuid4().hex[:10]}"
    saved_file_name = f"{document_id}_{filename}"
    saved_file_path = os.path.join(upload_folder, saved_file_name)

    doc_record = None

    try:
        # Read file contents and compute SHA-256 for duplicate detection
        file_bytes = file.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Save to disk
        with open(saved_file_path, "wb") as f:
            f.write(file_bytes)

        # Create database record
        doc_record = Document(
            id=document_id,
            user_id=current_user.id,
            filename=filename,
            filepath=saved_file_path,
            file_hash=file_hash,
            processed=False,
            chunk_count=0
        )
        db_session.add(doc_record)
        db_session.commit()

        # Extract text using pdf_service
        extraction_result = extract_pdf_text(saved_file_path)

        # Chunk pages
        chunk_size = current_app.config.get("CHUNK_SIZE", 1000)
        chunk_overlap = current_app.config.get("CHUNK_OVERLAP", 150)

        chunks = chunk_document_pages(
            pages=extraction_result["pages"],
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        total_chunks = len(chunks)

        # FAISS Vector Indexing with Deduplication Optimization
        gemini_api_key = current_app.config.get("GEMINI_API_KEY", "")
        embedding_model_name = current_app.config.get("EMBEDDING_MODEL", "models/text-embedding-004")
        vector_store_info = {}
        embedding_info = {}

        # Check if identical file was previously indexed by this user
        existing_doc = (
            db_session.query(Document)
            .filter_by(user_id=current_user.id, file_hash=file_hash, processed=True)
            .filter(Document.id != document_id)
            .first()
        )

        ai_api_key = current_app.config.get("OPENROUTER_API_KEY") or current_app.config.get("GEMINI_API_KEY", "")
        reused_cache = False
        if existing_doc and chunks:
            try:
                existing_store = get_document_vector_store_path(existing_doc.id)
                new_store = get_document_vector_store_path(document_id)
                if os.path.exists(os.path.join(existing_store, "index.faiss")) and os.path.exists(os.path.join(existing_store, "index.pkl")):
                    from services.rag_service import load_vector_store, get_embedding_model
                    from datetime import datetime, timezone
                    import json

                    # Load existing vector store without repeating embedding calculation
                    existing_vs = load_vector_store(
                        document_id=existing_doc.id,
                        model_name=embedding_model_name,
                        api_key=ai_api_key
                    )
                    # Refresh all chunk metadata to point to current document ID and current filename
                    for chunk_doc in existing_vs.docstore._dict.values():
                        chunk_doc.metadata["document_id"] = document_id
                        chunk_doc.metadata["filename"] = filename

                    # Save updated vector store to current document's path with metadata
                    os.makedirs(new_store, exist_ok=True)
                    existing_vs.save_local(new_store)

                    meta_info = {
                        "version": "2.0.0",
                        "document_id": document_id,
                        "dimension": getattr(get_embedding_model(model_name=embedding_model_name, api_key=ai_api_key), "dim", 768),
                        "model_name": embedding_model_name,
                        "chunk_count": total_chunks,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "reused_cache": True
                    }
                    with open(os.path.join(new_store, "index_meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta_info, f, indent=2)

                    reused_cache = True
                    vector_store_info = {
                        "created": True,
                        "document_id": document_id,
                        "vectors": total_chunks,
                        "model": embedding_model_name,
                        "reused_cache": True
                    }
                    embedding_info = {
                        "success": True,
                        "model": embedding_model_name,
                        "dimension": 768,
                        "total_embedded": total_chunks,
                        "reused_cache": True
                    }
            except Exception:
                reused_cache = False

        if not reused_cache:
            if chunks and gemini_api_key:
                try:
                    store_result = create_vector_store(
                        chunks=chunks,
                        document_id=document_id,
                        model_name=embedding_model_name,
                        api_key=gemini_api_key
                    )
                    vector_store_info = {
                        "created": True,
                        "document_id": document_id,
                        "vectors": store_result["vectors"],
                        "model": embedding_model_name
                    }
                    embedding_info = {
                        "success": True,
                        "model": embedding_model_name,
                        "dimension": 768,
                        "total_embedded": total_chunks
                    }
                except Exception as store_err:
                    vector_store_info = {
                        "created": False,
                        "document_id": document_id,
                        "error": str(store_err)
                    }
                    embedding_info = {
                        "success": False,
                        "model": embedding_model_name,
                        "error": str(store_err)
                    }
            else:
                vector_store_info = {
                    "created": False,
                    "document_id": document_id,
                    "message": "GEMINI_API_KEY not set in .env. FAISS index creation skipped."
                }
                embedding_info = {
                    "success": False,
                    "model": embedding_model_name,
                    "message": "GEMINI_API_KEY not set in .env. Embeddings skipped."
                }

        # Update record lifecycle status
        if not extraction_result["has_extractable_text"]:
            doc_record.status = "ocr_required"
            doc_record.processed = False
            doc_record.chunk_count = 0
            doc_record.error_message = extraction_result.get("message", "Scanned or image-only PDF requires OCR.")
            status_message = (
                "No extractable text was found in this PDF. It appears to be a scanned document or "
                "image-only PDF. OCR is required to process images."
            )
        elif vector_store_info.get("created"):
            doc_record.status = "indexed"
            doc_record.processed = True
            doc_record.chunk_count = total_chunks
            doc_record.error_message = None
            status_message = (
                f"Successfully extracted {extraction_result['total_pages']} page(s), "
                f"generated {total_chunks} chunk(s), and indexed in FAISS."
            )
        else:
            doc_record.status = "failed" if vector_store_info.get("error") else "processing"
            doc_record.processed = False
            doc_record.chunk_count = total_chunks
            doc_record.error_message = vector_store_info.get("error") or vector_store_info.get("message")
            status_message = (
                f"Successfully extracted {extraction_result['total_pages']} page(s) "
                f"and generated {total_chunks} text chunk(s)."
            )
        db_session.commit()

        return jsonify({
            "success": True,
            "document": doc_record.to_dict(),
            "document_id": document_id,
            "filename": filename,
            "pages": extraction_result["total_pages"],
            "total_pages": extraction_result["total_pages"],
            "total_chunks": total_chunks,
            "total_characters": extraction_result["total_characters"],
            "total_words": extraction_result["total_words"],
            "has_extractable_text": extraction_result["has_extractable_text"],
            "message": status_message,
            "chunk_config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            },
            "vector_store": vector_store_info,
            "embedding": embedding_info,
            "chunks": chunks,
            "extracted_text": extraction_result["full_text"]
        }), 200

    except PDFProcessingError as pdf_err:
        # Atomic cleanup on failure
        if os.path.exists(saved_file_path):
            try:
                os.remove(saved_file_path)
            except OSError:
                pass
        if doc_record:
            db_session.delete(doc_record)
            db_session.commit()
        return jsonify({
            "success": False,
            "error": str(pdf_err)
        }), 400

    except Exception:
        # Atomic cleanup on unexpected failure
        if os.path.exists(saved_file_path):
            try:
                os.remove(saved_file_path)
            except OSError:
                pass
        if doc_record:
            db_session.delete(doc_record)
            db_session.commit()
        else:
            db_session.rollback()
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while processing the PDF file. Please try again."
        }), 500


@documents_bp.route("/<document_id>", methods=["DELETE"])
@login_required
def delete_document(current_user: User, document_id: str):
    """
    Delete a document owned by the current user:
    - Removes database record (cascading to chats & messages)
    - Removes PDF file from uploads directory
    - Removes FAISS index folder from disk
    """
    if not document_id or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        return jsonify({
            "success": False,
            "error": "Invalid document ID format."
        }), 400

    doc = (
        db_session.query(Document)
        .filter_by(id=document_id, user_id=current_user.id)
        .first()
    )
    if not doc:
        return jsonify({
            "success": False,
            "error": "Document not found or access denied."
        }), 404

    try:
        # Delete on-disk PDF file
        if doc.filepath and os.path.exists(doc.filepath):
            try:
                os.remove(doc.filepath)
            except OSError:
                pass

        # Delete on-disk FAISS vector store directory
        try:
            vector_store_path = get_document_vector_store_path(document_id)
            if os.path.exists(vector_store_path):
                shutil.rmtree(vector_store_path, ignore_errors=True)
        except Exception:
            pass

        # Delete database record
        db_session.delete(doc)
        db_session.commit()

        return jsonify({
            "success": True,
            "message": f"Document '{doc.filename}' and its vector index have been deleted."
        }), 200

    except Exception:
        db_session.rollback()
        return jsonify({
            "success": False,
            "error": "Failed to delete document from database."
        }), 500


@documents_bp.route("/<document_id>/search", methods=["POST"])
@login_required
def search_document(current_user: User, document_id: str):
    """
    Perform semantic similarity search on a document's persistent FAISS index.
    Verifies user ownership before searching.
    """
    if not document_id or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        return jsonify({
            "success": False,
            "error": "Invalid document ID format."
        }), 400

    doc = (
        db_session.query(Document)
        .filter_by(id=document_id, user_id=current_user.id)
        .first()
    )
    if not doc:
        return jsonify({
            "success": False,
            "error": "Document not found or access denied."
        }), 404

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid request. Expected JSON object with 'query' field."
        }), 400

    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({
            "success": False,
            "error": "Search query cannot be empty. Please provide a question."
        }), 400

    k = data.get("k", 4)
    try:
        k = int(k)
        if k <= 0:
            k = 4
        elif k > 20:
            k = 20
    except (ValueError, TypeError):
        k = 4

    gemini_api_key = current_app.config.get("GEMINI_API_KEY", "")
    embedding_model_name = current_app.config.get("EMBEDDING_MODEL", "models/text-embedding-004")

    if not gemini_api_key:
        return jsonify({
            "success": False,
            "error": "GEMINI_API_KEY is not configured on the server (set OPENROUTER_API_KEY or GEMINI_API_KEY in .env)."
        }), 400

    try:
        results = search_similar_chunks(
            document_id=document_id,
            query=query,
            k=k,
            model_name=embedding_model_name,
            api_key=gemini_api_key
        )

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": doc.filename,
            "query": query,
            "k": k,
            "count": len(results),
            "results": results
        }), 200

    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": f"No vector store index found for document '{doc.filename}'. Please re-upload and process the document."
        }), 404
    except ValueError as val_err:
        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 400
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while querying the vector store."
        }), 500


@documents_bp.route("/<document_id>/summary", methods=["POST"])
@login_required
@rate_limit(max_requests=10, window_seconds=60, key_type="user")
def summarize_document(current_user: User, document_id: str):
    """
    Generate a structured study summary for an uploaded PDF document.
    """
    if not document_id or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        return jsonify({
            "success": False,
            "error": "Invalid document ID format."
        }), 400

    doc = (
        db_session.query(Document)
        .filter_by(id=document_id, user_id=current_user.id)
        .first()
    )
    if not doc:
        return jsonify({
            "success": False,
            "error": "Document not found or access denied."
        }), 404

    if not doc.filepath or not os.path.exists(doc.filepath):
        return jsonify({
            "success": False,
            "error": "Document PDF file not found on server storage."
        }), 404

    gemini_api_key = current_app.config.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        return jsonify({
            "success": False,
            "error": "GEMINI_API_KEY is not configured on the server (set OPENROUTER_API_KEY or GEMINI_API_KEY in .env)."
        }), 500

    gemini_model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        extraction_result = extract_pdf_text(doc.filepath)
    except PDFProcessingError as pdf_err:
        return jsonify({
            "success": False,
            "error": f"Failed to process PDF for summarization: {str(pdf_err)}"
        }), 400
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while reading the PDF file."
        }), 500

    if not extraction_result.get("has_extractable_text") or extraction_result.get("total_characters", 0) == 0:
        return jsonify({
            "success": False,
            "error": "No extractable text found in this PDF document. Summaries cannot be generated from empty or scanned image PDFs without OCR."
        }), 400

    try:
        summary_result = generate_study_summary(
            pages=extraction_result["pages"],
            filename=doc.filename,
            model_name=gemini_model_name,
            api_key=gemini_api_key
        )

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": doc.filename,
            "summary": summary_result["summary"],
            "sources": summary_result["sources"],
            "strategy": summary_result["strategy"],
            "total_pages": summary_result["total_pages"],
            "total_characters": summary_result["total_characters"],
            "message": f"Successfully generated study summary for '{doc.filename}'."
        }), 200

    except ValueError as val_err:
        err_msg = str(val_err)
        status = 500 if "api key" in err_msg.lower() else 400
        return jsonify({
            "success": False,
            "error": err_msg
        }), status
    except RuntimeError:
        return jsonify({
            "success": False,
            "error": "Failed to generate summary from Gemini API. Please check your API key and connection."
        }), 500
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while generating the study summary."
        }), 500


@documents_bp.route("/<document_id>/mcqs", methods=["POST"])
@login_required
@rate_limit(max_requests=10, window_seconds=60, key_type="user")
def generate_mcqs_for_document(current_user: User, document_id: str):
    """
    Generate multiple-choice practice questions (MCQs) for an uploaded PDF document.
    """
    if not document_id or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        return jsonify({
            "success": False,
            "error": "Invalid document ID format."
        }), 400

    doc = (
        db_session.query(Document)
        .filter_by(id=document_id, user_id=current_user.id)
        .first()
    )
    if not doc:
        return jsonify({
            "success": False,
            "error": "Document not found or access denied."
        }), 404

    if not doc.filepath or not os.path.exists(doc.filepath):
        return jsonify({
            "success": False,
            "error": "Document PDF file not found on server storage."
        }), 404

    req_data = request.get_json(silent=True) or {}

    count = req_data.get("count", 5)
    try:
        count = int(count)
        if count <= 0:
            count = 5
        elif count > 20:
            count = 20
    except (ValueError, TypeError):
        count = 5

    difficulty = str(req_data.get("difficulty", "medium")).strip().lower()
    if difficulty not in {"easy", "medium", "hard", "mixed"}:
        difficulty = "medium"

    gemini_api_key = current_app.config.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        return jsonify({
            "success": False,
            "error": "GEMINI_API_KEY is not configured on the server (set OPENROUTER_API_KEY or GEMINI_API_KEY in .env)."
        }), 500

    gemini_model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        extraction_result = extract_pdf_text(doc.filepath)
    except PDFProcessingError as pdf_err:
        return jsonify({
            "success": False,
            "error": f"Failed to process PDF for MCQ generation: {str(pdf_err)}"
        }), 400
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while reading the PDF file."
        }), 500

    if not extraction_result.get("has_extractable_text") or extraction_result.get("total_characters", 0) == 0:
        return jsonify({
            "success": False,
            "error": "No extractable text found in this PDF document. Questions cannot be generated from empty or scanned image PDFs without OCR."
        }), 400

    try:
        mcq_result = generate_document_mcqs(
            pages=extraction_result["pages"],
            filename=doc.filename,
            count=count,
            difficulty=difficulty,
            model_name=gemini_model_name,
            api_key=gemini_api_key
        )

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": doc.filename,
            "count": mcq_result["count"],
            "difficulty": mcq_result["difficulty"],
            "questions": mcq_result["questions"],
            "message": f"Successfully generated {mcq_result['count']} practice question(s) for '{doc.filename}'."
        }), 200

    except ValueError as val_err:
        err_msg = str(val_err)
        status = 500 if "api key" in err_msg.lower() else 400
        return jsonify({
            "success": False,
            "error": err_msg
        }), status
    except RuntimeError:
        return jsonify({
            "success": False,
            "error": "Failed to generate MCQs from Gemini API. Please check your API key and connection."
        }), 500
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while generating MCQs."
        }), 500


@documents_bp.route("/<document_id>/questions", methods=["POST"])
@login_required
@rate_limit(max_requests=10, window_seconds=60, key_type="user")
def generate_questions_for_document(current_user: User, document_id: str):
    """
    Generate exam-oriented important revision questions for an uploaded PDF document.
    """
    if not document_id or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        return jsonify({
            "success": False,
            "error": "Invalid document ID format."
        }), 400

    doc = (
        db_session.query(Document)
        .filter_by(id=document_id, user_id=current_user.id)
        .first()
    )
    if not doc:
        return jsonify({
            "success": False,
            "error": "Document not found or access denied."
        }), 404

    if not doc.filepath or not os.path.exists(doc.filepath):
        return jsonify({
            "success": False,
            "error": "Document PDF file not found on server storage."
        }), 404

    req_data = request.get_json(silent=True) or {}

    count = req_data.get("count", 10)
    try:
        count = int(count)
        if count < 1:
            count = 1
        elif count > 20:
            count = 20
    except (ValueError, TypeError):
        count = 10

    question_type = str(req_data.get("type", "mixed")).strip().lower()
    if question_type not in {"short", "long", "conceptual", "definition", "mixed"}:
        question_type = "mixed"

    gemini_api_key = current_app.config.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        return jsonify({
            "success": False,
            "error": "GEMINI_API_KEY is not configured on the server (set OPENROUTER_API_KEY or GEMINI_API_KEY in .env)."
        }), 500

    gemini_model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        extraction_result = extract_pdf_text(doc.filepath)
    except PDFProcessingError as pdf_err:
        return jsonify({
            "success": False,
            "error": f"Failed to process PDF for question generation: {str(pdf_err)}"
        }), 400
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while reading the PDF file."
        }), 500

    if not extraction_result.get("has_extractable_text") or extraction_result.get("total_characters", 0) == 0:
        return jsonify({
            "success": False,
            "error": "No extractable text found in this PDF document. Questions cannot be generated from empty or scanned image PDFs without OCR."
        }), 400

    try:
        questions_result = generate_important_questions(
            pages=extraction_result["pages"],
            filename=doc.filename,
            count=count,
            question_type=question_type,
            model_name=gemini_model_name,
            api_key=gemini_api_key
        )

        return jsonify({
            "success": True,
            "document_id": document_id,
            "filename": doc.filename,
            "count": questions_result["count"],
            "total_questions": questions_result["count"],
            "requested_count": count,
            "type": questions_result["type"],
            "questions": questions_result["questions"],
            "message": f"Successfully generated {questions_result['count']} important question(s) for '{doc.filename}'."
        }), 200

    except ValueError as val_err:
        err_msg = str(val_err)
        status = 500 if "api key" in err_msg.lower() else 400
        return jsonify({
            "success": False,
            "error": err_msg
        }), status
    except RuntimeError:
        return jsonify({
            "success": False,
            "error": "Failed to generate questions from Gemini API. Please check your API key and connection."
        }), 500
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while generating important questions."
        }), 500
