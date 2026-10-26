import re
from flask import Blueprint, request, jsonify, current_app
from models.database import db_session, Document, Chat, Message, User, login_required
from services.rag_service import search_similar_chunks
from services.ai_service import generate_rag_answer
from utils.rate_limiter import rate_limit

# Create a Blueprint for chat-related routes
chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@chat_bp.route("/status", methods=["GET"])
def chat_status():
    """Endpoint to verify the chat blueprint status."""
    return jsonify({
        "status": "success",
        "message": "Chat route is active and ready."
    }), 200


@chat_bp.route("", methods=["POST"], strict_slashes=False)
@login_required
@rate_limit(max_requests=20, window_seconds=60, key_type="user")
def chat(current_user: User):
    """
    Handle student questions via the RAG pipeline.
    Enforces user ownership on the selected document and persists conversation history.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid request. Expected JSON object with 'question' field."
        }), 400

    question = data.get("question") or data.get("query") or data.get("message")
    if not question or not isinstance(question, str) or not question.strip():
        return jsonify({
            "success": False,
            "error": "Question is required and cannot be empty."
        }), 400

    question = question.strip()

    document_id = data.get("document_id")
    if document_id is not None:
        if not isinstance(document_id, str) or not document_id.strip() or not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
            return jsonify({
                "success": False,
                "error": "Invalid document ID format. Only alphanumeric characters, hyphens, and underscores are allowed."
            }), 400
        document_id = document_id.strip()

        doc_record = (
            db_session.query(Document)
            .filter_by(id=document_id, user_id=current_user.id)
            .first()
        )
        if not doc_record:
            return jsonify({
                "success": False,
                "error": "Document not found or access denied."
            }), 404
    else:
        doc_record = (
            db_session.query(Document)
            .filter_by(user_id=current_user.id, processed=True)
            .order_by(Document.uploaded_at.desc())
            .first()
        )
        if not doc_record:
            return jsonify({
                "success": False,
                "error": "No uploaded document found. Please upload a PDF study document first."
            }), 404
        document_id = doc_record.id

    gemini_api_key = current_app.config.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        return jsonify({
            "success": False,
            "error": "GEMINI_API_KEY is not configured on the server. Please set it in your .env file."
        }), 500

    k = data.get("k", 4)
    try:
        k = int(k)
        if k <= 0:
            k = 4
        elif k > 20:
            k = 20
    except (ValueError, TypeError):
        k = 4

    embedding_model_name = current_app.config.get("EMBEDDING_MODEL", "models/text-embedding-004")
    gemini_model_name = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")

    try:
        retrieved_chunks = search_similar_chunks(
            document_id=document_id,
            query=question,
            k=k,
            model_name=embedding_model_name,
            api_key=gemini_api_key
        )
    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error": f"No vector store found for document ID '{document_id}'. Please upload and index the document first."
        }), 404
    except ValueError as val_err:
        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 400
    except Exception:
        return jsonify({
            "success": False,
            "error": "Failed to search vector store for relevant context."
        }), 500

    # Extract deduplicated source references
    sources = []
    seen_sources = set()
    for chunk in retrieved_chunks:
        doc_name = chunk.get("filename") or doc_record.filename
        page_num = chunk.get("page", 1)
        source_key = (doc_name, page_num)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "document": doc_name,
                "page": page_num
            })

    # Generate grounded answer via Gemini
    try:
        answer = generate_rag_answer(
            question=question,
            context_chunks=retrieved_chunks,
            model_name=gemini_model_name,
            api_key=gemini_api_key
        )
    except ValueError as val_err:
        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 500
    except RuntimeError:
        return jsonify({
            "success": False,
            "error": "Failed to generate answer from Gemini API. Please check your API key and connection."
        }), 500
    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred while generating the AI answer."
        }), 500

    # Save chat session and messages to database
    try:
        chat_session = (
            db_session.query(Chat)
            .filter_by(user_id=current_user.id, document_id=document_id)
            .order_by(Chat.created_at.desc())
            .first()
        )
        if not chat_session:
            chat_session = Chat(user_id=current_user.id, document_id=document_id)
            db_session.add(chat_session)
            db_session.commit()

        # Record user query
        user_msg = Message(chat_id=chat_session.id, role="user", content=question)
        db_session.add(user_msg)

        # Record assistant answer
        assistant_msg = Message(chat_id=chat_session.id, role="assistant", content=answer)
        db_session.add(assistant_msg)

        db_session.commit()
        chat_id = chat_session.id
    except Exception:
        db_session.rollback()
        chat_id = None

    return jsonify({
        "success": True,
        "answer": answer,
        "sources": sources,
        "chat_id": chat_id
    }), 200


@chat_bp.route("/history", methods=["GET"])
@login_required
def get_chat_history(current_user: User):
    """
    Retrieve past chat conversations and message logs for the current user.
    Optional query parameter: 'document_id' to filter history by document.
    """
    document_id = request.args.get("document_id", "").strip()

    query = db_session.query(Chat).filter_by(user_id=current_user.id)
    if document_id:
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
        query = query.filter_by(document_id=document_id)

    chats = query.order_by(Chat.created_at.desc()).all()
    chat_list = [c.to_dict(include_messages=True) for c in chats]

    return jsonify({
        "success": True,
        "count": len(chats),
        "chats": chat_list,
        "history": chat_list
    }), 200
