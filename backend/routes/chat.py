from flask import Blueprint, jsonify

# Create a Blueprint for chat & Q&A interaction routes
chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


@chat_bp.route("/status", methods=["GET"])
def chat_status():
    """Placeholder endpoint for chat & Q&A interaction."""
    return jsonify({
        "status": "success",
        "message": "Chat route placeholder ready."
    }), 200
