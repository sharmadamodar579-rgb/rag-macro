from flask import Blueprint, jsonify

# Create a Blueprint for document management routes (PDF uploads, processing)
documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


@documents_bp.route("/status", methods=["GET"])
def documents_status():
    """Placeholder endpoint for document operations."""
    return jsonify({
        "status": "success",
        "message": "Documents route placeholder ready."
    }), 200
