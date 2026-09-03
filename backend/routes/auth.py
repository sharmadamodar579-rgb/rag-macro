from flask import Blueprint, jsonify

# Create a Blueprint for authentication routes
auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/status", methods=["GET"])
def auth_status():
    """Placeholder endpoint for authentication status."""
    return jsonify({
        "status": "success",
        "message": "Auth route placeholder ready."
    }), 200
