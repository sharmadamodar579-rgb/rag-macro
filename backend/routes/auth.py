import re
from flask import Blueprint, request, jsonify, session
from models.database import db_session, User, get_current_user, login_required, generate_auth_token
from utils.rate_limiter import rate_limit

# Create a Blueprint for authentication routes
auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/status", methods=["GET"])
def auth_status():
    """Endpoint to verify the auth blueprint status."""
    return jsonify({
        "status": "success",
        "message": "Auth route is active and ready."
    }), 200


@auth_bp.route("/register", methods=["POST"])
@rate_limit(max_requests=10, window_seconds=60, key_type="ip")
def register():
    """
    Handle new student account registration.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid request. Expected JSON object with username, email, and password."
        }), 400

    username = str(data.get("username", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    # Input validations
    if not username:
        return jsonify({
            "success": False,
            "error": "Username is required."
        }), 400

    if len(username) < 3:
        return jsonify({
            "success": False,
            "error": "Username must be at least 3 characters long."
        }), 400

    if len(username) > 80:
        return jsonify({
            "success": False,
            "error": "Username cannot exceed 80 characters."
        }), 400

    # Strict email regex validation
    email_regex = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    if not email or not re.match(email_regex, email):
        return jsonify({
            "success": False,
            "error": "A valid email address is required."
        }), 400

    if not password or len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Password must be at least 6 characters long."
        }), 400

    # Check for duplicate email or username
    existing_user_by_email = db_session.query(User).filter_by(email=email).first()
    if existing_user_by_email:
        return jsonify({
            "success": False,
            "error": "An account with this email address already exists."
        }), 400

    existing_user_by_username = db_session.query(User).filter_by(username=username).first()
    if existing_user_by_username:
        return jsonify({
            "success": False,
            "error": "This username is already taken. Please choose another."
        }), 400

    try:
        new_user = User(username=username, email=email)
        new_user.set_password(password)

        db_session.add(new_user)
        db_session.commit()

        # Establish session upon successful registration
        session.clear()
        session["user_id"] = new_user.id
        session.permanent = True

        token = generate_auth_token(new_user.id)

        return jsonify({
            "success": True,
            "message": "Registration successful. Welcome to AI College Study Assistant!",
            "user": new_user.to_dict(),
            "token": token
        }), 201

    except Exception:
        db_session.rollback()
        return jsonify({
            "success": False,
            "error": "An unexpected error occurred during registration. Please try again."
        }), 500


@auth_bp.route("/login", methods=["POST"])
@rate_limit(max_requests=15, window_seconds=60, key_type="ip")
def login():
    """
    Handle student authentication and login.
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "success": False,
            "error": "Invalid request. Expected JSON object with email/username and password."
        }), 400

    identifier = str(data.get("email") or data.get("username") or data.get("username_or_email") or "").strip().lower()
    password = str(data.get("password", ""))

    if not identifier:
        return jsonify({
            "success": False,
            "error": "Please provide your email address or username."
        }), 400

    if not password:
        return jsonify({
            "success": False,
            "error": "Please provide your password."
        }), 400

    # Look up user by email or username
    user = (
        db_session.query(User).filter(
            (User.email == identifier) | (User.username.ilike(identifier))
        ).first()
    )

    if not user or not user.check_password(password):
        return jsonify({
            "success": False,
            "error": "Invalid email/username or password."
        }), 401

    # Store user id in session
    session.clear()
    session["user_id"] = user.id
    session.permanent = True

    token = generate_auth_token(user.id)

    return jsonify({
        "success": True,
        "message": f"Welcome back, {user.username}!",
        "user": user.to_dict(),
        "token": token
    }), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Clear session for the currently authenticated user.
    """
    session.clear()
    return jsonify({
        "success": True,
        "message": "Logged out successfully."
    }), 200


@auth_bp.route("/me", methods=["GET"])
@login_required
def get_current_profile(current_user: User):
    """
    Get profile information of the currently authenticated user.
    """
    return jsonify({
        "success": True,
        "user": current_user.to_dict()
    }), 200
