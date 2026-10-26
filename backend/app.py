import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from config import Config

from models.database import init_db, db_session

# Import route blueprints
from routes.auth import auth_bp
from routes.documents import documents_bp
from routes.chat import chat_bp


def create_app(config_class=Config):
    """
    Application factory function.
    Initializes and configures the Flask application, database, blueprints, and static assets.
    """
    # Resolve frontend directory path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.abspath(os.path.join(backend_dir, "..", "frontend"))

    app = Flask(
        __name__,
        static_folder=frontend_dir,
        static_url_path=""
    )

    # Load configuration settings
    app.config.from_object(config_class)

    # Configure CORS restricted to configured origins with credentials support
    allowed_origins = app.config.get("CORS_ORIGINS", ["http://127.0.0.1:5000", "http://localhost:5000"])
    CORS(app, origins=allowed_origins, supports_credentials=True)

    # Ensure required folders exist (uploads, vector store)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["VECTOR_STORE_FOLDER"], exist_ok=True)

    # Initialize SQLite database and tables
    init_db(app)

    # Register session removal at end of request
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()

    # Security Headers Middleware
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

    # Register blueprints with the application
    app.register_blueprint(auth_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(chat_bp)

    # Root endpoint: serves index.html for browsers or JSON for API clients
    @app.route("/", methods=["GET"])
    def index():
        accept_header = request.headers.get("Accept", "")
        if "text/html" in accept_header and os.path.exists(os.path.join(frontend_dir, "index.html")):
            return send_from_directory(frontend_dir, "index.html")
        return jsonify({
            "status": "success",
            "message": "AI College Study Assistant API is running!"
        }), 200

    # API Health check endpoint
    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy",
            "message": "AI College Study Assistant backend is operational.",
            "version": "1.0.0"
        }), 200

    # Static file routes for frontend HTML pages
    @app.route("/login.html", methods=["GET"])
    def serve_login():
        return send_from_directory(frontend_dir, "login.html")

    @app.route("/dashboard.html", methods=["GET"])
    def serve_dashboard():
        return send_from_directory(frontend_dir, "dashboard.html")

    # Custom error handlers without exposing internal stack traces
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "success": False,
            "error": "The uploaded file is too large. Maximum allowed size is 16 MB."
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "The requested endpoint or resource was not found."
        }), 404

    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return jsonify({
            "success": False,
            "error": "Too many requests. Please slow down and try again later."
        }), 429

    @app.errorhandler(500)
    def internal_server_error(error):
        return jsonify({
            "success": False,
            "error": "An internal server error occurred. Please try again later."
        }), 500

    return app


# Create default application instance
app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=app.config.get("DEBUG", False)
    )
