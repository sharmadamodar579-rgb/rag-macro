import os
from flask import Flask, jsonify
from flask_cors import CORS
from config import Config

# Import route blueprints
from routes.auth import auth_bp
from routes.documents import documents_bp
from routes.chat import chat_bp


def create_app(config_class=Config):
    """
    Application factory function.
    Initializes and configures the Flask application.
    """
    app = Flask(__name__)

    # Load configuration settings from config.py
    app.config.from_object(config_class)

    # Enable Cross-Origin Resource Sharing (CORS)
    # This allows the frontend (running on a different port/origin) to make API requests
    CORS(app)

    # Ensure required folders exist (uploads, vector store)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["VECTOR_STORE_FOLDER"], exist_ok=True)

    # Register blueprints with the application
    app.register_blueprint(auth_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(chat_bp)

    # Root endpoint for health-check / verification
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "message": "AI College Study Assistant API is running!"
        }), 200

    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    # Run the development server
    # Port 5000 is the standard Flask development port
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=app.config.get("DEBUG", True)
    )
