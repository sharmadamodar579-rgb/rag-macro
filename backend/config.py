import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the backend folder
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the .env file located at the project root
ROOT_DIR = BASE_DIR.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """
    Application configuration class.
    Reads values from environment variables or uses sensible defaults.
    """

    # Secret key used for session security and cryptographic signing
    SECRET_KEY = os.getenv("SECRET_KEY", "default-dev-secret-key-change-in-production")

    # API Key for Google Gemini AI service
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

    # SQLite Database connection URL
    DATABASE_URL = os.getenv(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'models' / 'app.db'}"
    )

    # Directory where uploaded study materials (PDFs) will be stored
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))

    # Directory where FAISS vector index files will be stored
    VECTOR_STORE_FOLDER = os.getenv(
        "VECTOR_STORE_FOLDER", str(BASE_DIR / "vector_store")
    )

    # Flask debug mode flag
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
