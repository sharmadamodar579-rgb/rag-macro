import os
import secrets
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the backend folder
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from potential .env file locations
ROOT_DIR = BASE_DIR.parent
for env_candidate in [ROOT_DIR / ".env", BASE_DIR / ".env", ROOT_DIR.parent / ".env"]:
    if env_candidate.exists():
        load_dotenv(dotenv_path=env_candidate)
load_dotenv()  # Fallback to standard search


class Config:
    """
    Application configuration class.
    Enforces security defaults and reads configuration from environment variables.
    """

    # Flask environment mode
    ENV = os.getenv("FLASK_ENV", "development").lower()
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    TESTING = False

    # Secret key validation & enforcement
    _raw_secret = os.getenv("SECRET_KEY", "")
    _DEFAULT_INSECURE_KEYS = {
        "",
        "default-dev-secret-key-change-in-production",
        "dev-secret-key-study-assistant-2025",
        "change-me",
        "secret"
    }

    if not DEBUG and ENV == "production" and _raw_secret in _DEFAULT_INSECURE_KEYS:
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: A strong, unique SECRET_KEY must be "
            "set in the environment (.env) when running in production mode."
        )

    # In dev mode, generate a fallback random secret if none or default is provided
    SECRET_KEY = _raw_secret if (_raw_secret and _raw_secret not in _DEFAULT_INSECURE_KEYS) else (
        _raw_secret or secrets.token_hex(32)
    )

    # Cookie Security Settings
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "False").lower() in ("true", "1", "t")
    PERMANENT_SESSION_LIFETIME = int(os.getenv("SESSION_LIFETIME_SECONDS", 7 * 86400))

    # Allowed CORS Origins (comma-separated string in .env)
    _raw_cors = os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5000,http://localhost:5000,http://127.0.0.1:5500,http://localhost:5500"
    )
    CORS_ORIGINS = [orig.strip() for orig in _raw_cors.split(",") if orig.strip()]

    # AI Provider Configuration: 'openrouter' or 'gemini'
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    _raw_gemini_key = os.getenv("GEMINI_API_KEY", "")

    # If an OpenRouter key (sk-or-...) is placed in GEMINI_API_KEY, auto-assign it
    if _raw_gemini_key.startswith("sk-or-") and not OPENROUTER_API_KEY:
        OPENROUTER_API_KEY = _raw_gemini_key

    # Active AI key for backward-compatible route access
    GEMINI_API_KEY = _raw_gemini_key or OPENROUTER_API_KEY

    # Auto-detect AI provider if not explicitly configured
    _configured_provider = os.getenv("AI_PROVIDER", "").strip().lower()
    if _configured_provider:
        AI_PROVIDER = _configured_provider
    elif OPENROUTER_API_KEY:
        AI_PROVIDER = "openrouter"
    else:
        AI_PROVIDER = "gemini"

    # OpenRouter Model Configuration (Default strictly to free router: openrouter/free)
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    # Gemini model for text generation / chat
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # SQLite Database connection URL (auto-switches to /tmp on serverless like Vercel)
    _default_db = "sqlite:////tmp/app.db" if os.getenv("VERCEL") else f"sqlite:///{BASE_DIR / 'models' / 'app.db'}"
    DATABASE_URL = os.getenv("DATABASE_URL", _default_db)

    # Storage paths (strictly contained; points to /tmp on serverless like Vercel)
    _default_upload = "/tmp/uploads" if os.getenv("VERCEL") else str(BASE_DIR / "uploads")
    _default_vectors = "/tmp/vector_store" if os.getenv("VERCEL") else str(BASE_DIR / "vector_store")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", _default_upload)
    VECTOR_STORE_FOLDER = os.getenv("VECTOR_STORE_FOLDER", _default_vectors)

    # Upload validation configuration
    ALLOWED_EXTENSIONS = {"pdf"}
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))

    # Text Chunking configuration (for RAG preparation)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

    # Embedding Model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
