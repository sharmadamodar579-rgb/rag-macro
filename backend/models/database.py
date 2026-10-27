import os
from datetime import datetime, timezone
from functools import wraps
from typing import Optional

from flask import current_app, request, session, jsonify
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    ForeignKey
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    scoped_session,
    relationship
)
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config

Base = declarative_base()

# Global database session placeholder
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False))


def get_utc_now():
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """
    User model representing registered student accounts.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(80), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        """Hash and set the user's password securely using PBKDF2-SHA256."""
        if not password or not password.strip():
            raise ValueError("Password cannot be empty.")
        self.password_hash = generate_password_hash(password.strip(), method="pbkdf2:sha256")

    def check_password(self, password: str) -> bool:
        """Verify the user's password against the stored hash."""
        if not password or not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password.strip())

    def to_dict(self) -> dict:
        """Return public user representation without sensitive fields."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class Document(Base):
    """
    Document model representing an uploaded PDF study material.
    Linked to the user who uploaded it.
    Includes lifecycle tracking (uploaded -> processing -> indexed / failed / ocr_required).
    """
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True)  # e.g., 'doc_abc123'
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(512), nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    uploaded_at = Column(DateTime, default=get_utc_now, nullable=False)
    status = Column(String(32), default="uploaded", nullable=False)  # uploaded, processing, indexed, failed, ocr_required
    processed = Column(Boolean, default=False, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="documents")
    chats = relationship("Chat", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Return document metadata dictionary with full lifecycle status."""
        return {
            "id": self.id,
            "document_id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "file_hash": self.file_hash,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "status": self.status or ("indexed" if self.processed else "uploaded"),
            "processed": self.processed,
            "chunk_count": self.chunk_count,
            "error_message": self.error_message
        }


class Chat(Base):
    """
    Chat conversation session associated with a user and a specific document.
    """
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(String(64), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    user = relationship("User", back_populates="chats")
    document = relationship("Document", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan", order_by="Message.created_at")

    def to_dict(self, include_messages: bool = True) -> dict:
        """Return chat session dictionary."""
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "document_id": self.document_id,
            "document_name": self.document.filename if self.document else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
        if include_messages:
            data["messages"] = [m.to_dict() for m in self.messages]
        return data


class Message(Base):
    """
    Individual message in a chat conversation (user question or AI assistant answer).
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_utc_now, nullable=False)

    # Relationships
    chat = relationship("Chat", back_populates="messages")

    def to_dict(self) -> dict:
        """Return message dictionary."""
        return {
            "id": self.id,
            "chat_id": self.chat_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


def init_db(app=None):
    """
    Initialize SQLite database engine, create all tables, bind scoped session,
    and apply safe backward-compatible schema migrations.
    """
    db_url = getattr(Config, "DATABASE_URL", "sqlite:///models/app.db")
    if app and "DATABASE_URL" in app.config:
        db_url = app.config["DATABASE_URL"]

    # Ensure SQLite directory exists if local file path
    if db_url.startswith("sqlite:///"):
        db_file = db_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(os.path.abspath(db_file))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    engine = create_engine(db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {})
    db_session.remove()
    db_session.configure(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Safe schema migration for existing SQLite databases
    if "sqlite" in db_url:
        with engine.connect() as conn:
            try:
                res = conn.exec_driver_sql("PRAGMA table_info(documents);")
                columns = [row[1] for row in res.fetchall()]
                if columns:
                    if "file_hash" not in columns:
                        conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN file_hash VARCHAR(64);")
                        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_documents_file_hash ON documents (file_hash);")
                    if "status" not in columns:
                        conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN status VARCHAR(32) DEFAULT 'uploaded';")
                    if "error_message" not in columns:
                        conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN error_message TEXT;")
                    conn.commit()
            except Exception:
                pass

    return engine


def _get_serializer() -> URLSafeTimedSerializer:
    """Return an itsdangerous serializer using the current application secret key."""
    secret = current_app.config.get("SECRET_KEY", "study-assistant-secure-salt")
    return URLSafeTimedSerializer(secret_key=secret, salt="auth-token-salt")


def generate_auth_token(user_id: int) -> str:
    """Generate a cryptographically signed auth token for a user."""
    s = _get_serializer()
    return s.dumps({"user_id": user_id})


def verify_auth_token(token: str, max_age: int = 7 * 86400) -> Optional[int]:
    """Verify a signed auth token and extract the user_id."""
    try:
        s = _get_serializer()
        payload = s.loads(token, max_age=max_age)
        return payload.get("user_id")
    except (BadSignature, SignatureExpired, Exception):
        return None


def get_current_user() -> Optional[User]:
    """
    Get the authenticated user.
    Sources supported:
    1. Flask cryptographic session cookie: session['user_id']
    2. Signed Bearer token: Authorization: Bearer <signed-token>

    Insecure, unverified headers (e.g. raw integer user ID) are strictly rejected.
    """
    user_id = session.get("user_id")

    if not user_id:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token:
                user_id = verify_auth_token(token)

    if not user_id:
        return None

    return db_session.get(User, user_id)


def login_required(f):
    """
    Decorator to protect routes requiring user authentication.
    Returns 401 Unauthorized if user is not authenticated.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        current_user = get_current_user()
        if not current_user:
            return jsonify({
                "success": False,
                "error": "Authentication required. Please log in to access this resource."
            }), 401
        return f(*args, current_user=current_user, **kwargs)
    return decorated_function
