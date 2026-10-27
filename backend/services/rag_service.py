"""
RAG Service: Document Chunking, Deterministic Offline Embeddings, FAISS Vector Store,
and Semantic Retrieval Utilities.
"""

import os
import re
import json
import shutil
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from config import Config

# Version identifier for persisted vector indexes
INDEX_VERSION = "2.0.0"


class IncompatibleIndexError(Exception):
    """Raised when an on-disk FAISS index was built with an incompatible model, version, or dimension."""
    pass


def _deterministic_hash(s: str) -> int:
    """
    Compute a stable, deterministic 64-bit integer hash from a text string.
    Unlike Python's built-in hash(), this uses SHA-256 and is completely invariant
    across Python process restarts, PYTHONHASHSEED randomization, and OS platforms.
    """
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


# ----------------------------------------------------------------------
# 1. Text Splitter & Chunking Utilities
# ----------------------------------------------------------------------

def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create and return a RecursiveCharacterTextSplitter configured with given
    chunk size and overlap parameters.
    """
    size = chunk_size or getattr(Config, "CHUNK_SIZE", 1000)
    overlap = chunk_overlap or getattr(Config, "CHUNK_OVERLAP", 150)

    if size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({size})"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )


def split_text_into_chunks(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[str]:
    """
    Split a raw string of text into chunks.
    """
    if not text or not text.strip():
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    filename: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Chunk extracted PDF pages while preserving page number and document metadata.
    """
    if not pages:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Dict[str, Any]] = []
    global_chunk_index = 0

    for page_dict in pages:
        page_num = page_dict.get("page_number", 1)
        page_text = page_dict.get("text", "")

        if not page_text or not page_text.strip():
            continue

        raw_chunks = splitter.split_text(page_text)

        for chunk_text in raw_chunks:
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            all_chunks.append({
                "chunk_index": global_chunk_index,
                "page": page_num,
                "text": cleaned_chunk,
                "char_count": len(cleaned_chunk),
                "word_count": len(cleaned_chunk.split()),
                "filename": filename
            })
            global_chunk_index += 1

    return all_chunks


# ----------------------------------------------------------------------
# 2. Embedding Model Utilities
# ----------------------------------------------------------------------

class LocalOfflineEmbeddings(Embeddings):
    """
    Deterministic offline term-frequency and character/word n-gram hash embeddings.
    Zero external API calls, 100% free, deterministic across process restarts,
    compatible with FAISS vector indexing.
    """
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            vec[0] = 1.0
            return vec.tolist()
        for w in words:
            h = _deterministic_hash(w) % self.dim
            vec[h] += 1.0
        for i in range(len(words) - 1):
            h2 = _deterministic_hash(f"{words[i]}_{words[i + 1]}") % self.dim
            vec[h2] += 0.5
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class DeterministicTestEmbeddings:
    """Offline, deterministic mock embeddings for automated unit testing."""
    def __init__(self, dim: int = 768):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        val = float((sum(ord(c) for c in text[:20]) % 100) + 1) / 100.0
        vec = [val] * self.dim
        norm = (len(vec) * (val ** 2)) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embedding_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
):
    """
    Initialize and return an embedding model:
    - DeterministicTestEmbeddings if a mock/test key is provided
    - LocalOfflineEmbeddings if OpenRouter key is provided or local mode
    - LangChain GoogleGenerativeAIEmbeddings client if Gemini key is provided
    """
    if api_key is not None:
        key = api_key
    else:
        key = getattr(Config, "OPENROUTER_API_KEY", "") or getattr(Config, "GEMINI_API_KEY", "")

    if not key or not key.strip():
        raise ValueError(
            "AI API key is missing or not configured. "
            "Please set your OPENROUTER_API_KEY or GEMINI_API_KEY in the .env file."
        )

    # Use offline deterministic mock for automated testing keys
    if key in ("test-gemini-key", "fake-key", "test-key", "fake-gemini-key") or key.startswith("test-"):
        return DeterministicTestEmbeddings()

    # If OpenRouter is the active provider or key is an OpenRouter key, use 100% free local embeddings
    provider = getattr(Config, "AI_PROVIDER", "gemini").lower()
    if provider == "openrouter" or key.startswith("sk-or-"):
        return LocalOfflineEmbeddings()

    selected_model = model_name or getattr(Config, "EMBEDDING_MODEL", "models/text-embedding-004")

    return GoogleGenerativeAIEmbeddings(
        model=selected_model,
        google_api_key=key
    )


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for each chunk dictionary.
    """
    if not chunks:
        return []

    embedding_model = get_embedding_model(model_name=model_name, api_key=api_key)
    texts_to_embed = [c.get("text", "") for c in chunks]

    try:
        raw_vectors = embedding_model.embed_documents(texts_to_embed)
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {str(e)}")

    enriched_chunks = []
    for chunk, vector in zip(chunks, raw_vectors):
        chunk_copy = dict(chunk)
        chunk_copy["embedding"] = vector
        chunk_copy["embedding_dim"] = len(vector)
        enriched_chunks.append(chunk_copy)

    return enriched_chunks


# ----------------------------------------------------------------------
# 3. FAISS Vector Store & Semantic Retrieval Utilities
# ----------------------------------------------------------------------

def get_document_vector_store_path(document_id: str) -> str:
    """
    Get the sanitized on-disk directory path for a specific document's FAISS index.
    Enforces strict path traversal prevention.
    """
    if not document_id or not isinstance(document_id, str):
        raise ValueError("Document ID must be a non-empty string.")

    if not re.match(r"^[a-zA-Z0-9_\-]+$", document_id):
        raise ValueError(f"Invalid document identifier format: '{document_id}'")

    base_vector_dir = os.path.abspath(getattr(Config, "VECTOR_STORE_FOLDER", "backend/vector_store"))

    direct_path = os.path.abspath(os.path.join(base_vector_dir, document_id))
    prefixed_path = os.path.abspath(os.path.join(base_vector_dir, f"doc_{document_id}"))

    common_prefix_direct = os.path.commonpath([base_vector_dir, direct_path])
    common_prefix_prefixed = os.path.commonpath([base_vector_dir, prefixed_path])
    if common_prefix_direct != base_vector_dir or common_prefix_prefixed != base_vector_dir:
        raise ValueError("Path traversal attempt detected in document ID.")

    if os.path.exists(os.path.join(direct_path, "index.faiss")):
        return direct_path
    if os.path.exists(os.path.join(prefixed_path, "index.faiss")):
        return prefixed_path

    if document_id.startswith("doc_"):
        return direct_path
    return prefixed_path


def create_vector_store(
    chunks: List[Dict[str, Any]],
    document_id: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a persistent FAISS vector store for a document from its text chunks and save it to disk.
    Applies atomic file writes and persists index configuration metadata (index_meta.json).
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from an empty list of chunks.")

    store_path = get_document_vector_store_path(document_id)
    temp_store_path = store_path + "_tmp"

    # 1. Convert chunks into LangChain Document instances with full metadata
    langchain_docs: List[Document] = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        metadata = {
            "chunk_index": chunk.get("chunk_index", 0),
            "page": chunk.get("page", 1),
            "filename": chunk.get("filename", ""),
            "document_id": document_id,
            "char_count": chunk.get("char_count", len(text)),
            "word_count": chunk.get("word_count", len(text.split()))
        }
        langchain_docs.append(Document(page_content=text, metadata=metadata))

    if not langchain_docs:
        raise ValueError("No valid text content found in chunks to index.")

    # 2. Get the embedding model client
    embedding_model = get_embedding_model(model_name=model_name, api_key=api_key)

    # 3. Create FAISS vector store from documents
    try:
        vector_store = FAISS.from_documents(langchain_docs, embedding_model)
    except Exception as e:
        raise RuntimeError(f"FAISS index creation failed: {str(e)}")

    # 4. Save index with atomic persistence and index_meta.json
    try:
        if os.path.exists(temp_store_path):
            shutil.rmtree(temp_store_path, ignore_errors=True)
        os.makedirs(temp_store_path, exist_ok=True)
        vector_store.save_local(temp_store_path)

        meta_info = {
            "version": INDEX_VERSION,
            "document_id": document_id,
            "dimension": getattr(embedding_model, "dim", 768),
            "model_name": model_name or getattr(Config, "EMBEDDING_MODEL", "local-offline-768"),
            "chunk_count": len(langchain_docs),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        with open(os.path.join(temp_store_path, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta_info, f, indent=2)

        # Atomic replacement: rename existing store to backup, rename temp to target, then purge backup
        if os.path.exists(store_path):
            backup_path = store_path + "_old"
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path, ignore_errors=True)
            os.rename(store_path, backup_path)
            os.rename(temp_store_path, store_path)
            shutil.rmtree(backup_path, ignore_errors=True)
        else:
            os.rename(temp_store_path, store_path)

    except Exception as e:
        if os.path.exists(temp_store_path):
            shutil.rmtree(temp_store_path, ignore_errors=True)
        raise RuntimeError(f"Failed to save FAISS index to disk at '{store_path}': {str(e)}")

    return {
        "created": True,
        "document_id": document_id,
        "vectors": len(langchain_docs),
        "store_path": store_path,
        "version": INDEX_VERSION
    }


def load_vector_store(
    document_id: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> FAISS:
    """
    Load an existing FAISS vector store from disk for a given document.
    Validates index existence, metadata version, and vector dimensionality.
    """
    store_path = get_document_vector_store_path(document_id)
    faiss_file = os.path.join(store_path, "index.faiss")
    pkl_file = os.path.join(store_path, "index.pkl")
    meta_file = os.path.join(store_path, "index_meta.json")

    if not os.path.exists(store_path) or not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
        raise FileNotFoundError(
            f"No FAISS vector store found for document ID '{document_id}' at path '{store_path}'."
        )

    embedding_model = get_embedding_model(model_name=model_name, api_key=api_key)
    expected_dim = getattr(embedding_model, "dim", 768)

    # Validate stored metadata if present
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
            stored_dim = meta_info.get("dimension")
            if stored_dim is not None and stored_dim != expected_dim:
                raise IncompatibleIndexError(
                    f"FAISS index dimension mismatch for document '{document_id}': "
                    f"stored index has {stored_dim}d, but current embedding model expects {expected_dim}d."
                )
        except IncompatibleIndexError:
            raise
        except Exception:
            pass

    try:
        vs = FAISS.load_local(
            folder_path=store_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index for document '{document_id}': {str(e)}")

    if hasattr(vs, "index") and hasattr(vs.index, "d") and vs.index.d != expected_dim:
        raise IncompatibleIndexError(
            f"FAISS index internal dimension ({vs.index.d}) does not match current embedding model ({expected_dim})."
        )

    return vs


def search_similar_chunks(
    document_id: str,
    query: str,
    k: int = 4,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Search a document's FAISS index for the top-k chunks most semantically similar to a query.
    Enforces vector dimension verification and chunk deduplication.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        k = 4
    elif k > 20:
        k = 20

    vector_store = load_vector_store(
        document_id=document_id,
        model_name=model_name,
        api_key=api_key
    )

    # Validate vector dimensions before search
    embedding_model = get_embedding_model(model_name=model_name, api_key=api_key)
    query_vector = embedding_model.embed_query(query)
    if hasattr(vector_store, "index") and hasattr(vector_store.index, "d"):
        if len(query_vector) != vector_store.index.d:
            raise ValueError(
                f"Vector dimension mismatch: query vector is {len(query_vector)}d, "
                f"but index expects {vector_store.index.d}d."
            )

    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=k * 2)
    except Exception as e:
        raise RuntimeError(f"Similarity search failed: {str(e)}")

    results: List[Dict[str, Any]] = []
    seen_texts = set()

    for doc, score in results_with_scores:
        cleaned_text = doc.page_content.strip()
        text_fingerprint = cleaned_text[:200]
        if text_fingerprint in seen_texts:
            continue
        seen_texts.add(text_fingerprint)

        meta = doc.metadata or {}
        results.append({
            "text": doc.page_content,
            "page": meta.get("page", 1),
            "chunk_index": meta.get("chunk_index", 0),
            "filename": meta.get("filename", ""),
            "document_id": meta.get("document_id", document_id),
            "score": round(float(score), 4)
        })
        if len(results) >= k:
            break

    return results
