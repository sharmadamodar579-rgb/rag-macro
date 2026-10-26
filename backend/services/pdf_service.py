"""
PDF Processing Service
Responsible for handling PDF file reading, validation, text extraction (using PyPDF),
and preparing structured page-by-page text with metadata.
"""

import os
from typing import Dict, Any, List
from pypdf import PdfReader
from pypdf.errors import PdfReadError, EmptyFileError


class PDFProcessingError(Exception):
    """Custom exception class for PDF processing errors."""
    pass


def extract_pdf_text(file_path: str) -> Dict[str, Any]:
    """
    Extract text page-by-page from a PDF file using PyPDF.

    Args:
        file_path (str): Absolute or relative path to the saved PDF file.

    Returns:
        dict: A structured dictionary containing:
            - total_pages (int): Total number of pages in the PDF.
            - pages (list): List of dictionaries for each page with:
                - page_number (int): 1-indexed page number.
                - text (str): Cleaned extracted text of the page.
                - char_count (int): Number of characters in the page text.
                - word_count (int): Number of words in the page text.
            - full_text (str): Complete concatenated text from all pages.
            - total_characters (int): Total character count across all pages.
            - total_words (int): Total word count across all pages.
            - has_extractable_text (bool): False if PDF contains no selectable text (e.g. scanned images).
            - message (str): Informational or warning message about the extraction.

    Raises:
        PDFProcessingError: If the file is missing, empty, corrupted, encrypted, or cannot be parsed.
    """
    # 1. Validate file existence on disk
    if not file_path or not os.path.exists(file_path):
        raise PDFProcessingError(f"File not found at path: {file_path}")

    # 2. Check for empty file (0 bytes)
    try:
        file_size = os.path.getsize(file_path)
    except OSError as e:
        raise PDFProcessingError(f"Unable to access file: {str(e)}")

    if file_size == 0:
        raise PDFProcessingError("The uploaded PDF file is empty (0 bytes).")

    # 3. Read and extract text using PyPDF
    try:
        reader = PdfReader(file_path)
    except EmptyFileError:
        raise PDFProcessingError("The uploaded PDF contains no data or is empty.")
    except PdfReadError as e:
        raise PDFProcessingError(f"Corrupted or invalid PDF format: {str(e)}")
    except Exception as e:
        raise PDFProcessingError(f"Failed to open PDF: {str(e)}")

    # 4. Check for encryption / password protection
    if getattr(reader, "is_encrypted", False):
        try:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise PDFProcessingError(
                    "The uploaded PDF is password protected / encrypted. "
                    "Please remove password encryption before uploading."
                )
        except Exception:
            raise PDFProcessingError(
                "The uploaded PDF is password protected / encrypted. "
                "Please remove password encryption before uploading."
            )

    try:
        total_pages = len(reader.pages)
    except Exception as e:
        raise PDFProcessingError(f"Failed to read PDF pages: {str(e)}")

    if total_pages == 0:
        raise PDFProcessingError("The PDF document contains 0 pages.")

    pages_data: List[Dict[str, Any]] = []
    full_text_list: List[str] = []
    total_characters = 0
    total_words = 0

    for idx, page in enumerate(reader.pages):
        page_number = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            # Handle rare page extraction failures gracefully without crashing entire batch
            raw_text = ""

        # Clean extracted text (normalize whitespace)
        cleaned_text = raw_text.strip()

        page_char_count = len(cleaned_text)
        page_word_count = len(cleaned_text.split()) if cleaned_text else 0

        total_characters += page_char_count
        total_words += page_word_count

        pages_data.append({
            "page_number": page_number,
            "text": cleaned_text,
            "char_count": page_char_count,
            "word_count": page_word_count
        })

        if cleaned_text:
            full_text_list.append(cleaned_text)

    full_text = "\n\n".join(full_text_list)
    has_extractable_text = total_characters > 0

    # Provide clear user-facing feedback if text extraction yielded no selectable text
    if not has_extractable_text:
        warning_message = (
            "No extractable text found in this PDF. It appears to be a scanned document or "
            "image-only PDF. Optical Character Recognition (OCR) is required to parse image-based notes."
        )
    else:
        warning_message = f"Successfully extracted text from {total_pages} page(s)."

    return {
        "total_pages": total_pages,
        "pages": pages_data,
        "full_text": full_text,
        "total_characters": total_characters,
        "total_words": total_words,
        "has_extractable_text": has_extractable_text,
        "message": warning_message
    }
