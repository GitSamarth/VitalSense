import pdfplumber
import streamlit as st
from config.app_config import MAX_PDF_PAGES
from utils.validators import validate_pdf_file, validate_pdf_content
from utils.pii_redactor import redact_pii, strip_injection_lines

def extract_text_from_pdf(pdf_file):
    """Extract and validate text from PDF file."""
    try:
        # Validate file first
        is_valid, error = validate_pdf_file(pdf_file)
        if not is_valid:
            return error

        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                return f"PDF exceeds maximum page limit of {MAX_PDF_PAGES}"
                
            for page in pdf.pages:
                extracted = page.extract_text()
                if not extracted:
                    return "Could not extract text from PDF. Please ensure it's not a scanned document."
                text += extracted + "\n"
        
        # Validate extracted content
        is_valid, error = validate_pdf_content(text)
        if not is_valid:
            return error

        # Sanitise: mask PII fields, then strip any injected lines.
        # Order matters — redact first so tokens like [REDACTED_NAME] are in
        # place before injection scanning runs on the remaining content.
        text = redact_pii(text)
        text = strip_injection_lines(text)

        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
