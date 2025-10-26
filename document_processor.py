from pypdf import PdfReader
from docx import Document
from io import BytesIO
from typing import Union
import streamlit as st


def extract_text_from_pdf(file) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file: Uploaded PDF file (Streamlit UploadedFile object)
    
    Returns:
        Extracted text as string
    """
    try:
        pdf_reader = PdfReader(file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")


def extract_text_from_docx(file) -> str:
    """
    Extract text from DOCX file.
    
    Args:
        file: Uploaded DOCX file (Streamlit UploadedFile object)
    
    Returns:
        Extracted text as string
    """
    try:
        # Read the file into BytesIO
        docx_bytes = BytesIO(file.read())
        doc = Document(docx_bytes)
        
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")


def extract_text_from_txt(file) -> str:
    """
    Extract text from TXT file.
    
    Args:
        file: Uploaded TXT file (Streamlit UploadedFile object)
    
    Returns:
        Extracted text as string
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # Reset file pointer
                file.seek(0)
                text = file.read().decode(encoding)
                return text.strip()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail
        raise Exception("Unable to decode text file. Please ensure it's a valid text file.")
    
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")


def extract_text_from_file(file) -> str:
    """
    Extract text from uploaded file based on file type.
    
    Args:
        file: Uploaded file (Streamlit UploadedFile object)
    
    Returns:
        Extracted text as string
    """
    file_extension = file.name.split('.')[-1].lower()
    
    # Reset file pointer to beginning
    file.seek(0)
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file)
    elif file_extension == 'docx':
        return extract_text_from_docx(file)
    elif file_extension == 'txt':
        return extract_text_from_txt(file)
    else:
        raise Exception(f"Unsupported file type: .{file_extension}")


def validate_document_content(text: str, min_length: int = 50) -> bool:
    """
    Validate that extracted text meets minimum requirements.
    
    Args:
        text: Extracted text
        min_length: Minimum character length
    
    Returns:
        True if valid, False otherwise
    """
    if not text or len(text.strip()) < min_length:
        return False
    return True