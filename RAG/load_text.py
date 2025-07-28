import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from logs.logging_config import logger

load_dotenv()

def load_pdf(file_path) -> str:
    try:
        return "\n".join(doc.page_content for doc in PyPDFLoader(file_path).load())
    except Exception as e:
        logger.error(f"PDF load error: {file_path} | {e}")
        raise

def load_txt(file_path) -> str:
    try:
        return "\n".join(doc.page_content for doc in TextLoader(file_path).load())
    except Exception as e:
        logger.error(f"TXT load error: {file_path} | {e}")
        raise

def load_email(file_path) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        lines = content.split('\n')
        email_content, in_body = [], False
        for line in lines:
            if line.startswith(('From:', 'To:', 'Subject:', 'Date:', 'Delivered-To:')):
                email_content.append(line)
            elif line.strip() == '' and not in_body:
                in_body = True
                email_content.append('\n--- Email Body ---\n')
            elif in_body:
                email_content.append(line)
        return '\n'.join(email_content)
    except Exception as e:
        logger.error(f"Email load error: {file_path} | {e}")
        raise

def detect_file_type(file_path: str, header: bytes, ext: str) -> str:
    if header.startswith(b'%PDF-'):
        return '.pdf'
    if any(tag in header for tag in [b'From:', b'To:', b'Subject:', b'Delivered-To:', b'MIME-Version:', b'Date:', b'Received:']):
        return '.eml'
    return ext or '.txt'

def load_by_extension(file_path: str, ext: str) -> str:
    if ext == '.pdf':
        return load_pdf(file_path)
    if ext == '.txt':
        return load_txt(file_path)
    if ext in ['.eml', '.msg']:
        return load_email(file_path)
    raise ValueError(f"Unsupported file type: {ext}")

def load_data(file_path: str) -> str:
    try:
        if not file_path:
            raise ValueError("Empty file path")
        file_path = str(Path(file_path)).replace('\\', '/')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Not found: {file_path}")
        with open(file_path, 'rb') as f:
            header = f.read(50)
        ext = os.path.splitext(file_path)[1].lower()
        return load_by_extension(file_path, detect_file_type(file_path, header, ext))
    except Exception as e:
        logger.error(f"Load error: {file_path} | {e}")
        raise

def load_and_store_to_pinecone(file_path: str, index_name: str = "document-index"):
    try:
        content = load_data(file_path)
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_type": os.path.splitext(file_path)[1].lower(),
                "filename": os.path.basename(file_path)
            }
        )
        from RAG.database import store_to_pinecone
        return store_to_pinecone([doc], index_name)
    except Exception as e:
        logger.error(f"Store error: {file_path} | {e}")
        raise
