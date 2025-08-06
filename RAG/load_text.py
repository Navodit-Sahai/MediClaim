import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import pdfplumber
from logs.logging_config import logger

load_dotenv()

def load_pdf(file_path: str) -> str:
    try:
        with pdfplumber.open(file_path) as pdf:
            content = []
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables or []:
                    table_text = "\n".join(["\t".join(cell or "" for cell in row) for row in table])
                    content.append(f"[TABLE]\n{table_text}\n[/TABLE]")
                text = page.extract_text()
                if text:
                    content.append(text)
            return "\n\n".join(content)
    except Exception as e:
        logger.warning(f"pdfplumber failed, falling back to PyPDFLoader: {e}")
        return "\n".join(doc.page_content for doc in PyPDFLoader(file_path).load())

def load_txt(file_path: str) -> str:
    return "\n".join(doc.page_content for doc in TextLoader(file_path).load())

def load_email(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.read().split('\n')
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

def detect_file_type(file_path: str, header: bytes, ext: str) -> str:
    if header.startswith(b'%PDF-'):
        return '.pdf'
    if any(tag in header for tag in [b'From:', b'To:', b'Subject:', b'Delivered-To:', b'MIME-Version:', b'Date:', b'Received:']):
        return '.eml'
    return ext or '.txt'

def load_by_extension(file_path: str, ext: str) -> str:
    loaders = {
        '.pdf': load_pdf,
        '.txt': load_txt,
        '.eml': load_email,
        '.msg': load_email
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    return loaders[ext](file_path)

def load_data(file_path: str) -> str:
    file_path = str(Path(file_path).resolve())
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        header = f.read(50)
    ext = os.path.splitext(file_path)[1].lower()
    return load_by_extension(file_path, detect_file_type(file_path, header, ext))

def load_and_store_to_pinecone(file_path: str, index_name: str = "document-index"):
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
