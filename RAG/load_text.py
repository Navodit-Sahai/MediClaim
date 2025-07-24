from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredEmailLoader, OutlookMessageLoader
from logs.logging_config import logger
import os
import requests
import cloudinary
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def load_pdf(file_path) -> str:
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Failed to load PDF: {file_path} | Error: {e}")
        raise

def load_txt(file_path) -> str:
    try:
        loader = TextLoader(file_path)
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs)
    except Exception as e:
        logger.error(f"Failed to load TXT: {file_path} | Error: {e}")
        raise

def load_email(file_path) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        lines = content.split('\n')
        email_content = []
        in_body = False
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
        logger.error(f"Failed to load Email: {file_path} | Error: {e}")
        raise

def detect_file_type(file_path: str, header: bytes, ext: str) -> str:
    if header.startswith(b'%PDF-'):
        return '.pdf'
    if any(tag in header for tag in [b'From:', b'To:', b'Subject:', b'Delivered-To:', b'MIME-Version:', b'Date:', b'Received:']):
        return '.eml'
    return ext if ext else '.txt'

def load_from_cloudinary(cloudinary_url: str) -> str:
    temp_file = "temp_file"
    try:
        response = requests.get(cloudinary_url)
        response.raise_for_status()

        ext = os.path.splitext(cloudinary_url.split("?")[0])[1].lower()
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        if os.path.getsize(temp_file) == 0:
            raise ValueError("Downloaded file is empty or corrupted")

        with open(temp_file, 'rb') as f:
            header = f.read(50)
        ext = detect_file_type(temp_file, header, ext)
        final_file = f"temp{ext}"
        os.rename(temp_file, final_file)

        return load_by_extension(final_file, ext)
    except Exception as e:
        logger.error(f"Failed to load from Cloudinary: {cloudinary_url} | Error: {e}")
        raise
    finally:
        for name in [temp_file, "temp.pdf", "temp.txt", "temp.eml"]:
            if os.path.exists(name):
                try:
                    os.remove(name)
                except:
                    pass

def load_by_extension(file_path: str, ext: str) -> str:
    if ext == '.pdf':
        return load_pdf(file_path)
    elif ext == '.txt':
        return load_txt(file_path)
    elif ext in ['.eml', '.msg']:
        return load_email(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_data(file_path) -> str:
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")

        file_path = str(Path(file_path)).replace('\\', '/')

        if file_path.startswith("https://res.cloudinary.com/"):
            return load_from_cloudinary(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, 'rb') as f:
            header = f.read(50)
        detected_ext = detect_file_type(file_path, header, ext)

        return load_by_extension(file_path, detected_ext)
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise
