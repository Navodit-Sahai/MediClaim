from langchain_community.document_loaders import TextLoader, PyPDFLoader
from logs.logging_config import logger
import os
import requests
import cloudinary
import cloudinary.api
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def load_pdf(file_path) -> str:
    try:
        file_path = str(Path(file_path)).replace('\\', '/')
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
        logger.info(f"Successfully loaded text from: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to load PDF: {file_path} | Error: {e}")
        raise

def load_txt(file_path) -> str:
    try:
        file_path = str(Path(file_path)).replace('\\', '/')
        
        loader = TextLoader(file_path)
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
        logger.info(f"Successfully loaded text from: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to load TXT: {file_path} | Error: {e}")
        raise

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def load_from_cloudinary(cloudinary_url: str, file_type: str = None) -> str:
    try:
        response = requests.get(cloudinary_url)
        response.raise_for_status()

        if not file_type:
            if '.pdf' in cloudinary_url.lower():
                file_type = 'pdf'
            elif '.txt' in cloudinary_url.lower():
                file_type = 'txt'
            else:
                raise ValueError("Cannot determine file type from URL")

        temp_file = f"temp.{file_type}"
        with open(temp_file, 'wb') as f:
            f.write(response.content)

        if file_type == 'pdf':
            text = load_pdf(temp_file)
        else:
            text = load_txt(temp_file)

        os.remove(temp_file)
        return text

    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        logger.error(f"Failed to load from Cloudinary: {cloudinary_url} | Error: {e}")
        raise

def load_data(file_path) -> str:
    try:
        if not file_path:
            raise ValueError("File path cannot be empty")

        file_path = str(Path(file_path)).replace('\\', '/')

        if file_path.startswith("https://res.cloudinary.com/"):
            return load_from_cloudinary(file_path)

        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".pdfg", ".pdff", ".pfd"]:
            corrected_path = file_path.replace(ext, ".pdf")
            logger.warning(f"Corrected file extension to .pdf for: {corrected_path}")
            file_path = corrected_path
            ext = ".pdf"
        elif ext in [".txtt", ".tx", ".text"]:
            corrected_path = file_path.replace(ext, ".txt")
            logger.warning(f"Corrected file extension to .txt for: {corrected_path}")
            file_path = corrected_path
            ext = ".txt"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if ext == ".txt":
            return load_txt(file_path)
        elif ext == ".pdf":
            return load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .txt")

    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        raise