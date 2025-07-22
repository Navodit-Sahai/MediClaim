from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pathlib import Path
from logs.logging_config import logger

def load_pdf(file_path: Path) -> str:
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n".join(doc.page_content for doc in docs)
        logger.info(f"Successfully loaded text from: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to load PDF: {file_path} | Error: {e}")
        raise


def load_txt(file_path :Path)-> str:
    try:
        loader=TextLoader(str(file_path))
        docs=loader.load()
        text="\n".join(doc.page_content for doc in docs)
        logger.info(f"Successfully loaded text from: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to load PDF: {file_path} | Error: {e}")
        raise

def load_data(file_path: Path) -> str:
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            return load_txt(file_path)
        elif suffix == ".pdf":
            return load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    except Exception as e:
        logger.error(f"Error loading file {file_path.name}: {e}")
        return ""