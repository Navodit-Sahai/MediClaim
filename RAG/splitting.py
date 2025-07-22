from langchain.text_splitter import RecursiveCharacterTextSplitter
from logs.logging_config import logger
from langchain_core.documents import Document

def split_text(text: str):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text)

        docs = [
            Document(
                page_content=chunk,
                metadata={"line": idx + 1, "source": "DOC3"}  
            )
            for idx, chunk in enumerate(chunks)
        ]
        logger.debug("Text splitted successfully with line numbers!")
        return docs
    except Exception as e:
        logger.error(f"Error in splitting text: {e}")
        raise

