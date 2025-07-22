from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from logs.logging_config import logger
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain_core.documents import Document


def store(docs: List[Document]):
    try:
        faiss_idx = FAISS.from_documents(docs, embedding=embeddings)
        save_path = "faiss_index"
        faiss_idx.save_local(save_path)
        logger.debug("Documents saved to Vectorstore successfully !!.")
    except Exception as e:
        logger.error(f"Failed to store documents in Vectorstore: {e}")
