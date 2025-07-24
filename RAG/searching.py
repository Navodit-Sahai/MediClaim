from logs.logging_config import logger
from RAG.database import load_from_cloudinary
from dotenv import load_dotenv
import os

load_dotenv()


def load_vectorstore(save_path="lightweight_index.zip", from_cloudinary=False):
    try:
        if from_cloudinary:
            return load_from_cloudinary(save_path)
        else:
            from RAG.database import vector_store
            try:
                vector_store.load_local(save_path)
                return vector_store
            except:
                logger.warning(f"Local vectorstore '{save_path}' not found, trying cloudinary...")
                return load_from_cloudinary(save_path)
    except Exception as e:
        logger.error(f"Failed to load Vectorstore: {e}")
        return None


def semantic_search(query, top_k=5, use_cloudinary=False):
    try:
        vectorstore = load_vectorstore(from_cloudinary=use_cloudinary)
        if not vectorstore:
            return "Vectorstore not available.", []

        results = vectorstore.similarity_search(query, k=top_k)
        if not results:
            return "No relevant documents found.", []

        answer = "\n\n".join([doc.page_content for doc in results])
        sources = "\n\n".join([
            f"{doc.metadata.get('source', 'Unknown Source')} [Line {doc.metadata.get('line', 'N/A')}]: {doc.page_content.strip()}"
            for doc in results
        ])

        print(sources, "\n\n\n")
        return answer, sources

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return "Something went wrong.", []
