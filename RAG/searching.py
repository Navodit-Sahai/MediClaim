from logs.logging_config import logger
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from RAG.database import embeddings
from FlagEmbedding import FlagReranker
from database import load_from_cloudinary
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True, 
                        trust_remote_code=True,
                        use_auth_token=os.environ["HF_TOKEN"])

def load_vectorstore(save_path: str = "faiss_index", from_cloudinary: bool = False):
    try:  
        if from_cloudinary:
            # Cloudinary se load karo
            db = load_from_cloudinary(save_path)
        else:
            # Original local loading
            db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
        
        logger.debug(f"Vectorstore loaded from '{save_path}' successfully!")
        return db
    except Exception as e:
        logger.error(f"Failed to load Vectorstore: {e}")
        return None

def semantic_search(query: str, top_k: int = 4, fetch_k: int = 8, use_cloudinary: bool = False):
    try:
        vectorstore = load_vectorstore(from_cloudinary=use_cloudinary)
        if not vectorstore:
            return "Vectorstore not available.", []

        initial_results = vectorstore.similarity_search(query, k=fetch_k)
        docs = [doc.page_content for doc in initial_results]

        if not docs:
            return "No relevant documents found.", []

        pairs = [(query, doc) for doc in docs]
        rerank_scores = reranker.compute_score(pairs)
        reranked = sorted(zip(initial_results, rerank_scores), key=lambda x: x[1], reverse=True)

        top_results = reranked[:top_k]
        answer = "\n\n".join([doc.page_content for doc, _ in top_results])
        sources = "\n\n".join(
    [
        f"{doc.metadata.get('source', 'Unknown Source')} [Line {doc.metadata.get('line', 'N/A')}]: {doc.page_content.strip()}"
        for doc, _ in top_results
    ]
)
        print(sources,"\n\n\n")
        return answer, sources

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise