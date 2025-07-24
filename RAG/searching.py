from logs.logging_config import logger
from RAG.database import load_from_cloudinary 
from dotenv import load_dotenv
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

load_dotenv()


class LightweightReranker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    
    def compute_score(self, pairs):
        """Lightweight reranking using TF-IDF similarity"""
        queries = [pair[0] for pair in pairs]
        docs = [pair[1] for pair in pairs]
        all_texts = queries + docs
        vectors = self.vectorizer.fit_transform(all_texts)
        scores = []
        for i, (query, doc) in enumerate(pairs):
            query_vec = vectors[i:i+1]  
            doc_vec = vectors[len(queries) + i:len(queries) + i + 1]   
            similarity = cosine_similarity(query_vec, doc_vec)[0][0]
            scores.append(similarity)
        
        return scores

reranker = LightweightReranker()

def load_vectorstore(save_path: str = "lightweight_index.zip", from_cloudinary: bool = False):
    try:  
        if from_cloudinary:
            db = load_from_cloudinary(save_path)
        else:
            from RAG.database import vector_store
            try:
                db = vector_store
                db.load_local(save_path)
            except:
                logger.warning(f"Local vectorstore '{save_path}' not found, trying cloudinary...")
                db = load_from_cloudinary(save_path)
        
        logger.debug(f"Vectorstore loaded from '{save_path}' successfully!")
        return db
    except Exception as e:
        logger.error(f"Failed to load Vectorstore: {e}")
        return None

def semantic_search(query: str, top_k: int = 5, fetch_k: int = 10, use_cloudinary: bool = False):
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