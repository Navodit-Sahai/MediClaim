import os
import re
import math
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import chromadb
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = genai.embed_content


class GoogleEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_texts(texts, "retrieval_document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text], "retrieval_query")[0]

    def _embed_texts(self, texts: List[str], task_type: str) -> List[List[float]]:
        def embed(text):
            try:
                clean_text = re.sub(r'\s+', ' ', text.strip())[:8000]
                result = embedding_model(
                    model="models/embedding-001",
                    content=clean_text,
                    task_type=task_type
                )
                return result["embedding"]
            except Exception as e:
                print(f"Embedding error: {e}")
                return [0.0] * 768

        with ThreadPoolExecutor(max_workers=10) as executor:
            embeddings = list(executor.map(embed, texts))
        return embeddings


class LightweightVectorStore(VectorStore):
    def __init__(self, index_name: str = "lightweight-index"):
        self.index_name = index_name.lower().replace('_', '-')
        self.db_path = f"./chroma_db_{self.index_name}"
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=self.index_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._embeddings = GoogleEmbeddings()
        super().__init__()

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        embeddings = self._embeddings.embed_documents(texts)
        metadatas = metadatas or [{}] * len(texts)
        ids = [f'doc_{i}_{len(texts)}' for i in range(len(texts))]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[Tuple[Document, float]]:
        try:
            query_emb = self._embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=min(k, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            if not results['documents'][0]:
                return []
            documents_with_scores = []
            for i in range(len(results['documents'][0])):
                distance = results['distances'][0][i]
                similarity_score = max(0, 1 - distance)
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] or {}
                )
                documents_with_scores.append((doc, similarity_score))
            return documents_with_scores
        except Exception as e:
            print(f"Similarity search error: {e}")
            return []

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs) -> "LightweightVectorStore":
        instance = cls(**kwargs)
        instance.add_texts(texts, metadatas)
        return instance

    def embed_batch(self, texts):
        return self._embeddings.embed_documents(texts)

    def embed_texts(self, texts: List[str]):
        return self._embeddings.embed_documents(texts)

    def _preprocess_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) < 3:
            return text
        return text[:8000]

    def embed_texts_old(self, texts: List[str]):
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Batch embedding error: {e}")
                all_embeddings.extend([[0.0] * 768] * len(batch))
        return all_embeddings

    def from_documents(self, docs: List[Document]):
        if not docs:
            return self
        if self.collection.count() > 0:
            return self
        valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        if not valid_docs:
            return self
        texts = [doc.page_content for doc in valid_docs]
        embeddings = self.embed_texts(texts)
        metadatas = [doc.metadata if doc.metadata else {} for doc in valid_docs]
        ids = [f'doc_{i}' for i in range(len(valid_docs))]
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding to collection: {e}")
        return self

    def similarity_search(self, query: str, k: int = 5, **kwargs):
        results_with_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, score in results_with_scores]

    def _advanced_keyword_score(self, query: str, doc_text: str) -> float:
        if not query or not doc_text:
            return 0.0
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        doc_words = re.findall(r'\b\w+\b', doc_text.lower())
        doc_word_set = set(doc_words)
        if not query_words or not doc_words:
            return 0.0
        word_freq = {}
        for word in doc_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        matched_words = query_words & doc_word_set
        if not matched_words:
            return 0.0
        score = 0.0
        for word in matched_words:
            tf = word_freq[word] / len(doc_words)
            idf = math.log(len(doc_words) / (word_freq[word] + 1))
            score += tf * idf
        return score / len(query_words)

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7):
        if not query or not query.strip():
            return []
        vector_results = self.similarity_search(query, k=min(15, k * 3))
        if not vector_results:
            return []
        for doc in vector_results:
            sim_score = doc.metadata.get("similarity_score", 0)
            kw_score = self._advanced_keyword_score(query, doc.page_content)
            hybrid_score = alpha * sim_score + (1 - alpha) * min(kw_score, 1.0)
            doc.metadata["hybrid_score"] = hybrid_score
            doc.metadata["keyword_score"] = kw_score
        vector_results.sort(key=lambda d: d.metadata.get("hybrid_score", 0), reverse=True)
        return vector_results[:k]

    def get_collection_info(self):
        try:
            count = self.collection.count()
            return {"collection_name": self.index_name, "document_count": count}
        except Exception as e:
            return {"error": str(e)}


def store(docs: List[Document], index_name: str = "lightweight_index"):
    vs = LightweightVectorStore(index_name)
    vs.from_documents(docs)
    return {"status": "stored_in_chroma", "index_name": index_name, "count": len(docs)}


def load_chroma(index_name: str = "lightweight_index"):
    return LightweightVectorStore(index_name)


def store_to_chroma(docs: List[Document], index_name: str = "lightweight_index"):
    return store(docs, index_name)


def load_from_chroma(index_name: str = "lightweight_index"):
    return load_chroma(index_name)


vector_store = LightweightVectorStore()
