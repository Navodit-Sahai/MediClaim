import os
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import chromadb
from langchain_core.documents import Document
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = genai.embed_content


class LightweightVectorStore:
    def __init__(self, index_name: str = "lightweight-index"):
        self.index_name = index_name.lower().replace('_', '-')
        self.db_path = f"./chroma_db_{self.index_name}"
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name=self.index_name,
            metadata={"hnsw:space": "cosine"}
        )



    def embed_batch(self, texts):
        def embed(text):
            try:
                result = embedding_model(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                return result["embedding"]
            except Exception:
                return [0.0] * 768

        with ThreadPoolExecutor(max_workers=15) as executor:
            embeddings = list(executor.map(embed, texts))
        return embeddings

    def embed_texts(self, texts: List[str]):
        batch_size = 40
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self.embed_batch(batch))
        return all_embeddings

    def from_documents(self, docs: List[Document]):
        if not docs:
            return self
        if self.collection.count() > 0:
            return self
        texts = [doc.page_content for doc in docs]
        embeddings = self.embed_texts(texts)
        metadatas = [doc.metadata for doc in docs]
        ids = [f'doc_{i}' for i in range(len(docs))]
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        return self

    def similarity_search(self, query: str, k: int = 5):
        query_emb = self.embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        return [
            Document(
                page_content=results['documents'][0][i],
                metadata={
                    **results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]
                }
            )
            for i in range(len(results['documents'][0]))
        ]

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.7):
        vector_results = self.similarity_search(query, k=11)

        def keyword_overlap_score(query, doc_text):
            query_words = set(query.lower().split())
            doc_words = set(doc_text.lower().split())
            return len(query_words & doc_words) / (len(query_words) + 1e-6)

        for doc in vector_results:
            sim_score = doc.metadata.get("similarity_score", 0)
            kw_score = keyword_overlap_score(query, doc.page_content)
            hybrid_score = alpha * sim_score + (1 - alpha) * kw_score
            doc.metadata["hybrid_score"] = hybrid_score

        vector_results.sort(key=lambda d: d.metadata["hybrid_score"], reverse=True)
        return vector_results[:k]


def store(docs: List[Document], index_name: str = "lightweight_index"):
    LightweightVectorStore(index_name).from_documents(docs)
    return {"status": "stored_in_chroma", "index_name": index_name}


def load_chroma(index_name: str = "lightweight_index"):
    return LightweightVectorStore(index_name)


def store_to_chroma(docs: List[Document], index_name: str = "lightweight_index"):
    return store(docs, index_name)


def load_from_chroma(index_name: str = "lightweight_index"):
    return load_chroma(index_name)


vector_store = LightweightVectorStore()
