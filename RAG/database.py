import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
import google.generativeai as genai
import time

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

embedding_model = genai.embed_content
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

class LightweightVectorStore:
    def __init__(self, index_name: str = "lightweight-index"):
        self.documents = []
        self.embeddings = []
        self.index_name = index_name.lower().replace('_', '-')
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = pc.Index(self.index_name)

    def embed_batch(self, texts_batch):
        embeddings = []
        for text in texts_batch:
            try:
                result = embedding_model(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])
            except Exception as e:
                print(f"Embedding failed for text, using fallback: {str(e)[:50]}")
                embeddings.append([0.0] * 768)
        return embeddings

    def embed_texts(self, texts: List[str]):
        batch_size = 15
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed_batch(batch)
            all_embeddings.extend(batch_embeddings)
            print(f"Processed {i+len(batch)}/{len(texts)} embeddings")
            time.sleep(0.5)
        
        return all_embeddings

    def from_documents(self, docs: List[Document]):
        if not docs:
            return self

        self.documents = docs
        texts = [doc.page_content for doc in docs]
        self.embeddings = self.embed_texts(texts)

        vectors = [
            {
                'id': f'doc_{i}',
                'values': emb,
                'metadata': {
                    'content': doc.page_content,
                    **doc.metadata
                }
            }
            for i, (doc, emb) in enumerate(zip(docs, self.embeddings))
        ]

        def batch(lst, batch_size=200):
            for i in range(0, len(lst), batch_size):
                yield lst[i:i + batch_size]

        for chunk in batch(vectors):
            self.index.upsert(chunk)

        return self

    def similarity_search(self, query: str, k: int = 4):
        query_emb = self.embed_texts([query])[0]
        results = self.index.query(vector=query_emb, top_k=k, include_metadata=True)
        return [
            Document(
                page_content=match['metadata'].pop('content', ''),
                metadata={**match['metadata'], 'similarity_score': float(match['score'])}
            )
            for match in results['matches']
        ]

def store(docs: List[Document], index_name: str = "lightweight_index"):
    return LightweightVectorStore(index_name).from_documents(docs) and {
        "status": "stored_in_pinecone", "index_name": index_name
    }

def load_pinecone(index_name: str = "lightweight_index"):
    return LightweightVectorStore(index_name)

def store_to_pinecone(docs: List[Document], index_name: str = "lightweight_index"):
    return store(docs, index_name)

def load_from_pinecone(index_name: str = "lightweight_index"):
    return load_pinecone(index_name)

vector_store = LightweightVectorStore()