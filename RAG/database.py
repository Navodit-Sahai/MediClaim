import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

load_dotenv()

embeddings_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

class LightweightVectorStore:
    def __init__(self, index_name: str = "lightweight-index"):
        self.documents = []
        self.embeddings = []
        self.index_name = index_name.lower().replace('_', '-')
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        self.index = pc.Index(self.index_name)

    def from_documents(self, docs: List[Document]):
        if not docs:
            return self
        self.documents = docs
        texts = [doc.page_content for doc in docs]
        self.embeddings = embeddings_model.encode(texts).tolist()
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
        self.index.upsert(vectors)
        return self

    def similarity_search(self, query: str, k: int = 4):
        query_emb = embeddings_model.encode([query])[0].tolist()
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
