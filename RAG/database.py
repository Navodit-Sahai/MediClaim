import os
import json
import pickle
import requests
import numpy as np
import base64
from dotenv import load_dotenv
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from langchain_core.documents import Document
from logs.logging_config import logger
import cloudinary
import cloudinary.uploader

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

class LightweightVectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.index = None

    def from_documents(self, docs: List[Document]):
        self.documents = docs
        texts = [doc.page_content for doc in docs]
        embeddings = self.model.encode(texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        return self

    def similarity_search(self, query: str, k: int = 4):
        if self.index is None:
            return []

        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results

    def save_local(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        docs_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in self.documents]
        with open(f"{path}/documents.json", 'w') as f:
            json.dump(docs_data, f)

    def load_local(self, path: str):
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/documents.json", 'r') as f:
            docs_data = json.load(f)
        self.documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in docs_data]
        return self

vector_store = LightweightVectorStore()

def store(docs: List[Document]):
    try:
        vector_store.from_documents(docs)
        vector_store.save_local("lightweight_index")
        logger.debug("Documents saved to Vectorstore successfully.")
    except Exception as e:
        logger.error(f"Failed to store documents in Vectorstore: {e}")

cloudinary_urls = {}

def _load_urls():
    try:
        if os.path.exists("urls.json"):
            with open("urls.json", 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def _save_urls():
    with open("urls.json", 'w') as f:
        json.dump(cloudinary_urls, f)

cloudinary_urls = _load_urls()

def store_to_cloudinary(docs: List[Document], index_name: str = "lightweight_index"):
    try:
        vector_store.from_documents(docs)
        vector_store.save_local(index_name)

        with open(f"{index_name}/index.faiss", "rb") as f:
            index_data = base64.b64encode(f.read()).decode()

        docs_data = json.dumps([
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ])

        json_data = json.dumps({
            'index': index_data,
            'documents': docs_data
        })

        base64_data = base64.b64encode(json_data.encode()).decode()
        data_uri = f"data:application/json;base64,{base64_data}"

        response = cloudinary.uploader.upload_large(
            data_uri,
            resource_type="raw",
            public_id=f"vector_indices/{index_name}",
            overwrite=True
        )

        cloudinary_urls[index_name] = response['secure_url']
        _save_urls()

        logger.info(f"Vector index uploaded to Cloudinary: {response['secure_url']}")
        return response['secure_url']

    except Exception as e:
        logger.error(f"Failed to store vector index to Cloudinary: {e}")

def load_from_cloudinary(index_name: str = "lightweight_index"):
    try:
        index_name = index_name.replace('.zip', '')
        if index_name not in cloudinary_urls:
            raise Exception(f"Index '{index_name}' not found. Upload first or check index name.")
        
        url = cloudinary_urls[index_name]
        response = requests.get(url)

        if response.status_code == 200:
            data = json.loads(response.text)

            if 'index' not in data or 'documents' not in data:
                raise Exception("Cloudinary data format is incorrect.")

            index_binary = base64.b64decode(data['index'])
            docs_json = json.loads(data['documents'])

            os.makedirs(index_name, exist_ok=True)
            with open(f"{index_name}/index.faiss", "wb") as f:
                f.write(index_binary)

            with open(f"{index_name}/documents.json", "w") as f:
                json.dump(docs_json, f)

            loaded_store = LightweightVectorStore()
            loaded_store.load_local(index_name)
            logger.info("Vector index loaded from Cloudinary successfully.")
            return loaded_store

        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    except Exception as e:
        logger.error(f"Failed to load vector index from Cloudinary: {e}")
        raise