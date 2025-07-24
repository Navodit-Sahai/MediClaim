import os
import json
import pickle
import requests
import numpy as np
import base64
from dotenv import load_dotenv
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.vectors = None

    def from_documents(self, docs: List[Document]):
        self.documents = docs
        texts = [doc.page_content for doc in docs]
        self.vectors = self.vectorizer.fit_transform(texts)
        return self

    def similarity_search(self, query: str, k: int = 4):
        if self.vectors is None:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.documents[i] for i in top_indices if similarities[i] > 0]

    def save_local(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f"{path}/vectors.pkl", 'wb') as f:
            pickle.dump(self.vectors, f)
        docs_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in self.documents]
        with open(f"{path}/documents.json", 'w') as f:
            json.dump(docs_data, f)

    def load_local(self, path: str):
        with open(f"{path}/vectorizer.pkl", 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(f"{path}/vectors.pkl", 'rb') as f:
            self.vectors = pickle.load(f)
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
        
        vectorizer_data = base64.b64encode(pickle.dumps(vector_store.vectorizer)).decode()
        vectors_data = base64.b64encode(pickle.dumps(vector_store.vectors)).decode()
        docs_data = json.dumps([{"page_content": doc.page_content, "metadata": doc.metadata} for doc in vector_store.documents])
        
        json_data = json.dumps({
            'vectorizer': vectorizer_data,
            'vectors': vectors_data, 
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
            
            loaded_store = LightweightVectorStore()
            loaded_store.vectorizer = pickle.loads(base64.b64decode(data['vectorizer']))
            loaded_store.vectors = pickle.loads(base64.b64decode(data['vectors']))
            docs_json = json.loads(data['documents'])
            loaded_store.documents = [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in docs_json]
            
            logger.info("Vector index loaded from Cloudinary successfully.")
            return loaded_store
        else:
            raise Exception(f"HTTP {response.status_code}")

    except Exception as e:
        logger.error(f"Failed to load vector index from Cloudinary: {e}")
        raise

def load_vector_store(save_path: str = "lightweight_index", from_cloudinary: bool = False):
    try:
        if from_cloudinary:
            return load_from_cloudinary(save_path)
        else:
            db = LightweightVectorStore()
            db.load_local(save_path)
            logger.debug(f"Vectorstore loaded from local '{save_path}' successfully!")
            return db
    except Exception as e:
        logger.warning(f"Loading failed: {e}. Using fresh vector store...")
        return LightweightVectorStore()