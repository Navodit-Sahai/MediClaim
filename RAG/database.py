import os
import json
import pickle
import shutil
import zipfile
import requests
import numpy as np
from dotenv import load_dotenv
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from logs.logging_config import logger
import cloudinary
import cloudinary.uploader
import cloudinary.api

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


def store_to_cloudinary(docs: List[Document], index_name: str = "lightweight_index"):
    try:
        vector_store.from_documents(docs)
        local_path = f"{index_name}_temp"
        vector_store.save_local(local_path)

        zip_path = f"{index_name}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, local_path)
                    zipf.write(file_path, arcname)

        response = cloudinary.uploader.upload(
            zip_path,
            resource_type="raw",
            public_id=f"vector_indices/{index_name}",
            overwrite=True
        )

        shutil.rmtree(local_path)
        os.remove(zip_path)

        logger.info(f"Vector index uploaded to Cloudinary: {response['secure_url']}")
        return response['secure_url']

    except Exception as e:
        logger.error(f"Failed to store vector index to Cloudinary: {e}")
        raise


def load_from_cloudinary(index_name: str = "lightweight_index"):
    try:
        resource_info = cloudinary.api.resource(f"vector_indices/{index_name}", resource_type="raw")
        zip_url = resource_info['secure_url']

        response = requests.get(zip_url)
        response.raise_for_status()

        zip_path = f"{index_name}_download.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        extract_path = f"{index_name}_extracted"
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_path)

        loaded_store = LightweightVectorStore()
        loaded_store.load_local(extract_path)

        os.remove(zip_path)
        shutil.rmtree(extract_path)

        logger.info("Vector index loaded from Cloudinary successfully.")
        return loaded_store

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