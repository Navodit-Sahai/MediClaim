from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from logs.logging_config import logger
from langchain_core.documents import Document
import os
import zipfile
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from dotenv import load_dotenv
import shutil

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store(docs: List[Document]):
    try:
        faiss_idx = FAISS.from_documents(docs, embedding=embeddings)
        save_path = "faiss_index"
        faiss_idx.save_local(save_path)
        logger.debug("Documents saved to Vectorstore successfully !!.")
    except Exception as e:
        logger.error(f"Failed to store documents in Vectorstore: {e}")

def store_to_cloudinary(docs: List[Document], index_name: str = "faiss_index"):
    try:
        faiss_idx = FAISS.from_documents(docs, embedding=embeddings)
        local_path = f"{index_name}_temp"
        faiss_idx.save_local(local_path)

        zip_path = f"{index_name}.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, local_path))

        response = cloudinary.uploader.upload(
            zip_path,
            resource_type="raw",
            public_id=f"faiss_indices/{index_name}",
            overwrite=True
        )

        shutil.rmtree(local_path)
        os.remove(zip_path)

        logger.info(f"FAISS index uploaded to Cloudinary: {response['secure_url']}")
        return response['secure_url']

    except Exception as e:
        logger.error(f"Failed to store FAISS index to Cloudinary: {e}")
        raise

def load_from_cloudinary(index_name: str = "faiss_index"):
    try:
        resource_info = cloudinary.api.resource(f"faiss_indices/{index_name}", resource_type="raw")
        zip_url = resource_info['secure_url']

        response = requests.get(zip_url)
        response.raise_for_status()

        zip_path = f"{index_name}_download.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)

        extract_path = f"{index_name}_extracted"
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_path)

        faiss_idx = FAISS.load_local(extract_path, embeddings, allow_dangerous_deserialization=True)

        os.remove(zip_path)
        shutil.rmtree(extract_path)

        logger.info(f"FAISS index loaded from Cloudinary successfully")
        return faiss_idx

    except Exception as e:
        logger.error(f"Failed to load FAISS index from Cloudinary: {e}")
        raise
