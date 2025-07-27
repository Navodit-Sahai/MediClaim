import os
import json
import requests
import base64
import gzip
import shutil
from dotenv import load_dotenv
from typing import List
import cloudinary
import cloudinary.uploader
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

class LightweightVectorStore:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = []

    def from_documents(self, docs: List[Document]):
        self.documents = docs
        texts = [doc.page_content for doc in docs]
        raw_embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.embeddings = raw_embeddings.tolist()
        return self

    def similarity_search(self, query: str, k: int = 4):
        if not self.embeddings:
            return []
        query_emb = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        sims = [(sum(a * b for a, b in zip(query_emb, emb)), i) for i, emb in enumerate(self.embeddings)]
        sims.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in sims[:k]:
            if 0 <= idx < len(self.documents):
                doc = self.documents[idx]
                doc.metadata['similarity_score'] = float(score)
                results.append(doc)
        return results

    def save_local(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/embeddings.json", 'w') as f:
            json.dump(self.embeddings, f)
        with open(f"{path}/documents.json", 'w') as f:
            json.dump(
                [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in self.documents], f
            )

    def load_local(self, path: str):
        with open(f"{path}/embeddings.json", 'r') as f:
            self.embeddings = json.load(f)
        with open(f"{path}/documents.json", 'r') as f:
            data = json.load(f)
        self.documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        return self

try:
    vector_store = LightweightVectorStore()
except Exception:
    vector_store = None

def store(docs: List[Document]):
    if vector_store:
        vector_store.from_documents(docs)
        vector_store.save_local("lightweight_index")

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

def store_to_cloudinary(docs: List[Document], index_name: str = "lightweight_index", compress: bool = False):
    if not vector_store:
        return None

    vector_store.from_documents(docs)
    vector_store.save_local(index_name)

    with open(f"{index_name}/embeddings.json", "rb") as f:
        embeddings_data = base64.b64encode(f.read()).decode()

    docs_data = json.dumps([{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs])

    if compress:
        combined_data = json.dumps({'embeddings': embeddings_data, 'documents': docs_data}, separators=(',', ':'))
        compressed_data = gzip.compress(combined_data.encode())
        base64_data = base64.b64encode(compressed_data).decode()
        data_uri = f"data:application/octet-stream;base64,{base64_data}"
    else:
        combined_data = json.dumps({'embeddings': embeddings_data, 'documents': docs_data})
        base64_data = base64.b64encode(combined_data.encode()).decode()
        data_uri = f"data:application/json;base64,{base64_data}"

    response = cloudinary.uploader.upload_large(
        data_uri,
        resource_type="raw",
        public_id=f"vector_indices/{index_name}",
        overwrite=True
    )

    cloudinary_urls[index_name] = response['secure_url']
    _save_urls()

    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    return response['secure_url']

def load_from_cloudinary(index_name: str = "lightweight_index"):
    index_name = index_name.replace('.zip', '')
    if index_name not in cloudinary_urls:
        raise Exception(f"Index '{index_name}' not found in cached URLs.")

    url = cloudinary_urls[index_name]
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch index from Cloudinary")

    try:
        compressed_data = base64.b64decode(response.content)
        decompressed_data = gzip.decompress(compressed_data).decode()
        data = json.loads(decompressed_data)
    except:
        try:
            data = json.loads(response.text)
        except:
            decoded_data = base64.b64decode(response.content).decode()
            data = json.loads(decoded_data)

    if 'embeddings' not in data or 'documents' not in data:
        raise Exception("Invalid format in downloaded index.")

    os.makedirs(index_name, exist_ok=True)

    with open(f"{index_name}/embeddings.json", "wb") as f:
        f.write(base64.b64decode(data['embeddings']))
    with open(f"{index_name}/documents.json", "w") as f:
        json.dump(json.loads(data['documents']), f)

    loaded_store = LightweightVectorStore().load_local(index_name)

    if os.path.exists(index_name):
        shutil.rmtree(index_name)

    return loaded_store
