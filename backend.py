import os
import tempfile
import logging
import requests
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI, File, UploadFile, Form, HTTPException,
    Depends, APIRouter, Body
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- Logging Setup ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Document Summarizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter(prefix="/api/v1")

# ---------------- Auth Setup ----------------
security = HTTPBearer()
VALID_TOKEN = os.getenv("EXPECTED_TOKEN") or "ff30391fef089ed361c4fd740566e8787e0b74f81be7deba92aedfb92a4a7af9"

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    return credentials

# ---------------- Models ----------------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------- Utilities ----------------
def parse_input(input_text: str) -> List[str]:
    text = input_text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            import ast
            return ast.literal_eval(text)
        except Exception as e:
            logger.warning(f"List parse failed, fallback to comma split: {e}")
    return [q.strip() for q in input_text.split(",") if q.strip()]

def process_request(input_text: List[str], file: UploadFile):
    try:
        from pydantic_models import state
        from lang import build_graph

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file.flush()
            temp_path = Path(temp_file.name)

        try:
            request_state = state(input=input_text, file_path=temp_path)
            graph = build_graph()
            result = graph.invoke(request_state)
            return result.get("rag_ans", [])
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        finally:
            if temp_path.exists():
                os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Endpoints ----------------
@router.get("/", tags=["Health Check"])
def root():
    return {"status": "ok", "message": "Welcome to the Retrieval System API v1!"}

@router.get("/health", tags=["Health Check"])
def health_check():
    return {"status": "healthy"}

@router.post("/summarize", tags=["Summarization"])
async def summarizer(input_text: str = Form(...), file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    parsed_questions = parse_input(input_text)
    result = process_request(parsed_questions, file)
    return {"result": result}

@router.post("/hackrx/run", tags=["HackRx"], dependencies=[Depends(verify_token)])
async def hackrx_run_json(payload: HackRxRequest = Body(...)):
    try:
        response = requests.get(payload.documents)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file.flush()
            temp_path = Path(temp_file.name)

        with open(temp_path, "rb") as f:
            upload_file = UploadFile(filename="file.pdf", file=f)
            result = process_request(payload.questions, upload_file)

        return {"answers": result}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch PDF from URL: {e}")
    except Exception as e:
        logger.error(f"HackRX processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------- Register Router ----------------
app.include_router(router)
