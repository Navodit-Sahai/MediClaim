from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from pathlib import Path
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Summarizer API", version="1.0.0")

VALID_TOKEN = "ff30391fef089ed361c4fd740566e8787e0b74f81be7deba92aedfb92a4a7af9"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def process_request(input_text: List[str], file: UploadFile):
    try:
        # Check if the modules exist before importing
        try:
            from pydantic_models import state
            from lang import build_graph
        except ImportError as e:
            logger.error(f"Import error: {e}")
            raise HTTPException(status_code=500, detail=f"Module import failed: {str(e)}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = file.file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_path = Path(temp_file.name)

            try:
                request_state = state(input=input_text, file_path=temp_path)
                graph = build_graph()
                response = graph.invoke(request_state)
                rag_answers = response["rag_ans"]  
                return rag_answers
            except Exception as e:
                logger.error(f"Processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_input(input_text: str) -> List[str]:
    if input_text.strip().startswith('[') and input_text.strip().endswith(']'):
        try:
            import ast
            return ast.literal_eval(input_text)
        except Exception as e:
            logger.warning(f"Failed to parse as list: {e}")
            pass
    return [item.strip() for item in input_text.split(',')]

@app.post("/summarize")
async def summarizer(input_text: str = Form(...), file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        parsed_input = parse_input(input_text)
        result = process_request(parsed_input, file)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summarizer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

@app.post("/hackrx/run", dependencies=[Depends(verify_token)])
async def hackrx_run(
    input_text: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        parsed_input = parse_input(input_text)
        result = process_request(parsed_input, file)
        return {"result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HackRX run error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)