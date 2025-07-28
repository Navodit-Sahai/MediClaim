from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from pydantic_models import state
from lang import build_graph

load_dotenv()

app = FastAPI()

VALID_TOKEN = os.getenv("HACKRX_API_TOKEN", "fallback-token")

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

def process_request(input_text: str, file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        content = file.file.read()
        temp_file.write(content)
        temp_file.flush()
        temp_path = Path(temp_file.name)

        request_state = state(input=input_text, file_path=temp_path)
        graph = build_graph()
        response = graph.invoke(request_state)
        decision_obj = response["rag_ans"]

        return {
            "Final Decision": decision_obj.decision,
            "Approved Amount": f"${decision_obj.approved_amount}" if isinstance(decision_obj.approved_amount, int) else decision_obj.approved_amount,
            "Justification": [
                {"Clause": j.clause, "Reason": j.reason} for j in decision_obj.justification
            ]
        }

@app.post("/summarize")
async def summarizer(input_text: str = Form(...), file: UploadFile = File(...)):
    try:
        result = process_request(input_text, file)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/run")
async def hackrx_run(
    input_text: str = Form(...),
    file: UploadFile = File(...),
    authorization: str = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    if authorization.replace("Bearer ", "") != VALID_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        result = process_request(input_text, file)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)
