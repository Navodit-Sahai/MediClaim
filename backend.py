from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os 
import tempfile
from pathlib import Path
from pydantic_models import state

app = FastAPI()

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

@app.post('/summarize')
async def summarizer(
    input_text: str = Form(...),
    file: UploadFile = File(...)
):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        request_state = state(
            input=input_text,
            file_path=temp_path
        )

        from lang import build_graph
        graph = build_graph()
        response = graph.invoke(request_state)

        decision_obj = response["rag_ans"]

        formatted_response = {
            "Final Decision": decision_obj.decision,
            "Approved Amount": f"${decision_obj.approved_amount}" if isinstance(decision_obj.approved_amount, int) else decision_obj.approved_amount,
            "Justification": [
                {
                    "Clause": j.clause,
                    "Reason": j.reason
                } for j in decision_obj.justification
            ]
        }

        return {"result": formatted_response}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=False)
