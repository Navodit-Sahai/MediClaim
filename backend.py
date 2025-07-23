from fastapi import FastAPI, File, UploadFile, Form
import uvicorn
import os 
import tempfile
from pathlib import Path
from pydantic_models import state

app = FastAPI()

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
        
        return {"summary": response["ref_ans"].content}
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
