from fastapi import FastAPI
from pydantic_models import state
from lang import build_graph
import uvicorn
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post('/summarize')
def summarizer(request: state):
    try:
        graph = build_graph()
        response = graph.invoke(request)
        return {"summary": response["ref_ans"].content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)