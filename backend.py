from fastapi import FastAPI
from pydantic_models import state
from lang import build_graph
import uvicorn
import os
app = FastAPI()

@app.post('/summarize')
def summarizer(request: state):
    graph = build_graph()
    response = graph.invoke(request)
    return response["ref_ans"].content

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)