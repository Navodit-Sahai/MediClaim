from fastapi import FastAPI
from pydantic_models import state
from lang import build_graph
app = FastAPI()

@app.post('/summarize')
def summarizer(request: state):
    graph = build_graph()
    response = graph.invoke(request)
    return response["ref_ans"].content
