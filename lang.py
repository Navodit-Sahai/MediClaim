from langgraph.graph import StateGraph
from agents.rag_reflector import rag_agent
from agents.query_generator import query_generator
from pydantic_models import state
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def build_graph():
    graph = StateGraph(state)
    graph.add_node("query_generator",query_generator)
    graph.add_node("rag_agent",rag_agent)
    
    graph.set_entry_point("query_generator")
    graph.add_edge("query_generator","rag_agent")
    graph.set_finish_point("rag_agent")
    
    return graph.compile()
