from langgraph.graph import StateGraph
from agents.query_generator import generate_query
from agents.rag_reflector import rag_agent,reflector
from pydantic_models import state
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def build_graph():
    graph = StateGraph(state)
    graph.add_node("query_generator", generate_query)
    graph.add_node("rag_agent",rag_agent)
    graph.add_node("reflector",reflector)
    
    graph.set_entry_point("query_generator")
    graph.add_edge("query_generator", "rag_agent")
    graph.add_edge("rag_agent","reflector")
    graph.set_finish_point("reflector")
    
    return graph.compile()

if __name__ == "__main__":
    st = state(
        input="Air ambulance outside India",
        file_path=Path(r"C:\Users\sahai\OneDrive\Desktop\hackathon\test\DOC3.pdf")
    )
    

# final_state = graph.invoke(st)
# print(final_state["ref_ans"].content)
# type(final_state)
