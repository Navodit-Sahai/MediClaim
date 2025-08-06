from langgraph.graph import StateGraph
from RAG.database import vector_Search
from agents.query_generator import query_generator
from pydantic_models import state
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def build_graph():
    graph = StateGraph(state)
    graph.add_node("query_generator", query_generator)
    graph.add_node("vector_search", vector_Search)
    graph.set_entry_point("query_generator")
    graph.add_edge("query_generator", "vector_search")
    graph.set_finish_point("vector_search")
    return graph.compile()

def process_questions(questions, file_path: str):
    single_question = isinstance(questions, str)
    questions_list = [questions] if single_question else questions
    app = build_graph()

    initial_state = state(
        input=questions_list,
        file_path=Path(file_path)
    )

    try:
        result = app.invoke(initial_state)
        if single_question:
            return result.rag_ans[0] if isinstance(result.rag_ans, list) else result.rag_ans
        else:
            return [
                f"**Question {i}:** {questions_list[i-1]}\n\n**Answer {i}:** {answer}\n"
                for i, answer in enumerate(result.rag_ans or [], 1)
            ]
    except Exception as e:
        return (
            f"Error processing question: {str(e)}"
            if single_question else
            [f"Error processing questions: {str(e)}"]
        )

# # Example usage
# def example_usage():
#     single_answer = process_questions(
#         "What is the coverage for heart surgery?",
#         "path/to/policy_document.pdf"
#     )
#     print("Single Question Result:")
#     print(single_answer)
#     print("\n" + "="*80 + "\n")

#     questions = [
#         "What is the coverage limit for emergency surgeries?",
#         "Are dental procedures covered under this policy?",
#         "What is the waiting period for pre-existing conditions?"
#     ]

#     multiple_answers = process_questions(questions, "path/to/policy_document.pdf")
#     print("Multiple Questions Result:")
#     for answer in multiple_answers:
#         print(answer)
#         print("-" * 60)

# if __name__ == "__main__":
#     example_usage()
