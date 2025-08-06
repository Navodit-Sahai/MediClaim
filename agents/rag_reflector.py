from RAG.database import vector_Search
from pydantic_models import state
from langchain.prompts import PromptTemplate
from llm import model
from concurrent.futures import ThreadPoolExecutor
from logs.logging_config import logger

def format_and_call_model(args):
    context, question, prompt, index = args
    try:
        logger.debug(f"[{index+1}] Processing question: {question}")
        final_prompt = prompt.format(context=context, question=question)
        ans = model.invoke(final_prompt)
        logger.debug(f"[{index+1}] Answer received successfully.")
        return ans.content
    except Exception as e:
        logger.error(f"[{index+1}] Error processing question: {str(e)}")
        return f"Error: {str(e)}"

def rag_agent(st: state) -> state:
    logger.info("Starting RAG agent pipeline.")
    response = vector_Search(st)
    logger.info("Vector search completed.")

    questions = [
        st.questions[i].procedure if st.questions[i].procedure else st.input[i]
        for i in range(len(st.input))
    ]
    logger.info(f"Total questions to process: {len(questions)}")

    prompt = PromptTemplate(
        template="""
You are an expert Document Analyst. Your task is to extract precise, factual answers from policy documents and summarize it.

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANSWERING GUIDELINES:**
- Answer strictly based on the information in the document.
- Include specific clause references.
- Quote exact phrases when helpful to improve clarity.
- Avoid assumptions or generalizations not supported by the document.
- Summarize in 20–30 words, ensuring the response is clear, concise, and complete.

**ANSWER:**
""",
        input_variables=["context", "question"]
    )

    args_list = [(response[i], questions[i], prompt, i) for i in range(len(questions))]

    with ThreadPoolExecutor(max_workers=min(10, len(args_list))) as executor:
        st.rag_ans = list(executor.map(format_and_call_model, args_list))

    logger.info("RAG agent pipeline completed.")
    return st
