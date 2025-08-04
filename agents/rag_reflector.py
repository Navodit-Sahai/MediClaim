from RAG.database import store_to_chroma, load_from_chroma
from langchain.prompts import PromptTemplate
from RAG.splitting import split_text
from RAG.load_text import load_data
from pydantic_models import state, query
from llm import model
import time

from concurrent.futures import ThreadPoolExecutor

def rag_agent(st: state) -> state:
    text = load_data(st.file_path)
    docs = split_text(text)
    docs = [doc for doc in docs if doc and isinstance(doc.page_content, str) and doc.page_content.strip()]

    store_to_chroma(docs, index_name="lightweight_index")
    vectorstore = load_from_chroma(index_name="lightweight_index")

    questions = st.questions if st.questions else [query(procedure=q) for q in (st.input or [])]

    prompt = PromptTemplate(
        template="""
You are a Policy Document Analyst specializing in insurance policy interpretation.

**POLICY DOCUMENT:**
{context}

**QUESTION:**
{question}

**INSTRUCTIONS:**
- Try to give a complete and summarized answer.
- Answer based strictly on the policy document.
- Quote exact policy terms if applicable.
- Your response should be factual and 15–30 words long.
**Answer:**""",
        input_variables=["context", "question"]
    )

    chain = prompt | model

    def process(q):
        try:
            query_text = q.procedure.strip() if q and q.procedure else ""
            if not query_text:
                return "Empty question provided."

            results = vectorstore.hybrid_search(query_text, k=4)
            context = "\n\n".join([doc.page_content for doc in results])

            response = chain.invoke({
                "context": context,
                "question": query_text
            })

            answer = response.content.strip()
            time.sleep(1)
            return answer if answer else "No answer generated."
        except Exception as e:
            return f"Error: {str(e)[:30]}"

    with ThreadPoolExecutor(max_workers=min(8, len(questions))) as executor:
        st.rag_ans = list(executor.map(process, questions))

    return st
