from RAG.database import store_to_chroma, load_from_chroma
from RAG.splitting import split_text
from RAG.load_text import load_data
from pydantic_models import state, query
from langchain.prompts import PromptTemplate
from llm import model
from concurrent.futures import ThreadPoolExecutor

def rag_agent(st: state) -> state:
    docs = [doc for doc in split_text(load_data(st.file_path))
            if getattr(doc, "page_content", "").strip()]
    
    store_to_chroma(docs, index_name="lightweight_index")
    vectorstore = load_from_chroma(index_name="lightweight_index")

    questions = st.questions or [query(procedure=q) for q in (st.input or [])]

    prompt = PromptTemplate(
        template = """
You are an expert PDF Analyst. Your task is to extract precise, factual answers from policy documents.

**POLICY DOCUMENT:**
{context}

**QUESTION:**
{question}

**ANSWERING GUIDELINES:**
- Answer strictly based on the information in the document.
- Include specific clause references or section numbers if mentioned.
- Quote exact phrases when helpful to improve clarity.
- Avoid assumptions or generalizations not supported by the document.
- Summarize in 20–30 words, ensuring the response is clear, concise, and complete.

**ANSWER:**
""",
        input_variables=["context", "question"]
    )

    chain = prompt | model

    def process(q):
        try:
            q_text = (q.procedure or "").strip()
            if not q_text:
                return "Empty question."

            results = vectorstore.hybrid_search(q_text, k=4)
            context = "\n\n".join(doc.page_content for doc in results if doc.page_content.strip())
            res = chain.invoke({"context": context, "question": q_text})
            return res.content.strip() or "No answer generated."
        except Exception as e:
            return f"Error: {str(e)[:30]}"

    with ThreadPoolExecutor(max_workers=min(8, len(questions))) as pool:
        st.rag_ans = list(pool.map(process, questions))

    return st
