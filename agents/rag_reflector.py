from RAG.load_text import load_data
from RAG.splitting import split_text
from RAG.database import store
from RAG.searching import semantic_search
from pathlib import Path
from pydantic_models import state
from langchain.prompts import PromptTemplate
from llm import model
from logs.logging_config import logger
from langchain_tavily import TavilySearch

tool=TavilySearch(max_results=2)

def rag_agent(st: state) -> state:
    try:
        file_path = st.file_path
        dic = st.question
        doubt = dic.procedure
        duration = dic.duration
        

        text = load_data(file_path=file_path)
        chunks = split_text(text=text)
        store(chunks)
        content, sources = semantic_search(doubt)
        
        template = """
You are a medical policy coverage assistant with access to the user's medical policy document.

User Query: "{doubt}" (Duration: {duration})
Document Context: {context}

Your Task:

1. Generate a precise, point-wise answer strictly based on the **document context**.
2. For **each point**, precede the answer with a **relevant quoted line** from the document context (i.e., the supporting evidence).
3. Ensure the answer is **clear, short, and relevant** to the user's query.
4. DO NOT include any information not found in the document.
5. Final output format should be:

- "**[Source Line]**" → Your refined answer point.
- Repeat for all supporting points.

Make sure the answer is helpful and formatted professionally.


Instructions:

1. Refine and shorten the rag-answer and keep it straight forward adn relevant to the medical policy coverage.
2. Justify you answer by adding the source(line no. of the doc keep it same-exact line number mentioned in the rag source) of your answer from the Document source clearly.
3. Improve the quality of rag-answer and keep it crisp.
4. Answer using ONLY the document context provided.
5. Give precise point-wise answers.
6. Also mention what is covered and what is not if mentioned in the rag-context.
7. Display the source like e.g [DOC-Line x].
8. Don't skip any necessary information that may lead to the loss of the user.
9. Include any amount listed in the context and highlight it.

        """
        
        prompt = PromptTemplate(template=template, input_variables=["doubt", "duration", "context"])
        inp = prompt.format(
            doubt=doubt,
            duration=duration,
            context=sources,
        )

        response = model.invoke(inp) 

        st.rag_ans = response.content
        logger.debug("RAG ANSWER GENERATED SUCCESSFULLY. !!")
        return st

    except Exception as e:
        logger.error(f"error in generating RAG answer : {e}")
        raise

def reflector(st:state)->state:
    try:
        rag_ans=st.rag_ans
        input=st.input
        template = """
You are a medical policy expert assistant.

Task: Refine the RAG response and provide only relevant information.

User Query: "{input}"
RAG Answer: "{rag_ans}"

Instructions:
1. extract only relevant policy details:
2. follow a professional medical policy struture while answering (headings and coverages and its source(line no. - exact line number mentioned in the rag_source) mention in the policy)
3. Don't add information not in RAG answer
4. Keep the answer short , crisp and stright forward
5. Add additional points mentioned in the policy only the one which you find relevant to the user query.
Provide refined answer:
CRITICAL: If the RAG answer only mentions procedure codes without actual coverage details, clarify that procedure codes alone do not indicate coverage terms, amounts, or eligibility.
"""
        prompt=PromptTemplate(template=template,input_variables=["rag_ans","input"])
        inp=prompt.format(
            rag_ans=rag_ans,
            input=input
        )
        response=model.invoke(inp)
        st.ref_ans=response
        logger.debug("REFLECTOR ANSWER GENERATED SUCCESSFULLY.!!")
        return st

    except Exception as e:
        logger.error("failed to generate reflector answer")
        raise