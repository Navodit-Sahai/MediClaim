from RAG.load_text import load_data
from RAG.splitting import split_text
from RAG.database import store_to_cloudinary,load_from_cloudinary
from RAG.searching import semantic_search
from pathlib import Path
from pydantic_models import state, Justification, PolicyDecision
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from llm import model
from logs.logging_config import logger

parser = PydanticOutputParser(pydantic_object=PolicyDecision)

def rag_agent(st: state) -> state:
    try:
        file_path = st.file_path
        doubt = st.input
        text = load_data(file_path=file_path)
        chunks = split_text(text=text)

        store_to_cloudinary(chunks, index_name="lightweight_index")
        db=load_from_cloudinary()
        content, sources = semantic_search(doubt, use_cloudinary=True)

        template = """
You are an expert AI system acting as an insurance claims processor.
Your task is to analyze the provided insurance policy context and evaluate the user's query based *only* on the information within that document.

**INSURANCE POLICY DOCUMENT:**
---
{context}
---

**USER QUERY:**
---
{doubt}
---

Based strictly on the document, evaluate the query. You must make a final decision of either "Approved", "Rejected", or "Pending".
Do mention the context of your reason from the line no. and clause mentioned in the rag answer .
Consider all the points mentionned in the context and dont miss any point which can be relevant .
If the information is insufficient to make a clear decision, you MUST select 'Pending' and your 'reason' must clearly state what specific information is missing from the user's query. For example, if a rule depends on the cause being an accident and the query does not specify the cause, you must state that the cause of the injury (illness or accident) is required.
Do mention any details regarding the amount insured,covered aur not covered.
Don't remove any point that might be relevant. try to cover all the relevant points around the query.
For the 'Approved Amount', if the claim is not approved or if no specific amount is mentioned, you must state 'NA'.

Provide a structured JSON response according to the schema. Keep the 'reason' for each justification as simple and concise as possible.

{format_instructions}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["doubt", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | model | parser

        response = chain.invoke({
            "doubt": doubt,
            "context": content
        })

        st.rag_ans = response
        logger.debug("RAG ANSWER GENERATED SUCCESSFULLY!!")
        return st

    except Exception as e:
        logger.error(f"Error in generating RAG answer: {e}")
        raise
