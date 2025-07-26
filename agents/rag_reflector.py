from RAG.load_text import load_data
from RAG.splitting import split_text
from RAG.database import store_to_cloudinary,load_from_cloudinary
from RAG.searching import semantic_search
from pathlib import Path
from pydantic_models import state, PolicyDecision
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from llm import model
from logs.logging_config import logger

parser = PydanticOutputParser(pydantic_object=PolicyDecision)

def rag_agent(st: state) -> state:
    try:
        file_path = st.file_path
        doubt = st.question.procedure
        age= st.question.age
        location=st.question.duration
        duration=st.question.Location

        text = load_data(file_path=file_path)
        chunks = split_text(text=text)

        store_to_cloudinary(chunks, index_name="lightweight_index")
        db=load_from_cloudinary()
        content, sources = semantic_search(doubt, use_cloudinary=True)

       
        template = """
You are a Medical Insurance Claims Analyst specializing in health policy interpretation and claim assessment.

**POLICY DOCUMENT:**
{context}

**MEDICAL CLAIM:**
{doubt}

**POLICYHOLDER DETAILS:**
- Age: {age}
- Location: {location}
- Policy Tenure: {duration}

**ANALYSIS REQUIREMENTS:**

1. **Coverage Assessment:**
   - Check if the medical procedure/treatment is covered
   - Identify applicable sum insured or coverage limits
   - Note any pre-authorization requirements

2. **Exclusion Review:**
   - Look for specific medical exclusions
   - Check waiting period compliance
   - Verify pre-existing disease conditions

3. **Financial Evaluation:**
   - Determine coverage percentage (if mentioned)
   - Check for co-payment or deductibles
   - Note room rent limits or sub-limits
   - Extract specific monetary amounts from policy

4. **Decision Criteria:**
   - "Approved": If procedure is covered and no exclusions apply
   - "Rejected": If procedure is specifically excluded
   - "Pending": If additional documentation or verification needed

**CRITICAL REASONING INSTRUCTIONS:**
- In your justification, ALWAYS quote the exact text from the policy document
- Use phrases like: "According to the policy document: '[exact policy text]'"
- Explain WHY something is covered or excluded by referencing specific clauses
- If a procedure is covered, mention the exact coverage terms from the document
- If excluded, quote the exact exclusion clause
- Be very specific about which section/clause applies to this claim
- Connect the claim directly to the policy language

**Example Reasoning Format:**
- "This procedure is covered because the policy states: '[exact quote from policy]'"
- "This claim is excluded as per clause X which specifically mentions: '[exact exclusion text]'"
- "The coverage limit applies as the policy clearly states: '[exact financial clause]'"

**Important:** 
- For approved_amount: Extract exact amount from policy or calculate based on coverage percentage
- If no specific amount available, use "NA"
- ALWAYS reference the exact policy text that led to your decision

{format_instructions}
"""
        prompt = PromptTemplate(
            template=template,
            input_variables=["doubt", "context","age","location","duration"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        chain = prompt | model | parser

        response = chain.invoke({
            "doubt": doubt,
            "context": content,
            "age":age,
            "location":location,
            "duration":duration
        })

        st.rag_ans = response
        logger.debug("RAG ANSWER GENERATED SUCCESSFULLY!!")
        return st

    except Exception as e:
        logger.error(f"Error in generating RAG answer: {e}")
        raise
