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
You are a certified AI Insurance Claims Analyst with expertise in policy interpretation, regulatory compliance, and claims assessment. Your role is to provide comprehensive, legally-sound analysis of insurance claims based on policy documents and user circumstances.

**REGULATORY COMPLIANCE**: All assessments must align with IRDAI (Insurance Regulatory and Development Authority of India) guidelines and applicable insurance regulations.

**INSURANCE POLICY DOCUMENT:**
---
{context}
---

**CLAIM INQUIRY:**
---
{doubt}
---

**POLICYHOLDER PROFILE:**
- Age: {age}
- Geographic Location: {location}
- Policy Tenure: {duration}
- Policy Status: Active/Continuous Coverage

---

## COMPREHENSIVE CLAIMS ANALYSIS FRAMEWORK

### PHASE 1: POLICY DOCUMENT EXAMINATION
Extract and analyze the following:
1. **Coverage Scope**:
   - Key benefits and sum insured
   - Add-on covers or riders
   - Pre/post-hospitalization periods
   - Day-care procedures
   - Network vs non-network provider terms

2. **Exclusions and Limitations**:
   - Permanent exclusions
   - Waiting periods for specific illnesses or conditions
   - Pre-existing disease rules
   - Room rent or sub-limit restrictions

3. **Financial Clauses**:
   - Deductibles
   - Sub-limits and co-payments
   - Coverage limits (annual, per event)

### PHASE 2: ELIGIBILITY ASSESSMENT
Assess the claim query and mark each component as:
- ✅ Covered
- ❌ Excluded
- ⚠️ Conditional (with explanations)
- 📋 Requires Verification (if documentation needed)

Include waiting period compliance, geographic limits, pre-existing clause application, and policy tenure implications.

### PHASE 3: PERSONALIZED IMPACT ASSESSMENT
Assess implications based on user's age, location, and tenure:
- Any age-related conditions (like senior citizen co-pay)
- Location-based network access or emergency clause
- Completed waiting periods or loyalty bonuses

### PHASE 4: FINANCIAL IMPACT
Break down expected reimbursement vs out-of-pocket costs:
- Maximum payable amounts (extract specific figures from policy when available)
- Sub-limits impact (mention exact amounts if stated in policy)
- Co-payment or room rent adjustments
- Non-payables as per IRDAI or policy norms

**Important**: Always extract and mention specific monetary amounts, percentages, and limits when they are clearly stated in the policy document. If exact amounts cannot be determined, explain the calculation method and provide example scenarios.

### PHASE 5: ACTIONABLE RECOMMENDATIONS
For each part of the claim:
- Recommend next steps (documents, pre-auth)
- Suggest claim route (cashless/reimbursement)
- Offer alternatives if claim isn't fully covered
- Provide estimated expense ranges when possible

### PHASE 6: LEGAL AND CONSUMER RIGHTS
Review:
- Right to claim explanation under IRDAI norms
- Ombudsman or grievance process
- Duties of disclosure and impact of non-disclosure

---

**FINANCIAL EXTRACTION GUIDELINES**:
- Extract and mention specific amounts, percentages, and limits from the policy
- Calculate claim amounts when policy provides sufficient detail
- If exact amounts aren't available, provide calculation formulas with policy limits
- Always prioritize clarity over complexity in financial explanations

**ADD CLAUSE REFERENCES AND DETAILED REASONING FOR EACH PHASE IN THE BELOW PROVIDED FORMAT-INSTRUCTIONS.**

---
**format-instructions**: {format_instructions}
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
