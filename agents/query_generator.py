from langchain.output_parsers import PydanticOutputParser
from pydantic_models import state, query
from langchain.prompts import PromptTemplate
from llm import model
from logs.logging_config import logger

def query_generator(st: state):
    try:
        combined_input = "\n".join([f"{i+1}. {inp}" for i, inp in enumerate(st.input)])
        parser = PydanticOutputParser(pydantic_object=query)

        template = """
You are an expert at extracting key information from user queries about insurance claims.

For each user input, extract the CORE QUESTION/PROCEDURE by:
- Removing filler words (what, how, can, please, etc.)
- Identifying the main medical procedure, claim type, or specific question
- Keeping only the essential information needed for search

Examples:
Input: "What is the coverage for knee replacement surgery?"
Core: "knee replacement surgery coverage"

Input: "How to claim for diabetic medication expenses?"
Core: "diabetic medication claim process"

Input: "Can you help me understand cardiac surgery benefits?"
Core: "cardiac surgery benefits"

{format_instructions}

User inputs:
{user_input}

Extract core procedure/question for each input above.
""".strip()

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["user_input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        formatted_prompt = prompt_template.format(user_input=combined_input)
        response = model.invoke(formatted_prompt)
        
        try:
            parsed_response = parser.parse(response.content)
            st.questions = parsed_response if isinstance(parsed_response, list) else [parsed_response]
        except:
            st.questions = [query(procedure=inp.strip()) for inp in st.input]

        logger.debug("All questions parsed successfully!")
        return st

    except Exception as e:
        logger.error(f"Failed to generate question(s): {e}")
        st.questions = [query(procedure=inp) for inp in st.input] if st.input else []
        return st