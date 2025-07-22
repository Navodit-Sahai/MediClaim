from langchain.tools import tool
from pydantic_models import state, query
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from llm import model
from logs.logging_config import logger


def generate_query(st: state) -> state:
    """
    Parses user input string and returns a structured query (age, procedure, duration).
    """
    try:
        input_text = st.input  
        

        if not input_text or not input_text.strip():
            logger.warning("Empty input provided")
            st.question = query(age=None, procedure="unknown", duration=None)
            return st
            
        parser = PydanticOutputParser(pydantic_object=query)

        prompt = ChatPromptTemplate.from_template(
            """
You are an intelligent information extractor. Extract the following details from the user's input:
1. Name - Name of the policy holder
2. Age - Age of the policy holder (should be a number).
3. Procedure - Medical procedure to be covered in the policy or asked in the query.
4. Duration - Duration of the policy.


User Input:
-----------
{input}

Instructions:
Return output as JSON matching this format:
{format_instructions}
"""
        )

        chain = prompt | model | parser
        raw_response = chain.invoke({
            "input": input_text,
            "format_instructions": parser.get_format_instructions()
        })

        if hasattr(raw_response, 'content'):
            ques = parser.parse(raw_response.content)
        else:
            ques = raw_response

        st.question = ques
        logger.debug(f"Parsed question object: {ques}")
        logger.debug("question created successfully!!")
        return st

    except Exception as e:
        logger.error(f"Error parsing input: {e}")

        st.question = query(age=None, procedure="unknown", duration=None)
        return st