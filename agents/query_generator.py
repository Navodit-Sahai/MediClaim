from langchain.output_parsers import PydanticOutputParser
from pydantic_models import state, query
from langchain.prompts import PromptTemplate
from llm import model
from logs.logging_config import logger
def query_generator(st: state):
    try:
        parser = PydanticOutputParser(pydantic_object=query)

        template = """
    You are an intelligent assistant that extracts structured information from user queries related to insurance claims.

    Extract the following fields from the user's input:
    - Age of the person
    - Medical procedure (or query related to the claim)
    - Location of the claimant
    - Duration or validity of the insurance policy

    {format_instructions}

    User input: "{user_input}"
    """.strip()
        
        user_input = st.input

        prompt = PromptTemplate(
            template=template,
            input_variables=["user_input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        formatted_prompt = prompt.format(user_input=user_input)
        response = model.invoke(formatted_prompt)
        parsed_question = parser.parse(response.content)
        st.question = parsed_question
        logger.debug("question created successfully.!!")
        return st
    
    except Exception as e:
        logger.error("failed to generate question : {e}")
        

