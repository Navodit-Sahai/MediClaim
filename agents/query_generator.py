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
You are an intelligent assistant that extracts structured information from multiple user queries related to insurance claims.

Extract the following fields from EACH user input:
- Age of the person  
- Medical procedure (or query related to the claim)
- Location of the claimant
- Duration or validity of the insurance policy

Process all queries below and return a list of extracted information:

{format_instructions}

Multiple User inputs:
{user_input}

Return as a list where each item corresponds to each numbered query above.
""".strip()

        prompt_template = PromptTemplate(
            template=template,
            input_variables=["user_input"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        formatted_prompt = prompt_template.format(user_input=combined_input)
        response = model.invoke(formatted_prompt)

        try:
            parsed_questions = []
            for i, inp in enumerate(st.input):
                parsed_questions.append(query(
                    age=None,
                    procedure=inp,
                    location=None,
                    duration=None
                ))
            st.questions = parsed_questions
        except:
            st.questions = [query(procedure=inp) for inp in st.input]

        logger.debug("All questions parsed successfully!")
        return st

    except Exception as e:
        logger.error(f"Failed to generate question(s): {e}")
        st.questions = [query(procedure=inp) for inp in st.input] if st.input else []
        return st
