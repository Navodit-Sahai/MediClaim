from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
model=ChatGroq(model_name="llama3-70b-8192")