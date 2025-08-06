import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, DocumentCompressorPipeline
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder
)
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from RAG.load_text import load_data
from RAG.splitting import split_text
from pydantic_models import state
from llm import model

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

class GoogleEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [genai.embed_content(
            model="models/embedding-001",
            content=doc,
            task_type="retrieval_document"
        )["embedding"] for doc in texts]

    def embed_query(self, text):
        return genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

def vector_Search(st: state):
    text = load_data(st.file_path)
    chunks = split_text(text)
    embedding = GoogleEmbedding()

    if not os.path.exists("chroma_db"):
        Chroma.from_documents(chunks, embedding=embedding, persist_directory="chroma_db").persist()

    vector_retriever = Chroma(
        persist_directory="chroma_db", 
        embedding_function=embedding
    ).as_retriever(search_kwargs={"k": 8})
    
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5

    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever],
        weights=[0.7, 0.3]
    )
    
    num_docs = len(chunks)
    clusters = min(4, num_docs)
    
    filter1 = EmbeddingsClusteringFilter(embeddings=embedding, num_clusters=clusters)
    filter2 = EmbeddingsRedundantFilter(embeddings=embedding, threshold=0.8)
    
    reordering1 = LongContextReorder()

    pipeline = DocumentCompressorPipeline(
        transformers=[filter1, filter2, reordering1]
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline,
        base_retriever=retriever
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a highly skilled Policy Document Analyst. Your task is to extract accurate, legally worded summaries from insurance policy documents.

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANSWERING INSTRUCTIONS:**
- Use **formal and professional insurance terminology**.
- Your answer must be **complete, self-contained, and policy-style**.
- **Summarize in 25–40 words**, using language similar to that found in policy booklets.
- Reference **specific eligibility rules, limits, conditions, or periods** when present.
- Where applicable, **state caps, durations, criteria**, and other relevant limitations.
- Avoid conversational tone or unnecessary legal citations.
- Do **not** include "**Section 4.2.c**" unless essential; prefer quoting or paraphrasing the clause content.

**FINAL ANSWER:**"""
    )

    hybrid_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )

    questions = [{"query": q} for q in st.input]
    results = hybrid_chain.apply(questions)
    response = [r["result"] for r in results]
    st.rag_ans=response
    return st