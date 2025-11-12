# 📋 Insurance Policy Document Analyzer

> **AI-powered RAG system for instant policy document analysis and claim verification**

An intelligent document analysis system that extracts precise information from insurance policy documents using advanced RAG (Retrieval-Augmented Generation) techniques. Built with LangGraph, LangChain, and Groq LLM, it provides accurate, context-aware answers to policy-related queries with citation support.

![Python Version](https://img.shields.io/badge/python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-orange)
![Status](https://img.shields.io/badge/status-development-yellow)

## 🌟 Features

- **📄 Multi-Format Support**: Process PDF, TXT, EML, and MSG files
- **🧠 Hybrid Retrieval**: Combines vector search (ChromaDB) + keyword search (BM25)
- **🎯 Query Understanding**: Automatic extraction of core questions from user input
- **🔍 Advanced Compression**: Multi-stage document filtering and reranking
- **📊 Table Extraction**: Specialized handling of tabular data in PDFs
- **⚡ Parallel Processing**: Concurrent question processing for faster responses
- **🎨 Context Reordering**: Long-context optimization for better accuracy
- **🔐 Secure API**: Token-based authentication for production endpoints
- **📝 Citation Support**: Answers include specific clause references
- **🌐 RESTful API**: Easy integration with any frontend application

## 🎯 Use Cases

### For Insurance Companies
- **Policy Q&A Systems**: Instant answers to policy coverage questions
- **Claim Verification**: Automated eligibility checking against policy terms
- **Customer Support**: 24/7 self-service policy information
- **Compliance Checking**: Verify policy terms and conditions

### For Insurance Agents
- **Quick Policy Lookup**: Find specific coverage details instantly
- **Client Inquiries**: Answer client questions with accurate citations
- **Policy Comparison**: Extract comparable information from multiple policies

### For Policyholders
- **Coverage Understanding**: Clear explanations of policy terms
- **Claim Eligibility**: Check if specific procedures are covered
- **Benefit Verification**: Understand limits, caps, and exclusions

## 🏗️ Architecture

```
PDF/TXT Upload → Document Loading → Text Splitting → Embeddings
                                                          ↓
User Query → Query Generator → Hybrid Retrieval → Compression → LLM → Answer
                                      ↓                    ↓
                              Vector (ChromaDB)    Clustering +
                                   +               Redundancy Filter
                              BM25 (Keyword)       + Reordering
```

### RAG Pipeline Components

1. **Document Processing**
   - PDF extraction with table detection (pdfplumber)
   - Email parsing (.eml, .msg)
   - Text chunking with overlap

2. **Embedding & Storage**
   - Google Generative AI embeddings
   - ChromaDB vector storage
   - Persistent database caching

3. **Retrieval Strategy**
   - Ensemble retriever (70% vector, 30% keyword)
   - Top-K retrieval (8 vector + 5 BM25)
   - Contextual compression pipeline

4. **Answer Generation**
   - Groq Llama 3 70B model
   - Policy-specific prompt engineering
   - Structured output with citations

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **LLM Orchestration**: LangChain, LangGraph
- **Vector Database**: ChromaDB
- **Embeddings**: Google Generative AI (embedding-001)
- **LLM**: Groq (Llama 3 70B)
- **Document Processing**: pdfplumber, PyPDF, LangChain loaders
- **Search**: BM25 (rank_bm25), Vector Search
- **Compression**: Cohere Rerank (optional)

## 📋 Prerequisites

- Python 3.9 or higher
- API Keys:
  - Groq API Key (required)
  - Google AI API Key (required)
  - Cohere API Key (optional, for reranking)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Navodit-Sahai/MediClaim.git
cd MediClaim
```

### 2. Create Virtual Environment

```bash
python -m venv hackenv
source hackenv/bin/activate  # On Windows: hackenv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Optional
EXPECTED_TOKEN=your_secure_token_here  # For API authentication
```

**Get Your API Keys:**
- Groq: [console.groq.com](https://console.groq.com)
- Google AI: [ai.google.dev](https://ai.google.dev)
- Cohere: [cohere.com](https://cohere.com)

## 🎮 Usage

### Starting the Backend Server

```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Access interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Health Check
```bash
GET http://localhost:8000/api/v1/health
```

#### 2. Summarize Policy Document
```bash
POST http://localhost:8000/api/v1/summarize
Content-Type: multipart/form-data

Form Data:
- input_text: "What is the coverage for heart surgery?, What is the waiting period?"
- file: policy_document.pdf
```

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/summarize" \
  -F "input_text=What is the coverage for cardiac surgery?" \
  -F "file=@policy.pdf"
```

**Response:**
```json
{
  "result": [
    "Cardiac surgery is covered up to $50,000 per policy year with a 90-day waiting period for pre-existing conditions as per Section 4.2.c.",
    "..."
  ]
}
```

#### 3. HackRx Endpoint (Authenticated)
```bash
POST http://localhost:8000/api/v1/hackrx/run
Authorization: Bearer your_token_here
Content-Type: application/json

{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is covered under emergency care?",
    "What is the deductible amount?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "Emergency care is covered 100% after $500 deductible as per Section 3.1...",
    "Annual deductible is $500 per individual, $1000 per family as stated in Section 2.4..."
  ]
}
```

## 📂 Project Structure

```
insurance-policy-analyzer/
├── RAG/
│   ├── __init__.py
│   ├── database.py              # Vector DB, retrieval, RAG pipeline
│   ├── load_text.py             # Document loaders (PDF, TXT, email)
│   └── splitting.py             # Text chunking logic
├── agents/
│   ├── query_generator.py       # Query extraction and formatting
│   └── rag_reflector.py         # RAG agent with parallel processing
├── logs/
│   └── logging_config.py        # Logging configuration
├── backend.py                   # FastAPI application
├── lang.py                      # LangGraph workflow builder
├── llm.py                       # LLM configuration
├── pydantic_models.py          # Data models and schemas
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🔧 Configuration

### Customizing Retrieval Settings

Edit `RAG/database.py`:

```python
# Adjust retrieval counts
vector_retriever = Chroma(...).as_retriever(search_kwargs={"k": 8})  # Change k
keyword_retriever.k = 5  # Change BM25 count

# Adjust ensemble weights
retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.7, 0.3]  # Adjust weights (must sum to 1.0)
)
```

### Modifying Chunking Strategy

Edit `RAG/splitting.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Tokens per chunk
    chunk_overlap=100    # Overlap between chunks
)
```

### Changing LLM Model

Edit `llm.py`:

```python
model = ChatGroq(model_name="llama3-70b-8192")  # Options:
# - llama3-70b-8192 (current, best accuracy)
# - llama3-8b-8192 (faster, lighter)
# - mixtral-8x7b-32768 (good balance)
```

### Custom Prompts

Edit prompt templates in `RAG/database.py` or `agents/rag_reflector.py`:

```python
prompt_template = PromptTemplate(
    template="""Your custom prompt here...
    
    Context: {context}
    Question: {question}
    
    Instructions: ...
    """,
    input_variables=["context", "question"]
)
```

## 🧪 Testing

### Test with Sample Policy

```python
from lang import process_questions

# Single question
answer = process_questions(
    "What is the coverage for knee replacement?",
    "path/to/policy.pdf"
)
print(answer)

# Multiple questions
questions = [
    "What is the deductible amount?",
    "Are dental procedures covered?",
    "What is the waiting period for surgery?"
]
answers = process_questions(questions, "path/to/policy.pdf")
for ans in answers:
    print(ans)
```

### Using Postman

1. Import the API into Postman
2. Create a POST request to `http://localhost:8000/api/v1/summarize`
3. Select `form-data` body type
4. Add `input_text` and `file` fields
5. Send request

## 🔍 Advanced Features

### Query Generator

Automatically extracts core questions from natural language:

**Input:** "Can you tell me what the coverage is for heart surgery procedures?"

**Extracted:** "heart surgery coverage"

### Contextual Compression Pipeline

1. **Clustering Filter**: Groups similar chunks (reduces redundancy)
2. **Redundancy Filter**: Removes near-duplicate content
3. **Long Context Reorder**: Optimizes chunk ordering for LLM

### Parallel Processing

Multiple questions are processed concurrently using `ThreadPoolExecutor`:

```python
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_function, questions))
```

## 📊 Performance Considerations

- **ChromaDB Persistence**: First query creates DB, subsequent queries are faster
- **Embedding Cache**: Embeddings are cached in `chroma_db/` directory
- **Concurrent Processing**: 10 parallel workers for multi-question requests
- **Document Size**: Optimized for policies up to 100 pages

## 🐛 Troubleshooting

### "GROQ_API_KEY not found"
**Solution:** Ensure `.env` file exists with valid API key

### ChromaDB errors
**Solution:** Delete `chroma_db/` directory and restart

### "Failed to load PDF"
**Solution:** 
- Ensure PDF is not password-protected
- Try re-saving PDF (some PDFs have encoding issues)
- Check file isn't corrupted

### Slow response times
**Solutions:**
- Reduce chunk count in retrieval settings
- Use lighter LLM model (llama3-8b)
- Reduce chunk_size in splitting config

### Out of memory errors
**Solutions:**
- Process documents in smaller batches
- Reduce chunk_size and retrieval k values
- Use pagination for large documents

## 🔒 Security

- ⚠️ Never commit `.env` file or API keys
- 🔐 Token-based authentication for production endpoints
- 🗑️ Temporary files automatically cleaned up
- 📁 Uploaded files processed in temp directory

## 💰 API Costs

### Per 1000 Requests (Approximate)

- **Groq LLM**: ~$0.10-0.50 (70B model)
- **Google Embeddings**: ~$0.01-0.05
- **Cohere Rerank**: ~$0.02 (if enabled)

*Actual costs depend on document size and query complexity*

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. Frontend UI development
2. Additional document formats (DOCX, HTML)
3. Multi-language support
4. Enhanced caching strategies
5. Query result caching
6. Batch processing optimization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Navodit Sahai**
- GitHub: [@Navodit-Sahai](https://github.com/Navodit-Sahai)
- Email: sahainavodit781@gmail.com

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) for RAG framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Groq](https://groq.com/) for fast LLM inference
- [Google AI](https://ai.google.dev/) for embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for backend framework

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Email: sahainavodit781@gmail.com

## 🗺️ Roadmap

- [ ] Web frontend interface (React/Streamlit)
- [ ] User authentication and session management
- [ ] Document comparison features
- [ ] Multi-document analysis
- [ ] Export answers to PDF/DOCX
- [ ] Real-time collaborative querying
- [ ] Integration with insurance databases
- [ ] Mobile app development
- [ ] Conversation history and context
- [ ] Custom policy templates
- [ ] Batch document processing
- [ ] Advanced analytics dashboard
- [ ] Multi-language support
- [ ] Voice query input
- [ ] Deployment on cloud platforms

## 📚 Example Queries

### Coverage Questions
```
"What is covered under maternity benefits?"
"Is mental health treatment included?"
"What are the limits for emergency care?"
```

### Eligibility Questions
```
"What is the waiting period for pre-existing conditions?"
"Am I eligible for dental coverage?"
"What age restrictions apply to dependents?"
```

### Financial Questions
```
"What is my deductible amount?"
"What are the co-payment requirements?"
"What is the maximum coverage limit?"
```

### Procedure-Specific
```
"Is knee replacement surgery covered?"
"What about cardiac procedures?"
"Are diagnostic tests included?"
```

---

⭐ If this project helps you build better insurance systems, please give it a star on GitHub!

**Making insurance policies understandable! 📋✨**
