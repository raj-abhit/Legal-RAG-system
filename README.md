# Legal RAG System

A Retrieval-Augmented Generation (RAG) system for querying Indian legal documents using Groq's API and LangChain.

## Features

- Free and fast: uses Groq's free API for quick responses
- Accurate retrieval: semantic search across legal documents using FAISS
- Source citations: answers include relevant document sections
- Easy to use: interactive CLI, API server, and simple HTML UI
- Extensible: add more legal documents easily

## Prerequisites

- Python 3.8 or higher
- Groq API key (free at console.groq.com)
- Legal documents (TXT or PDF) in the `documents/` folder

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd Legal-RAG-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Get Your Groq API Key

1. Visit console.groq.com
2. Sign up for a free account
3. Create an API key from the dashboard
4. Save it securely

### 3. Prepare Documents

- Place your `.txt` or `.pdf` files in the `documents/` folder
- Example files are already included in this repo

### 4. Set Up Environment Variables (Optional)

Create a `.env` file:
```bash
GROQ_API_KEY=your_api_key_here
```

## Usage

### Interactive CLI Mode

```bash
python interactive_legal_rag_v2.py
```

This will:
1. Prompt for your Groq API key (if not in .env)
2. Load or create the vector store
3. Start an interactive Q&A session

Example session:
```
Your question: What are the fundamental rights in Article 19?

Searching and analyzing...

ANSWER:
Article 19 of the Constitution of India guarantees six fundamental 
rights to all citizens:
1. Freedom of speech and expression
2. Right to assemble peaceably and without arms
...
```

### Python API Mode

```python
from legal_rag_system import LegalRAGSystem

# Initialize
rag = LegalRAGSystem(groq_api_key="your-key-here")

# First time: Load and process documents
documents = rag.load_documents([
    "documents/constitution_of_india.pdf",
    "documents/bns.pdf",
    "documents/bnss.pdf",
    "documents/bsa.pdf",
    "documents/landmark_cases.txt",
])
chunks = rag.chunk_documents(documents)
rag.create_vectorstore(chunks)
rag.save_vectorstore("legal_vectorstore")

# Setup QA chain
rag.setup_qa_chain()

# Query
result = rag.query("Explain Section 103 of BNS")
rag.print_answer(result)
```

### Subsequent Uses

```python
from legal_rag_system import LegalRAGSystem

# Initialize
rag = LegalRAGSystem(groq_api_key="your-key-here")

# Load existing vectorstore (much faster)
rag.load_vectorstore("legal_vectorstore")
rag.setup_qa_chain()

# Query
result = rag.query("What is Article 21?")
print(result["answer"])
```

## Example Questions

Try asking:
- "What are the fundamental rights guaranteed by the Constitution?"
- "Explain Section 103 of BNS"
- "What is the difference between Articles 14 and 21?"
- "What are the punishments for theft under BNS?"
- "What is the right to equality?"
- "Explain culpable homicide vs murder under BNS"

## Configuration

### Adjust Chunk Size

In `legal_rag_system.py`, modify the `chunk_documents` method:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer context
    chunk_overlap=200,  # Increase for more context preservation
)
```

### Change Number of Retrieved Documents

In `setup_qa_chain` method:

```python
retriever=self.vectorstore.as_retriever(
    search_kwargs={"k": 4}  # Retrieve top 4 chunks (increase for more context)
)
```

### Use Different Groq Models

Available models:
- `mixtral-8x7b-32768` (default - balanced)
- `llama3-70b-8192` (more powerful)
- `llama3-8b-8192` (faster, less accurate)

Change in `__init__` method:

```python
self.llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-70b-8192",  # Change here
)
```

## API Server + Simple UI

Run the API server:

```bash
python api_server.py
```

Open `frontend.html` in a browser and query the API.

## Project Structure

```
legal-rag-system/
|
|-- legal_rag_system.py          # Main RAG system class
|-- interactive_legal_rag_v2.py  # Interactive CLI
|-- api_server.py                # API server
|-- frontend.html                # Simple web UI
|-- requirements.txt             # Python dependencies
|-- README.md                    # This file
|
|-- documents/                   # Your legal documents (PDF/TXT)
|   |-- constitution_of_india.pdf
|   |-- bns.pdf
|   |-- bnss.pdf
|   |-- bsa.pdf
|   `-- landmark_cases.txt
|
`-- legal_vectorstore/           # Generated vector store (auto-created)
    |-- index.faiss
    `-- index.pkl
```

## Troubleshooting

### "No module named 'langchain'"
```bash
pip install -r requirements.txt
```

## Security Notes

- Never commit your `.env` file with API keys
- Keep your Groq API key private

## Disclaimer

This system is for informational purposes only and does not constitute legal advice. Always consult qualified legal professionals for legal matters.
