# Legal RAG System - Simplified Edition

> Intelligent Q&A over Indian Legal Documents using LangChain + FAISS + Groq

## 📋 Quick Start (2 minutes)

### 1. Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key: https://console.groq.com

### 3. Place Documents
Add PDF/text legal documents to `documents/` folder:
```
documents/
  ├── Constitution.pdf
  ├── BNS_2023.pdf
  └── other_laws.pdf
```

### 4. Run System
**Interactive CLI:**
```bash
python interactive_legal_rag_v2.py
```

**REST API:**
```bash
python api_server.py
# Access: http://localhost:8000
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   Your Question (English/Hindi)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Legal RAG System (legal_rag_system.py)             │
│   ┌──────────────────────────────┐  │
│   │ 1. Language Detection        │  │
│   │    (Hindi/English/Hinglish)  │  │
│   └──────────────────────────────┘  │
│   ┌──────────────────────────────┐  │
│   │ 2. Conversation Memory       │  │
│   │    (Last 5 Q&A pairs)        │  │
│   └──────────────────────────────┘  │
│   ┌──────────────────────────────┐  │
│   │ 3. Semantic Search (FAISS)   │  │
│   │    (Find relevant chunks)    │  │
│   └──────────────────────────────┘  │
│   ┌──────────────────────────────┐  │
│   │ 4. LLM Response (Groq)       │  │
│   │    (Generate answer)         │  │
│   └──────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Answer + Sources + Metadata        │
│   (in requested language)            │
└──────────────────────────────────────┘
```

---

## 🎯 Core Features

✅ **Multi-Language Support**
- English  
- Hindi (Devanagari)
- Hinglish (romanized Hindi)

✅ **Conversation Memory**
- Remembers last 5 Q&A pairs
- Detects follow-up questions
- Injects context into prompts

✅ **Source Attribution**
- Every answer cites source documents
- Page numbers included
- Document excerpts shown

✅ **Indian Legal Focus**
- Constitution of India
- BNS 2023 (Current criminal law)
- BNSS 2023 (Criminal procedure)
- BSA 2023 (Evidence law)
- Special acts: RTI, POCSO, IT Act, NDPS, etc.

---

## 📝 Usage Examples

### Interactive CLI
```bash
$ python interactive_legal_rag_v2.py

📋 Your question: What is Article 21?
🔍 Searching legal documents...

📌 ANSWER:
Article 21 of the Indian Constitution states: "No person shall be 
deprived of his life or personal liberty except according to procedure 
established by law." This article guarantees the right to life and 
personal liberty to all persons...

📚 Sources:
  1. documents/Constitution.pdf (Page 45)
```

### REST API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is BNS Section 103?",
    "session_id": "user-123"
  }'
```

**Response:**
```json
{
  "answer": "BNS Section 103 provides punishment for murder...",
  "sources": [
    {
      "document": "documents/BNS_2023.pdf",
      "page": 102,
      "excerpt": "Murder..."
    }
  ],
  "session_id": "user-123",
  "language": "English",
  "is_followup": false
}
```

### Python Integration
```python
from legal_rag_system import LegalRAGSystem

# Initialize
rag = LegalRAGSystem()
rag.load_vectorstore()  # Use pre-built embeddings

# Query
result = rag.query("What is the RTI Act?")
print(result['answer'])
print(result['sources'])
```

---

## 🚀 First-Time Setup (Embeddings)

The first time you run the system with new documents:

```bash
python interactive_legal_rag_v2.py

# System will detect no vectorstore and ask:
# "Create embeddings from these PDFs? (y/n):"
# → Answer 'y'

# ⏳ Wait 10-15 minutes for embedding generation
# (This is a one-time process - future runs are instant!)
```

**What's happening:**
1. PDFs are split into 1000-char chunks
2. Each chunk gets embedded with all-MiniLM-L6-v2 model
3. Embeddings are stored in FAISS index
4. Index is saved to `legal_vectorstore/` folder

**Next time:**
- System loads pre-computed embeddings instantly ⚡

---

## 🔧 Configuration

Edit `.env` file:

```env
# Required
GROQ_API_KEY=your_key_here

# Optional - Change LLM model
# GROQ_MODEL=llama-3.3-70b-versatile  (default)
# GROQ_MODEL=mixtral-8x7b-32768
# GROQ_MODEL=llama3-70b-8192

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Chat Data Storage
CHAT_DATA_FILE=chat_data.json

# Vector Store
VECTOR_STORE_PATH=./legal_vectorstore
```

---

## 📚 Supported Laws

| Category | Laws |
|----------|------|
| **Constitution** | Constitution of India |
| **Criminal Law** | BNS 2023 (new), IPC 1860 (old - comparison) |
| **Procedure** | BNSS 2023 (new), CrPC 1973 (old - comparison) |
| **Evidence** | BSA 2023 (new), Evidence Act 1872 (old - comparison) |
| **Special Acts** | RTI, POCSO, IT Act, NDPS, Motor Vehicles Act |

---

## ⚡ Performance

| Operation | Time |
|-----------|------|
| Load vector store | 2-3 sec |
| Single query | 1-3 sec |
| Embedding generation (first time) | 10-15 min |

---

## 🛠️ Troubleshooting

### "GROQ_API_KEY not found"
```bash
# Add to .env file
GROQ_API_KEY=your_key_here
```

### "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "Slow query responses"
- Ensure vector store is loaded from disk (not regenerating)
- Close other CPU-intensive programs
- Use faster machine if available

### "Out of memory during embedding"
- Reduce PDF file size
- Process documents in smaller batches
- Increase RAM/swap space

---

## 📂 Project Structure

```
law/
├── legal_rag_system.py          # Core RAG engine (~400 lines)
├── api_server.py                # REST API (~300 lines)
├── interactive_legal_rag_v2.py  # CLI interface (~150 lines)
├── requirements.txt             # Python dependencies
├── .env                         # Configuration (create this)
├── README.md                    # This file
│
├── documents/                   # Place legal PDFs here
│   ├── Constitution.pdf
│   ├── BNS_2023.pdf
│   └── ...
│
├── legal_vectorstore/           # Generated embeddings (auto-created)
│   ├── index.faiss
│   └── index.pkl
│
└── chat_data.json               # Session data (auto-created)
```

---

## 🎓 How It Works

### 1️⃣ Document Loading
- PDFs extracted via PyPDF2
- Text split into 1000-char chunks (with 200-char overlap)
- Each chunk preserves document metadata (source, page)

### 2️⃣ Embeddings
- Chunks converted to 384-dim vectors
- Uses HuggingFace's all-MiniLM-L6-v2 (free, fast, local)
- Stored in FAISS index for semantic search

### 3️⃣ Language Detection
- Checks for Devanagari script → Hindi
- Checks for Hindi keywords → Hindi
- Checks for Hinglish patterns → Hindi
- Default: English

### 4️⃣ Conversation Context
- Stores last 5 Q&A pairs in memory
- Detects if current question is "follow-up"
- Injects context into LLM prompt

### 5️⃣ Retrieval
- Convert question to embedding
- Find 4 most similar chunks via FAISS
- Pass chunks as context to LLM

### 6️⃣ Generation
- LLM (Groq) receives: context + conversation history + question
- Generates answer grounded in retrieved documents
- Returns answer + source citations

---

## 💡 Tips for Better Results

**Question:** "What is Article 21?"  
**Better:** "What does Article 21 of the Constitution guarantee?"

**Question:** "Tell me about theft"  
**Better:** "What are the elements of theft under BNS Section 378?"

**Question:** "Compare old and new laws"  
**Better:** "How did BNS Section 103 change from IPC Section 302?"

---

## 🔐 Security Notes

- API keys stored in `.env` (never commit to git)
- `.env` file added to `.gitignore` by default
- No data sent elsewhere except Groq API
- Chat history stored locally in `chat_data.json`

---

## 📝 What Changed (From Previous Version)

### Removed Complexity ❌
- ❌ Web scraping & RSS feeds  
- ❌ MongoDB support (JSON file storage instead)
- ❌ WebSocket connections
- ❌ Advanced search algorithms
- ❌ Evidence scoring heuristics
- ❌ Multiple quality modes

### Kept Core Features ✅
- ✅ Multi-language (especially Hindi/Hinglish)
- ✅ Conversation memory  
- ✅ Source attribution
- ✅ RAG with semantic search
- ✅ Production-ready API

### Result 📊
- **Code reduced:** 1,500 → 400 lines (core)
- **Files simplified:** 7 → 3 main files
- **Easier to understand:** Clear, documented code
- **Easier to maintain:** Fewer dependencies
- **Faster startup:** No complex initialization

---

## 📖 API Documentation

Full OpenAPI docs available at:
```
http://localhost:8000/docs
```

## 🤝 Contributing

Improvements welcome! Focus areas:
- Better Hindi response generation
- Improved follow-up detection
- Additional Indian laws
- Performance optimizations

## 📄 License

MIT License - See LICENSE file

## ⚖️ Disclaimer

**This system is for informational purposes only.**  
- Not a substitute for legal advice
- Consult qualified lawyers for actual legal matters
- No attorney-client relationship created
- Information may be incomplete or outdated

---

**Made for Indian legal research and education** ⚖️
