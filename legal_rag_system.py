"""
Legal RAG System - Production Version
Supports: Constitution, BNS, BNSS, BSA, IPC, CrPC, RTI, POCSO, IT Act, NDPS, MV Act
Features: Conversation memory, Advanced search, IPC vs BNS comparison
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from collections import deque

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """Remembers last 5 Q&A pairs for context"""
    
    def __init__(self, max_history: int = 5):
        self.history = deque(maxlen=max_history)
        self.max_history = max_history
    
    def add_exchange(self, question: str, answer: str):
        self.history.append({"question": question, "answer": answer})
    
    def get_context_string(self) -> str:
        if not self.history:
            return ""
        
        context = "\n\nPrevious conversation:\n"
        for i, ex in enumerate(self.history, 1):
            context += f"Q{i}: {ex['question']}\n"
            preview = ex['answer'][:200] + "..." if len(ex['answer']) > 200 else ex['answer']
            context += f"A{i}: {preview}\n"
        return context
    
    def is_follow_up(self, question: str) -> bool:
        if not self.history:
            return False
        
        indicators = ['what about', 'how about', 'can you', 'tell me more', 'elaborate', 
                     'explain that', 'give me examples', 'example', 'clarify', 'more details', 
                     'why', 'how', 'also']
        
        q_lower = question.lower().strip()
        pronouns = ['it', 'that', 'this', 'these', 'those', 'they']
        
        if any(q_lower.startswith(p) for p in pronouns):
            return True
        
        if len(question.split()) <= 7:
            return any(ind in q_lower for ind in indicators)
        
        return False
    
    def clear(self):
        self.history.clear()


class AdvancedSearch:
    """Hybrid search with legal term extraction"""
    
    @staticmethod
    def extract_legal_terms(query: str) -> List[str]:
        patterns = [
            r'Article\s+\d+[A-Z]*', r'Section\s+\d+[A-Z]*',
            r'BNS\s+\d+', r'BNSS\s+\d+', r'BSA\s+\d+',
            r'IPC\s+\d+', r'CrPC\s+\d+',
        ]
        
        terms = []
        for pattern in patterns:
            terms.extend(re.findall(pattern, query, re.IGNORECASE))
        
        common = {'what', 'when', 'where', 'which', 'under', 'about', 'explain', 'tell', 'give', 'compare', 'difference'}
        words = [w for w in query.lower().split() if len(w) > 4 and w not in common]
        terms.extend(words)
        
        return list(set(terms))
    
    @staticmethod
    def hybrid_search(query: str, vectorstore, k: int = 10) -> List[Document]:
        try:
            results = vectorstore.similarity_search(query, k=k*2)
            terms = AdvancedSearch.extract_legal_terms(query)
            
            scored = []
            for doc in results:
                score = 1.0
                content = doc.page_content.lower()
                
                for term in terms:
                    if term.lower() in content:
                        score += 0.3
                if any(term.lower() in content[:200] for term in terms):
                    score += 0.2
                
                scored.append((doc, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored[:k]]
        except:
            return vectorstore.similarity_search(query, k=k)
    
    @staticmethod
    def dynamic_k(query: str) -> int:
        q = query.lower()
        if any(w in q for w in ['what is', 'define', 'meaning']):
            return 2
        if any(w in q for w in ['compare', 'difference', 'vs', 'versus']):
            return 6
        if any(w in q for w in ['detail', 'comprehensive', 'all', 'list']):
            return 8
        return 4
    
    @staticmethod
    def remove_duplicates(documents: List[Document]) -> List[Document]:
        unique = []
        seen = set()
        for doc in documents:
            fp = doc.page_content[:100].lower().strip()
            if fp not in seen:
                seen.add(fp)
                unique.append(doc)
        return unique


class LegalRAGSystem:
    """Production Legal RAG System"""
    
    def __init__(self, vectorstore_path: str = "legal_vectorstore"):
        load_dotenv()
        
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationMemory(max_history=5)
        
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY not in .env")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
            groq_api_key=groq_key
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("System initialized")
    
    def load_documents(self, pdf_paths: List[str] = None, txt_paths: List[str] = None):
        documents = []
        successful = []
        failed = []
        
        if pdf_paths:
            for path in pdf_paths:
                print(f"Loading: {os.path.basename(path)}")
                try:
                    docs = PyPDFLoader(path).load()
                    documents.extend(docs)
                    successful.append(path)
                    print(f"  ✓ {len(docs)} pages")
                except Exception as e:
                    failed.append((path, str(e)))
                    print(f"  ✗ {str(e)}")
        
        if txt_paths:
            for path in txt_paths:
                print(f"Loading: {os.path.basename(path)}")
                try:
                    docs = TextLoader(path, encoding='utf-8').load()
                    documents.extend(docs)
                    successful.append(path)
                    print(f"  ✓ Loaded")
                except Exception as e:
                    failed.append((path, str(e)))
                    print(f"  ✗ {str(e)}")
        
        if not documents:
            raise ValueError("No documents loaded")
        
        print(f"\n✓ Loaded {len(documents)} documents")
        if failed:
            print(f"✗ Failed: {len(failed)}")
        
        print("\nSplitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        print(f"✓ Created {len(splits)} chunks")
        
        print("\nCreating vectorstore (10-15 minutes)...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.vectorstore.save_local(self.vectorstore_path)
        print(f"✓ Saved to {self.vectorstore_path}")
        
        self._initialize_qa_chain()
        return successful, failed
    
    def load_vectorstore(self):
        if not os.path.exists(self.vectorstore_path):
            raise ValueError(f"Vectorstore not found: {self.vectorstore_path}")
        
        print(f"Loading vectorstore...")
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Loaded")
        self._initialize_qa_chain()
    
    def _initialize_qa_chain(self):
        prompt = """You are a professional Indian legal research assistant with expertise in both OLD and NEW Indian laws.

YOUR KNOWLEDGE BASE:
📜 CURRENT LAWS:
- Constitution of India (current, unchanged)
- BNS (Bharatiya Nyaya Sanhita) - NEW criminal law, replaced IPC on July 1, 2024
- BNSS (Bharatiya Nagarik Suraksha Sanhita) - NEW criminal procedure, replaced CrPC
- BSA (Bharatiya Sakshya Adhiniyam) - NEW evidence law, replaced Evidence Act

📜 OLD LAWS (For comparison only - NO LONGER IN FORCE):
- IPC (Indian Penal Code, 1860) - OBSOLETE since July 1, 2024
- CrPC (Criminal Procedure Code, 1973) - OBSOLETE since July 1, 2024
- Evidence Act (1872) - OBSOLETE since July 1, 2024

📜 SPECIAL LAWS (Still in force):
- Motor Vehicles Act, 1988
- RTI Act, 2005 (Right to Information)
- POCSO Act, 2012 (Protection of Children from Sexual Offences)
- IT Act, 2000 (Information Technology)
- NDPS Act, 1985 (Narcotic Drugs and Psychotropic Substances)

CRITICAL INSTRUCTIONS:
1. When asked about criminal law: ALWAYS cite BNS (NOT IPC), unless specifically asked about IPC
2. When asked about procedure: ALWAYS cite BNSS (NOT CrPC)
3. When asked about evidence: ALWAYS cite BSA (NOT Evidence Act)
4. For comparison questions (IPC vs BNS, CrPC vs BNSS): Explain BOTH clearly
5. Be concise (2-3 sentences) unless detail is requested
6. NEVER say "Namaste again" or repeat greetings
7. Use conversation context for follow-up questions

COMPARISON FORMAT (when asked "compare" or "difference"):
"Old Law: IPC Section [X] stated [provision]
New Law: BNS Section [Y] states [provision]
Key Changes: [clearly list what changed]"

EXAMPLES:
Q: "What is murder?"
A: "Under BNS Section 103, murder is punishable with death or life imprisonment..."

Q: "Compare IPC 302 with BNS 103"
A: "Old Law: IPC Section 302 provided punishment for murder...
New Law: BNS Section 103 provides the same punishment...
Key Changes: The substantive law remains same, but section number changed from 302 to 103."

{context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def validate_question(self, question: str) -> Tuple[bool, Optional[str]]:
        if not question or not question.strip():
            return False, "Question cannot be empty"
        if len(question.strip()) < 2:
            return False, "Question too short"
        if len(question) > 1000:
            return False, "Question too long (max 1000 chars)"
        
        bad = ['<script', 'javascript:', 'onerror=']
        if any(b in question.lower() for b in bad):
            return False, "Invalid input"
        
        return True, None
    
    def is_greeting(self, question: str) -> bool:
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'namaste']
        q = question.lower().strip()
        return any(g == q or q.startswith(g + ' ') for g in greetings) and len(question.split()) <= 4
    
    def query(self, question: str) -> Dict:
        # Validate
        valid, err = self.validate_question(question)
        if not valid:
            return {'answer': f"❌ {err}", 'sources': [], 'error': True}
        
        if not self.qa_chain:
            return {'answer': "System not initialized", 'sources': [], 'error': True}
        
        # Greetings
        if self.is_greeting(question):
            return {
                'answer': "Hello! I'm your Indian Legal Research Assistant. I can help with:\n• Constitution of India\n• BNS, BNSS, BSA (New laws from July 2024)\n• IPC, CrPC comparison with new laws\n• RTI Act, POCSO, IT Act, NDPS Act, Motor Vehicles Act\n• Landmark Supreme Court cases\n\nWhat legal question can I help you with?",
                'sources': [],
                'error': False
            }
        
        try:
            # Follow-up detection
            enhanced_q = question
            if self.memory.is_follow_up(question):
                enhanced_q = f"{self.memory.get_context_string()}\n\nCurrent: {question}"
            
            # Search
            k = AdvancedSearch.dynamic_k(question)
            docs = AdvancedSearch.hybrid_search(enhanced_q, self.vectorstore, k=k)
            docs = AdvancedSearch.remove_duplicates(docs)
            
            # Query
            result = self.qa_chain.invoke({"query": enhanced_q})
            answer = result['result']
            
            # Sources
            sources = []
            for doc in result.get('source_documents', []):
                src = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', '?')
                sources.append(f"{src} (Page {page})")
            
            # Store
            self.memory.add_exchange(question, answer)
            
            return {
                'answer': answer,
                'sources': list(set(sources))[:5],
                'error': False
            }
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                'answer': f"Error: {str(e)[:100]}",
                'sources': [],
                'error': True
            }
    
    def clear_conversation(self):
        self.memory.clear()
        return {"message": "Cleared"}


if __name__ == "__main__":
    print("="*80)
    print("INDIAN LEGAL RAG SYSTEM - PRODUCTION VERSION")
    print("="*80)
    
    rag = LegalRAGSystem()
    
    # Load or create vectorstore
    try:
        rag.load_vectorstore()
    except ValueError:
        print("\n⚠️  Vectorstore not found. Creating new...")
        print("This will take 10-15 minutes (one-time only)\n")
        
        docs_folder = "documents"
        pdfs = []
        txts = []
        
        if os.path.exists(docs_folder):
            for f in os.listdir(docs_folder):
                path = os.path.join(docs_folder, f)
                if f.endswith('.pdf'):
                    pdfs.append(path)
                elif f.endswith('.txt'):
                    txts.append(path)
            
            print(f"Found {len(pdfs)} PDFs, {len(txts)} text files\n")
            rag.load_documents(pdf_paths=pdfs, txt_paths=txts)
        else:
            print("❌ documents/ folder not found!")
            exit(1)
    
    # Test
    print("\n" + "="*80)
    print("TESTING")
    print("="*80)
    
    tests = [
        "Hi",
        "What is Article 21?",
        "Can you explain more?",
        "What is murder under BNS?",
        "How does BNS 103 differ from IPC 302?",
        "What is RTI Act?",
        "What is POCSO about?",
    ]
    
    for q in tests:
        print(f"\n{'='*80}")
        print(f"Q: {q}")
        print(f"{'='*80}")
        res = rag.query(q)
        print(f"A: {res['answer']}")
        if res['sources']:
            print(f"\n📚 {', '.join(res['sources'][:3])}")