"""
Legal RAG System - Enhanced Version
Features:
1. Conversation Memory - Remembers context across messages
2. Advanced Search - Hybrid search with re-ranking
3. Error Handling - Graceful failures and validation
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationMemory:
    """Manages conversation history for context-aware responses"""
    
    def __init__(self, max_history: int = 5):
        self.history = deque(maxlen=max_history)
        self.max_history = max_history
    
    def add_exchange(self, question: str, answer: str):
        """Add a question-answer pair to history"""
        self.history.append({
            "question": question,
            "answer": answer
        })
    
    def get_context_string(self) -> str:
        """Get formatted conversation history"""
        if not self.history:
            return ""
        
        context = "\n\nPrevious conversation:\n"
        for i, exchange in enumerate(self.history, 1):
            context += f"Q{i}: {exchange['question']}\n"
            answer_preview = exchange['answer'][:200] + "..." if len(exchange['answer']) > 200 else exchange['answer']
            context += f"A{i}: {answer_preview}\n"
        
        return context
    
    def is_follow_up(self, question: str) -> bool:
        """Detect if question is a follow-up"""
        if not self.history:
            return False
        
        follow_up_indicators = [
            'what about', 'how about', 'can you', 'tell me more',
            'elaborate', 'explain that', 'give me examples', 'example',
            'clarify', 'more details', 'why', 'how', 'also'
        ]
        
        question_lower = question.lower().strip()
        
        # Pronouns indicate follow-up
        pronouns = ['it', 'that', 'this', 'these', 'those', 'they']
        if any(question_lower.startswith(pronoun) for pronoun in pronouns):
            return True
        
        # Short questions with indicators
        if len(question.split()) <= 7:
            return any(indicator in question_lower for indicator in follow_up_indicators)
        
        return False
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()


class AdvancedSearch:
    """Enhanced search with hybrid retrieval and re-ranking"""
    
    @staticmethod
    def extract_legal_terms(query: str) -> List[str]:
        """Extract legal references from query"""
        patterns = [
            r'Article\s+\d+[A-Z]*',
            r'Section\s+\d+[A-Z]*',
            r'BNS\s+\d+',
            r'BNSS\s+\d+',
            r'BSA\s+\d+',
            r'IPC\s+\d+',
        ]
        
        terms = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            terms.extend(matches)
        
        # Add significant words
        common_words = {'what', 'when', 'where', 'which', 'under', 'about', 'explain', 'tell', 'give'}
        words = [w for w in query.lower().split() if len(w) > 4 and w not in common_words]
        terms.extend(words)
        
        return list(set(terms))
    
    @staticmethod
    def hybrid_search(query: str, vectorstore, k: int = 10) -> List[Document]:
        """Hybrid search with keyword boosting"""
        try:
            # Get semantic results
            semantic_results = vectorstore.similarity_search(query, k=k*2)
            
            # Extract legal terms
            legal_terms = AdvancedSearch.extract_legal_terms(query)
            
            # Re-rank based on term presence
            scored_results = []
            for doc in semantic_results:
                score = 1.0
                content_lower = doc.page_content.lower()
                
                # Boost for legal terms
                for term in legal_terms:
                    if term.lower() in content_lower:
                        score += 0.3
                
                # Boost for terms in first 200 chars (likely headers/important)
                if any(term.lower() in content_lower[:200] for term in legal_terms):
                    score += 0.2
                
                scored_results.append((doc, score))
            
            # Sort and return top k
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_results[:k]]
            
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            # Fallback to basic search
            return vectorstore.similarity_search(query, k=k)
    
    @staticmethod
    def dynamic_k(query: str) -> int:
        """Determine optimal number of documents based on query"""
        query_lower = query.lower()
        
        # Simple queries need fewer docs
        if any(word in query_lower for word in ['what is', 'define', 'meaning']):
            return 2
        
        # Comparison queries need more
        if any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return 6
        
        # Detailed queries need more
        if any(word in query_lower for word in ['detail', 'comprehensive', 'all', 'list']):
            return 8
        
        return 4
    
    @staticmethod
    def remove_duplicates(documents: List[Document]) -> List[Document]:
        """Remove duplicate chunks"""
        unique_docs = []
        seen_fingerprints = set()
        
        for doc in documents:
            # Create fingerprint from first 100 chars
            fingerprint = doc.page_content[:100].lower().strip()
            
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_docs.append(doc)
        
        return unique_docs


class LegalRAGSystem:
    """Enhanced RAG system with memory, advanced search, and error handling"""
    
    def __init__(self, vectorstore_path: str = "legal_vectorstore"):
        """Initialize the enhanced RAG system"""
        load_dotenv()
        
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=5)
        
        # Get API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        # Initialize LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1024,
            groq_api_key=groq_api_key
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("LegalRAGSystem initialized successfully")
    
    def load_documents(self, pdf_paths: List[str] = None, txt_paths: List[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Load documents with error handling"""
        documents = []
        successful = []
        failed = []
        
        # Load PDFs
        if pdf_paths:
            for pdf_path in pdf_paths:
                print(f"Loading PDF: {pdf_path}")
                try:
                    if not os.path.exists(pdf_path):
                        raise FileNotFoundError(f"File not found: {pdf_path}")
                    
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    if not docs:
                        raise ValueError(f"No content extracted from {pdf_path}")
                    
                    documents.extend(docs)
                    successful.append(pdf_path)
                    print(f"  ✓ Loaded {len(docs)} pages")
                    logger.info(f"Successfully loaded {pdf_path}")
                    
                except Exception as e:
                    failed.append((pdf_path, str(e)))
                    print(f"  ✗ Error: {str(e)}")
                    logger.error(f"Failed to load {pdf_path}: {e}")
        
        # Load text files
        if txt_paths:
            for txt_path in txt_paths:
                print(f"Loading text file: {txt_path}")
                try:
                    if not os.path.exists(txt_path):
                        raise FileNotFoundError(f"File not found: {txt_path}")
                    
                    loader = TextLoader(txt_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                    successful.append(txt_path)
                    print(f"  ✓ Loaded successfully")
                    logger.info(f"Successfully loaded {txt_path}")
                    
                except Exception as e:
                    failed.append((txt_path, str(e)))
                    print(f"  ✗ Error: {str(e)}")
                    logger.error(f"Failed to load {txt_path}: {e}")
        
        if not documents:
            raise ValueError("No documents were loaded successfully")
        
        print(f"\nTotal documents loaded: {len(documents)}")
        print(f"Successful: {len(successful)}, Failed: {len(failed)}")
        
        # Split documents
        print("\nSplitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        
        # Create vectorstore
        print("\nCreating vectorstore...")
        try:
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            print(f"Saving vectorstore to {self.vectorstore_path}...")
            self.vectorstore.save_local(self.vectorstore_path)
            print("✓ Vectorstore saved successfully!")
            logger.info("Vectorstore created and saved")
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            raise
        
        # Initialize QA chain
        self._initialize_qa_chain()
        
        return successful, failed
    
    def load_vectorstore(self):
        """Load existing vectorstore with error handling"""
        if not os.path.exists(self.vectorstore_path):
            raise ValueError(f"Vectorstore not found at {self.vectorstore_path}")
        
        print(f"Loading vectorstore from {self.vectorstore_path}...")
        try:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ Vectorstore loaded successfully!")
            logger.info("Vectorstore loaded successfully")
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
        except Exception as e:
            logger.error(f"Error loading vectorstore: {e}")
            raise ValueError(f"Failed to load vectorstore: {str(e)}")
    
    def _initialize_qa_chain(self):
        """Initialize QA chain with conversational prompt"""
        prompt_template = """You are a friendly and knowledgeable Indian law assistant specializing in the Constitution of India and new criminal codes (BNS, BSA, BNSS).

PERSONALITY:
- Conversational and approachable
- Concise by default, detailed when asked
- Use simple language
- Acknowledge conversation context when relevant

RESPONSE RULES:
1. For greetings: Respond warmly
2. For simple questions: 2-3 sentence answers
3. For "detailed" requests: Comprehensive explanations
4. Always cite specific sections/articles
5. If unsure, say so honestly
6. Reference previous conversation when relevant

{context}

Question: {question}

Answer (concise unless detail requested):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("QA chain initialized")
    
    def validate_question(self, question: str) -> Tuple[bool, Optional[str]]:
        """Validate user input"""
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if len(question.strip()) < 2:
            return False, "Question is too short. Please provide more details."
        
        if len(question) > 1000:
            return False, "Question is too long. Please keep it under 1000 characters."
        
        # Check for suspicious patterns (basic security)
        suspicious_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=']
        if any(pattern in question.lower() for pattern in suspicious_patterns):
            return False, "Invalid input detected"
        
        return True, None
    
    def is_greeting(self, question: str) -> bool:
        """Check if question is a greeting"""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'namaste']
        question_lower = question.lower().strip()
        return any(greeting == question_lower or question_lower.startswith(greeting + ' ') for greeting in greetings) and len(question.split()) <= 4
    
    def query(self, question: str) -> Dict:
        """Enhanced query with memory, advanced search, and error handling"""
        
        # Validate input
        is_valid, error_msg = self.validate_question(question)
        if not is_valid:
            logger.warning(f"Invalid question: {error_msg}")
            return {
                'answer': f"❌ {error_msg}",
                'sources': [],
                'error': True
            }
        
        # Check system initialization
        if not self.qa_chain or not self.vectorstore:
            logger.error("System not initialized")
            return {
                'answer': "System not initialized. Please load documents first.",
                'sources': [],
                'error': True
            }
        
        # Handle greetings
        if self.is_greeting(question):
            response = "Hello! 👋 I'm your Indian Law Assistant. I can help you with:\n• Constitution of India\n• BNS (Bharatiya Nyaya Sanhita)\n• BNSS (Bharatiya Nagarik Suraksha Sanhita)\n• BSA (Bharatiya Sakshya Adhiniyam)\n• Landmark Supreme Court cases\n\nWhat would you like to know?"
            return {
                'answer': response,
                'sources': [],
                'error': False
            }
        
        try:
            # Check if follow-up question
            enhanced_question = question
            if self.memory.is_follow_up(question):
                # Add conversation context
                context = self.memory.get_context_string()
                enhanced_question = f"{context}\n\nCurrent question: {question}"
                logger.info("Follow-up question detected, using conversation context")
            
            # Advanced search
            k = AdvancedSearch.dynamic_k(question)
            logger.info(f"Using k={k} for retrieval")
            
            documents = AdvancedSearch.hybrid_search(enhanced_question, self.vectorstore, k=k)
            documents = AdvancedSearch.remove_duplicates(documents)
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Query with retrieved documents
            result = self.qa_chain.invoke({"query": enhanced_question})
            
            answer = result['result']
            
            # Extract sources
            sources = []
            for doc in result.get('source_documents', []):
                source_info = doc.metadata.get('source', 'Unknown')
                page_info = doc.metadata.get('page', '?')
                sources.append(f"{os.path.basename(source_info)} (Page {page_info})")
            
            # Check if answer is meaningful
            if len(answer.strip()) < 20:
                logger.warning("Answer too short, may be invalid")
                answer = "I couldn't find enough relevant information to answer your question properly. Could you rephrase or provide more context?"
            
            # Store in memory (only store actual Q&A, not greetings)
            self.memory.add_exchange(question, answer)
            logger.info(f"Query processed successfully. Answer length: {len(answer)}")
            
            return {
                'answer': answer,
                'sources': list(set(sources))[:5],  # Unique sources, max 5
                'error': False
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                'answer': f"I encountered an error while processing your question. Please try rephrasing or contact support if the issue persists. (Error: {str(e)[:100]})",
                'sources': [],
                'error': True
            }
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.memory.clear()
        logger.info("Conversation history cleared")
        return {"message": "Conversation history cleared"}


# Example usage
if __name__ == "__main__":
    # Initialize
    rag = LegalRAGSystem()
    
    # Load vectorstore
    rag.load_vectorstore()
    
    # Test conversation memory
    print("\n" + "="*70)
    print("Testing Conversation Memory")
    print("="*70)
    
    queries = [
        "Hi",
        "What is Article 21?",
        "Can you give me examples?",  # Follow-up
        "How does it relate to privacy?",  # Follow-up
        "What is BNS Section 103?",  # New topic
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        result = rag.query(q)
        print(f"A: {result['answer'][:300]}...")
        if result['sources']:
            print(f"Sources: {result['sources'][:2]}")
