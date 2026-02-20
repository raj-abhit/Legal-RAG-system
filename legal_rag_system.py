"""
Simplified Legal RAG System - Indian Law Document Analysis

This system provides intelligent Q&A over Indian legal documents using:
- LangChain for document processing and LLM chains
- FAISS for semantic search
- Groq API for fast LLM inference
- Multi-language support (English, Hindi, Hinglish)

Main Classes:   
- ConversationMemory: Tracks previous Q&A pairs for context
- LegalRAGSystem: Core RAG engine for legal document querying
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque
from dotenv import load_dotenv

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
    """
    Stores and manages conversation history for multi-turn Q&A.
    
    Keeps the last N Q&A pairs to provide context for follow-up questions.
    This helps the system understand references to previous answers.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of Q&A pairs to retain
        """
        self.history = deque(maxlen=max_history)
        self.max_history = max_history
    
    def add_exchange(self, question: str, answer: str):
        """Store a Q&A pair in conversation history."""
        self.history.append({"question": question, "answer": answer})
    
    def get_context_string(self) -> str:
        """
        Generate conversation context string for prompt injection.
        
        Returns the full previous Q&A pairs formatted for inclusion in prompts.
        This helps the LLM understand follow-up questions better.
        """
        if not self.history:
            return ""
        
        context = "\n## PREVIOUS CONVERSATION CONTEXT\n"
        for i, ex in enumerate(self.history, 1):
            context += f"\nQ{i}: {ex['question']}\n"
            context += f"A{i}: {ex['answer']}\n"
        context += "\n---\n"
        return context
    
    def is_follow_up(self, question: str) -> bool:
        """
        Detect if current question is a follow-up to previous answers.
        
        Uses linguistic cues (pronouns, continuation words) to identify
        questions that depend on previous answers for context.
        """
        if not self.history:
            return False
        
        # Words indicating a follow-up question
        followup_indicators = [
            'what about', 'how about', 'tell me more', 'elaborate',
            'explain that', 'give me examples', 'example', 'clarify', 'more details',
            'why', 'how', 'also',
            # Hindi/Hinglish cues
            'aur', 'aur details', 'aur batao', 'samjhao', 'vistar', 'और', 'समझाओ'
        ]
        
        q_lower = question.lower().strip()
        pronouns = ['it', 'that', 'this', 'these', 'those', 'they']
        
        # Questions starting with pronouns (e.g., "It means...?")
        if any(q_lower.startswith(p) for p in pronouns):
            return True
        
        # Short questions with followup keywords
        if len(question.split()) <= 7:
            return any(ind in q_lower for ind in followup_indicators)
        
        # Translation / language-switch requests are always follow-ups
        if self.is_translation_request(question):
            return True
        
        return False
    
    def is_translation_request(self, question: str) -> bool:
        """
        Detect if the user is asking to repeat/translate the previous answer
        in a different language (e.g. "tell me this in hindi", "write this in tamil").
        """
        if not self.history:
            return False
        
        q = question.lower().strip()
        
        # Patterns like "in hindi", "hindi me", "hindi mein batao", etc.
        lang_names = [
            'hindi', 'english', 'tamil', 'telugu', 'kannada', 'malayalam',
            'bengali', 'marathi', 'gujarati', 'punjabi', 'odia', 'urdu', 'assamese',
            'hinglish'
        ]
        
        translate_cues = [
            'in {lang}', '{lang} me', '{lang} mein', '{lang} mai',
            '{lang} m ', '{lang} main',
        ]
        
        has_lang = any(lang in q for lang in lang_names)
        if not has_lang:
            return False
        
        # Short request referencing a language = likely translation
        action_cues = [
            'tell me', 'write', 'say', 'batao', 'bataye', 'bolo', 'likho',
            'translate', 'convert', 'repeat', 'same', 'this', 'again',
            'yeh', 'ye', 'isko', 'upar wala', 'wahi',
        ]
        if any(cue in q for cue in action_cues):
            return True
        
        # Very short query with just a language name
        if len(question.split()) <= 5 and has_lang:
            return True
        
        return False
    
    def get_last_answer(self) -> str:
        """Get the most recent answer from conversation history."""
        if self.history:
            return self.history[-1]["answer"]
        return ""
    
    @staticmethod
    def detect_response_language(question: str) -> str:
        """
        Detect which language the user is writing in, based on script detection
        and explicit language requests. Defaults to English for romanized text.
        
        Supports: English, Hindi, Tamil, Telugu, Kannada, Malayalam,
                  Bengali, Marathi, Gujarati, Punjabi, Odia, Urdu, Assamese.
        
        Returns:
            Language name string (e.g. 'Hindi', 'Tamil') or '' for English (default).
        """
        q = question.strip()
        ql = q.lower()
        
        # 1. Check for EXPLICIT language request (highest priority)
        #    e.g. "answer in Tamil", "reply in Hindi", "in Telugu batao"
        explicit_map = {
            r'\b(in\s+hindi|hindi\s+m[ei]n|हिंदी)\b': "Hindi",
            r'\b(in\s+tamil|tamil[il]?\s+l[ei]|தமிழ்)\b': "Tamil",
            r'\b(in\s+telugu|telugu\s+lo|తెలుగు)\b': "Telugu",
            r'\b(in\s+kannada|kannada\s+d?alli|ಕನ್ನಡ)\b': "Kannada",
            r'\b(in\s+malayalam|malayalath+il|മലയാളം)\b': "Malayalam",
            r'\b(in\s+bengali|bangla[yt]?|বাংলা)\b': "Bengali",
            r'\b(in\s+marathi|marathi\s+madh[ey]|मराठी)\b': "Marathi",
            r'\b(in\s+gujarati|gujarati\s+m[ae]|ગુજરાતી)\b': "Gujarati",
            r'\b(in\s+punjabi|punjabi\s+vich|ਪੰਜਾਬੀ)\b': "Punjabi",
            r'\b(in\s+odia|odia\s+re|ଓଡ଼ିଆ)\b': "Odia",
            r'\b(in\s+urdu|urdu\s+m[ei]n|اردو)\b': "Urdu",
            r'\b(in\s+assamese|অসমীয়া)\b': "Assamese",
            r'\b(in\s+english|english\s+m[ei]n|अंग्रेजी)\b': "",
        }
        for pattern, lang in explicit_map.items():
            if re.search(pattern, q, re.IGNORECASE):
                return lang
        
        # 2. Detect language by Unicode script (non-Latin scripts)
        script_map = [
            (r'[\u0900-\u097F]', "Hindi"),       # Devanagari (Hindi/Marathi)
            (r'[\u0B80-\u0BFF]', "Tamil"),        # Tamil script
            (r'[\u0C00-\u0C7F]', "Telugu"),       # Telugu script
            (r'[\u0C80-\u0CFF]', "Kannada"),      # Kannada script
            (r'[\u0D00-\u0D7F]', "Malayalam"),     # Malayalam script
            (r'[\u0980-\u09FF]', "Bengali"),       # Bengali script
            (r'[\u0A80-\u0AFF]', "Gujarati"),      # Gujarati script
            (r'[\u0A00-\u0A7F]', "Punjabi"),       # Gurmukhi script
            (r'[\u0B00-\u0B7F]', "Odia"),          # Odia script
            (r'[\u0600-\u06FF]', "Urdu"),          # Arabic script (Urdu)
        ]
        for pattern, lang in script_map:
            if re.search(pattern, q):
                # Distinguish Marathi from Hindi (both use Devanagari)
                if lang == "Hindi" and re.search(r'\b(in\s+marathi|मराठी)\b', q, re.IGNORECASE):
                    return "Marathi"
                return lang
        
        # 3. Default: English for all romanized text
        #    No Hinglish guessing — user must explicitly ask for Hindi
        return ""
    
    def clear(self):
        """Clear all conversation history."""
        self.history.clear()


class LegalRAGSystem:
    """
    Core Retrieval-Augmented Generation system for Indian legal documents.
    
    Provides Q&A capabilities over legal documents using semantic search
    and LLM-based answer generation. Supports multiple Indian laws including
    Constitution, BNS, BNSS, BSA, IPC, CrPC, and special acts.
    """
    
    def __init__(self, vectorstore_path: str = "legal_vectorstore"):
        """
        Initialize the Legal RAG System.
        
        Args:
            vectorstore_path: Path to pre-built FAISS vector store
            
        Raises:
            ValueError: If GROQ_API_KEY is not set in environment
        """
        load_dotenv()
        
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationMemory(max_history=5)
        
        # Initialize LLM (Groq for fast inference)
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,  # Lower temperature for factual legal answers
            max_tokens=1500,
            groq_api_key=groq_key
        )
        
        # Initialize embeddings using HuggingFace (free, local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("Legal RAG System initialized")
    
    def load_documents(self, pdf_paths: List[str] = None, txt_paths: List[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Load and index legal documents from PDFs and text files.
        
        Creates a vector store for semantic search. This is a one-time
        operation that may take 5-15 minutes for large document sets.
        
        Args:
            pdf_paths: List of PDF file paths
            txt_paths: List of text file paths
            
        Returns:
            (successful_files, failed_files)
            
        Raises:
            ValueError: If no documents could be loaded
        """
        documents = []
        successful = []
        failed = []
        
        # Load PDFs
        if pdf_paths:
            for path in pdf_paths:
                print(f"Loading: {os.path.basename(path)}")
                try:
                    docs = PyPDFLoader(path).load()
                    documents.extend(docs)
                    successful.append(path)
                    print(f"  ✓ Loaded {len(docs)} pages")
                except Exception as e:
                    failed.append((path, str(e)))
                    print(f"  ✗ Error: {str(e)}")
        
        # Load text files
        if txt_paths:
            for path in txt_paths:
                print(f"Loading: {os.path.basename(path)}")
                try:
                    docs = TextLoader(path, encoding='utf-8').load()
                    documents.extend(docs)
                    successful.append(path)
                    print("  ✓ Loaded")
                except Exception as e:
                    failed.append((path, str(e)))
                    print(f"  ✗ Error: {str(e)}")
        
        if not documents:
            raise ValueError("No documents could be loaded")
        
        print(f"\n✓ Loaded {len(documents)} documents")
        if failed:
            print(f"✗ Failed: {len(failed)} documents")
        
        # Split documents into chunks
        print("\nSplitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = splitter.split_documents(documents)
        print(f"✓ Created {len(splits)} chunks")
        
        # Create vector embeddings
        print("\nCreating embeddings (this may take 5-15 minutes)...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        self.vectorstore.save_local(self.vectorstore_path)
        print(f"✓ Saved to {self.vectorstore_path}")
        
        self._initialize_qa_chain()
        return successful, failed
    
    def load_vectorstore(self):
        """
        Load pre-computed vector store from disk.
        
        Call this if you've already created a vector store and want
        to reuse it instead of regenerating embeddings.
        """
        if not os.path.exists(self.vectorstore_path):
            raise ValueError(f"Vectorstore not found: {self.vectorstore_path}")
        
        print(f"Loading vector store from {self.vectorstore_path}...")
        allow_dangerous = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "false").strip().lower() == "true"
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=allow_dangerous
        )
        print("✓ Vector store loaded")
        self._initialize_qa_chain()
    
    def _initialize_qa_chain(self):
        """Set up the LLM question-answering chain with retrieval."""
        prompt_template = """You are a friendly Indian legal assistant. Help ordinary citizens
understand their rights in plain, simple language.

RULES:
1. Give practical, actionable advice — explain rights simply, suggest steps to take,
   mention helplines/authorities when relevant. Cite specific section numbers.
2. Never invent section numbers — only cite what is in the context.
3. Prefer BNS/BNSS/BSA over IPC/CrPC unless user asks to compare.
4. Respond in the SAME language as the question. Default to English.
5. Use previous conversation if this is a follow-up.
6. If asked to draft a legal notice, generate a complete ready-to-use notice
   using the templates and facts provided by the user.

PREVIOUS CONVERSATION (if any):
{context}

Based on the legal documents above, answer this question:
{question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str) -> Dict:
        """
        Query the legal RAG system for answers.
        
        Args:
            question: Legal question (can be in English or Hindi)
            
        Returns:
            Dictionary with:
            - 'answer': The generated answer
            - 'sources': List of source documents used
            - 'language': Detected language preference
            - 'is_followup': Whether question detected as follow-up
        """
        # Validate input
        is_valid, error_msg = self.validate_question(question)
        if not is_valid:
            return {
                "answer": f"Error: {error_msg}",
                "sources": [],
                "language": "",
                "is_followup": False,
                "error": True
            }
        
        # Detect language preference
        language = ConversationMemory.detect_response_language(question)
        is_followup = self.memory.is_follow_up(question)
        is_translate = self.memory.is_translation_request(question)
        
        # Handle translation requests — skip retrieval, just translate
        if is_translate:
            prev_answer = self.memory.get_last_answer()
            if prev_answer:
                target = language if language else "Hindi"  # fallback
                try:
                    translate_prompt = (
                        f"Translate the following answer to {target}. "
                        f"Keep all legal section numbers, law names, and formatting intact. "
                        f"Only translate the language, do not add or remove any information.\n\n"
                        f"{prev_answer}"
                    )
                    translated = self.llm.invoke(translate_prompt)
                    answer = translated.content.strip() if hasattr(translated, 'content') else str(translated).strip()
                    self.memory.add_exchange(question, answer)
                    return {
                        "answer": answer,
                        "sources": [],
                        "language": target,
                        "is_followup": True,
                        "error": False
                    }
                except Exception as e:
                    logger.warning(f"Translation failed, falling back to RAG: {e}")
        
        # Get conversation context if this is a follow-up
        conversation_context = ""
        if is_followup:
            conversation_context = self.memory.get_context_string()
        
        try:
            # Inject conversation context if available
            full_question = question
            if conversation_context:
                full_question = f"{conversation_context}\nCurrent question: {question}"
            
            # If a non-English language was detected, add explicit instruction
            if language:
                full_question += f"\n[RESPOND IN {language.upper()}]"
            
            # Run RAG chain
            result = self.qa_chain.invoke({"query": full_question})
            
            # Get answer
            answer = result.get("result", "").strip()
            
            # Extract sources
            source_documents = result.get("source_documents", [])
            sources = [
                {
                    "document": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", 0),
                    "excerpt": doc.page_content[:200]
                }
                for doc in source_documents
            ]
            
            # Store in memory for follow-up questions
            self.memory.add_exchange(question, answer)
            
            return {
                "answer": answer,
                "sources": sources,
                "language": language,
                "is_followup": is_followup,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            return {
                "answer": "Error processing query. Please check your question and try again.",
                "sources": [],
                "language": language,
                "is_followup": is_followup,
                "error": True
            }
    
    @staticmethod
    def validate_question(question: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user question before processing.
        
        Returns:
            (is_valid, error_message)
        """
        if not question or not question.strip():
            return False, "Question cannot be empty"
        
        if len(question.strip()) < 2:
            return False, "Question too short (minimum 2 characters)"
        
        if len(question) > 5000:
            return False, "Question too long (maximum 5000 characters)"
        
        # Basic injection protection
        if any(bad in question.lower() for bad in ['<script', 'javascript:', 'onerror=']):
            return False, "Invalid input detected"
        
        return True, None
    
    def reset_conversation(self):
        """Clear conversation history for fresh start."""
        self.memory.clear()
        logger.info("Conversation history cleared")


def main():
    """Example usage of the Legal RAG System."""
    try:
        # Initialize system
        rag = LegalRAGSystem()
        
        # First time: load documents and create embeddings
        # rag.load_documents(
        #     pdf_paths=["./documents/Constitution.pdf", "./documents/BNS_2023.pdf"],
        #     txt_paths=None
        # )
        
        # Subsequent times: load pre-built vector store
        rag.load_vectorstore()
        
        # Example queries
        print("\n" + "="*70)
        print("LEGAL RAG SYSTEM - Interactive Query")
        print("="*70)
        
        while True:
            question = input("\n📋 Your question (or 'quit' to exit): ").strip()
            
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            result = rag.query(question)
            
            print(f"\n✓ Answer:")
            print(result["answer"])
            
            if result["sources"]:
                print(f"\n📚 Sources:")
                for i, src in enumerate(result["sources"], 1):
                    print(f"  {i}. {src['document']} (Page {src['page']})")
            
            if result["is_followup"]:
                print("\n💡 (Detected as follow-up question - using conversation context)")
    
    except Exception as e:
        print(f"Error: {e}")
        logger.error(str(e), exc_info=True)


if __name__ == "__main__":
    main()
