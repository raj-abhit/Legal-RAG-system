"""
Legal RAG System - Conversational & Concise Version
Updated prompt for better user interaction
"""

import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

class LegalRAGSystem:
    """RAG system for Indian legal documents with Groq LLM"""
    
    def __init__(self, vectorstore_path: str = "legal_vectorstore"):
        """Initialize the RAG system - reads API key from .env file"""
        load_dotenv()
        
        self.vectorstore_path = vectorstore_path
        self.vectorstore = None
        self.qa_chain = None
        
        # Get API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        # Initialize Groq LLM with adjusted parameters for conciseness
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,  # Slightly increased for more natural responses
            max_tokens=1024,  # Reduced from 2048 for more concise answers
            groq_api_key=groq_api_key
        )
        
        # Initialize embeddings (free HuggingFace model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def load_documents(self, pdf_paths: List[str] = None, txt_paths: List[str] = None):
        """Load and process documents from PDFs and text files"""
        documents = []
        
        # Load PDFs
        if pdf_paths:
            for pdf_path in pdf_paths:
                print(f"Loading PDF: {pdf_path}")
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"  ✓ Loaded {len(docs)} pages")
                except Exception as e:
                    print(f"  ✗ Error loading {pdf_path}: {str(e)}")
        
        # Load text files
        if txt_paths:
            for txt_path in txt_paths:
                print(f"Loading text file: {txt_path}")
                try:
                    loader = TextLoader(txt_path, encoding='utf-8')
                    documents.extend(loader.load())
                    print(f"  ✓ Loaded successfully")
                except Exception as e:
                    print(f"  ✗ Error loading {txt_path}: {str(e)}")
        
        if not documents:
            raise ValueError("No documents were loaded successfully")
        
        print(f"\nTotal documents loaded: {len(documents)}")
        
        # Split documents into chunks
        print("\nSplitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        
        # Create vectorstore
        print("\nCreating vectorstore (this may take a few minutes)...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        # Save vectorstore
        print(f"Saving vectorstore to {self.vectorstore_path}...")
        self.vectorstore.save_local(self.vectorstore_path)
        print("✓ Vectorstore saved successfully!")
        
        # Initialize QA chain
        self._initialize_qa_chain()
        
    def load_vectorstore(self):
        """Load existing vectorstore from disk"""
        if not os.path.exists(self.vectorstore_path):
            raise ValueError(f"Vectorstore not found at {self.vectorstore_path}")
        
        print(f"Loading vectorstore from {self.vectorstore_path}...")
        self.vectorstore = FAISS.load_local(
            self.vectorstore_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Vectorstore loaded successfully!")
        
        # Initialize QA chain
        self._initialize_qa_chain()
        
    def _initialize_qa_chain(self):
        """Initialize the QA chain with conversational prompt"""
        prompt_template = """You are a friendly and knowledgeable Indian law assistant. You specialize in the Constitution of India and the new criminal codes (BNS, BSA, BNSS).

PERSONALITY & TONE:
- Be conversational and approachable
- Greet users warmly when they say hi/hello
- Be concise by default - give brief, clear answers
- Only provide detailed explanations when explicitly asked for "details", "in detail", "explain fully", etc.
- Use simple language, avoid excessive legal jargon unless necessary

RESPONSE RULES:
1. For greetings (hi, hello, hey): Respond warmly and ask how you can help with legal questions
2. For simple questions: Give concise 2-3 sentence answers with key points
3. For "detailed" requests: Provide comprehensive explanations with examples
4. For comparisons: Use bullet points to show differences clearly
5. Always cite specific sections/articles when referencing laws
6. If you don't know based on the documents, say so honestly

CONTEXT HANDLING:
- Use the provided context to answer accurately
- Don't make up information not in the documents
- If context is insufficient, acknowledge limitations

Context: {context}

Question: {question}

Answer (be concise unless detail is requested):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),  # Reduced from 4 to 3 for more focused context
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def is_greeting(self, question: str) -> bool:
        """Check if the question is a greeting"""
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'namaste']
        question_lower = question.lower().strip()
        return any(greeting in question_lower for greeting in greetings) and len(question.split()) <= 4
    
    def is_detailed_request(self, question: str) -> bool:
        """Check if user is asking for detailed explanation"""
        detail_keywords = ['detail', 'detailed', 'explain', 'elaborate', 'comprehensive', 'thoroughly', 'in depth', 'full explanation']
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in detail_keywords)
        
    def query(self, question: str) -> Dict:
        """Query the RAG system with smart handling"""
        if not self.qa_chain:
            raise ValueError("System not initialized. Load documents or vectorstore first.")
        
        # Handle pure greetings without RAG search
        if self.is_greeting(question):
            return {
                'answer': "Hello! 👋 I'm your Indian Law Assistant. I can help you with questions about the Constitution of India, BNS (Bharatiya Nyaya Sanhita), BNSS (Bharatiya Nagarik Suraksha Sanhita), BSA (Bharatiya Sakshya Adhiniyam), and landmark Supreme Court cases. What would you like to know?",
                'sources': []
            }
        
        # For legal questions, use RAG
        result = self.qa_chain.invoke({"query": question})
        
        # Extract source information
        sources = []
        for doc in result.get('source_documents', []):
            source_info = doc.metadata.get('source', 'Unknown source')
            page_info = doc.metadata.get('page', 'Unknown page')
            sources.append(f"{source_info} (Page {page_info})")
        
        answer = result['result']
        
        # Add note for detailed requests
        if not self.is_detailed_request(question) and len(answer) > 500:
            # If answer is long but detail wasn't requested, suggest they can ask for more
            answer += "\n\n💡 Want more details? Ask me to 'explain in detail' or 'elaborate'."
        
        return {
            'answer': answer,
            'sources': sources
        }

# Example usage
if __name__ == "__main__":
    # Initialize system
    rag = LegalRAGSystem()
    
    # Load existing vectorstore
    rag.load_vectorstore()
    
    # Test queries
    test_questions = [
        "Hi",
        "What is Article 21?",
        "Explain Article 21 in detail",
        "What is the punishment for murder under BNS?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = rag.query(question)
        print(f"A: {result['answer']}")
        if result['sources']:
            print(f"Sources: {result['sources'][:2]}")