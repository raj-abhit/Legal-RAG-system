"""
Interactive CLI for Legal RAG System

Provides an interactive command-line interface for querying Indian legal documents.
Supports multi-language (English and Hindi) with conversation history.

Features:
- Direct querying of legal documents
- Conversation memory for follow-up questions
- Source attribution for all answers
- Multi-language support (English/Hindi)
"""

import os
import sys
from pathlib import Path
from legal_rag_system import LegalRAGSystem


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("LEGAL RAG SYSTEM - INDIAN LAW DATABASE")
    print("="*70)
    print("\nKnowledge Base:")
    print("  ✓ Constitution of India")
    print("  ✓ BNS 2023 (Bharatiya Nyaya Sanhita) - Current Criminal Law")
    print("  ✓ BNSS 2023 (Criminal Procedure)")
    print("  ✓ BSA 2023 (Evidence Law)")
    print("  ✓ Special Acts: RTI, POCSO, IT Act, NDPS, Motor Vehicles Act")
    print("\nLanguage Support: English & Hindi (Hinglish friendly)")
    print("="*70 + "\n")


def display_help():
    """Display usage examples and tips."""
    print("\n" + "="*70)
    print("HELP & EXAMPLES")
    print("="*70)
    print("\n📚 Sample Questions:")
    print("\n  Constitutional Law:")
    print("    - What are fundamental rights under Article 21?")
    print("    - Explain the right to privacy")
    
    print("\n  Criminal Law (BNS):")
    print("    - What is punishment for murder under BNS Section 103?")
    print("    - Difference between BNS and old IPC")
    print("    - Provisions for cybercrime")
    
    print("\n  Multi-language Questions:")
    print("    - बाल विवाह क्या है? (in Hindi)")
    print("    - bnss mein arrest kaise hota hai? (in Hinglish)")
    
    print("\n⌨️  Commands:")
    print("    help   - Show this help")
    print("    reset  - Clear conversation history")
    print("    quit   - Exit program")
    print("="*70 + "\n")


def display_initial_setup():
    """Show vectorstore setup options."""
    print("VECTORSTORE SETUP")
    print("-" * 70)
    
    if os.path.exists("legal_vectorstore"):
        print("✓ Found existing vectorstore")
        print("\nOptions:")
        print("  1. Use existing vectorstore (instant, recommended)")
        print("  2. Rebuild with PDFs from documents/ folder")
        
        choice = input("\nChoice (1 or 2): ").strip()
        return choice == "2"
    else:
        print("⚠ No vectorstore found - first time setup\n")
        auto_pdf = os.path.exists("documents") and list(Path("documents").glob("*.pdf"))
        
        if auto_pdf:
            print(f"Found {len(auto_pdf)} PDF(s) in documents/ folder")
            use_them = input("Create embeddings from these PDFs? (y/n): ").strip().lower()
            return use_them == 'y'
        else:
            print("No PDFs found in documents/ folder")
            return False


def main():
    """Main interactive loop."""
    print_banner()
    
    # Initialize RAG system
    print("Initializing Legal RAG System...")
    try:
        rag = LegalRAGSystem()
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("Please set GROQ_API_KEY in .env file")
        sys.exit(1)
    
    # Handle vectorstore
    should_rebuild = display_initial_setup()
    
    if should_rebuild:
        pdf_paths = list(Path("documents").glob("*.pdf"))
        if pdf_paths:
            print(f"\n📄 Creating embeddings for {len(pdf_paths)} PDF(s)")
            print("⏳ This may take 10-15 minutes (one-time process)...\n")
            try:
                rag.load_documents([str(p) for p in pdf_paths])
            except Exception as e:
                print(f"✗ Error loading documents: {e}")
                sys.exit(1)
        else:
            print("✗ No PDFs found in documents/ folder")
            sys.exit(1)
    else:
        print("Loading vectorstore...")
        try:
            rag.load_vectorstore()
        except Exception as e:
            print(f"✗ Error: {e}")
            print("Please ensure documents are in legal_vectorstore/")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ System Ready!")
    print("="*70)
    print("Type 'help' for examples or questions directly")
    print("="*70 + "\n")
    
    # Interactive loop
    while True:
        try:
            question = input("📋 Your question (or 'quit'): ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n✓ Thank you! Goodbye.")
                break
            
            if question.lower() in ['help', 'h', '?']:
                display_help()
                continue
            
            if question.lower() in ['reset']:
                rag.reset_conversation()
                print("✓ Conversation history cleared\n")
                continue
            
            # Query RAG system
            print("\n🔍 Searching legal documents...")
            result = rag.query(question)
            
            if result.get('error'):
                print(f"\n✗ Error: {result['answer']}\n")
                continue
            
            print("\n" + "-"*70)
            print("📌 ANSWER:")
            print("-"*70)
            print(result['answer'])
            
            # Show sources
            if result.get('sources'):
                print("\n📚 Sources:")
                for i, src in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {src['document']} (Page {src['page']})")
            
            # Show metadata
            if result.get('is_followup'):
                print("\n💡 (Using conversation context)")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\nUse 'quit' to exit properly.\n")
        except Exception as e:
            print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)
