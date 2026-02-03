"""
Enhanced Interactive Legal RAG System
Optimized for handling multiple large PDFs:
- Constitution of India
- Bharatiya Nyaya Sanhita (BNS) 2023
- Bharatiya Sakshya Adhiniyam (BSA) 2023  
- Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
"""

import os
import sys
from pathlib import Path
from legal_rag_system import LegalRAGSystem

def print_banner():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("         LEGAL RAG SYSTEM - INDIAN LAW DATABASE")
    print("="*70)
    print("\nThis system provides answers based on:")
    print("  • Constitution of India")
    print("  • Bharatiya Nyaya Sanhita (BNS) 2023 - Criminal Law")
    print("  • Bharatiya Sakshya Adhiniyam (BSA) 2023 - Evidence Law")
    print("  • Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 - Criminal Procedure")
    print("  • 26 Landmark Supreme Court Cases")
    print("="*70 + "\n")

def get_pdf_paths():
    """Get PDF file paths from user"""
    print("PDF FILE SETUP")
    print("-" * 70)
    
    # Check if documents folder exists
    if os.path.exists("documents"):
        print("Found 'documents' folder. Looking for PDFs...\n")
        
        # Auto-detect PDFs in documents folder
        pdf_files = list(Path("documents").glob("*.pdf"))
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF(s) in documents folder:")
            for i, pdf in enumerate(pdf_files, 1):
                size_mb = pdf.stat().st_size / (1024 * 1024)
                print(f"  {i}. {pdf.name} ({size_mb:.2f} MB)")
            
            use_auto = input("\nUse these PDFs automatically? (y/n): ").strip().lower()
            if use_auto == 'y':
                return [str(pdf) for pdf in pdf_files]
    
    # Manual entry
    print("\nPlease provide paths to your PDF files.")
    print("(Press Enter without input to skip a file)\n")
    
    pdf_paths = []
    pdf_names = [
        ("Constitution of India", "e.g., documents/Constitution.pdf"),
        ("Bharatiya Nyaya Sanhita (BNS)", "e.g., documents/BNS_2023.pdf"),
        ("Bharatiya Sakshya Adhiniyam (BSA)", "e.g., documents/BSA_2023.pdf"),
        ("Bharatiya Nagarik Suraksha Sanhita (BNSS)", "e.g., documents/BNSS_2023.pdf")
    ]
    
    for name, example in pdf_names:
        while True:
            path = input(f"{name} PDF path ({example}): ").strip()
            
            if not path:  # Skip this file
                print(f"  Skipping {name}")
                break
            
            if os.path.exists(path) and path.lower().endswith('.pdf'):
                pdf_paths.append(path)
                print(f"  ✓ Found {name}")
                break
            else:
                print(f"  ✗ File not found or not a PDF. Try again.")
    
    if not pdf_paths:
        print("\n⚠ No valid PDFs provided. Cannot proceed.")
        sys.exit(1)
    
    return pdf_paths

def display_help():
    """Display help information"""
    print("\n" + "="*70)
    print("HELP & EXAMPLES")
    print("="*70)
    print("\nSample Questions:")
    print("\n1. Constitutional Law:")
    print("   - What are the fundamental rights under Article 19?")
    print("   - Explain the basic structure doctrine")
    print("   - What is judicial review?")
    
    print("\n2. Criminal Law (BNS):")
    print("   - What is the punishment for murder under BNS?")
    print("   - Difference between culpable homicide and murder")
    print("   - What are the provisions for cybercrime?")
    
    print("\n3. Evidence Law (BSA):")
    print("   - What is the best evidence rule?")
    print("   - Can electronic records be used as evidence?")
    print("   - Rules for examination of witnesses")
    
    print("\n4. Criminal Procedure (BNSS):")
    print("   - What is the procedure for arrest?")
    print("   - Powers of magistrates")
    print("   - Anticipatory bail provisions")
    
    print("\n5. Comparative Questions:")
    print("   - How does BNS differ from IPC Section 420?")
    print("   - Changes in new evidence law vs old Evidence Act")
    print("   - Comparison of arrest procedures")
    
    print("\nCommands:")
    print("  help  - Show this help message")
    print("  quit  - Exit the system")
    print("="*70 + "\n")

def main():
    """Main interactive loop"""
    print_banner()
    
    # Initialize the RAG system
    print("Initializing Legal RAG System...")
    rag = LegalRAGSystem()
    
    # Check if vectorstore already exists
    if os.path.exists(rag.vectorstore_path):
        print(f"\n✓ Found existing vectorstore at: {rag.vectorstore_path}")
        rebuild = input("Rebuild vectorstore with new PDFs? (y/n): ").strip().lower()
        
        if rebuild == 'y':
            # Get PDF paths and rebuild
            pdf_paths = get_pdf_paths()
            print(f"\n📄 Processing {len(pdf_paths)} PDF(s)...")
            print("⏳ This may take 10-15 minutes for large PDFs...")
            print("=" * 70 + "\n")
            
            rag.load_documents(pdf_paths)
        else:
            # Load existing vectorstore
            print("Loading existing vectorstore...")
            rag.load_vectorstore()
    else:
        # First time setup
        print("\n⚠ No existing vectorstore found. First-time setup required.")
        pdf_paths = get_pdf_paths()
        
        print(f"\n📄 Processing {len(pdf_paths)} PDF(s)...")
        print("⏳ This may take 10-15 minutes for large PDFs...")
        print("💡 Tip: This is a one-time process. Future runs will be instant!")
        print("=" * 70 + "\n")
        
        rag.load_documents(pdf_paths)
    
    print("\n" + "="*70)
    print("✓ System Ready!")
    print("="*70)
    print("Type 'help' for examples, or 'quit' to exit.")
    print("="*70 + "\n")
    
    # Interactive query loop
    while True:
        try:
            query = input("Your Question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Legal RAG System. Goodbye!")
                break
            
            if query.lower() in ['help', 'h', '?']:
                display_help()
                continue
            
            # Process query
            print("\n🔍 Searching legal documents...")
            result = rag.query(query)
            
            print("\n" + "-"*70)
            print("ANSWER:")
            print("-"*70)
            print(result['answer'])
            
            # Show sources if available
            if result.get('sources'):
                print("\n" + "-"*70)
                print("RELEVANT SOURCES:")
                print("-"*70)
                for i, source in enumerate(result['sources'][:3], 1):  # Show top 3
                    print(f"\n{i}. {source}")
            
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit properly.")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            print("Please try rephrasing your question.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        print("Please check your setup and try again.")
