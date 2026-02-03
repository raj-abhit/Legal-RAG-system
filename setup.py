#!/usr/bin/env python3
"""
Automated Setup Script for Legal RAG System
Helps users get started quickly
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def check_python_version():
    """Check if Python version is adequate"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    print("   This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies!")
        print("   Try running: pip install -r requirements.txt")
        return False


def setup_env_file():
    """Create .env file from template"""
    print("\n🔑 Setting up environment variables...")
    
    if os.path.exists(".env"):
        print("✅ .env file already exists")
        return True
    
    # Copy from example
    if os.path.exists(".env.example"):
        with open(".env.example", "r") as f:
            content = f.read()
        
        print("\n📝 Let's set up your Groq API key")
        print("   Get a free API key at: https://console.groq.com")
        print()
        
        api_key = input("Enter your Groq API key (or press Enter to skip): ").strip()
        
        if api_key:
            content = content.replace("your_groq_api_key_here", api_key)
            with open(".env", "w") as f:
                f.write(content)
            print("✅ .env file created with your API key!")
        else:
            with open(".env", "w") as f:
                f.write(content)
            print("⚠️  .env file created without API key")
            print("   You'll need to add your API key later")
        
        return True
    
    print("⚠️  .env.example not found, skipping .env setup")
    return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["documents", "legal_vectorstore"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified: {directory}/")
    
    return True


def download_sample_docs():
    """Ask user if they want to download sample documents"""
    print("\n📚 Document Setup")
    print("   You need legal documents (Constitution of India & IPC) to use the system.")
    print()
    print("   Options:")
    print("   1. Create sample documents with key excerpts (quick start)")
    print("   2. I'll add my own documents later")
    print()
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        try:
            import download_documents
            download_documents.create_sample_documents()
            print("✅ Sample documents created!")
            return True
        except Exception as e:
            print(f"❌ Error creating sample documents: {e}")
            return False
    else:
        print("📝 Note: Place your PDF/TXT documents in the 'documents/' folder")
        print("   before running the system.")
        return True


def verify_installation():
    """Verify all components are installed"""
    print("\n🔍 Verifying installation...")
    
    try:
        import langchain
        import langchain_groq
        import langchain_community
        import faiss
        import sentence_transformers
        print("✅ All packages verified!")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Run: pip install -r requirements.txt")
        return False


def print_next_steps():
    """Print instructions for next steps"""
    print_header("🎉 Setup Complete!")
    
    print("Next steps:\n")
    print("1. If you haven't already, add your Groq API key to .env file")
    print("   Get free API key: https://console.groq.com\n")
    
    print("2. Add your legal documents to documents/ folder:")
    print("   - Constitution of India (PDF/TXT)")
    print("   - Indian Penal Code (PDF/TXT)\n")
    
    print("3. Run the interactive system:")
    print("   python interactive_legal_rag.py\n")
    
    print("4. Or try the demo:")
    print("   python demo.py\n")
    
    print("For more information, see README.md")
    print("\n" + "="*80 + "\n")


def main():
    """Main setup workflow"""
    print_header("🏛️  LEGAL RAG SYSTEM - SETUP")
    
    print("This script will help you set up the Legal RAG System")
    print("for querying Constitution of India and IPC documents.\n")
    
    # Step 1: Check Python
    if not check_python_version():
        return False
    
    # Step 2: Install dependencies
    install_deps = input("\n📦 Install dependencies from requirements.txt? (y/n): ").strip().lower()
    if install_deps == 'y':
        if not install_dependencies():
            return False
    else:
        print("⏭️  Skipping dependency installation")
    
    # Step 3: Setup environment
    if not setup_env_file():
        print("⚠️  Environment setup incomplete")
    
    # Step 4: Create directories
    if not create_directories():
        return False
    
    # Step 5: Download sample docs
    if not download_sample_docs():
        print("⚠️  Document setup incomplete")
    
    # Step 6: Verify
    if install_deps == 'y':
        if not verify_installation():
            return False
    
    # Done!
    print_next_steps()
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. Run again when ready!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
