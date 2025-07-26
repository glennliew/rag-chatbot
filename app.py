"""
Main Application Entry Point

This module serves as the main entry point for the RAG Chatbot application,
providing command-line interface options to run different components.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import all modules
from load_kb import KnowledgeBaseLoader
from rag_pipeline import RAGPipelineManager
from interface import ChatbotInterface, create_streamlit_interface
from evaluate import RAGEvaluator
from utils import LanguageTranslator, OutOfScopeDetector


def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment setup...")
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   Please add your OpenAI API key to a .env file or environment")
        return False
    else:
        print("✅ OpenAI API key found")
    
    # Check for required files
    required_files = ["requirements.txt", "test_data.json"]
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True


def setup_knowledge_base(pdf_path: str = None, force_rebuild: bool = False):
    """Set up knowledge base from PDF"""
    print("📚 Setting up knowledge base...")
    
    if not pdf_path:
        # Look for default PDF files in current directory and /data
        default_locations = [
            # Current directory
            "sample_knowledge_base.pdf",
            "knowledge_base.pdf", 
            "docs.pdf",
            # /data directory (absolute)
            "/data/sample_knowledge_base.pdf",
            "/data/knowledge_base.pdf",
            "/data/docs.pdf",
            # ./data directory (relative)
            "./data/sample_knowledge_base.pdf",
            "./data/knowledge_base.pdf",
            "./data/docs.pdf"
        ]
        
        # Also scan /data and ./data directories for any PDF files
        for data_dir in ["/data", "./data", "data"]:
            if os.path.exists(data_dir):
                try:
                    for file in os.listdir(data_dir):
                        if file.lower().endswith('.pdf'):
                            default_locations.append(os.path.join(data_dir, file))
                except PermissionError:
                    pass
        
        for pdf in default_locations:
            if os.path.exists(pdf):
                pdf_path = pdf
                break
        
        if not pdf_path:
            print("❌ No PDF file found. Please provide a PDF file path.")
            print("   You can place a PDF in one of these locations:")
            print("   Current directory:")
            print("   - sample_knowledge_base.pdf")
            print("   - knowledge_base.pdf") 
            print("   - docs.pdf")
            print("   Or in /data directory:")
            print("   - /data/your_document.pdf")
            return False
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Initialize knowledge base
        pipeline = RAGPipelineManager()
        pipeline.initialize_from_pdf(pdf_path, force_rebuild=force_rebuild)
        
        print(f"✅ Knowledge base successfully created from: {pdf_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up knowledge base: {e}")
        return False


def run_chat_interface(interface_type: str = "gradio"):
    """Run the chat interface"""
    print(f"🤖 Starting {interface_type} chat interface...")
    
    try:
        if interface_type.lower() == "streamlit":
            # Import and run Streamlit
            import streamlit as st
            print("🌐 Streamlit interface starting...")
            print("   Run this command in your terminal:")
            print("   streamlit run app.py -- --mode streamlit")
            create_streamlit_interface()
            
        else:
            # Default to Gradio
            chatbot_interface = ChatbotInterface()
            interface = chatbot_interface.create_interface()
            
            print("🌐 Gradio interface starting...")
            print("   Open your browser and go to: http://127.0.0.1:7860")
            
            interface.launch(
                share=False,
                debug=False,
                server_name="127.0.0.1",
                server_port=7860,
                show_error=True
            )
            
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped by user")
    except Exception as e:
        print(f"❌ Error running interface: {e}")


def run_evaluation():
    """Run RAGAS evaluation"""
    print("🧪 Starting evaluation...")
    
    # Check if knowledge base exists
    pipeline = RAGPipelineManager()
    
    if not pipeline.initialize_from_existing_db():
        print("❌ No existing knowledge base found.")
        print("   Please set up a knowledge base first using: python app.py --setup")
        return
    
    try:
        evaluator = RAGEvaluator(pipeline)
        report = evaluator.run_comprehensive_evaluation()
        
        print("\n📊 Evaluation completed!")
        print("   Check the generated report file for detailed results.")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


def run_simple_test():
    """Run a simple test to verify everything works"""
    print("🧪 Running simple test...")
    
    # Initialize pipeline
    pipeline = RAGPipelineManager()
    
    if not pipeline.initialize_from_existing_db():
        print("❌ No knowledge base found. Please set up first.")
        return
    
    # Test questions
    test_questions = [
        "What is this document about?",
        "Hello!",  # Should trigger greeting response
        "What is quantum physics?",  # Should be out of scope
    ]
    
    print("\n" + "="*50)
    print("SIMPLE TEST RESULTS")
    print("="*50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Q: {question}")
        try:
            result = pipeline.ask_question(question)
            print(f"   A: {result['answer'][:100]}...")
            print(f"   Relevant: {result['relevant']}, Score: {result['similarity_score']:.3f}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("-" * 30)
    
    print("\n✅ Simple test completed!")


def show_help():
    """Show detailed help information"""
    help_text = """
🤖📚 RAG Chatbot - Help Guide

QUICK START:
1. Set up environment: cp .env.example .env (add your OpenAI API key)
2. Install dependencies: pip install -r requirements.txt
3. Set up knowledge base: python app.py --setup your_document.pdf
4. Run chat interface: python app.py --chat

COMMANDS:
  --setup [PDF_PATH]     : Set up knowledge base from PDF
  --chat [gradio|streamlit] : Run chat interface (default: gradio)
  --evaluate            : Run RAGAS evaluation
  --test                : Run simple functionality test
  --help                : Show this help message

EXAMPLES:
  python app.py --setup sample.pdf
  python app.py --chat gradio
  python app.py --evaluate
  python app.py --test

FILES:
  .env                  : Environment variables (API keys)
  sample_knowledge_base.pdf : Your PDF document
  chroma_db/           : Vector database (auto-created)
  test_data.json       : Test cases for evaluation

FEATURES:
  ✅ PDF knowledge base loading
  ✅ OpenAI GPT-4 powered responses
  ✅ Kid-friendly language (ages 8-12)
  ✅ Multilingual support (EN, ZH, MS, etc.)
  ✅ Out-of-scope detection
  ✅ RAGAS evaluation metrics
  ✅ Gradio & Streamlit interfaces

EVALUATION METRICS:
  - Faithfulness (≥80%)
  - Answer Relevancy (≥80%)
  - Context Precision (≥80%)
  - Context Recall (≥80%)

For more information, check the README.md file.
"""
    print(help_text)


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - A kid-friendly AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--setup", 
        nargs="?", 
        const="auto",
        help="Set up knowledge base from PDF file"
    )
    
    parser.add_argument(
        "--chat", 
        nargs="?", 
        const="gradio",
        choices=["gradio", "streamlit"],
        help="Run chat interface (gradio or streamlit)"
    )
    
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Run RAGAS evaluation"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run simple functionality test"
    )
    
    parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild of knowledge base"
    )
    
    parser.add_argument(
        "--help-detailed", 
        action="store_true",
        help="Show detailed help information"
    )
    
    # Handle Streamlit mode
    if len(sys.argv) > 1 and sys.argv[-1] == "streamlit":
        create_streamlit_interface()
        return
    
    args = parser.parse_args()
    
    # Show detailed help
    if args.help_detailed:
        show_help()
        return
    
    # Show welcome message
    print("🤖📚 Welcome to RAG Chatbot!")
    print("A kid-friendly AI assistant powered by your documents")
    print("-" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above.")
        return
    
    # Handle different modes
    if args.setup:
        pdf_path = args.setup if args.setup != "auto" else None
        success = setup_knowledge_base(pdf_path, args.force_rebuild)
        if success:
            print("\n💡 Next steps:")
            print("   1. Run chat interface: python app.py --chat")
            print("   2. Run evaluation: python app.py --evaluate")
    
    elif args.chat:
        run_chat_interface(args.chat)
    
    elif args.evaluate:
        run_evaluation()
    
    elif args.test:
        run_simple_test()
    
    else:
        # Default behavior - show help
        print("\n💡 No command specified. Here are your options:\n")
        show_help()


if __name__ == "__main__":
    main()