#!/usr/bin/env python3
"""
Hugging Face Spaces App Entry Point - Production Ready
This file will be automatically run when your Space starts
"""

import os
import sys
import gradio as gr
from pathlib import Path

# Set up environment for Hugging Face Spaces
os.environ.setdefault('GRADIO_TEMP_DIR', '/tmp')

def check_requirements():
    """Check if all requirements are met"""
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        return False, "❌ OpenAI API key not found. Please add OPENAI_API_KEY to your Hugging Face Spaces secrets."
    
    # Check for PDF file
    pdf_paths = [
        "data/Cells and Chemistry of Life.pdf",
        "Cells and Chemistry of Life.pdf"
    ]
    
    pdf_found = False
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            pdf_found = True
            break
    
    if not pdf_found:
        return False, f"❌ PDF file not found. Please upload your PDF to one of these locations: {pdf_paths}"
    
    return True, "✅ All requirements met!"

def create_error_interface(error_message):
    """Create a simple error interface"""
    with gr.Blocks(title="RAG Chatbot - Setup Required") as interface:
        gr.Markdown(f"""
        # 🤖 RAG Chatbot Setup Required
        
        {error_message}
        
        ## Setup Instructions:
        1. **Add API Key**: Go to your Space settings → Secrets → Add `OPENAI_API_KEY`
        2. **Upload PDF**: Upload your knowledge base PDF file
        3. **Restart Space**: The space will automatically restart and load your chatbot
        
        ## Need Help?
        Check the README.md file for detailed setup instructions.
        """)
    
    return interface

def main():
    """Main function for Hugging Face Spaces deployment"""
    print("🤖 Starting RAG Chatbot on Hugging Face Spaces...")
    
    # Check requirements
    requirements_ok, message = check_requirements()
    
    if not requirements_ok:
        print(message)
        interface = create_error_interface(message)
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        return
    
    # Import and create the main interface
    try:
        from interface import ChatbotInterface
        
        print("✅ Requirements met, initializing chatbot...")
        chatbot_interface = ChatbotInterface()
        interface = chatbot_interface.create_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            show_tips=False
        )
        
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        error_interface = create_error_interface(f"❌ Error initializing chatbot: {str(e)}")
        error_interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

if __name__ == "__main__":
    main()