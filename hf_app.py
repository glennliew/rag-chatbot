#!/usr/bin/env python3
"""
Hugging Face Spaces App Entry Point
This file will be automatically run when your Space starts
"""

import os
from interface import ChatbotInterface

# Set up environment for Hugging Face Spaces
os.environ.setdefault('GRADIO_TEMP_DIR', '/tmp')

def main():
    """Main function for Hugging Face Spaces deployment"""
    print("🤖 Starting RAG Chatbot on Hugging Face Spaces...")
    
    # Create and launch the chatbot interface
    chatbot_interface = ChatbotInterface()
    interface = chatbot_interface.create_interface()
    
    # Launch with public sharing enabled for HF Spaces
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # HF Spaces handles sharing
        show_error=True,
        show_tips=False
    )

if __name__ == "__main__":
    main()