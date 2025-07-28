#!/usr/bin/env python3
"""
RAG Chatbot - Main Entry Point for Hugging Face Spaces
Educational AI Assistant for Primary School Students
"""

import os
from interface import ChatbotInterface

def main():
    """Main entry point for Hugging Face Spaces deployment"""
    print("🌟 Starting Smart Learning Assistant for Hugging Face Spaces...")
    print("🚀 Designed for young learners with professional UI/UX")
    
    # Create chatbot interface
    chatbot_interface = ChatbotInterface()
    interface = chatbot_interface.create_interface()
    
    # Launch for Hugging Face Spaces
    interface.launch(
        share=False,
        debug=False,
        show_error=True,
        server_name="0.0.0.0",  # Important for HF Spaces
        server_port=7860,
        inbrowser=False,  # Don't try to open browser on HF Spaces
        favicon_path=None
    )

if __name__ == "__main__":
    main()