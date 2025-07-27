"""
Interface Module - Gradio Compatible Version

This module provides a Gradio-based chat interface that works with multiple Gradio versions
"""

import os
import gradio as gr
from typing import List, Tuple, Optional
import time

from rag_pipeline import RAGPipelineManager
from utils import LanguageTranslator, OutOfScopeDetector, create_friendly_response


class ChatbotInterface:
    """Gradio interface for the RAG chatbot - Compatible version"""
    
    def __init__(self):
        self.pipeline_manager = RAGPipelineManager()
        self.translator = LanguageTranslator()
        self.out_of_scope_detector = OutOfScopeDetector()
        self.is_initialized = False
        self.chat_history = []
        
        # Try to initialize from existing database
        self._try_initialize()
    
    def _try_initialize(self):
        """Try to initialize from existing database or sample PDF"""
        try:
            # First try existing database
            if self.pipeline_manager.initialize_from_existing_db():
                self.is_initialized = True
                return
        except Exception as e:
            print(f"Warning: Could not initialize from existing database: {e}")
        
        # Then try sample PDF in current directory and /data
        sample_pdf_paths = [
            "sample_knowledge_base.pdf",
            "knowledge_base.pdf",
            "docs.pdf",
            "/data/sample_knowledge_base.pdf",
            "/data/knowledge_base.pdf",
            "/data/docs.pdf",
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
                            sample_pdf_paths.append(os.path.join(data_dir, file))
                except PermissionError:
                    pass
        
        for pdf_path in sample_pdf_paths:
            if os.path.exists(pdf_path):
                try:
                    self.pipeline_manager.initialize_from_pdf(pdf_path)
                    self.is_initialized = True
                    return
                except Exception as e:
                    print(f"Failed to initialize from {pdf_path}: {e}")
                    continue
    
    def upload_pdf_and_initialize(self, pdf_file) -> str:
        """Handle PDF upload and initialize the RAG pipeline"""
        if pdf_file is None:
            return "Please upload a PDF file first!"
        
        try:
            # Initialize from uploaded PDF
            self.pipeline_manager.initialize_from_pdf(
                pdf_file.name, 
                force_rebuild=True
            )
            self.is_initialized = True
            
            # Clear chat history when new PDF is loaded
            self.chat_history = []
            
            return f"✅ Successfully loaded knowledge base from: {os.path.basename(pdf_file.name)}"
            
        except Exception as e:
            return f"❌ Error loading PDF: {str(e)}"
    
    def process_message(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Process user message and return response using traditional format"""
        if not message.strip():
            return "", history
        
        # Check if system is initialized
        if not self.is_initialized:
            bot_response = "🤖 Hi there! I need you to upload a PDF document first so I can learn and help answer your questions!"
            history.append([message, bot_response])
            return "", history
        
        # Handle greetings and out-of-scope questions
        if self.out_of_scope_detector.is_greeting_or_general(message):
            bot_response = "👋 Hello! I'm here to help answer questions about the materials I've learned. What would you like to know?"
            history.append([message, bot_response])
            return "", history
        
        # Check for obvious out-of-scope patterns
        if self.out_of_scope_detector.is_out_of_scope_by_patterns(message):
            bot_response = self.out_of_scope_detector.generate_helpful_response(message)
            history.append([message, bot_response])
            return "", history
        
        try:
            # Detect language and translate to English if needed
            detected_lang, english_question = self.translator.translate_to_english(message)
            
            # Get response from RAG pipeline
            result = self.pipeline_manager.ask_question(english_question)
            
            # Make response more friendly
            friendly_response = create_friendly_response(result["answer"], detected_lang)
            
            # Translate response back to user's language if needed
            if detected_lang != 'en':
                friendly_response = self.translator.translate_from_english(friendly_response, detected_lang)
            
            # Add debugging info for non-relevant responses
            if not result["relevant"]:
                debug_info = f" (Similarity: {result['similarity_score']:.2f})"
                friendly_response += debug_info
            
            history.append([message, friendly_response])
            
        except Exception as e:
            error_response = f"🤖 Oops! I had trouble understanding that. Could you try asking in a different way? (Error: {str(e)})"
            history.append([message, error_response])
        
        return "", history
    
    def clear_chat(self) -> List[List[str]]:
        """Clear chat history"""
        return []
    
    def get_system_status(self) -> str:
        """Get current system status"""
        if self.is_initialized:
            return "✅ Ready to answer questions!"
        else:
            return "⏳ Please upload a PDF to get started"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface - Compatible version"""
        
        # Custom CSS for kid-friendly design
        custom_css = """
        .gradio-container {
            font-family: 'Comic Sans MS', cursive, sans-serif !important;
        }
        .chatbot {
            border-radius: 15px !important;
            border: 3px solid #4CAF50 !important;
        }
        .message {
            border-radius: 10px !important;
            padding: 10px !important;
        }
        h1 {
            color: #2196F3 !important;
            text-align: center !important;
        }
        h2 {
            color: #FF9800 !important;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Kids RAG Chatbot 🤖📚") as interface:
            
            # Header
            gr.Markdown(
                """
                # 🤖📚 Smart Learning Buddy
                ### Your friendly AI assistant that answers questions from your documents!
                
                **How to use:**
                1. 📁 Upload a PDF document below (or it will auto-load if available)
                2. 💬 Ask me questions about what you've learned
                3. 🌍 You can ask in English, Chinese, Malay, or other languages!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface - using traditional format for compatibility
                    chatbot = gr.Chatbot(
                        height=400,
                        label="💬 Chat with your Learning Buddy"
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your question here... 🤔",
                            label="Your Question",
                            scale=4
                        )
                        send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                
                with gr.Column(scale=1):
                    # PDF upload and status
                    gr.Markdown("### 📁 Upload Your Learning Material")
                    
                    pdf_upload = gr.File(
                        file_types=[".pdf"],
                        label="Choose a PDF file",
                        type="filepath"
                    )
                    
                    upload_status = gr.Textbox(
                        label="Status",
                        value=self.get_system_status(),
                        interactive=False
                    )
                    
                    # Example questions
                    gr.Markdown(
                        """
                        ### 💡 Example Questions to Try:
                        - "What is this document about?"
                        - "Can you summarize the main points?"
                        - "Tell me about..." (specific topic)
                        - "How does ... work?"
                        - "Why is ... important?"
                        
                        ### 🌍 Language Support:
                        - English: "What is photosynthesis?"
                        - Chinese: "什么是光合作用？"
                        - Malay: "Apakah fotosintesis?"
                        """
                    )
            
            # Event handlers
            pdf_upload.change(
                fn=self.upload_pdf_and_initialize,
                inputs=[pdf_upload],
                outputs=[upload_status]
            )
            
            msg_input.submit(
                fn=self.process_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                fn=self.process_message,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot]
            )
            
            # Add footer
            gr.Markdown(
                """
                ---
                **Tips for better answers:**
                - Ask specific questions about the content
                - Use simple, clear language
                - If you don't get a good answer, try rephrasing your question
                
                Made with ❤️ for young learners everywhere! 🌟
                """
            )
        
        return interface


def main():
    """Main function to run the interface"""
    chatbot_interface = ChatbotInterface()
    interface = chatbot_interface.create_interface()
    
    # Launch interface
    interface.launch(
        share=False,
        debug=True,
        server_name="127.0.0.1",
        server_port=7860
    )


if __name__ == "__main__":
    main()