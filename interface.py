"""
Interface Module - Enhanced Professional UI for Primary School Students

This module provides a beautiful, clean, and professional Gradio interface 
specifically designed for young learners.
"""

import os
import gradio as gr
from typing import List, Tuple, Optional
import time

from rag_pipeline import RAGPipelineManager
from utils import LanguageTranslator, OutOfScopeDetector, create_friendly_response


class ChatbotInterface:
    """Enhanced Gradio interface with professional design for primary school students"""
    
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
            return "🔴 Please upload a PDF file first!"
        
        try:
            # Initialize from uploaded PDF
            self.pipeline_manager.initialize_from_pdf(
                pdf_file.name, 
                force_rebuild=True
            )
            self.is_initialized = True
            
            # Clear chat history when new PDF is loaded
            self.chat_history = []
            
            return f"🟢 Successfully loaded: {os.path.basename(pdf_file.name)}"
            
        except Exception as e:
            return f"🔴 Error loading PDF: {str(e)}"
    
    def process_message(self, message: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """Process user message and return response using new messages format"""
        if not message.strip():
            return "", history
        
        # Check if system is initialized
        if not self.is_initialized:
            bot_response = "🤖 Hi there! I need you to upload a PDF document first so I can learn and help answer your questions!"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            return "", history
        
        try:
            # Detect language and translate to English if needed
            detected_lang, english_question = self.translator.translate_to_english(message)
            
            # Handle greetings and out-of-scope questions with language-aware responses
            if self.out_of_scope_detector.is_greeting_or_general(message):
                bot_response = self.out_of_scope_detector.generate_helpful_response(message, detected_lang)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": bot_response})
                return "", history
            
            # Check for obvious out-of-scope patterns
            if self.out_of_scope_detector.is_out_of_scope_by_patterns(message):
                bot_response = self.out_of_scope_detector.generate_helpful_response(message, detected_lang)
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": bot_response})
                return "", history
            
            # Get response from RAG pipeline with language context
            result = self.pipeline_manager.ask_question(english_question, detected_lang)
            
            # Make response more friendly only if it's a successful answer
            if result["relevant"]:
                friendly_response = create_friendly_response(result["answer"], detected_lang)
                
                # Translate response back to user's language if needed
                if detected_lang != 'en':
                    friendly_response = self.translator.translate_from_english(friendly_response, detected_lang)
            else:
                # For out-of-scope responses, don't add friendly starters as they're already handled
                friendly_response = result["answer"]
                
                # Add debugging info for non-relevant responses
                debug_info = f" (Similarity: {result['similarity_score']:.2f})"
                friendly_response += debug_info
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": friendly_response})
            
        except Exception as e:
            # Language-specific error messages
            error_messages = {
                'en': f"🤖 Oops! I had trouble understanding that. Could you try asking in a different way? (Error: {str(e)})",
                'zh': f"🤖 哎呀！我在理解这个问题时遇到了困难。你能换个方式问吗？（错误：{str(e)}）",
                'ms': f"🤖 Ops! Saya menghadapi masalah memahami itu. Bolehkah anda cuba bertanya dengan cara yang berbeza? (Ralat: {str(e)})",
                'ta': f"🤖 அய்யோ! அதைப் புரிந்துகொள்வதில் எனக்கு சிரமம் இருந்தது. வேறு வழியில் கேட்க முடியுமா? (பிழை: {str(e)})",
                'hi': f"🤖 अरे! मुझे उसे समझने में परेशानी हुई। क्या आप अलग तरीके से पूछ सकते हैं? (त्रुटि: {str(e)})"
            }
            
            # Detect language for error message
            detected_lang, _ = self.translator.translate_to_english(message)
            error_response = error_messages.get(detected_lang, error_messages['en'])
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_response})
        
        return "", history
    
    def clear_chat(self) -> List[dict]:
        """Clear chat history"""
        return []
    
    def get_system_status(self) -> str:
        """Get current system status"""
        if self.is_initialized:
            return "🟢 Ready to answer questions!"
        else:
            return "🟡 Please upload a PDF to get started"
    
    def create_interface(self) -> gr.Blocks:
        """Create the enhanced professional Gradio interface for primary school students"""
        
        # Professional dark CSS matching the Higgs Audio interface style
        custom_css = """
        /* Global Styles - Dark Theme */
        .gradio-container {
            font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
            background: #1a1a1a !important;
            min-height: 100vh;
            color: #ffffff !important;
        }
        
        /* Main Container */
        .block-container {
            background: #2d2d2d !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
            margin: 20px !important;
            padding: 24px !important;
            border: 1px solid #404040 !important;
        }
        
        /* Header Styles */
        h1 {
            color: #ffffff !important;
            text-align: center !important;
            font-size: 2.2em !important;
            font-weight: 600 !important;
            margin-bottom: 10px !important;
        }
        
        h2 {
            color: #ffffff !important;
            font-size: 1.3em !important;
            font-weight: 500 !important;
            margin: 16px 0 12px 0 !important;
        }
        
        h3 {
            color: #ffffff !important;
            font-size: 1.1em !important;
            font-weight: 500 !important;
            margin: 12px 0 8px 0 !important;
        }
        
        /* Dark Cards and Content Areas */
        .chatbot {
            border: 1px solid #404040 !important;
            border-radius: 12px !important;
            background: #363636 !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.2) !important;
        }
        
        .message {
            border-radius: 8px !important;
            padding: 12px !important;
            margin: 6px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            background: #404040 !important;
            color: #ffffff !important;
        }
        
        /* Green Button Styles - Matching Higgs Interface */
        .btn {
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
            border: none !important;
        }
        
        .btn-primary {
            background: #00d46a !important;
            color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(0, 212, 106, 0.3) !important;
        }
        
        .btn-primary:hover {
            background: #00c462 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(0, 212, 106, 0.4) !important;
        }
        
        .btn-secondary {
            background: #404040 !important;
            color: #ffffff !important;
            border: 1px solid #606060 !important;
        }
        
        .btn-secondary:hover {
            background: #505050 !important;
            transform: translateY(-1px) !important;
        }
        
        /* Input Styles - Dark Theme */
        .textbox input, .textbox textarea {
            background: #404040 !important;
            border: 1px solid #606060 !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-size: 14px !important;
            color: #ffffff !important;
            transition: all 0.2s ease !important;
        }
        
        .textbox input:focus, .textbox textarea:focus {
            border-color: #00d46a !important;
            box-shadow: 0 0 0 2px rgba(0, 212, 106, 0.2) !important;
            outline: none !important;
            background: #4a4a4a !important;
        }
        
        .textbox input::placeholder, .textbox textarea::placeholder {
            color: #999999 !important;
        }
        
        /* File Upload Area - Enhanced */
        .file-upload, .upload, input[type="file"] {
            border: 2px dashed #606060 !important;
            border-radius: 12px !important;
            background: #363636 !important;
            padding: 30px !important;
            text-align: center !important;
            transition: all 0.2s ease !important;
            color: #ffffff !important;
            min-height: 120px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        .file-upload:hover, .upload:hover {
            background: #404040 !important;
            border-color: #00d46a !important;
        }
        
        /* Upload button styling */
        .upload-btn {
            background: #5865f2 !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 24px !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            margin-bottom: 16px !important;
            cursor: pointer !important;
        }
        
        /* Upload area text */
        .upload-text {
            color: #cccccc !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            line-height: 1.4 !important;
        }
        
        /* File component specific styles */
        .file-component {
            background: #363636 !important;
            border: 2px dashed #606060 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
        
        .file-component:hover {
            border-color: #00d46a !important;
            background: #404040 !important;
        }
        
        /* Gradio file upload specific */
        .gradio-file {
            background: #363636 !important;
            border: 2px dashed #606060 !important;
            border-radius: 12px !important;
        }
        
        .gradio-file:hover {
            border-color: #00d46a !important;
            background: #404040 !important;
        }
        
        /* Dark Info Cards */
        .info-card {
            background: #363636 !important;
            border: 1px solid #404040 !important;
            border-radius: 12px !important;
            padding: 16px !important;
            margin: 12px 0 !important;
            color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        }
        
        .example-card {
            background: #363636 !important;
            border: 1px solid #404040 !important;
            border-radius: 12px !important;
            padding: 16px !important;
            margin: 12px 0 !important;
            color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
        }
        
        .example-card h3 {
            color: #00d46a !important;
            margin-bottom: 8px !important;
        }
        
        .info-card h3 {
            color: #00d46a !important;
            margin-bottom: 8px !important;
        }
        
        /* Status Indicators - Green Theme */
        .status-ready {
            background: #00d46a !important;
            color: #ffffff !important;
            padding: 8px 16px !important;
            border-radius: 20px !important;
            font-weight: 500 !important;
            text-align: center !important;
            font-size: 13px !important;
        }
        
        .status-waiting {
            background: #ff9500 !important;
            color: #ffffff !important;
            padding: 8px 16px !important;
            border-radius: 20px !important;
            font-weight: 500 !important;
            text-align: center !important;
            font-size: 13px !important;
        }
        
        /* Header Banner */
        .header-banner {
            background: linear-gradient(135deg, #363636 0%, #404040 100%) !important;
            border: 1px solid #505050 !important;
            color: #ffffff !important;
            padding: 20px !important;
            border-radius: 12px !important;
            margin: 16px auto !important;
            text-align: center !important;
        }
        
        .header-banner strong {
            color: #00d46a !important;
        }
        
        /* Footer */
        .footer {
            background: #363636 !important;
            border: 1px solid #404040 !important;
            color: #ffffff !important;
            padding: 20px !important;
            border-radius: 12px !important;
            text-align: center !important;
            margin-top: 24px !important;
        }
        
        .footer h3 {
            color: #00d46a !important;
            margin-bottom: 12px !important;
        }
        
        /* Labels and Text */
        label {
            color: #ffffff !important;
            font-weight: 500 !important;
        }
        
        .markdown {
            color: #ffffff !important;
        }
        
        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px !important;
            height: 8px !important;
        }
        
        ::-webkit-scrollbar-track {
            background: #2d2d2d !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #606060 !important;
            border-radius: 4px !important;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #00d46a !important;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 8px !important;
            }
            
            h1 {
                font-size: 1.8em !important;
            }
            
            .btn {
                font-size: 13px !important;
                padding: 8px 16px !important;
            }
            
            .block-container {
                margin: 8px !important;
                padding: 16px !important;
            }
        }
        """
        
        # Create the interface with enhanced design
        with gr.Blocks(
            css=custom_css, 
            title="🌟 Learning Assistant",
            theme=gr.themes.Soft()
        ) as interface:
            
            # Enhanced Header Section
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    <div style="text-align: center; padding: 20px;">
                        <h1>🌟 Learning Assistant</h1>
                        <p style="font-size: 1.2em; color: #cccccc; margin-top: 10px;">
                            Your friendly AI companion for learning and discovery! 🚀
                        </p>
                        <div class="header-banner">
                            <strong>🎯 How it works:</strong><br>
                            1️⃣ Upload your learning material (PDF) or use the default "Cells and Chemistry of Life" <br>
                            2️⃣ Ask questions about what you want to learn <br>
                            3️⃣ Get instant, helpful answers! 
                        </div>
                    </div>
                    """)
            
            # Main Content Area
            with gr.Row(equal_height=True):
                # Left Column - Chat Interface
                with gr.Column(scale=2):
                    gr.HTML('<h2>💬 Chat with Your Learning Buddy</h2>')
                    
                    # Chat Interface
                    chatbot = gr.Chatbot(
                        height=450,
                        label="",
                        show_label=False,
                        avatar_images=("👦", "🤖"),
                        type="messages"
                    )
                    
                    # Input Area
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="💭 Ask me anything about your document... (e.g., 'What is this about?')",
                            label="",
                            show_label=False,
                            scale=4,
                            max_lines=3
                        )
                        send_btn = gr.Button(
                            "Send 🚀", 
                            variant="primary", 
                            scale=1,
                            size="lg"
                        )
                    
                    # Control Buttons
                    with gr.Row():
                        clear_btn = gr.Button(
                            "🗑️ Clear Chat", 
                            variant="secondary",
                            size="lg",
                        
                        )
                        gr.HTML('<div style="flex-grow: 1;"></div>')  # Spacer
                
                # Right Column - Upload and Examples
                with gr.Column(scale=1):
                    # Upload Section
                    gr.HTML('<h2>📚 Upload Your Learning Material</h2>')
                    
                    # Custom upload area
                    gr.HTML("""
                    <div style="margin-bottom: 16px;">
                        <div style="background: #5865f2; color: white; border: none; border-radius: 8px; 
                                   padding: 12px 24px; font-size: 14px; font-weight: 500; 
                                   display: inline-block; margin-bottom: 0px;">
                            📄 Choose your PDF document
                        </div>
                    </div>
                    """)
                    
                    pdf_upload = gr.File(
                        file_types=[".pdf"],
                        label="",
                        show_label=False,
                        file_count="single",
                        height=100,
                        elem_classes=["file-component"]
                    )
                    
                    upload_status = gr.HTML(
                        value=f'<div class="status-waiting">{self.get_system_status()}</div>'
                    )
                    
                    # Language Support
                    gr.HTML("""
                    <div class="info-card">
                        <h3>🌍 Multi-Language Support</h3>
                        <p style="margin: 10px 0; text-align: left;">
                            <strong>English:</strong> "What is photosynthesis?"<br>
                            <strong>中文:</strong> "什么是光合作用？"<br>
                            <strong>Bahasa:</strong> "Apakah fotosintesis?"<br>
                            <strong>தமிழ்:</strong> "ஒளிச்சேர்க்கை என்றால் என்ன?"
                        </p>
                    </div>
                    """)
            
            # Enhanced Footer with Tips
            gr.HTML("""
            <div class="footer">
                <h3>🎓 Learning Tips for Better Results</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 15px;">
                    <div>
                        <strong>✨ Be Specific</strong><br>
                        Instead of "Tell me about this", try "What are the main parts of a cell?"
                    </div>
                    <div>
                        <strong>🔄 Try Different Ways</strong><br>
                        If you don't get a good answer, rephrase your question!
                    </div>
                    <div>
                        <strong>🌟 Stay Curious</strong><br>
                        Ask follow-up questions to learn more deeply!
                    </div>
                </div>
                <p style="margin-top: 20px; font-size: 0.9em; opacity: 0.9;">
                    Made with ❤️ for young learners everywhere! Happy learning! 🌈📖
                </p>
            </div>
            """)
            
            # Enhanced Event Handlers
            def update_status(file):
                status = self.upload_pdf_and_initialize(file)
                if "Successfully" in status:
                    return f'<div class="status-ready">{status}</div>'
                else:
                    return f'<div class="status-waiting">{status}</div>'
            
            # Connect events
            pdf_upload.change(
                fn=update_status,
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
        
        return interface


def main():
    """Main function to run the enhanced interface"""
    print("🌟 Starting Smart Learning Assistant...")
    print("🚀 Designed for young learners with professional UI/UX")
    
    chatbot_interface = ChatbotInterface()
    interface = chatbot_interface.create_interface()
    
    # Launch with enhanced settings
    interface.launch(
        share=False,
        debug=False,
        show_error=True,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        favicon_path=None,
        app_kwargs={"docs_url": None, "redoc_url": None}
    )


if __name__ == "__main__":
    main()