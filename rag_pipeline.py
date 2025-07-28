"""
RAG Pipeline Module

This module implements the Retrieval-Augmented Generation pipeline
using LangChain with OpenAI GPT-4 and custom prompts for primary school students.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores.chroma import Chroma

from load_kb import KnowledgeBaseLoader

# Load environment variables
load_dotenv()


class RAGChatbot:
    def __init__(self, 
                 vector_store: Optional[Chroma] = None,
                 model_name: str = "gpt-4-1106-preview",
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 similarity_threshold: float = 0.4):  # Optimized value for best balance
        """
        Initialize RAG Chatbot
        
        Args:
            vector_store: Chroma vector store for retrieval
            model_name: OpenAI model to use
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens in response
            similarity_threshold: Minimum similarity score for relevance (optimized for precision)
        """
        # Initialize with optimized parameters
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature  
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold  # Fine-tuned from 0.4 to 0.45
        
        # Get API key from environment or fail gracefully
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize components
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=api_key
        )
        
        self.retriever = None
        self.rag_chain = None
        self.is_initialized = False
        
        # Initialize retriever if vector store is provided
        if vector_store:
            # Optimized retrieval: more documents for better recall
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 6})  # Increased from 4 to 6
        
        # Create prompts
        self.system_prompt = self._create_system_prompt()
        self.chat_prompt = self._create_chat_prompt()
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for primary school friendly responses with strong context adherence"""
        return """You are a friendly and helpful AI assistant designed to answer questions for primary school students (ages 8-12). 

CRITICAL CONTEXT RULES:
1. ONLY use information provided in the context below
2. DO NOT add any information that is not explicitly mentioned in the context
3. If the context doesn't contain enough information to answer the question completely, say: "I'm not sure how to answer that based on the information I have."
4. Quote or paraphrase directly from the context when possible
5. NEVER make assumptions or add details not in the context

COMMUNICATION GUIDELINES:
1. Use simple, clear language that children can understand
2. Be encouraging and positive in your tone  
3. Break down complex ideas into simple steps
4. Use examples from the context when available
5. Keep answers concise but complete
6. Start with friendly phrases like "Great question!" or "Let me help you with that!"

RESPONSE STRUCTURE:
- Base your entire answer on the provided context
- If context is incomplete, acknowledge what you can answer and what you cannot
- Never fabricate details not present in the context
- Prioritize accuracy over completeness

Remember: Your primary goal is to be helpful AND factually accurate based only on the provided context!"""
    
    def _create_chat_prompt(self) -> ChatPromptTemplate:
        """Create chat prompt template"""
        system_message = SystemMessagePromptTemplate.from_template(self.system_prompt)
        
        human_template = """Context information:
{context}

Question: {question}

Please provide a helpful answer based on the context above but do not mention about the context provided. Remember to use simple language suitable for primary school students."""
        
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def _create_rag_chain(self):
        """Create the RAG chain"""
        if not self.retriever:
            return None
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.chat_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def set_vector_store(self, vector_store: Chroma):
        """Set vector store and update retriever with optimized parameters"""
        self.vector_store = vector_store
        # Optimized retrieval: more documents for better recall 
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 6})  # Increased from 4 to 6
        self.rag_chain = self._create_rag_chain()
    
    def retrieve_context(self, question: str, k: int = 6) -> Tuple[List[Document], float]:  # Increased from 4 to 6
        """
        Retrieve relevant context and calculate relevance score with optimized parameters
        
        Args:
            question: User question
            k: Number of documents to retrieve (default increased to 6)
            
        Returns:
            Tuple of (documents, average_similarity_score)
        """
        if not self.vector_store:
            return [], 0.0
        
        # Get documents with similarity scores
        docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        if not docs_with_scores:
            return [], 0.0
        
        # Extract documents and calculate average similarity
        documents = [doc for doc, score in docs_with_scores]
        
        # Note: Chroma returns distance (lower is better), convert to similarity
        similarities = [1 - score for doc, score in docs_with_scores if score <= 1.0]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        return documents, avg_similarity
    
    def check_relevance(self, question: str) -> bool:
        """
        Check if the question has relevant context in the knowledge base
        
        Args:
            question: User question
            
        Returns:
            True if relevant context exists, False otherwise
        """
        docs, avg_similarity = self.retrieve_context(question, k=3)
        return avg_similarity >= self.similarity_threshold
    
    def generate_answer(self, question: str, user_language: str = 'en') -> Dict[str, Any]:
        """
        Generate answer for a question with context checking
        
        Args:
            question: User question
            user_language: Detected language of the user
            
        Returns:
            Dictionary containing answer, context, and metadata
        """
        if not self.rag_chain:
            # Language-specific "no KB" messages
            no_kb_messages = {
                'en': "I need a knowledge base to answer questions. Please load a PDF first!",
                'zh': "我需要一个知识库来回答问题。请先加载一个PDF文件！",
                'ms': "Saya memerlukan pangkalan pengetahuan untuk menjawab soalan. Sila muat naik fail PDF dahulu!",
                'ta': "கேள்விகளுக்கு பதிலளிக்க எனக்கு அறிவுத் தளம் தேவை. முதலில் PDF கோப்பை ஏற்றவும்!",
                'hi': "प्रश्नों का उत्तर देने के लिए मुझे एक ज्ञान आधार चाहिए। कृपया पहले एक PDF लोड करें!"
            }
            
            return {
                "answer": no_kb_messages.get(user_language, no_kb_messages['en']),
                "context": [],
                "relevant": False,
                "similarity_score": 0.0
            }
        
        # Retrieve context and check relevance
        context_docs, similarity_score = self.retrieve_context(question)
        
        if similarity_score < self.similarity_threshold:
            # Language-specific out-of-scope messages
            out_of_scope_messages = {
                'en': "I'm not sure how to answer that based on the information I have.",
                'zh': "根据我掌握的信息，我不确定如何回答这个问题。",
                'ms': "Saya tidak pasti bagaimana untuk menjawab itu berdasarkan maklumat yang saya ada.",
                'ta': "என்னிடம் உள்ள தகவல்களின் அடிப்படையில் அதற்கு எவ்வாறு பதிலளிப்பது என்று எனக்குத் தெரியவில்லை. ",
                'hi': "मेरे पास जो जानकारी है उसके आधार पर मुझे யकीन नहीं है कि इसका உதவி எப்படி செய்வது."
            }
            
            return {
                "answer": out_of_scope_messages.get(user_language, out_of_scope_messages['en']),
                "context": context_docs,
                "relevant": False,
                "similarity_score": similarity_score
            }
        
        # Generate answer using RAG chain
        try:
            answer = self.rag_chain.invoke(question)
            
            return {
                "answer": answer,
                "context": context_docs,
                "relevant": True,
                "similarity_score": similarity_score
            }
        
        except Exception as e:
            # Language-specific error messages
            error_messages = {
                'en': f"Sorry, I had trouble generating an answer. Error: {str(e)}",
                'zh': f"抱歉，我在生成答案时遇到了问题。错误：{str(e)}",
                'ms': f"Maaf, saya menghadapi masalah untuk menghasilkan jawapan. Ralat: {str(e)}",
                'ta': f"மன்னிக்கவும், பதில் உருவாக்குவதில் எனக்குச் சிக்கல் ஏற்பட்டது. பிழை: {str(e)}",
                'hi': f"क्षमा करें, मुझे उत्तर बनाने में समस्या हुई। त्रुटि: {str(e)}"
            }
            
            return {
                "answer": error_messages.get(user_language, error_messages['en']),
                "context": context_docs,
                "relevant": False,
                "similarity_score": similarity_score
            }
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface that returns just the answer
        
        Args:
            question: User question
            
        Returns:
            Generated answer string
        """
        result = self.generate_answer(question)
        return result["answer"]


class RAGPipelineManager:
    """Manager class for the complete RAG pipeline"""
    
    def __init__(self):
        self.kb_loader = KnowledgeBaseLoader()
        self.chatbot = RAGChatbot()
        self.is_initialized = False
    
    def initialize_from_pdf(self, 
                          pdf_path: str, 
                          persist_directory: str = "./chroma_db",
                          force_rebuild: bool = False):
        """
        Initialize the RAG pipeline from a PDF
        
        Args:
            pdf_path: Path to PDF file
            persist_directory: Directory to persist vector store
            force_rebuild: Whether to force rebuild of vector store
        """
        print("Initializing RAG Pipeline...")
        
        # Load knowledge base
        vector_store = self.kb_loader.process_pdf_to_vector_store(
            pdf_path, persist_directory, force_rebuild
        )
        
        # Set up chatbot
        self.chatbot.set_vector_store(vector_store)
        self.is_initialized = True
        
        print("RAG Pipeline initialized successfully!")
    
    def initialize_from_existing_db(self, persist_directory: str = "./chroma_db"):
        """
        Initialize from existing vector database
        
        Args:
            persist_directory: Directory where vector store is persisted
        """
        vector_store = self.kb_loader.load_existing_vector_store(persist_directory)
        
        if vector_store:
            self.chatbot.set_vector_store(vector_store)
            self.is_initialized = True
            print("RAG Pipeline initialized from existing database!")
            return True
        else:
            print("No existing database found!")
            return False
    
    def ask_question(self, question: str, user_language: str = 'en') -> Dict[str, Any]:
        """
        Ask a question to the chatbot
        
        Args:
            question: User question
            user_language: Detected language of the user
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.is_initialized:
            # Language-specific initialization messages
            init_messages = {
                'en': "Please initialize the RAG pipeline first by loading a PDF!",
                'zh': "请先加载PDF文件来初始化RAG管道！",
                'ms': "Sila mulakan saluran RAG dahulu dengan memuatkan PDF!",
                'ta': "முதலில் PDF ஐ ஏற்றி RAG பைப்லைனை துவக்கவும்!",
                'hi': "कृपया पहले एक PDF लोड करके RAG पाइपलाइन को प्रारंभ करें!"
            }
            
            return {
                "answer": init_messages.get(user_language, init_messages['en']),
                "context": [],
                "relevant": False,
                "similarity_score": 0.0
            }
        
        return self.chatbot.generate_answer(question, user_language)
    
    def simple_chat(self, question: str) -> str:
        """Simple chat interface"""
        result = self.ask_question(question)
        return result["answer"]


def main():
    """Test the RAG pipeline"""
    
    # Initialize pipeline manager
    pipeline = RAGPipelineManager()
    
    # Try to initialize from existing database first
    if not pipeline.initialize_from_existing_db():
        # If no existing database, try to create from PDF
        pdf_path = "sample_knowledge_base.pdf"
        if os.path.exists(pdf_path):
            pipeline.initialize_from_pdf(pdf_path)
        else:
            print("No PDF file found for testing. Please add a 'sample_knowledge_base.pdf' file.")
            return
    
    # Test questions
    test_questions = [
        "What is this document about?",
        "Can you summarize the main points?",
        "What is quantum physics?",  # Should trigger out-of-scope response
    ]
    
    print("\n" + "="*50)
    print("Testing RAG Chatbot")
    print("="*50)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = pipeline.ask_question(question)
        print(f"A: {result['answer']}")
        print(f"Relevant: {result['relevant']}, Similarity: {result['similarity_score']:.3f}")
        print("-" * 50)


if __name__ == "__main__":
    main()