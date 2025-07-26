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
                 similarity_threshold: float = 0.5):
        """
        Initialize RAG Chatbot
        
        Args:
            vector_store: Chroma vector store for retrieval
            model_name: OpenAI model name
            temperature: LLM temperature for generation
            max_tokens: Maximum tokens for response
            similarity_threshold: Minimum similarity score for relevant context
        """
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize retriever if vector store is provided
        self.retriever = None
        if vector_store:
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Create prompts
        self.system_prompt = self._create_system_prompt()
        self.chat_prompt = self._create_chat_prompt()
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for primary school friendly responses"""
        return """You are a friendly and helpful AI assistant designed to answer questions for primary school students (ages 8-12). 

IMPORTANT GUIDELINES:
1. Use simple, clear language that children can understand
2. Be encouraging and positive in your tone
3. Break down complex ideas into simple steps
4. Use examples and analogies that kids can relate to
5. If you don't have enough information in the context to answer properly, say: "I'm not sure how to answer that based on the information I have."
6. Always stay on topic and only use the provided context
7. Keep answers concise but complete
8. Use friendly phrases like "Great question!" or "Let me help you with that!"

CONTEXT RELEVANCE:
- Only answer questions if the provided context contains relevant information
- If the context doesn't match the question well, politely say you don't have that information
- Never make up information that isn't in the context

Remember: You're helping young learners, so be patient, kind, and educational!"""
    
    def _create_chat_prompt(self) -> ChatPromptTemplate:
        """Create chat prompt template"""
        system_message = SystemMessagePromptTemplate.from_template(self.system_prompt)
        
        human_template = """Context information:
{context}

Question: {question}

Please provide a helpful answer based on the context above. Remember to use simple language suitable for primary school students."""
        
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
        """Set vector store and update retriever"""
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        self.rag_chain = self._create_rag_chain()
    
    def retrieve_context(self, question: str, k: int = 4) -> Tuple[List[Document], float]:
        """
        Retrieve relevant context and calculate relevance score
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
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
    
    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate answer for a question with context checking
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing answer, context, and metadata
        """
        if not self.rag_chain:
            return {
                "answer": "I need a knowledge base to answer questions. Please load a PDF first!",
                "context": [],
                "relevant": False,
                "similarity_score": 0.0
            }
        
        # Retrieve context and check relevance
        context_docs, similarity_score = self.retrieve_context(question)
        
        if similarity_score < self.similarity_threshold:
            return {
                "answer": "I'm not sure how to answer that based on the information I have. Could you ask about something from the materials I've learned?",
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
            return {
                "answer": f"Sorry, I had trouble generating an answer. Error: {str(e)}",
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
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot
        
        Args:
            question: User question
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.is_initialized:
            return {
                "answer": "Please initialize the RAG pipeline first by loading a PDF!",
                "context": [],
                "relevant": False,
                "similarity_score": 0.0
            }
        
        return self.chatbot.generate_answer(question)
    
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