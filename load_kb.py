"""
Knowledge Base Loading Module

This module handles loading PDF documents, splitting them into chunks,
creating embeddings, and storing them in a vector database.
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document


class KnowledgeBaseLoader:
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the Knowledge Base Loader
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model: OpenAI embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Get API key from environment or fail gracefully
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )
        self.vector_store = None
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load and process a PDF document
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"Loaded {len(documents)} pages from PDF")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects
        """
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks")
        return chunks
    
    def create_vector_store(self, 
                          chunks: List[Document], 
                          persist_directory: str = "./chroma_db") -> Chroma:
        """
        Create and populate vector store with document chunks
        
        Args:
            chunks: List of Document chunks
            persist_directory: Directory to persist the vector store
            
        Returns:
            Chroma vector store instance
        """
        print("Creating embeddings and vector store...")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        # Persist the vector store
        self.vector_store.persist()
        print(f"Vector store created and persisted to {persist_directory}")
        
        return self.vector_store
    
    def load_existing_vector_store(self, persist_directory: str = "./chroma_db") -> Optional[Chroma]:
        """
        Load an existing vector store from disk
        
        Args:
            persist_directory: Directory where vector store is persisted
            
        Returns:
            Chroma vector store instance or None if not found
        """
        if os.path.exists(persist_directory):
            print(f"Loading existing vector store from {persist_directory}")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            return self.vector_store
        else:
            print(f"No existing vector store found at {persist_directory}")
            return None
    
    def process_pdf_to_vector_store(self, 
                                  pdf_path: str, 
                                  persist_directory: str = "./chroma_db",
                                  force_rebuild: bool = False) -> Chroma:
        """
        Complete pipeline to process PDF and create vector store
        
        Args:
            pdf_path: Path to the PDF file
            persist_directory: Directory to persist the vector store
            force_rebuild: Whether to force rebuild even if vector store exists
            
        Returns:
            Chroma vector store instance
        """
        # Check if vector store already exists and we don't want to force rebuild
        if not force_rebuild:
            existing_store = self.load_existing_vector_store(persist_directory)
            if existing_store:
                return existing_store
        
        # Load and process PDF
        documents = self.load_pdf(pdf_path)
        chunks = self.split_documents(documents)
        vector_store = self.create_vector_store(chunks, persist_directory)
        
        return vector_store
    
    def get_vector_store(self) -> Optional[Chroma]:
        """Get the current vector store instance"""
        return self.vector_store


def main():
    """
    Main function for testing the knowledge base loader
    """
    
    loader = KnowledgeBaseLoader()
    
    pdf_path = "data/Cells and Chemistry of Life.pdf"
    
    if os.path.exists(pdf_path):
        try:
            vector_store = loader.process_pdf_to_vector_store(pdf_path)
            print("Knowledge base successfully loaded!")
            
            # Test similarity search
            test_query = "What is this document about?"
            docs = vector_store.similarity_search(test_query, k=3)
            
            print(f"\nTest query: {test_query}")
            print(f"Found {len(docs)} relevant chunks:")
            for i, doc in enumerate(docs, 1):
                print(f"\nChunk {i}:")
                print(doc.page_content[:200] + "...")
                
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please place a PDF file in the project directory to test.")


if __name__ == "__main__":
    main()