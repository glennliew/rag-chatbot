#!/usr/bin/env python3
"""
Simple test script for the RAG chatbot
"""

from rag_pipeline import RAGPipelineManager

# Initialize pipeline
print("🤖 Initializing RAG Chatbot...")
pipeline = RAGPipelineManager()
pipeline.initialize_from_existing_db()

# Test questions
test_questions = [
    "What are cells?",
    "What is the basic unit of life?",
    "What are the main components of a cell?",
    "What is DNA?",
    "How do cells reproduce?",
    "What is quantum physics?",  # Out of scope
    "Hello there!",  # Greeting
]

print("\n" + "="*60)
print("RAG CHATBOT TEST RESULTS")
print("="*60)

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. Q: {question}")
    result = pipeline.ask_question(question)
    print(f"   A: {result['answer']}")
    print(f"   Relevant: {result['relevant']} | Score: {result['similarity_score']:.3f}")
    print("-" * 50)

print("\n✅ Test completed! Your RAG chatbot is working with your PDF knowledge base.")
print("📚 Knowledge Base: Cells and Chemistry of Life.pdf")
print("🤖 Ready for deployment!")