#!/usr/bin/env python3
"""
Test script for multilingual functionality

This script tests the multilingual response capabilities to ensure
encouraging phrases and out-of-scope messages appear in the correct language.
"""

from rag_pipeline import RAGPipelineManager
from utils import LanguageTranslator, OutOfScopeDetector, create_friendly_response

def test_multilingual_responses():
    """Test multilingual response generation"""
    print("🧪 Testing Multilingual Response Generation")
    print("=" * 50)
    
    # Initialize components
    pipeline = RAGPipelineManager()
    translator = LanguageTranslator()
    out_of_scope_detector = OutOfScopeDetector()
    
    # Try to initialize from existing database
    if pipeline.initialize_from_existing_db():
        print("✅ RAG Pipeline initialized from existing database")
    else:
        print("❌ No existing database found - cannot test in-scope responses")
        return
    
    # Test cases: [question, expected_language]
    test_cases = [
        ("What is photosynthesis?", "en"),  # English - in scope
        ("什么是光合作用？", "zh"),  # Chinese - in scope  
        ("Apakah fotosintesis?", "ms"),  # Malay - in scope
        ("Hello there!", "en"),  # English greeting
        ("你好！", "zh"),  # Chinese greeting
        ("What is the weather today?", "en"),  # English out-of-scope
        ("今天天气怎么样？", "zh"),  # Chinese out-of-scope
        ("Cuaca hari ini bagaimana?", "ms"),  # Malay out-of-scope
    ]
    
    for question, expected_lang in test_cases:
        print(f"\n🔍 Testing: {question}")
        print(f"Expected Language: {expected_lang}")
        
        # Detect language
        detected_lang, english_question = translator.translate_to_english(question)
        print(f"Detected Language: {detected_lang}")
        
        # Check if it's a greeting or out-of-scope
        if out_of_scope_detector.is_greeting_or_general(question):
            response = out_of_scope_detector.generate_helpful_response(question, detected_lang)
            print(f"🎯 Greeting Response: {response}")
        elif out_of_scope_detector.is_out_of_scope_by_patterns(question):
            response = out_of_scope_detector.generate_helpful_response(question, detected_lang)
            print(f"🚫 Out-of-Scope Response: {response}")
        else:
            # Test RAG pipeline response
            result = pipeline.ask_question(english_question, detected_lang)
            
            if result["relevant"]:
                # Test friendly response generation
                friendly_response = create_friendly_response(result["answer"], detected_lang)
                
                # Translate back if needed
                if detected_lang != 'en':
                    friendly_response = translator.translate_from_english(friendly_response, detected_lang)
                
                print(f"✅ In-Scope Response: {friendly_response}")
                print(f"Similarity Score: {result['similarity_score']:.3f}")
            else:
                print(f"🚫 Out-of-Scope RAG Response: {result['answer']}")
                print(f"Similarity Score: {result['similarity_score']:.3f}")
        
        print("-" * 40)

def test_language_detection():
    """Test language detection accuracy"""
    print("\n🔍 Testing Language Detection")
    print("=" * 30)
    
    translator = LanguageTranslator()
    
    test_texts = [
        ("Hello, how are you?", "en"),
        ("你好吗？", "zh"),
        ("Apa khabar?", "ms"),
        ("வணக்கம், எப்படி இருக்கிறீர்கள்?", "ta"),
        ("नमस्ते, आप कैसे हैं?", "hi"),
    ]
    
    for text, expected_lang in test_texts:
        detected = translator.detect_language(text)
        status = "✅" if detected == expected_lang else "❌"
        print(f"{status} '{text}' -> Detected: {detected}, Expected: {expected_lang}")

if __name__ == "__main__":
    print("🚀 Starting Multilingual Functionality Tests")
    print("=" * 60)
    
    try:
        test_language_detection()
        test_multilingual_responses()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()