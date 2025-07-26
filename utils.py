"""
Utility Module

This module provides helper functions for language translation,
out-of-scope detection, and other utility functions for the RAG chatbot.
"""

import re
from typing import Dict, List, Optional, Tuple
from deep_translator import GoogleTranslator
import tiktoken


class LanguageTranslator:
    """Handles multi-language support for the chatbot"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'english',
            'zh': 'chinese',
            'ms': 'malay',
            'ta': 'tamil',
            'hi': 'hindi',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'ja': 'japanese',
            'ko': 'korean'
        }
        
        # Cache for translations to avoid repeated API calls
        self.translation_cache = {}
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Simple heuristic-based language detection
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese characters
            return 'zh'
        elif re.search(r'[\u0d80-\u0dff]', text):  # Tamil characters
            return 'ta'
        elif re.search(r'[\u0900-\u097f]', text):  # Hindi characters
            return 'hi'
        else:
            # Default to English or try to detect using common words
            malay_words = ['apa', 'adalah', 'dengan', 'untuk', 'dari', 'ini', 'itu', 'dan', 'atau']
            text_lower = text.lower()
            
            if any(word in text_lower for word in malay_words):
                return 'ms'
            
            return 'en'  # Default to English
    
    def translate_text(self, text: str, source_lang: str = 'auto', target_lang: str = 'en') -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # Create cache key
        cache_key = f"{text}_{source_lang}_{target_lang}"
        
        # Check cache first
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            # Skip translation if source and target are the same
            if source_lang == target_lang:
                return text
            
            # Skip translation if source is auto-detected as English and target is English
            if source_lang == 'auto' and target_lang == 'en':
                detected_lang = self.detect_language(text)
                if detected_lang == 'en':
                    return text
            
            # Perform translation
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(text)
            
            # Cache the result
            self.translation_cache[cache_key] = translated
            
            return translated
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def translate_to_english(self, text: str) -> Tuple[str, str]:
        """
        Translate text to English and return both original language and translated text
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (detected_language, translated_text)
        """
        detected_lang = self.detect_language(text)
        
        if detected_lang == 'en':
            return detected_lang, text
        
        translated = self.translate_text(text, detected_lang, 'en')
        return detected_lang, translated
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate from English to target language
        
        Args:
            text: English text
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if target_lang == 'en':
            return text
        
        return self.translate_text(text, 'en', target_lang)


class OutOfScopeDetector:
    """Detects if a question is outside the scope of the knowledge base"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        
        # Common out-of-scope question patterns
        self.out_of_scope_patterns = [
            # Current events and time-sensitive questions
            r'\b(today|yesterday|tomorrow|this week|this month|this year|recently|latest|current|now)\b',
            r'\b(2024|2025|2026)\b',  # Specific recent years
            
            # Personal questions
            r'\b(my|mine|i am|i have|i want|tell me about myself)\b',
            
            # Real-time data requests
            r'\b(weather|stock price|news|live|real-time)\b',
            
            # Computation requests
            r'\b(calculate|compute|solve|math|arithmetic)\b',
            
            # Inappropriate content
            r'\b(password|hack|illegal|inappropriate)\b',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.out_of_scope_patterns]
    
    def is_out_of_scope_by_patterns(self, question: str) -> bool:
        """
        Check if question matches out-of-scope patterns
        
        Args:
            question: User question
            
        Returns:
            True if likely out of scope, False otherwise
        """
        for pattern in self.compiled_patterns:
            if pattern.search(question):
                return True
        return False
    
    def is_greeting_or_general(self, question: str) -> bool:
        """
        Check if question is a greeting or general chat
        
        Args:
            question: User question
            
        Returns:
            True if greeting/general, False otherwise
        """
        greetings = [
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', 'what\'s up', 'whats up', 'thank you', 'thanks', 'bye', 'goodbye'
        ]
        
        question_lower = question.lower().strip()
        
        return any(greeting in question_lower for greeting in greetings)
    
    def generate_helpful_response(self, question: str) -> str:
        """
        Generate a helpful response for out-of-scope questions
        
        Args:
            question: User question
            
        Returns:
            Helpful response string
        """
        if self.is_greeting_or_general(question):
            return "Hello! I'm here to help answer questions about the materials I've learned. What would you like to know?"
        
        return "I'm not sure how to answer that based on the information I have. Could you ask about something from the materials I've learned?"


class TextUtils:
    """Utility functions for text processing"""
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for a given model
        
        Args:
            text: Input text
            model: Model name for tokenization
            
        Returns:
            Number of tokens
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    @staticmethod
    def truncate_text(text: str, max_tokens: int = 1000, model: str = "gpt-4") -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            model: Model name for tokenization
            
        Returns:
            Truncated text
        """
        current_tokens = TextUtils.count_tokens(text, model)
        
        if current_tokens <= max_tokens:
            return text
        
        # Rough truncation based on character ratio
        ratio = max_tokens / current_tokens
        target_chars = int(len(text) * ratio * 0.9)  # Leave some buffer
        
        return text[:target_chars] + "..."
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-.,!?;:()"\']', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text (simple implementation)
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'this', 'that', 'these', 'those'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words and count frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]


def create_friendly_response(original_response: str, user_language: str = 'en') -> str:
    """
    Make response more friendly for primary school students
    
    Args:
        original_response: Original response text
        user_language: User's preferred language
        
    Returns:
        Friendlier response
    """
    # Add encouraging phrases
    encouraging_starters = [
        "Great question! ",
        "That's a wonderful thing to ask about! ",
        "I'm happy to help you learn about that! ",
        "Let me explain that for you! "
    ]
    
    # Simple way to make responses friendlier
    if not original_response.startswith(("Great", "That's", "I'm", "Hello", "Hi")):
        import random
        starter = random.choice(encouraging_starters)
        original_response = starter + original_response
    
    return original_response


def main():
    """Test the utility functions"""
    
    print("Testing Language Translator...")
    translator = LanguageTranslator()
    
    # Test translation
    test_texts = [
        "Hello, how are you?",
        "你好吗？",  # Chinese
        "Apa khabar?",  # Malay
    ]
    
    for text in test_texts:
        lang, translated = translator.translate_to_english(text)
        print(f"Original: {text}")
        print(f"Detected Language: {lang}")
        print(f"Translated: {translated}")
        print("-" * 30)
    
    print("\nTesting Out-of-Scope Detector...")
    detector = OutOfScopeDetector()
    
    test_questions = [
        "What is the weather today?",  # Out of scope
        "Hello, how are you?",  # Greeting
        "What is photosynthesis?",  # Potentially in scope
        "Calculate 2 + 2",  # Out of scope
    ]
    
    for question in test_questions:
        is_oos = detector.is_out_of_scope_by_patterns(question)
        is_greeting = detector.is_greeting_or_general(question)
        print(f"Q: {question}")
        print(f"Out of scope: {is_oos}, Greeting: {is_greeting}")
        if is_oos or is_greeting:
            response = detector.generate_helpful_response(question)
            print(f"Response: {response}")
        print("-" * 30)
    
    print("\nTesting Text Utils...")
    sample_text = "This is a sample text for testing various utility functions."
    tokens = TextUtils.count_tokens(sample_text)
    keywords = TextUtils.extract_keywords(sample_text)
    
    print(f"Text: {sample_text}")
    print(f"Token count: {tokens}")
    print(f"Keywords: {keywords}")


if __name__ == "__main__":
    main()