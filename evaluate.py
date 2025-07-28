"""
Evaluation Module

This module uses RAGAS (Retrieval-Augmented Generation Assessment) to evaluate
the performance of the RAG chatbot on various metrics.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Any, Tuple
import time
from datetime import datetime

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Dataset creation
from datasets import Dataset

from rag_pipeline import RAGPipelineManager
from utils import LanguageTranslator


class RAGEvaluator:
    """Evaluates RAG chatbot performance using RAGAS metrics"""
    
    def __init__(self, pipeline_manager: RAGPipelineManager = None):
        """
        Initialize the evaluator
        
        Args:
            pipeline_manager: Initialized RAG pipeline manager
        """
        self.pipeline_manager = pipeline_manager or RAGPipelineManager()
        self.translator = LanguageTranslator()
        
        # RAGAS metrics
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        
        # Load test data
        self.test_data = self._load_test_data()
    
    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data from JSON file"""
        try:
            with open("test_data.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: test_data.json not found. Using default test cases.")
            return self._get_default_test_data()
    
    def _get_default_test_data(self) -> Dict[str, Any]:
        """Default test data if file is not found"""
        return {
            "test_cases": [
                {
                    "question": "What is photosynthesis?",
                    "expected_answer": "Photosynthesis is the process by which plants make their own food using sunlight, water, and carbon dioxide.",
                    "contexts": ["Photosynthesis is a biological process where plants convert light energy into chemical energy."],
                    "ground_truth": "Photosynthesis is the process plants use to make food from sunlight, water, and carbon dioxide."
                }
            ]
        }
    
    def generate_responses(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Generate responses for a list of questions
        
        Args:
            questions: List of questions to ask
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        print(f"Generating responses for {len(questions)} questions...")
        
        for i, question in enumerate(questions, 1):
            print(f"Processing question {i}/{len(questions)}: {question[:50]}...")
            
            try:
                # Get response from pipeline
                result = self.pipeline_manager.ask_question(question)
                
                responses.append(result)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                responses.append({
                    "answer": f"Error: {str(e)}",
                    "context": [],
                    "relevant": False,
                    "similarity_score": 0.0
                })
        
        return responses
    
    def prepare_evaluation_dataset(self, test_cases: List[Dict[str, Any]] = None) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation
        
        Args:
            test_cases: Optional list of test cases. If None, uses loaded test data.
            
        Returns:
            Dataset formatted for RAGAS evaluation
        """
        if test_cases is None:
            test_cases = self.test_data["test_cases"]
        
        questions = [case["question"] for case in test_cases]
        
        # Generate responses
        responses = self.generate_responses(questions)
        
        # Prepare data for RAGAS
        evaluation_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for i, (test_case, response) in enumerate(zip(test_cases, responses)):
            evaluation_data["question"].append(test_case["question"])
            evaluation_data["answer"].append(response["answer"])
            
            # Extract context from retrieved documents
            contexts = []
            if response["context"]:
                contexts = [doc.page_content for doc in response["context"]]
            else:
                # Use expected contexts if no retrieval happened
                contexts = test_case.get("contexts", ["No context retrieved"])
            
            evaluation_data["contexts"].append(contexts)
            evaluation_data["ground_truth"].append(test_case["ground_truth"])
        
        # Create dataset
        dataset = Dataset.from_dict(evaluation_data)
        return dataset
    
    def run_evaluation(self, test_cases: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Run RAGAS evaluation
        
        Args:
            test_cases: Optional list of test cases
            
        Returns:
            Dictionary of evaluation scores
        """
        print("🧪 Starting RAGAS evaluation...")
        
        # Check if pipeline is initialized
        if not self.pipeline_manager.is_initialized:
            print("❌ Pipeline not initialized. Please load a PDF first.")
            return {}
        
        try:
            # Prepare dataset
            print("📊 Preparing evaluation dataset...")
            dataset = self.prepare_evaluation_dataset(test_cases)
            
            print(f"📝 Evaluating {len(dataset)} test cases...")
            print("Metrics being evaluated:")
            for metric in self.metrics:
                print(f"  - {metric.name}")
            
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics
            )
            
            # Convert EvaluationResult to dictionary
            scores = {}
            try:
                # Get scores from pandas DataFrame
                df = result.to_pandas()
                # Extract mean scores for each metric
                for metric in self.metrics:
                    if metric.name in df.columns:
                        scores[metric.name] = df[metric.name].mean()
                
                print(f"✅ Evaluation completed successfully!")
                print("📊 Raw scores extracted:")
                for metric, score in scores.items():
                    print(f"  {metric}: {score:.3f}")
                
            except Exception as e:
                print(f"❌ Could not extract scores: {str(e)}")
                return {}
            
            return scores
            
        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            print("This might be due to API issues or dataset format problems.")
            return {}
    
    def evaluate_multilingual_support(self) -> Dict[str, Any]:
        """
        Evaluate multilingual support capabilities
        
        Returns:
            Dictionary with multilingual evaluation results
        """
        print("🌍 Evaluating multilingual support...")
        
        multilingual_cases = self.test_data.get("languages", {}).get("multilingual_test_cases", [])
        
        if not multilingual_cases:
            print("No multilingual test cases found.")
            return {}
        
        results = {}
        
        for case in multilingual_cases:
            lang_results = {}
            
            # Test each language
            for lang in ["en", "zh", "ms"]:
                question_key = f"question_{lang}"
                if question_key in case:
                    question = case[question_key]
                    
                    print(f"Testing {lang}: {question}")
                    
                    try:
                        response = self.pipeline_manager.ask_question(question)
                        lang_results[lang] = {
                            "question": question,
                            "answer": response["answer"],
                            "relevant": response["relevant"],
                            "similarity_score": response["similarity_score"]
                        }
                    except Exception as e:
                        lang_results[lang] = {
                            "question": question,
                            "error": str(e)
                        }
            
            results[case.get("question_en", "unknown")] = lang_results
        
        return results
    
    def generate_evaluation_report(self, 
                                 ragas_scores: Dict[str, float],
                                 multilingual_results: Dict[str, Any] = None,
                                 save_to_file: bool = True) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            ragas_scores: RAGAS evaluation scores
            multilingual_results: Multilingual evaluation results
            save_to_file: Whether to save report to file
            
        Returns:
            Report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("RAG CHATBOT EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # RAGAS Scores
        if ragas_scores:
            report_lines.append("📊 RAGAS EVALUATION SCORES")
            report_lines.append("-" * 30)
            
            target_scores = self.test_data.get("evaluation_metrics", {}).get("target_scores", {})
            
            for metric, score in ragas_scores.items():
                target = target_scores.get(metric, 0.8)
                status = "✅ PASS" if score >= target else "❌ FAIL"
                report_lines.append(f"{metric.replace('_', ' ').title():<20}: {score:.3f} (Target: {target:.1f}) {status}")
            
            # Overall assessment
            avg_score = sum(ragas_scores.values()) / len(ragas_scores)
            overall_status = "✅ OVERALL PASS" if avg_score >= 0.8 else "⚠️  NEEDS IMPROVEMENT"
            report_lines.append("")
            report_lines.append(f"Average Score: {avg_score:.3f} {overall_status}")
            
        else:
            report_lines.append("❌ RAGAS evaluation failed or not completed")
        
        report_lines.append("")
        
        # Multilingual Results
        if multilingual_results:
            report_lines.append("🌍 MULTILINGUAL SUPPORT EVALUATION")
            report_lines.append("-" * 40)
            
            for question, lang_results in multilingual_results.items():
                report_lines.append(f"\nQuestion: {question}")
                
                for lang, result in lang_results.items():
                    if "error" in result:
                        report_lines.append(f"  {lang.upper()}: ❌ Error - {result['error']}")
                    else:
                        relevant_status = "✅" if result["relevant"] else "❌"
                        score = result["similarity_score"]
                        report_lines.append(f"  {lang.upper()}: {relevant_status} (Score: {score:.3f})")
        
        # Recommendations
        report_lines.append("")
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-" * 20)
        
        if ragas_scores:
            if ragas_scores.get("faithfulness", 0) < 0.8:
                report_lines.append("• Improve faithfulness: Ensure answers stick closely to provided context")
            
            if ragas_scores.get("answer_relevancy", 0) < 0.8:
                report_lines.append("• Improve answer relevancy: Make sure answers directly address the questions")
            
            if ragas_scores.get("context_precision", 0) < 0.8:
                report_lines.append("• Improve context precision: Enhance retrieval to get more relevant context")
            
            if ragas_scores.get("context_recall", 0) < 0.8:
                report_lines.append("• Improve context recall: Ensure all relevant information is retrieved")
        
        report_lines.append("• Consider adjusting chunk size and overlap for better retrieval")
        report_lines.append("• Fine-tune similarity threshold for out-of-scope detection")
        report_lines.append("• Expand test cases to cover more diverse topics")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            
            print(f"📄 Report saved to: {filename}")
        
        return report
    
    def run_comprehensive_evaluation(self) -> str:
        """
        Run comprehensive evaluation including RAGAS and multilingual testing
        
        Returns:
            Evaluation report as string
        """
        print("🚀 Starting comprehensive evaluation...")
        
        # Run RAGAS evaluation
        ragas_scores = self.run_evaluation()
        
        # Run multilingual evaluation
        multilingual_results = self.evaluate_multilingual_support()
        
        # Generate report
        report = self.generate_evaluation_report(ragas_scores, multilingual_results)
        
        print("\n" + report)
        
        return report


def main():
    """Main function to run evaluation"""
    
    # Initialize pipeline
    pipeline = RAGPipelineManager()
    
    # Try to initialize from existing database or sample PDF
    if not pipeline.initialize_from_existing_db():
        sample_pdf = "sample_knowledge_base.pdf"
        if os.path.exists(sample_pdf):
            print(f"Initializing from {sample_pdf}...")
            pipeline.initialize_from_pdf(sample_pdf)
        else:
            print("❌ No knowledge base found. Please ensure you have:")
            print("  1. An existing vector database in ./chroma_db, OR")
            print("  2. A sample PDF file named 'sample_knowledge_base.pdf'")
            return
    
    # Create evaluator
    evaluator = RAGEvaluator(pipeline)
    
    # Run comprehensive evaluation
    report = evaluator.run_comprehensive_evaluation()
    
    print("\n🎉 Evaluation completed!")


if __name__ == "__main__":
    main()