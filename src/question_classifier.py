"""
Question Classification Module
Automatically detects the type of question to route to appropriate prompt
"""
from typing import Optional
import re


class QuestionClassifier:
    """Classifies user questions to determine appropriate response format"""
    
    def __init__(self):
        """Initialize classification patterns"""
        
        # Behavioral interview patterns
        self.behavioral_patterns = [
            r"tell me about a time",
            r"describe a situation",
            r"give me an example",
            r"can you share an example",
            r"walk me through",
            r"how did you handle",
            r"what did you do when",
            r"describe when you"
        ]
        
        # Cover letter patterns
        self.cover_letter_patterns = [
            r"cover letter",
            r"why are you interested",
            r"why do you want",
            r"why should we hire",
            r"why are you a good fit",
            r"write.*paragraph.*about.*fit"
        ]
        
        # Resume bullet patterns
        self.resume_bullet_patterns = [
            r"resume bullet",
            r"create bullet",
            r"write bullet",
            r"achievement",
            r"accomplishment"
        ]
    
    def classify(self, question: str) -> Optional[str]:
        """
        Classify a question and return the appropriate purpose
        
        Args:
            question: The user's question
            
        Returns:
            One of: 'behavioral_interview', 'cover_letter', 'resume_bullet', None
        """
        question_lower = question.lower()
        
        # Check behavioral interview patterns
        for pattern in self.behavioral_patterns:
            if re.search(pattern, question_lower):
                return "behavioral_interview"
        
        # Check cover letter patterns
        for pattern in self.cover_letter_patterns:
            if re.search(pattern, question_lower):
                return "cover_letter"
        
        # Check resume bullet patterns
        for pattern in self.resume_bullet_patterns:
            if re.search(pattern, question_lower):
                return "resume_bullet"
        
        # Default: general Q&A (None)
        return None
    
    def get_recommended_tone(self, purpose: Optional[str]) -> str:
        """
        Get recommended tone based on purpose
        
        Args:
            purpose: The detected purpose
            
        Returns:
            Recommended tone string
        """
        tone_map = {
            "behavioral_interview": "professional",
            "cover_letter": "enthusiastic",
            "resume_bullet": "professional",
            None: "professional"
        }
        return tone_map.get(purpose, "professional")


# Test function
if __name__ == "__main__":
    """Test the classifier with example questions"""
    
    classifier = QuestionClassifier()
    
    test_questions = [
        "Tell me about a time you solved a difficult technical problem",
        "Write a cover letter paragraph for a Data Scientist role at Rappi",
        "What's my Python experience?",
        "Create resume bullets about my database work",
        "What is my educational background?",
        "Describe a situation where you had to work under pressure",
        "Why are you interested in this position?",
        "Give me an example of a project you're proud of"
    ]
    
    print("=" * 80)
    print("QUESTION CLASSIFIER TEST")
    print("=" * 80)
    
    for question in test_questions:
        purpose = classifier.classify(question)
        tone = classifier.get_recommended_tone(purpose)
        
        print(f"\n📝 Question: {question}")
        print(f"🎯 Detected Purpose: {purpose}")
        print(f"🎭 Recommended Tone: {tone}")