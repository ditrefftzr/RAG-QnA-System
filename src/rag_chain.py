"""
RAG Chain - Combines retrieval and generation for question answering
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from google import genai
from langchain_core.documents import Document
from datetime import datetime 

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_logger, MODEL_CONFIG, RETRIEVAL_CONFIG, GENERATION_CONFIG

from src.vector_store import VectorStoreManager
from src.question_classifier import QuestionClassifier

# Load environment
load_dotenv()

# Create logger for this module
logger = get_logger(__name__)


class RAGChain:
    def __init__(self, vectorstore_manager: VectorStoreManager):
        """Initialize RAG chain"""
        self.vectorstore = vectorstore_manager
        
        # Initialize Gemini with new API
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = MODEL_CONFIG['generation_model']
        
        # Initialize question classifier
        self.classifier = QuestionClassifier()
        
        logger.info(f"RAG Chain initialized with model: {self.model_name}")

    def create_prompt(
        self,
        question: str,
        context_chunks: List[Document],
        purpose: Optional[str] = None,
        tone: str = "professional"
    ) -> str:
        """
        Create a well-structured prompt for the LLM
        
        Args:
            question: User's question
            context_chunks: Retrieved relevant chunks
            purpose: Optional purpose (cover_letter, resume_bullet, behavioral_interview)
            tone: Desired tone (professional, enthusiastic, technical)
            
        Returns:
            Formatted prompt string
        """
        # STEP 0: Get current date dynamically
        current_date = datetime.now().strftime("%B %d, %Y")

        # STEP 1: Extract text from chunks with source attribution
        context_text = "\n\n---\n\n".join([
            f"Source: {chunk.metadata.get('source_file', 'unknown')}\n{chunk.page_content}"
            for chunk in context_chunks
        ])
        
        # STEP 2: Tone-specific instructions
        tone_instructions = {
            "professional": "Maintain a professional, factual tone.",
            "enthusiastic": "Show enthusiasm and passion for the work.",
            "technical": "Focus on technical details, tools, and methodologies."
        }
        tone_instruction = tone_instructions.get(tone, tone_instructions["professional"])
        
        # STEP 3: Purpose-specific output format instructions
        purpose_instructions = {
            "cover_letter": """
OUTPUT FORMAT: Write 2-3 paragraphs in FIRST PERSON ("I") showing enthusiasm and fit.
- Start with why you're excited about this opportunity
- Connect David's specific experience to the role requirements  
- Show passion and alignment
- End with forward-looking statement
Example: "I'm excited to apply because my experience building NLP systems..."
""",
            "resume_bullet": """
OUTPUT FORMAT: Write 1-2 concise bullet points (max 2 sentences each).
- Start with strong action verb (Developed, Built, Engineered, Led, Achieved)
- Include specific technologies/tools
- ALWAYS include quantifiable metrics/results
- Use past tense, third person
Example: "• Engineered ML pipeline achieving 85% accuracy on 100K+ documents"
""",
            "behavioral_interview": """
OUTPUT FORMAT: Structure answer in STAR format (Situation-Task-Action-Result).
- Situation: Brief context (1-2 sentences)
- Task: Your specific responsibility/challenge
- Action: What YOU did (specific, use "I" statements)
- Result: Quantifiable outcome, what you learned
Keep it story-like but concise (2-3 paragraphs total).
""",
            "general": """
OUTPUT FORMAT: Provide a clear, comprehensive answer in 2-3 paragraphs.
Include specific projects, technologies, and achievements with metrics.
"""
        }
        purpose_instruction = purpose_instructions.get(purpose, purpose_instructions["general"])
        
        # STEP 4: Build the complete prompt
        prompt = f"""You are a professional career assistant helping David Trefftz answer questions about his background for job applications and career development.

    CURRENT DATE: {current_date}

    TEMPORAL REASONING:
    When describing activities or credentials with date ranges:
    - If the end date has already passed, use past tense (graduated, worked, completed, achieved)
    - If marked as "Present" or the end date is in the future, use present tense (working, studying, developing)

    INSTRUCTIONS:
    - Answer the question based ONLY on the information provided in the context below
    - Be specific and cite concrete examples, projects, and achievements
    - Include metrics, numbers, and results whenever available
    - {tone_instruction}
    - If the context doesn't contain enough information, acknowledge this clearly

    {purpose_instruction}

    GUARDRAILS:
    - Do not make up or infer information not explicitly stated in the context
    - Do not share sensitive personal information (phone numbers, personal addresses)
    - Keep answers relevant to professional/career contexts
    - If the context does not contain information to answer the question, respond ONLY with:
    "I don't have information about that in my documents."
    Do NOT describe what is in the context - just state you cannot answer the question.

    CONTEXT (David's Background):
    {context_text}

    {f"ADDITIONAL CONTEXT: This answer will be used for {purpose}" if purpose else ""}

    QUESTION: {question}

    ANSWER:"""
        
        return prompt
    


    def generate_answer(
        self,
        question: str,
        purpose: Optional[str] = None,
        tone: Optional[str] = None,
        k: int = None
    ) -> Dict[str, any]:
        """
        Generate an answer using RAG (Retrieval + Generation)
        
        Args:
            question: User's question
            purpose: Optional purpose - if None, will auto-detect
            tone: Optional tone - if None, will auto-detect based on purpose
            k: Number of context chunks to retrieve (uses default from config if None)
            
        Returns:
            Dictionary with 'answer', 'sources', 'detected_purpose', 'detected_tone'
        """
        # Use default k from config if not provided
        if k is None:
            k = RETRIEVAL_CONFIG['default_k']
        
        # AUTO-DETECT purpose if not provided
        if purpose is None:
            purpose = self.classifier.classify(question)
            detected_purpose = True
        else:
            detected_purpose = False
        
        # AUTO-DETECT tone if not provided
        if tone is None:
            tone = self.classifier.get_recommended_tone(purpose)
            detected_tone = True
        else:
            detected_tone = False
        
        # Show what was detected
        if detected_purpose or detected_tone:
            logger.debug(f"Auto-detected: purpose={purpose}, tone={tone}")
        
        # STEP 1: RETRIEVE relevant chunks from vector database
        logger.info(f"Retrieving top {k} relevant chunks...")
        context_chunks = self.vectorstore.similarity_search(question, k=k)
        logger.info(f"Retrieved {len(context_chunks)} chunks")
        
        # Show what was retrieved (debug level)
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.metadata.get('source_file', 'unknown')
            logger.debug(f"Chunk {i}: {source}")
        
        # STEP 2: CREATE the prompt using our method
        logger.debug(f"Creating prompt (purpose={purpose}, tone={tone})...")
        prompt = self.create_prompt(
            question=question,
            context_chunks=context_chunks,
            purpose=purpose,
            tone=tone
        )
        logger.debug("Prompt created")
        
        # STEP 3: GENERATE answer using Gemini
        logger.info(f"Generating answer with {self.model_name}...")
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'temperature': GENERATION_CONFIG['temperature'],
                    'max_output_tokens': GENERATION_CONFIG['max_tokens'],
                    'top_p': GENERATION_CONFIG['top_p'],
                }
            )
            answer = response.text
            logger.info("Answer generated successfully")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = f"Error: Could not generate answer. {str(e)}"
        
        # STEP 4: RETURN structured result
        return {
            "answer": answer,
            "sources": [chunk.metadata.get('source_file', 'unknown') 
                    for chunk in context_chunks],
            "detected_purpose": purpose if detected_purpose else None,
            "detected_tone": tone if detected_tone else None,
            "prompt": prompt  # Include prompt for debugging if needed
        }

if __name__ == "__main__":
    """Test the RAG chain with actual generation"""
    from vector_store import VectorStoreManager
    
    print("=" * 80)
    print("TESTING RAG CHAIN - COMPLETE PIPELINE")
    print("=" * 80)
    
    # Initialize RAG chain
    print("\n[Initializing RAG Chain]")
    vs_manager = VectorStoreManager()
    vs_manager.load_vectorstore()
    rag = RAGChain(vs_manager)
    
    # Test 1: General Q&A
    print("\n" + "=" * 80)
    print("TEST 1: GENERAL Q&A")
    print("=" * 80)
    
    result = rag.generate_answer(
        question="What's my NLP experience?",
        purpose=None,
        tone="professional",
        k=3
    )
    
    print("\n📄 ANSWER:")
    print("-" * 80)
    print(result["answer"])
    print("\n📚 SOURCES:")
    print("-" * 80)
    for source in result["sources"]:
        print(f"  • {source}")
    
    # Test 2: Cover Letter
    print("\n" + "=" * 80)
    print("TEST 2: COVER LETTER PARAGRAPH")
    print("=" * 80)
    
    result = rag.generate_answer(
        question="Write a cover letter paragraph about why I'm a good fit for a Data Scientist role at a Colombian fintech company",
        purpose="cover_letter",
        tone="enthusiastic",
        k=5
    )
    
    print("\n📄 ANSWER:")
    print("-" * 80)
    print(result["answer"])
    print("\n📚 SOURCES:")
    print("-" * 80)
    for source in result["sources"]:
        print(f"  • {source}")
    
    # Test 3: Resume Bullet
    print("\n" + "=" * 80)
    print("TEST 3: RESUME BULLETS")
    print("=" * 80)
    
    result = rag.generate_answer(
        question="Create resume bullets about my database and data engineering work",
        purpose="resume_bullet",
        tone="professional",
        k=3
    )
    
    print("\n📄 ANSWER:")
    print("-" * 80)
    print(result["answer"])
    print("\n📚 SOURCES:")
    print("-" * 80)
    for source in result["sources"]:
        print(f"  • {source}")