"""
Configuration module for RAG Job Application Optimizer

Centralizes all configuration settings for models, retrieval, and paths.
Dynamic functions avoid hardcoding values that can be determined programmatically.

Usage:
    from config import get_logger, MODEL_CONFIG, RETRIEVAL_CONFIG
    
    logger = get_logger(__name__)
    logger.info(f"Using model: {MODEL_CONFIG['generation_model']}")
"""
import logging
import sys
from typing import Dict

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the entire application
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        
    Returns:
        Configured root logger
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter with timestamp, module name, level, and message
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Console handler (prints to terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Optional: File handler (saves to file)
    # Uncomment the lines below to also save logs to a file
    # file_handler = logging.FileHandler('rag_system.log')
    # file_handler.setLevel(numeric_level)
    # file_handler.setFormatter(formatter)
    # root_logger.addHandler(file_handler)
    
    return root_logger


# Initialize logging (default to INFO level)
setup_logging(level="INFO")


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG: Dict[str, str] = {
    # Generation model (for answering questions, writing cover letters)
    "generation_model": "gemma-3-27b-it",
    
    # Embedding model (for converting text to vectors)
    "embedding_model": "gemini-embedding-001",
}


# =============================================================================
# RETRIEVAL CONFIGURATION  
# =============================================================================

RETRIEVAL_CONFIG: Dict[str, any] = {
    # Default number of documents to retrieve
    "default_k": 5,
    
    # ChromaDB settings
    "collection_name": "job_application_docs",
    "persist_directory": "./chroma_db",
}


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

CHUNKING_CONFIG: Dict[str, any] = {
    # Chunk size for document splitting
    "chunk_size": 500,
    "chunk_overlap": 50,
    "encoding_name": "cl100k_base",
}


# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

GENERATION_CONFIG: Dict[str, any] = {
    # Temperature (0.0 = deterministic, 1.0 = creative)
    "temperature": 0.7,
    
    # Maximum tokens to generate
    "max_tokens": 1000,
    
    # Top-p sampling (nucleus sampling)
    "top_p": 0.9,
}


# =============================================================================
# FILE PATHS
# =============================================================================

PATHS: Dict[str, str] = {
    "data_directory": "./data/raw",
    "output_directory": "./outputs",
    "cover_letters_directory": "./outputs/cover_letters",
}


# =============================================================================
# DYNAMIC HELPER FUNCTIONS
# =============================================================================


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        Logger instance for that module
        
    Example:
        logger = get_logger(__name__)
        logger.info("Module initialized")
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """
    Change logging level at runtime
    
    Args:
        level: "DEBUG", "INFO", "WARNING", or "ERROR"
        
    Example:
        set_log_level("DEBUG")   # See everything during development
        set_log_level("WARNING") # Only see problems in production
    """
    setup_logging(level=level)


# =============================================================================
# EXPORTS
# =============================================================================

# Create a default logger that modules can import
logger = get_logger(__name__)

__all__ = [
    'logger',
    'get_logger', 
    'set_log_level',
    'MODEL_CONFIG',
    'RETRIEVAL_CONFIG',
    'CHUNKING_CONFIG',
    'GENERATION_CONFIG',
    'PATHS',
]