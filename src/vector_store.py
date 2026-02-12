"""
Vector store module for RAG system
Creates embeddings and stores them in ChromaDB for semantic search
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_logger, MODEL_CONFIG, RETRIEVAL_CONFIG

# Load environment variables
load_dotenv()

# Create logger for this module
logger = get_logger(__name__)


class VectorStoreManager:
    """Manages document embeddings and ChromaDB vector store"""
    
    def __init__(self):
        """
        Initialize the vector store manager
        Uses collection_name and persist_directory from RETRIEVAL_CONFIG
        """
        self.collection_name = RETRIEVAL_CONFIG['collection_name']
        self.persist_directory = RETRIEVAL_CONFIG['persist_directory']
        
        # Initialize Google embeddings
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        logger.info("Initializing embedding model...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=MODEL_CONFIG['embedding_model'],
            google_api_key=api_key
        )
        logger.info(f"Embedding model ready: {MODEL_CONFIG['embedding_model']}")
        
        self.vectorstore = None
        
    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """
        Create vector store from document chunks
        
        Args:
            chunks: List of document chunks to embed and store
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store with {len(chunks)} chunks...")
        logger.info(f"This will create {len(chunks)} embeddings")
        logger.info(f"Estimated time: ~{len(chunks) * 0.5:.0f} seconds")
        logger.info("Please wait...")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        logger.info("Vector store created!")
        logger.info(f"Persisted to: {self.persist_directory}")
        
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        """
        Load existing vector store from disk
        
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Loading existing vector store from: {self.persist_directory}")
        
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Get collection stats
        collection = self.vectorstore._collection
        count = collection.count()
        
        logger.info(f"Loaded vector store with {count} embeddings")
        
        return self.vectorstore

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filter (e.g., {"category": "cv"})
            
        Returns:
            List of most similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call create_vectorstore() or load_vectorstore() first")
        
        # Perform similarity search
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        Search with similarity scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results

    def get_collection_stats(self) -> dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with statistics
        """
        if not self.vectorstore:
            return {"error": "Vector store not initialized"}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        # Sample some documents to get categories
        sample_results = self.vectorstore.similarity_search("sample", k=min(10, count))
        categories = set()
        for doc in sample_results:
            if "category" in doc.metadata:
                categories.add(doc.metadata["category"])
        
        return {
            "total_embeddings": count,
            "collection_name": self.collection_name,
            "categories_found": list(categories)
        }


# Test function
if __name__ == "__main__":
    """Test the vector store with chunked documents"""
    from data_loader import DocumentLoader
    from text_splitter import DocumentChunker
    
    print("=" * 80)
    print("TESTING VECTOR STORE")
    print("=" * 80)
    
    # Load and chunk documents
    print("\n[Step 1: Load Documents]")
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    
    print("\n[Step 2: Chunk Documents]")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split_documents(documents)
    
    # Create vector store
    print("\n[Step 3: Create Vector Store]")
    vs_manager = VectorStoreManager()
    vectorstore = vs_manager.create_vectorstore(chunks)
    
    # Get stats
    print("\n" + "=" * 80)
    print("VECTOR STORE STATISTICS")
    print("=" * 80)
    stats = vs_manager.get_collection_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test similarity search
    print("\n" + "=" * 80)
    print("TESTING SIMILARITY SEARCH")
    print("=" * 80)
    
    test_queries = [
        "What programming languages does David know?",
        "What is David's NLP experience?",
        "Tell me about David's education",
        "What databases has David worked with?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = vs_manager.similarity_search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Source: {doc.metadata.get('source_file', 'unknown')}")
            print(f"   Category: {doc.metadata.get('category', 'unknown')}")
            print(f"   Content preview: {doc.page_content[:150]}...")
    
    print("\n" + "=" * 80)
    print("✅ Vector store test complete!")
    print("=" * 80)