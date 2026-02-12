"""
Text splitting module for RAG system
Chunks documents into optimal sizes for embedding and retrieval
"""
import sys
from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tiktoken

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_logger, CHUNKING_CONFIG

# Create logger for this module
logger = get_logger(__name__)


class DocumentChunker:
    """Splits documents into chunks optimized for RAG retrieval"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = None
    ):
        """
        Initialize the document chunker
        
        Args:
            chunk_size: Target size of each chunk in tokens (uses config default if None)
            chunk_overlap: Number of tokens to overlap between chunks (uses config default if None)
            encoding_name: Tiktoken encoding to use for token counting (uses config default if None)
        """
        # Use config defaults if not provided
        self.chunk_size = chunk_size if chunk_size is not None else CHUNKING_CONFIG['chunk_size']
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNKING_CONFIG['chunk_overlap']
        encoding = encoding_name if encoding_name is not None else CHUNKING_CONFIG['encoding_name']
        
        # Initialize token encoder
        self.encoding = tiktoken.get_encoding(encoding)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentences
                ", ",    # Clauses
                " ",     # Words
                ""       # Characters (last resort)
            ]
        )
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents with preserved metadata
        """
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        logger.info(f"Chunk size: {self.chunk_size} tokens, Overlap: {self.chunk_overlap} tokens")
        
        # Split all documents
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = self._count_tokens(chunk.page_content)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {}
        
        token_counts = [self._count_tokens(chunk.page_content) for chunk in chunks]
        
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(token_counts) / len(token_counts),
            "min_chunk_size": min(token_counts),
            "max_chunk_size": max(token_counts),
            "total_tokens": sum(token_counts),
            "chunks_by_category": {}
        }
        
        # Count by category
        for chunk in chunks:
            category = chunk.metadata.get("category", "unknown")
            stats["chunks_by_category"][category] = \
                stats["chunks_by_category"].get(category, 0) + 1
        
        return stats


# Test function
if __name__ == "__main__":
    """Test the chunker with loaded documents"""
    from data_loader import DocumentLoader
    
    print("=" * 80)
    print("TESTING DOCUMENT CHUNKER")
    print("=" * 80)
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    
    # Initialize chunker
    chunker = DocumentChunker(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Split documents
    chunks = chunker.split_documents(documents)
    
    # Get statistics
    stats = chunker.get_chunk_stats(chunks)
    
    print("\n" + "=" * 80)
    print("CHUNK STATISTICS")
    print("=" * 80)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average chunk size: {stats['avg_chunk_size']:.1f} tokens")
    print(f"Min chunk size: {stats['min_chunk_size']} tokens")
    print(f"Max chunk size: {stats['max_chunk_size']} tokens")
    
    print(f"\nChunks by category:")
    for category, count in stats['chunks_by_category'].items():
        print(f"  • {category}: {count} chunks")
    
    # Show sample chunks
    print("\n" + "=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)
    
    # Show first chunk
    print("\n[CHUNK 1]")
    print(f"Source: {chunks[0].metadata.get('source_file')}")
    print(f"Category: {chunks[0].metadata.get('category')}")
    print(f"Size: {chunks[0].metadata.get('chunk_size')} tokens")
    print(f"Content:\n{chunks[0].page_content[:300]}...")
    
    # Show a chunk from a different category
    for chunk in chunks[10:20]:  # Look in middle
        if chunk.metadata.get('category') != chunks[0].metadata.get('category'):
            print(f"\n[SAMPLE CHUNK - {chunk.metadata.get('category').upper()}]")
            print(f"Source: {chunk.metadata.get('source_file')}")
            print(f"Size: {chunk.metadata.get('chunk_size')} tokens")
            print(f"Content:\n{chunk.page_content[:300]}...")
            break