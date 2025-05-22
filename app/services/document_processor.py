"""
Document processing service for preparing text for vector embeddings
"""
import re
import uuid
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing documents for vector embeddings"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Document processor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a document into chunks with metadata
        
        Args:
            document: Document to process
            
        Returns:
            List[Dict[str, Any]]: List of processed chunks
        """
        try:
            doc_id = document.get("id")
            title = document.get("title", "Untitled")
            content = document.get("content", "")
            
            if not content:
                logger.warning(f"Empty content for document {doc_id}")
                return []
            
            # Get document metadata
            metadata = {
                "doc_id": doc_id,
                "title": title,
                "created_time": document.get("created_time"),
                "last_edited_time": document.get("last_edited_time"),
                "source": "notion"
            }
            
            # Process tags if available
            if "tags" in document and document["tags"]:
                metadata["tags"] = document["tags"]
                
            # Create chunks
            chunks = self._create_chunks(content)
            
            # Create processed chunks with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                chunk_data = {
                    "id": chunk_id,
                    "text": chunk,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                processed_chunks.append(chunk_data)
            
            logger.info(f"Processed document {doc_id} into {len(processed_chunks)} chunks")
            return processed_chunks
        
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return []
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        
        # Create chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If paragraph is too large, split it into sentences
            if paragraph_size > self.chunk_size:
                sentences = self._split_into_sentences(paragraph)
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    # If sentence fits in current chunk
                    if current_size + sentence_size <= self.chunk_size:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                    # If sentence is too large, split into smaller chunks
                    elif sentence_size > self.chunk_size:
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                            current_chunk = []
                            current_size = 0
                        
                        # Split sentence into fixed-size chunks
                        for i in range(0, len(sentence), self.chunk_size - self.chunk_overlap):
                            chunk = sentence[i:i + self.chunk_size]
                            if chunk:
                                chunks.append(chunk)
                    # Start a new chunk with this sentence
                    else:
                        if current_chunk:
                            chunks.append("\n".join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sentence_size
            # If paragraph fits in current chunk
            elif current_size + paragraph_size <= self.chunk_size:
                current_chunk.append(paragraph)
                current_size += paragraph_size
            # Start a new chunk with this paragraph
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [paragraph]
                current_size = paragraph_size
        
        # Add the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting - handle abbreviations better in production
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text