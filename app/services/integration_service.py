"""
Integration service to sync data from Notion to Qdrant
"""
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.services.notion_service import NotionService
from app.services.vector_service import VectorService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class IntegrationService:
    """Service for integrating Notion and Qdrant"""
    
    def __init__(
        self, 
        notion_service: NotionService, 
        vector_service: VectorService, 
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        state_file_path: str = "data/sync_state.json"
    ):
        """
        Initialize the integration service
        
        Args:
            notion_service: Notion service
            vector_service: Vector service
            document_processor: Document processor
            embedding_service: Embedding service
            state_file_path: Path to state file for tracking sync state
        """
        self.notion_service = notion_service
        self.vector_service = vector_service
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.state_file_path = state_file_path
        
        # Create directory for state file if it doesn't exist
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True)
        
        logger.info("Integration service initialized")
    
    async def sync_documents(self, force_full_sync: bool = False) -> Dict[str, Any]:
        """
        Sync documents from Notion to Qdrant
        
        Args:
            force_full_sync: Force a full sync
            
        Returns:
            Dict[str, Any]: Sync stats
        """
        try:
            start_time = datetime.now()
            
            # Get last sync timestamp
            last_sync_time = None
            if not force_full_sync:
                last_sync_time = self._get_last_sync_time()
            
            # Process documents
            logger.info(f"Starting document sync. Last sync time: {last_sync_time}")
            documents = await self.notion_service.process_all_documents(last_sync_time=last_sync_time)
            
            # Process chunks
            total_chunks = 0
            for document in documents:
                # Process document
                chunks = self.document_processor.process_document(document)
                total_chunks += len(chunks)
                
                # Store chunks
                if chunks:
                    texts = [chunk["text"] for chunk in chunks]
                    ids = [chunk["id"] for chunk in chunks]
                    
                    # Create payloads for each chunk with metadata and text
                    payloads = []
                    for chunk in chunks:
                        # Include both metadata and the actual text in the payload
                        payload = {
                            **chunk["metadata"],
                            "text": chunk["text"]  # Include the text in payload for retrieval
                        }
                        payloads.append(payload)

                    # Convert texts to vectors
                    vectors = self.embedding_service.generate_embeddings(texts)

                    # Store in vector database
                    await self.vector_service.store_vectors(
                        vectors=vectors,
                        ids=ids,
                        payloads=payloads
                    )
            
            # Update sync state
            sync_time = datetime.now().isoformat()
            self._update_last_sync_time(sync_time)
            
            # Get stats
            collection_info = await self.vector_service.get_collection_info()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            stats = {
                "documents_processed": len(documents),
                "chunks_processed": total_chunks,
                "vectors_count": collection_info.get("vectors_count"),
                "points_count": collection_info.get("points_count"),
                "sync_time": sync_time,
                "duration_seconds": duration
            }
            
            logger.info(f"Sync completed: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error syncing documents: {e}")
            raise
    
    async def search_similar_texts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar texts
        
        Args:
            query: Query text
            limit: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: List of similar texts
        """
        try:
            # Convert query to vector
            query_vector = self.embedding_service.generate_embedding(query)

            # Search
            results = await self.vector_service.search_similar(
                query_vector=query_vector,
                top_k=limit
            )
            
            # Process results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result["id"],
                    "score": result["score"],
                    "text": result["payload"].get("text", ""),  # Get text from payload
                    "metadata": {
                        "title": result["payload"].get("title", ""),
                        "doc_id": result["payload"].get("doc_id", ""),
                        "tags": result["payload"].get("tags", [])
                    }
                })

            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching similar texts: {e}")
            raise
    
    def _get_last_sync_time(self) -> Optional[str]:
        """
        Get the last sync time
        
        Returns:
            Optional[str]: Last sync time
        """
        try:
            if os.path.exists(self.state_file_path):
                with open(self.state_file_path, "r") as f:
                    state = json.load(f)
                    return state.get("last_sync_time")
            return None
        
        except Exception as e:
            logger.error(f"Error getting last sync time: {e}")
            return None
    
    def _update_last_sync_time(self, sync_time: str) -> None:
        """
        Update the last sync time
        
        Args:
            sync_time: Sync time
        """
        try:
            state = {"last_sync_time": sync_time}
            with open(self.state_file_path, "w") as f:
                json.dump(state, f)
        
        except Exception as e:
            logger.error(f"Error updating last sync time: {e}")