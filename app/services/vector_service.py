"""
Vector database service using Qdrant
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

def init_vector_db(url: str, collection_name: str, vector_size: int = 384) -> QdrantClient:
    """
    Initialize the Qdrant vector database client and ensure collection exists
    
    Args:
        url: Qdrant server URL
        collection_name: Name of the collection to use
        vector_size: Dimension of vectors (default: 384 for all-MiniLM-L6-v2)
        
    Returns:
        QdrantClient: Initialized Qdrant client
    """
    try:
        # Initialize Qdrant client
        client = QdrantClient(url=url)
        logger.info(f"Connected to Qdrant at {url}")
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' not found, creating...")
            
            # Create collection with the specified parameters
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection '{collection_name}'")
            
            # Create payload index for faster filtering
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source_id",
                field_schema="keyword"
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="source_type",
                field_schema="keyword"
            )
            logger.info(f"Created payload indices for collection '{collection_name}'")
        else:
            logger.info(f"Collection '{collection_name}' already exists")
        
        return client
    
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant: {e}")
        raise


class VectorService:
    """Service for managing vector operations with Qdrant"""
    
    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Initialize the vector service
        
        Args:
            client: Qdrant client
            collection_name: Name of the collection to use
        """
        self.client = client
        self.collection_name = collection_name
        logger.info(f"Vector service initialized for collection: {collection_name}")
    
    async def store_vectors(self, 
                     vectors: List[List[float]], 
                     ids: List[str],
                     payloads: List[Dict[str, Any]]) -> bool:
        """
        Store vectors in the vector database
        
        Args:
            vectors: List of embedding vectors
            ids: List of unique IDs for each vector
            payloads: List of metadata payloads for each vector
            
        Returns:
            bool: Success status
        """
        if not vectors or not ids or not payloads:
            logger.warning("Attempted to store empty vectors")
            return False
        
        if len(vectors) != len(ids) or len(vectors) != len(payloads):
            logger.error("Mismatch in lengths of vectors, ids, and payloads")
            raise ValueError("vectors, ids, and payloads must have the same length")
        
        try:
            # Create point objects
            points = [
                PointStruct(
                    id=id_,
                    vector=vector,
                    payload=payload
                )
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Upsert points (insert or update)
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(vectors)} vectors in collection '{self.collection_name}'")
            return True
        
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise
    
    async def search_similar(self, 
                      query_vector: List[float], 
                      top_k: int = 5,
                      filter_conditions: Optional[Dict[str, Any]] = None,
                      score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional filter conditions for the search
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List[Dict[str, Any]]: List of search results with payloads and scores
        """
        try:
            # Build filter if provided
            search_filter = None
            if filter_conditions:
                conditions = []
                for field, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(
                    must=conditions
                )
            
            # Search for similar vectors
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            logger.info(f"Found {len(results)} similar vectors in collection '{self.collection_name}'")
            return results
        
        except Exception as e:
            logger.error(f"Error searching for similar vectors: {e}")
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Dict[str, Any]: Collection information
        """
        try:
            # Get collection info
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            # Format info - adapted to match actual Qdrant response structure
            info = {
                "name": self.collection_name,  # Use the name we already know
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }

            logger.info(f"Retrieved information for collection '{self.collection_name}'")
            return info
        
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise