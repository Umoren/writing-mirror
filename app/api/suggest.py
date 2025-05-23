from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from ..services.embedding_service import EmbeddingService
from ..services.vector_service import VectorService
from ..services.integration_service import IntegrationService
from ..services.llm_service import LLMService


from ..models.api_models import (
    SuggestRequest, 
    SuggestResponse, 
    Suggestion, 
    Source, 
    PerformanceStats,
    HealthResponse,
    ErrorResponse
)


# Load environment variables
load_dotenv()


logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
vector_service = VectorService()
integration_service = IntegrationService()
llm_service = LLMService()


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(request: SuggestRequest) -> SuggestResponse:
    """Generate writing suggestions based on user input"""
    start_time = time.time()
    trace_id = f"suggest_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{trace_id}] Processing suggestion request: {request.text[:50]}...")
        
        # Step 1: Generate embedding for the query
        embedding_start = time.time()
        query_embedding = embedding_service.embed_text(request.text)
        embedding_time_ms = int((time.time() - embedding_start) * 1000)
        
        # Step 2: Search for similar content
        search_start = time.time()
        search_results = vector_service.search(
            query_vector=query_embedding,
            top_k=min(10, request.num_suggestions * 3),  # Get more results for better context
            score_threshold=0.2
        )
        search_time_ms = int((time.time() - search_start) * 1000)
        
        # Step 3: Extract text chunks for LLM context
        retrieved_chunks = []
        sources = []
        
        for result in search_results:
            payload = result.get('payload', {})
            chunk_text = payload.get('text', '')
            if chunk_text:
                retrieved_chunks.append(chunk_text)
                
                sources.append(Source(
                    doc_id=payload.get('doc_id', 'unknown'),
                    title=payload.get('doc_title', 'Unknown Document'),
                    similarity=result.get('score', 0),
                    chunk_text=chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                ))
        
        # Step 4: Generate suggestions using LLM
        generation_start = time.time()
        llm_suggestions = await llm_service.generate_suggestions(
            user_text=request.text,
            context=request.context or "General writing",
            retrieved_chunks=retrieved_chunks,
            task=request.task,
            num_suggestions=request.num_suggestions,
            max_tokens=request.max_length
        )
        generation_time_ms = int((time.time() - generation_start) * 1000)
        
        # Step 5: Convert LLM suggestions to API format
        suggestions = []
        for llm_suggestion in llm_suggestions:
            suggestions.append(Suggestion(
                text=llm_suggestion['text'],
                score=llm_suggestion['score'],
                reasoning=llm_suggestion['reasoning']
            ))
        
        # Step 6: Compile performance stats
        total_time_ms = int((time.time() - start_time) * 1000)
        stats = PerformanceStats(
            total_time_ms=total_time_ms,
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_time_ms,
            generation_time_ms=generation_time_ms,
            chunks_retrieved=len(search_results),
            chunks_processed=len(suggestions)
        )
        
        logger.info(f"[{trace_id}] Generated {len(suggestions)} suggestions in {total_time_ms}ms")
        
        return SuggestResponse(
            trace_id=trace_id,
            suggestions=suggestions,
            sources=sources[:len(suggestions)],  # Match sources to suggestions
            stats=stats
        )
        
    except Exception as e:
        logger.error(f"[{trace_id}] Error generating suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=f"Failed to generate suggestions: {str(e)}",
                trace_id=trace_id
            ).dict()
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    try:
        # Check vector service
        vector_status = vector_service.get_collection_info()
        vector_healthy = vector_status.get('status') == 'green'
        
        # Check embedding service
        test_embedding = embedding_service.embed_text("test")
        embedding_healthy = len(test_embedding) > 0
        
        # Check LLM service (simple test)
        llm_healthy = bool(llm_service.api_key and llm_service.model_name)
        
        # Overall status
        overall_status = "healthy" if vector_healthy and embedding_healthy and llm_healthy else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            services={
                "vector_database": {
                    "status": "healthy" if vector_healthy else "unhealthy",
                    "details": vector_status
                },
                "embedding_service": {
                    "status": "healthy" if embedding_healthy else "unhealthy",
                    "model": os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
                },
                "llm_service": {
                    "status": "healthy" if llm_healthy else "unhealthy",
                    "model": os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
                    "provider": "Together AI"
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            services={
                "error": str(e)
            }
        )