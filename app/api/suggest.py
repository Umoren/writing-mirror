from fastapi import APIRouter, HTTPException
from typing import List
import time
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ..models.api_models import (
    SuggestRequest, 
    SuggestResponse, 
    Suggestion, 
    PerformanceStats,
    HealthResponse,
    ErrorResponse
)
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize only the LLM service for now
llm_service = LLMService()


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(request: SuggestRequest) -> SuggestResponse:
    """Generate writing suggestions based on user input - SIMPLIFIED VERSION"""
    start_time = time.time()
    trace_id = f"suggest_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{trace_id}] Processing suggestion request: {request.text[:50]}...")
        
        # For now, we'll generate suggestions without vector search
        # Using dummy retrieved chunks until vector service is set up
        retrieved_chunks = [
            "This is a sample writing piece that shows good style.",
            "Another example of clear, engaging writing.",
            "Professional writing should be concise and clear."
        ]
        
        # Generate suggestions using LLM
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
        
        # Convert LLM suggestions to API format
        suggestions = []
        for llm_suggestion in llm_suggestions:
            suggestions.append(Suggestion(
                text=llm_suggestion['text'],
                score=llm_suggestion['score'],
                reasoning=llm_suggestion['reasoning']
            ))
        
        # Compile performance stats
        total_time_ms = int((time.time() - start_time) * 1000)
        stats = PerformanceStats(
            total_time_ms=total_time_ms,
            embedding_time_ms=0,  # No embedding for now
            search_time_ms=0,     # No search for now
            generation_time_ms=generation_time_ms,
            chunks_retrieved=len(retrieved_chunks),
            chunks_processed=len(suggestions)
        )
        
        logger.info(f"[{trace_id}] Generated {len(suggestions)} suggestions in {total_time_ms}ms")
        
        return SuggestResponse(
            trace_id=trace_id,
            suggestions=suggestions,
            sources=[],  # No sources for now
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
    """Health check endpoint - SIMPLIFIED VERSION"""
    try:
        # Check LLM service (simple test)
        llm_healthy = bool(llm_service.api_key and llm_service.model_name)
        
        # Overall status
        overall_status = "healthy" if llm_healthy else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            services={
                "llm_service": {
                    "status": "healthy" if llm_healthy else "unhealthy",
                    "model": os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
                    "provider": "Together AI"
                },
                "vector_database": {
                    "status": "disabled",
                    "details": "Vector service temporarily disabled for initial setup"
                },
                "embedding_service": {
                    "status": "disabled",
                    "details": "Embedding service temporarily disabled for initial setup"
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