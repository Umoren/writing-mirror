"""
Enhanced Suggest API with hybrid search and context-aware suggestions
"""
import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from ..services.vector_service import VectorService, init_vector_db
from ..services.embedding_service import EmbeddingService
from ..services.enhanced_text_processor import AdvancedTextProcessor, SemanticChunker
from ..services.hybrid_search_engine import HybridSearchEngine, ContextAwareScoring
from ..services.context_suggestion_engine import ContextSuggestionEngine

# Setup logging
logger = logging.getLogger(__name__)

# Initialize services
qdrant_client = init_vector_db()
vector_service = VectorService(client=qdrant_client)
embedding_service = EmbeddingService()
hybrid_search = HybridSearchEngine(vector_service, embedding_service)
suggestion_engine = ContextSuggestionEngine()
context_scoring = ContextAwareScoring()

router = APIRouter()

class SuggestRequest(BaseModel):
    text: str
    context: str = ""
    suggestion_type: str = "continuation"  # continuation, completion, enhancement
    source_filter: Optional[str] = None  # gmail, notion, or None for all
    content_type_hint: Optional[str] = None  # job_related, technical, personal, etc.
    max_results: int = 3

class SuggestResponse(BaseModel):
    trace_id: str
    suggestions: List[dict]
    sources: List[dict]
    stats: dict
    timestamp: str
    search_strategy: dict

@router.get("/health")
async def health_check():
    """Enhanced health check with service status"""
    try:
        # Test vector database connection
        collection_info = await vector_service.get_collection_info()

        # Test embedding service
        test_embedding = embedding_service.generate_embedding("test")

        return {
            "status": "healthy",
            "services": {
                "vector_database": {
                    "status": "healthy",
                    "vectors_count": collection_info.get("points_count", 0)
                },
                "embedding_service": {
                    "status": "healthy",
                    "vector_dimension": len(test_embedding)
                },
                "hybrid_search": {
                    "status": "healthy"
                },
                "suggestion_engine": {
                    "status": "healthy"
                }
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

@router.post("/suggest", response_model=SuggestResponse)
async def suggest_text(request: SuggestRequest):
    """Enhanced suggest endpoint with hybrid search and context-aware suggestions"""

    # Generate trace ID for debugging
    trace_id = f"suggest_{int(time.time() * 1000)}"
    start_time = time.time()
    
    try:
        logger.info(f"[{trace_id}] Processing suggestion request")
        
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Step 1: Hybrid Search with multiple signals
        search_start = time.time()

        search_results = await hybrid_search.search(
            query=request.text,
            context=request.context,
            top_k=min(request.max_results * 3, 15),  # Get more for better variety
            source_filter=request.source_filter,
            content_type_hint=request.content_type_hint
        )

        search_time = (time.time() - search_start) * 1000

        # Step 2: Apply context-aware scoring adjustments
        if search_results:
            for result in search_results:
                # Adjust for query specificity
                result.final_score = context_scoring.adjust_for_query_specificity(
                    result.final_score, request.text
                )

                # Boost exact matches
                result.final_score = context_scoring.boost_exact_matches(
                    result.final_score, request.text, result.content
                )

            # Re-sort after adjustments
            search_results.sort(key=lambda x: x.final_score, reverse=True)

            # Apply source diversification
            search_results = context_scoring.diversify_sources(search_results)

        # Step 3: Generate context-aware suggestions
        suggestion_start = time.time()

        suggestions = suggestion_engine.generate_suggestions(
            current_text=request.text,
            context=request.context,
            search_results=search_results,
            suggestion_type=request.suggestion_type
        )
        
        suggestion_time = (time.time() - suggestion_start) * 1000
        
        # Step 4: Format response
        formatted_suggestions = [
            {
                "text": suggestion.text,
                "score": suggestion.confidence,
                "reasoning": suggestion.reasoning,
                "type": suggestion.suggestion_type,
                "source_context": suggestion.source_context
            }
            for suggestion in suggestions[:request.max_results]
        ]
        
        formatted_sources = [
            {
                "doc_id": result.doc_id,
                "title": result.title,
                "similarity": result.vector_score,
                "final_score": result.final_score,
                "source": result.source,
                "ranking_factors": result.ranking_factors,
                "chunk_text": result.content[:200] + "..." if len(result.content) > 200 else result.content
            }
            for result in search_results[:5]  # Show top 5 sources
        ]
        
        total_time = (time.time() - start_time) * 1000

        # Step 5: Build comprehensive stats
        stats = {
            "total_time_ms": round(total_time, 2),
            "search_time_ms": round(search_time, 2),
            "suggestion_time_ms": round(suggestion_time, 2),
            "chunks_retrieved": len(search_results),
            "suggestions_generated": len(suggestions),
            "sources_analyzed": len(set(r.source for r in search_results)),
            "avg_similarity_score": round(
                sum(r.vector_score for r in search_results) / len(search_results), 3
            ) if search_results else 0,
            "avg_final_score": round(
                sum(r.final_score for r in search_results) / len(search_results), 3
            ) if search_results else 0
        }

        # Search strategy information
        search_strategy = {
            "hybrid_search": True,
            "context_aware_scoring": True,
            "source_diversification": True,
            "temporal_weighting": True,
            "content_type_matching": bool(request.content_type_hint),
            "source_filtering": bool(request.source_filter)
        }

        logger.info(f"[{trace_id}] Successfully processed request in {total_time:.2f}ms")

        return SuggestResponse(
            trace_id=trace_id,
            suggestions=formatted_suggestions,
            sources=formatted_sources,
            stats=stats,
            search_strategy=search_strategy,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{trace_id}] Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/suggest/debug")
async def debug_suggest(request: SuggestRequest):
    """Debug endpoint that shows detailed scoring breakdown"""

    trace_id = f"debug_{int(time.time() * 1000)}"

    try:
        # Get search results with full detail
        search_results = await hybrid_search.search(
            query=request.text,
            context=request.context,
            top_k=10,
            source_filter=request.source_filter,
            content_type_hint=request.content_type_hint
        )
        
        # Show detailed scoring breakdown
        debug_info = {
            "trace_id": trace_id,
            "query_analysis": {
                "text_length": len(request.text),
                "word_count": len(request.text.split()),
                "query_specificity": "high" if len(request.text.split()) >= 5 else "medium" if len(request.text.split()) >= 2 else "low"
            },
            "search_results": [
                {
                    "doc_id": result.doc_id,
                    "title": result.title[:50],
                    "source": result.source,
                    "vector_score": result.vector_score,
                    "final_score": result.final_score,
                    "ranking_factors": result.ranking_factors,
                    "content_preview": result.content[:100] + "..."
                }
                for result in search_results[:5]
            ],
            "scoring_weights": hybrid_search.weights,
            "total_results": len(search_results)
        }
        
        return debug_info

    except Exception as e:
        logger.error(f"[{trace_id}] Debug error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

# Additional utility endpoints
@router.get("/stats")
async def get_system_stats():
    """Get overall system statistics"""
    try:
        collection_info = await vector_service.get_collection_info()
        
        return {
            "collection_stats": collection_info,
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_dimension": 384,
            "search_features": [
                "hybrid_search",
                "temporal_scoring",
                "source_diversification",
                "context_aware_ranking",
                "exact_match_boosting"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@router.post("/suggest/batch")
async def batch_suggest(requests: List[SuggestRequest]):
    """Process multiple suggestion requests in batch"""
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")

    results = []
    for i, request in enumerate(requests):
        try:
            result = await suggest_text(request)
            result.trace_id = f"batch_{int(time.time() * 1000)}_{i}"
            results.append(result)
        except Exception as e:
            results.append({
                "trace_id": f"batch_error_{i}",
                "error": str(e),
                "request_index": i
            })

    return {"batch_results": results}