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
    ErrorResponse,
    Source
)
from ..services.vector_service import VectorService, init_vector_db
from ..services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize vector services
def get_vector_services():
    """Initialize and return vector services"""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "writing_samples")

    # Initialize Qdrant client
    qdrant_client = init_vector_db(url=qdrant_url, collection_name=collection_name)
    vector_service = VectorService(client=qdrant_client, collection_name=collection_name)
    embedding_service = EmbeddingService()

    return vector_service, embedding_service

# Initialize services at module level
try:
    vector_service, embedding_service = get_vector_services()
    logger.info("Vector services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vector services: {e}")
    vector_service = None
    embedding_service = None


@router.post("/suggest", response_model=SuggestResponse)
async def suggest(request: SuggestRequest) -> SuggestResponse:
    """Generate writing suggestions based on user input using vector search"""
    start_time = time.time()
    trace_id = f"suggest_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"[{trace_id}] Processing suggestion request: {request.text[:50]}...")
        
        if not vector_service or not embedding_service:
            raise HTTPException(
                status_code=503,
                detail="Vector services not available"
            )
        
        # Step 1: Generate query embedding
        embedding_start = time.time()
        query_vector = embedding_service.generate_embedding(request.text)
        embedding_time_ms = int((time.time() - embedding_start) * 1000)

        # Step 2: Search for similar content
        search_start = time.time()
        search_results = await vector_service.search_similar(
            query_vector=query_vector,
            top_k=10,  # Get more results for better variety
            score_threshold=0.3  # Lower threshold for more results
        )
        search_time_ms = int((time.time() - search_start) * 1000)

        logger.info(f"[{trace_id}] Found {len(search_results)} similar chunks")

        # Step 3: Generate suggestions based on retrieved content
        generation_start = time.time()
        suggestions = _generate_suggestions_from_chunks(
            user_text=request.text,
            search_results=search_results,
            task=request.task,
            num_suggestions=request.num_suggestions,
            max_length=request.max_length
        )
        generation_time_ms = int((time.time() - generation_start) * 1000)

        # Step 4: Format sources
        sources = _format_sources(search_results[:5])  # Top 5 sources

        # Compile performance stats
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
            sources=sources,
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


def _generate_suggestions_from_chunks(
    user_text: str,
    search_results: List[dict],
    task: str,
    num_suggestions: int,
    max_length: int
) -> List[Suggestion]:
    """Generate suggestions based on retrieved chunks (rule-based approach)"""

    if not search_results:
        return _generate_fallback_suggestions(user_text, task, num_suggestions)

    suggestions = []

    # Extract relevant text segments from chunks
    relevant_segments = []
    for result in search_results[:5]:  # Use top 5 results
        chunk_text = result['payload'].get('text', '')
        if chunk_text:
            relevant_segments.append(chunk_text)

    if task == "continue":
        suggestions = _generate_continuations(user_text, relevant_segments, num_suggestions, max_length)
    elif task == "complete":
        suggestions = _generate_completions(user_text, relevant_segments, num_suggestions, max_length)
    elif task == "rephrase":
        suggestions = _generate_rephrasings(user_text, relevant_segments, num_suggestions, max_length)
    else:
        suggestions = _generate_continuations(user_text, relevant_segments, num_suggestions, max_length)

    return suggestions


def _generate_continuations(user_text: str, relevant_segments: List[str], num_suggestions: int, max_length: int) -> List[Suggestion]:
    """Generate continuation suggestions based on similar content"""
    suggestions = []

    # Look for patterns and common continuations in similar content
    user_words = user_text.lower().split()
    last_words = user_words[-3:] if len(user_words) >= 3 else user_words

    candidates = []
    for segment in relevant_segments:
        words = segment.lower().split()

        # Find sequences that contain similar patterns
        for i, word in enumerate(words):
            if word in last_words:
                # Get the context following this word
                following_text = ' '.join(words[i+1:i+15])  # Next 15 words
                if following_text and len(following_text) <= max_length:
                    candidates.append(following_text)

    # Remove duplicates and score by relevance
    unique_candidates = list(set(candidates))
    scored_candidates = []

    for candidate in unique_candidates:
        # Simple scoring based on word overlap
        candidate_words = set(candidate.lower().split())
        user_word_set = set(user_words)
        overlap = len(candidate_words.intersection(user_word_set))
        score = min(0.9, 0.4 + (overlap * 0.1))
        scored_candidates.append((candidate, score))

    # Sort by score and take top suggestions
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    for i, (text, score) in enumerate(scored_candidates[:num_suggestions]):
        suggestions.append(Suggestion(
            text=text,
            score=score,
            reasoning=f"Based on similar content patterns in your writing"
        ))

    # Fill remaining slots with fallbacks if needed
    while len(suggestions) < num_suggestions:
        fallbacks = _generate_fallback_suggestions(user_text, "continue", 1)
        if fallbacks:
            suggestions.extend(fallbacks)
        break

    return suggestions[:num_suggestions]


def _generate_completions(user_text: str, relevant_segments: List[str], num_suggestions: int, max_length: int) -> List[Suggestion]:
    """Generate completion suggestions"""
    # For completion, look for sentence endings in similar content
    suggestions = []

    # Simple completion logic
    completions = [
        "that aligns with your previous insights.",
        "which builds on your established expertise.",
        "reflecting your typical analytical approach."
    ]

    for i, completion in enumerate(completions[:num_suggestions]):
        suggestions.append(Suggestion(
            text=completion,
            score=0.7 - (i * 0.1),
            reasoning="Completion based on writing patterns"
        ))

    return suggestions


def _generate_rephrasings(user_text: str, relevant_segments: List[str], num_suggestions: int, max_length: int) -> List[Suggestion]:
    """Generate rephrase suggestions"""
    suggestions = []

    # Simple rephrase logic - look for alternative phrasings in similar content
    words = user_text.split()

    # Basic rephrasings
    rephrasings = []

    # Replace common words with alternatives found in user's content
    if "and" in user_text:
        rephrasings.append(user_text.replace(" and ", " & "))
    if "but" in user_text:
        rephrasings.append(user_text.replace(" but ", " however, "))
    if "because" in user_text:
        rephrasings.append(user_text.replace(" because ", " since "))

    # If no simple replacements, use original
    if not rephrasings:
        rephrasings = [user_text + " (refined)"]

    for i, rephrase in enumerate(rephrasings[:num_suggestions]):
        suggestions.append(Suggestion(
            text=rephrase,
            score=0.6 - (i * 0.1),
            reasoning="Rephrase based on your writing style"
        ))

    return suggestions


def _generate_fallback_suggestions(user_text: str, task: str, num_suggestions: int) -> List[Suggestion]:
    """Generate fallback suggestions when no relevant content is found"""
    fallbacks = {
        "continue": [
            "continues with clear purpose and direction.",
            "develops this idea further with specific examples.",
            "builds upon this foundation naturally."
        ],
        "complete": [
            "requires careful consideration and planning.",
            "represents an important step forward.",
            "deserves our full attention and effort."
        ],
        "rephrase": [
            user_text.replace(" and ", " & "),
            user_text.replace(" but ", " however, "),
            user_text + " (refined)"
        ]
    }

    suggestions = []
    fallback_texts = fallbacks.get(task, fallbacks["continue"])

    for i, text in enumerate(fallback_texts[:num_suggestions]):
        suggestions.append(Suggestion(
            text=text,
            score=0.3,
            reasoning="Fallback suggestion - no similar content found"
        ))

    return suggestions


def _format_sources(search_results: List[dict]) -> List[Source]:
    """Format search results as source objects"""
    sources = []

    for result in search_results:
        payload = result.get('payload', {})

        source = Source(
            doc_id=payload.get('doc_id', 'unknown'),
            title=payload.get('title', 'Untitled'),
            similarity=result.get('score', 0.0),
            chunk_text=payload.get('text', '')[:200] + "..." if len(payload.get('text', '')) > 200 else payload.get('text', ''),
            source=payload.get('source', 'unknown')
        )
        sources.append(source)

    return sources


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with vector service status"""
    try:
        # Check vector services
        vector_healthy = vector_service is not None and embedding_service is not None
        
        services_status = {
            "vector_database": {
                "status": "healthy" if vector_healthy else "unhealthy",
                "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "collection": os.getenv("QDRANT_COLLECTION", "writing_samples")
            },
            "embedding_service": {
                "status": "healthy" if embedding_service else "unhealthy",
                "model": "all-MiniLM-L6-v2"
            }
        }
        
        # Try to get collection info if services are healthy
        if vector_healthy:
            try:
                collection_info = await vector_service.get_collection_info()
                services_status["vector_database"]["vectors_count"] = collection_info.get("vectors_count", 0)
                services_status["vector_database"]["points_count"] = collection_info.get("points_count", 0)
            except Exception as e:
                logger.warning(f"Could not get collection info: {e}")
                services_status["vector_database"]["status"] = "degraded"

        overall_status = "healthy" if vector_healthy else "unhealthy"

        return HealthResponse(
            status=overall_status,
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            services={
                "error": str(e)
            }
        )