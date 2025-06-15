"""
Hybrid search engine that combines vector similarity with metadata signals
"""
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SearchResult:
    doc_id: str
    content: str
    title: str
    source: str
    vector_score: float
    final_score: float
    metadata: Dict[str, Any]
    ranking_factors: Dict[str, float]

class HybridSearchEngine:
    """Advanced search that combines multiple signals"""
    
    def __init__(self, vector_service, embedding_service):
        self.vector_service = vector_service
        self.embedding_service = embedding_service
        
        # Scoring weights
        self.weights = {
            "vector_similarity": 0.6,    # Base semantic similarity
            "temporal_relevance": 0.15,  # How recent the content is
            "source_preference": 0.1,    # Prefer certain sources
            "content_type_match": 0.1,   # Match content type to query
            "engagement_score": 0.05     # Historical user engagement
        }
    
    async def search(self, 
                    query: str, 
                    context: str = "", 
                    top_k: int = 10,
                    source_filter: Optional[str] = None,
                    content_type_hint: Optional[str] = None) -> List[SearchResult]:
        """Execute hybrid search with multiple ranking signals"""
        
        # Generate query embedding
        query_vector = self.embedding_service.generate_embedding(f"{query} {context}")
        
        # Get initial vector search results (cast wider net)
        vector_results = await self.vector_service.search_similar(
            query_vector=query_vector,
            top_k=top_k * 3,  # Get 3x results for re-ranking
            score_threshold=0.2  # Lower threshold for initial retrieval
        )
        
        if not vector_results:
            return []
        
        # Re-rank with hybrid scoring
        scored_results = []
        for result in vector_results:
            hybrid_score = self._calculate_hybrid_score(
                result, query, context, content_type_hint
            )
            
            scored_results.append(SearchResult(
                doc_id=result.get('id', ''),
                content=result.get('payload', {}).get('text', ''),
                title=result.get('payload', {}).get('title', ''),
                source=result.get('payload', {}).get('source', ''),
                vector_score=result.get('score', 0.0),
                final_score=hybrid_score['total'],
                metadata=result.get('payload', {}),
                ranking_factors=hybrid_score['factors']
            ))
        
        # Sort by final score and return top results
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Apply source filtering if specified
        if source_filter:
            scored_results = [r for r in scored_results if r.source == source_filter]
        
        return scored_results[:top_k]
    
    def _calculate_hybrid_score(self, 
                               result: Dict[str, Any], 
                               query: str, 
                               context: str,
                               content_type_hint: Optional[str]) -> Dict[str, Any]:
        """Calculate hybrid score combining multiple signals"""
        
        payload = result.get('payload', {})
        vector_score = result.get('score', 0.0)
        
        factors = {
            "vector_similarity": vector_score,
            "temporal_relevance": self._calculate_temporal_score(payload),
            "source_preference": self._calculate_source_score(payload, context),
            "content_type_match": self._calculate_content_type_score(payload, content_type_hint),
            "engagement_score": self._calculate_engagement_score(payload)
        }
        
        # Calculate weighted total
        total_score = sum(
            factors[factor] * self.weights[factor] 
            for factor in factors
        )
        
        return {
            "total": total_score,
            "factors": factors
        }
    
    def _calculate_temporal_score(self, payload: Dict[str, Any]) -> float:
        """Score based on how recent the content is"""
        timestamp_str = payload.get('timestamp')
        if not timestamp_str:
            return 0.5  # Neutral score for missing timestamps
        
        try:
            # Parse timestamp
            if isinstance(timestamp_str, str):
                content_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                content_date = timestamp_str
            
            # Calculate age in days
            age_days = (datetime.now() - content_date.replace(tzinfo=None)).days
            
            # Exponential decay: more recent = higher score
            if age_days <= 7:
                return 1.0  # Very recent
            elif age_days <= 30:
                return 0.8  # Recent
            elif age_days <= 90:
                return 0.6  # Somewhat recent
            elif age_days <= 365:
                return 0.4  # Older
            else:
                return 0.2  # Very old
                
        except Exception:
            return 0.5  # Default for parsing errors
    
    def _calculate_source_score(self, payload: Dict[str, Any], context: str) -> float:
        """Score based on source relevance to context"""
        source = payload.get('source', '').lower()
        context_lower = context.lower()
        
        # Context-based source preferences
        if 'job' in context_lower or 'career' in context_lower:
            return 1.0 if source == 'gmail' else 0.7  # Prefer emails for job context
        
        elif 'knowledge' in context_lower or 'research' in context_lower:
            return 1.0 if source == 'notion' else 0.7  # Prefer notes for research
        
        else:
            # Equal preference by default
            return 0.8
    
    def _calculate_content_type_score(self, payload: Dict[str, Any], content_type_hint: Optional[str]) -> float:
        """Score based on content type matching"""
        if not content_type_hint:
            return 0.5  # Neutral if no hint provided
        
        content_type = payload.get('content_type', '').lower()
        hint_lower = content_type_hint.lower()
        
        # Direct matches
        if content_type == hint_lower:
            return 1.0
        
        # Related matches
        type_relationships = {
            'job_related': ['career', 'work', 'professional'],
            'technical': ['development', 'programming', 'code'],
            'personal': ['communication', 'conversation'],
            'newsletter': ['information', 'updates', 'news']
        }
        
        for content_t, related_terms in type_relationships.items():
            if content_type == content_t and any(term in hint_lower for term in related_terms):
                return 0.8
        
        return 0.3  # Low score for mismatched types
    
    def _calculate_engagement_score(self, payload: Dict[str, Any]) -> float:
        """Score based on historical user engagement (placeholder for now)"""
        # This would track user interactions in a real system
        # For now, use simple heuristics
        
        content_length = len(payload.get('text', ''))
        title_length = len(payload.get('title', ''))
        
        # Longer content might be more substantial
        length_score = min(content_length / 1000, 1.0)
        
        # Meaningful titles might indicate important content  
        title_score = 0.8 if title_length > 10 else 0.5
        
        return (length_score + title_score) / 2

class ContextAwareScoring:
    """Additional context-aware scoring improvements"""
    
    @staticmethod
    def adjust_for_query_specificity(score: float, query: str) -> float:
        """Adjust scores based on query specificity"""
        word_count = len(query.split())
        
        if word_count == 1:
            # Single word queries are broad, be more permissive
            return score * 0.9
        elif word_count >= 5:
            # Long queries are specific, be more strict
            return score * 1.1
        else:
            return score
    
    @staticmethod
    def boost_exact_matches(score: float, query: str, content: str) -> float:
        """Boost scores for exact phrase matches"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        if query_lower in content_lower:
            return min(score * 1.2, 1.0)  # 20% boost, capped at 1.0
        
        # Check for partial matches
        query_words = query_lower.split()
        content_words = content_lower.split()
        
        matches = sum(1 for word in query_words if word in content_words)
        match_ratio = matches / len(query_words)
        
        if match_ratio > 0.7:
            return score * 1.1  # 10% boost for high word overlap
        
        return score
    
    @staticmethod
    def diversify_sources(results: List[SearchResult], max_per_source: int = 3) -> List[SearchResult]:
        """Ensure source diversity in results"""
        source_counts = {}
        diversified = []
        
        for result in results:
            source = result.source
            current_count = source_counts.get(source, 0)
            
            if current_count < max_per_source:
                diversified.append(result)
                source_counts[source] = current_count + 1
        
        return diversified