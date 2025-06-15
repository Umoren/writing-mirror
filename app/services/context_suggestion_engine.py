"""
Context-aware suggestion generator that creates personalized writing suggestions
"""
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter

@dataclass
class WritingSuggestion:
    text: str
    confidence: float
    reasoning: str
    source_context: str
    suggestion_type: str

class ContextSuggestionEngine:
    """Generate personalized suggestions based on user's writing history"""
    
    def __init__(self):
        self.suggestion_patterns = {
            'continuation': [
                self._suggest_continuation,
                self._suggest_elaboration,
                self._suggest_transition
            ],
            'completion': [
                self._suggest_completion,
                self._suggest_closing,
                self._suggest_call_to_action
            ],
            'enhancement': [
                self._suggest_specificity,
                self._suggest_examples,
                self._suggest_tone_match
            ]
        }
    
    def generate_suggestions(self, 
                           current_text: str,
                           context: str,
                           search_results: List[Any],
                           suggestion_type: str = "continuation") -> List[WritingSuggestion]:
        """Generate contextual suggestions based on search results"""
        
        if not search_results:
            return self._generate_fallback_suggestions(current_text, context)
        
        # Analyze current text and context
        text_analysis = self._analyze_current_text(current_text, context)
        
        # Extract patterns from search results
        patterns = self._extract_writing_patterns(search_results, text_analysis)
        
        # Generate suggestions using extracted patterns
        suggestions = []
        
        # Use appropriate suggestion methods based on type
        suggestion_methods = self.suggestion_patterns.get(suggestion_type, 
                                                         self.suggestion_patterns['continuation'])
        
        for method in suggestion_methods:
            method_suggestions = method(current_text, context, patterns, search_results)
            suggestions.extend(method_suggestions)
        
        # Rank and filter suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, text_analysis)
        
        return ranked_suggestions[:3]  # Return top 3 suggestions
    
    def _analyze_current_text(self, text: str, context: str) -> Dict[str, Any]:
        """Analyze the current text to understand writing context"""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'ends_with_punctuation': text.strip().endswith(('.', '!', '?')),
            'is_question': text.strip().endswith('?'),
            'is_list': '•' in text or any(text.strip().startswith(f"{i}.") for i in range(1, 10)),
            'tone': self._detect_tone(text),
            'context_type': self._classify_context(context),
            'last_sentence': text.split('.')[-1].strip() if '.' in text else text.strip(),
            'key_topics': self._extract_key_topics(text)
        }
    
    def _extract_writing_patterns(self, search_results: List[Any], text_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract writing patterns from search results"""
        patterns = {
            'common_phrases': [],
            'sentence_starters': [],
            'transitions': [],
            'closings': [],
            'tone_examples': [],
            'topic_connections': []
        }
        
        for result in search_results[:5]:  # Analyze top 5 results
            content = result.content
            
            # Extract sentence starters
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            for sentence in sentences:
                words = sentence.split()
                if len(words) >= 3:
                    starter = ' '.join(words[:3])
                    patterns['sentence_starters'].append(starter)
            
            # Extract transition phrases
            transitions = self._find_transitions(content)
            patterns['transitions'].extend(transitions)
            
            # Extract common phrases (2-4 word n-grams)
            phrases = self._extract_ngrams(content, 2, 4)
            patterns['common_phrases'].extend(phrases)
            
            # Extract topic connections
            if text_analysis['key_topics']:
                connections = self._find_topic_connections(content, text_analysis['key_topics'])
                patterns['topic_connections'].extend(connections)
        
        # Deduplicate and rank patterns
        for pattern_type in patterns:
            patterns[pattern_type] = self._rank_patterns(patterns[pattern_type])
        
        return patterns
    
    def _suggest_continuation(self, current_text: str, context: str, patterns: Dict, results: List) -> List[WritingSuggestion]:
        """Suggest natural continuations"""
        suggestions = []
        
        # Use common sentence starters from user's writing
        for starter in patterns['sentence_starters'][:3]:
            suggestion = WritingSuggestion(
                text=starter + "...",
                confidence=0.7,
                reasoning="Based on your typical sentence patterns",
                source_context=f"Found in {len([r for r in results if starter.lower() in r.content.lower()])} similar documents",
                suggestion_type="continuation"
            )
            suggestions.append(suggestion)
        
        # Use topic connections
        for connection in patterns['topic_connections'][:2]:
            suggestion = WritingSuggestion(
                text=connection,
                confidence=0.6,
                reasoning="Builds on topics from your previous writing",
                source_context="Connected to your knowledge base",
                suggestion_type="continuation"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_elaboration(self, current_text: str, context: str, patterns: Dict, results: List) -> List[WritingSuggestion]:
        """Suggest ways to elaborate on current text"""
        suggestions = []
        
        # Find examples from similar content
        for result in results[:2]:
            if self._is_elaborative_content(result.content):
                elaboration = self._extract_elaboration_phrase(result.content)
                if elaboration:
                    suggestion = WritingSuggestion(
                        text=elaboration,
                        confidence=0.6,
                        reasoning="Elaborative pattern from your writing style",
                        source_context=f"From: {result.title[:50]}...",
                        suggestion_type="elaboration"
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_transition(self, current_text: str, context: str, patterns: Dict, results: List) -> List[WritingSuggestion]:
        """Suggest transition phrases"""
        suggestions = []
        
        for transition in patterns['transitions'][:2]:
            suggestion = WritingSuggestion(
                text=transition,
                confidence=0.5,
                reasoning="Transition style from your writing",
                source_context="Based on your typical transitions",
                suggestion_type="transition"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_completion(self, current_text: str, context: str, patterns: Dict, results: List) -> List[WritingSuggestion]:
        """Suggest completions for unfinished sentences"""
        suggestions = []
        
        # Analyze incomplete text
        if not current_text.strip().endswith(('.', '!', '?')):
            # Find similar incomplete patterns in user's writing
            for result in results:
                completion = self._find_completion_pattern(current_text, result.content)
                if completion:
                    suggestion = WritingSuggestion(
                        text=completion,
                        confidence=0.7,
                        reasoning="Completion pattern from similar context",
                        source_context=f"Pattern from: {result.title[:40]}...",
                        suggestion_type="completion"
                    )
                    suggestions.append(suggestion)
                    break
        
        return suggestions
    
    def _suggest_closing(self, current_text: str, context: str, patterns: Dict, results: List) -> List[WritingSuggestion]:
        """Suggest closing phrases"""
        suggestions = []
        
        # Extract common closings from user's writing
        closings = []
        for result in results:
            content_sentences = result.content.split('.')
            if content_sentences:
                last_sentence = content_sentences[-1].strip()
                if len(last_sentence.split()) <= 10 and last_sentence:
                    closings.append(last_sentence)
        
        # Use most common closings
        common_closings = Counter(closings).most_common(2)
        for closing, count in common_closings:
            suggestion = WritingSuggestion(
                text=closing,
                confidence=0.6,
                reasoning="Closing style from your writing",
                source_context=f"Used {count} times in your writing",
                suggestion_type="closing"
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_call_to_action(self, current_text: str, context: str, patterns: Dict, results: List) -> List[WritingSuggestion]:
        """Suggest call-to-action phrases"""
        suggestions = []
        
        # Look for action-oriented phrases in user's writing
        action_patterns = [
            r"let me know.*",
            r"please.*",
            r"would you.*",
            r"could you.*",
            r"feel free.*"
        ]
        
        for result in results:
            for pattern in action_patterns:
                matches = re.findall(pattern, result.content, re.IGNORECASE)
                for match in matches[:1]:  # Take first match
                    suggestion = WritingSuggestion(
                        text=match,
                        confidence=0.5,
                        reasoning="Call-to-action from your communication style",
                        source_context=f"From: {result.title[:40]}...",
                        suggestion_type="call_to_action"
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _detect_tone(self, text: str) -> str:
        """Detect the tone of current text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['please', 'thanks', 'thank you', 'kindly']):
            return 'polite'
        elif any(word in text_lower for word in ['!', 'excited', 'amazing', 'great']):
            return 'enthusiastic'
        elif any(word in text_lower for word in ['however', 'unfortunately', 'issue', 'problem']):
            return 'serious'
        else:
            return 'neutral'
    
    def _classify_context(self, context: str) -> str:
        """Classify the writing context"""
        context_lower = context.lower()
        
        if 'email' in context_lower:
            return 'email'
        elif 'job' in context_lower or 'career' in context_lower:
            return 'professional'
        elif 'personal' in context_lower:
            return 'personal'
        else:
            return 'general'
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from current text"""
        # Simple keyword extraction (could be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Filter common words
        stop_words = {'that', 'this', 'with', 'from', 'they', 'have', 'will', 'been', 'your', 'what', 'when', 'where'}
        keywords = [w for w in words if w not in stop_words]
        
        return list(set(keywords))[:5]  # Return top 5 unique keywords
    
    def _rank_suggestions(self, suggestions: List[WritingSuggestion], text_analysis: Dict) -> List[WritingSuggestion]:
        """Rank suggestions by quality and relevance"""
        
        for suggestion in suggestions:
            # Adjust confidence based on text analysis
            if text_analysis['tone'] == 'polite' and 'please' in suggestion.text.lower():
                suggestion.confidence *= 1.2
            
            # Boost suggestions that match context type
            if text_analysis['context_type'] == 'professional' and suggestion.suggestion_type in ['call_to_action', 'closing']:
                suggestion.confidence *= 1.1
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates
        seen_texts = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion.text not in seen_texts:
                unique_suggestions.append(suggestion)
                seen_texts.add(suggestion.text)
        
        return unique_suggestions
    
    def _generate_fallback_suggestions(self, current_text: str, context: str) -> List[WritingSuggestion]:
        """Generate fallback suggestions when no search results available"""
        return [
            WritingSuggestion(
                text="continues with clear purpose and direction.",
                confidence=0.3,
                reasoning="Fallback suggestion - no similar content found",
                source_context="Generic suggestion",
                suggestion_type="fallback"
            )
        ]
    
    # Helper methods for pattern extraction
    def _find_transitions(self, content: str) -> List[str]:
        """Find transition phrases in content"""
        transition_patterns = [
            r"however,?\s+.*?[.!?]",
            r"additionally,?\s+.*?[.!?]",
            r"furthermore,?\s+.*?[.!?]",
            r"on the other hand,?\s+.*?[.!?]"
        ]
        
        transitions = []
        for pattern in transition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            transitions.extend(matches)
        
        return transitions[:5]  # Return top 5
    
    def _extract_ngrams(self, content: str, min_len: int, max_len: int) -> List[str]:
        """Extract n-grams from content"""
        words = content.split()
        ngrams = []
        
        for length in range(min_len, max_len + 1):
            for i in range(len(words) - length + 1):
                ngram = ' '.join(words[i:i + length])
                if len(ngram) > 10:  # Skip very short phrases
                    ngrams.append(ngram)
        
        return ngrams
    
    def _rank_patterns(self, patterns: List[str]) -> List[str]:
        """Rank patterns by frequency and quality"""
        if not patterns:
            return []
        
        # Count frequencies
        pattern_counts = Counter(patterns)
        
        # Sort by frequency, take top items
        ranked = [pattern for pattern, count in pattern_counts.most_common(5)]
        
        return ranked
    
    def _find_topic_connections(self, content: str, topics: List[str]) -> List[str]:
        """Find connections between current topics and content"""
        connections = []
        content_lower = content.lower()
        
        for topic in topics:
            # Find sentences containing the topic
            sentences = content.split('.')
            for sentence in sentences:
                if topic.lower() in sentence.lower() and len(sentence.split()) > 5:
                    connections.append(sentence.strip()[:100] + "...")
                    break
        
        return connections
    
    def _is_elaborative_content(self, content: str) -> bool:
        """Check if content contains elaborative patterns"""
        elaborative_markers = ['for example', 'specifically', 'in particular', 'furthermore', 'additionally']
        return any(marker in content.lower() for marker in elaborative_markers)
    
    def _extract_elaboration_phrase(self, content: str) -> str:
        """Extract elaborative phrase from content"""
        sentences = content.split('.')
        for sentence in sentences:
            if any(marker in sentence.lower() for marker in ['for example', 'specifically']):
                return sentence.strip()[:80] + "..."
        return ""
    
    def _find_completion_pattern(self, incomplete_text: str, reference_content: str) -> str:
        """Find completion pattern for incomplete text"""
        # Simple pattern matching (could be enhanced)
        incomplete_words = incomplete_text.split()
        if len(incomplete_words) >= 2:
            last_two_words = ' '.join(incomplete_words[-2:])
            
            # Look for similar patterns in reference content
            sentences = reference_content.split('.')
            for sentence in sentences:
                if last_two_words.lower() in sentence.lower():
                    # Extract completion part
                    start_idx = sentence.lower().find(last_two_words.lower())
                    if start_idx != -1:
                        completion = sentence[start_idx + len(last_two_words):].strip()
                        if completion and len(completion.split()) <= 8:
                            return completion
        
        return ""