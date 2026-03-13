"""
Abstention Controller for Perfect Recall.

Implements intelligent abstention when the system is uncertain about
retrieval results. Based on confidence thresholds and result quality metrics.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from ..models.retrieval import RetrievedMemory


@dataclass
class AbstentionDecision:
    """Decision on whether to abstain from providing an answer."""
    abstain: bool
    confidence: float
    reason: str
    suggestion: Optional[str] = None
    
    def __bool__(self):
        return not self.abstain


class AbstentionController:
    """
    Controls when the system should abstain from providing answers.
    
    Abstention is triggered when:
    1. No relevant memories are found (empty results)
    2. All results have low salience scores (below threshold)
    3. Results are contradictory or conflicting
    4. Semantic similarity is too low
    5. The query is outside the system's knowledge domain
    
    The controller provides graceful degradation with suggestions
    for how to proceed when abstaining.
    """
    
    def __init__(
        self,
        min_salience_threshold: float = 0.3,
        min_semantic_similarity: float = 0.4,
        min_results: int = 1,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the abstention controller.
        
        Args:
            min_salience_threshold: Minimum salience score to consider relevant
            min_semantic_similarity: Minimum semantic similarity for results
            min_results: Minimum number of results required
            confidence_threshold: Overall confidence threshold for answering
        """
        self.min_salience_threshold = min_salience_threshold
        self.min_semantic_similarity = min_semantic_similarity
        self.min_results = min_results
        self.confidence_threshold = confidence_threshold
    
    def should_abstain(
        self,
        query: str,
        results: List[RetrievedMemory],
        context: Optional[Dict[str, Any]] = None,
    ) -> AbstentionDecision:
        """
        Decide whether to abstain from answering based on retrieval results.
        
        Args:
            query: The original query
            results: Retrieved memories
            context: Additional context for decision
            
        Returns:
            AbstentionDecision with abstain flag and reasoning
        """
        # Check 1: No results at all
        if not results:
            return AbstentionDecision(
                abstain=True,
                confidence=0.0,
                reason="No relevant memories found",
                suggestion="The system has no information about this query. "
                          "Consider storing relevant memories first."
            )
        
        # Check 2: Not enough results
        if len(results) < self.min_results:
            return AbstentionDecision(
                abstain=True,
                confidence=0.1,
                reason=f"Insufficient results ({len(results)} found, "
                       f"{self.min_results} required)",
                suggestion="Try broadening your search or storing more context."
            )
        
        # Check 3: Low salience scores
        top_salience = results[0].salience_score if results else 0
        if top_salience < self.min_salience_threshold:
            return AbstentionDecision(
                abstain=True,
                confidence=top_salience,
                reason=f"Top result salience ({top_salience:.2f}) below threshold "
                       f"({self.min_salience_threshold})",
                suggestion="The retrieved memories don't seem strongly relevant. "
                          "Consider rephrasing your query."
            )
        
        # Check 4: Low semantic similarity
        top_semantic = results[0].semantic_similarity if results else 0
        if top_semantic < self.min_semantic_similarity:
            return AbstentionDecision(
                abstain=True,
                confidence=top_semantic,
                reason=f"Semantic similarity ({top_semantic:.2f}) below threshold "
                       f"({self.min_semantic_similarity})",
                suggestion="The query doesn't semantically match stored memories. "
                          "Try using different terminology."
            )
        
        # Check 5: High score variance (conflicting results)
        if len(results) >= 2:
            score_gap = results[0].salience_score - results[1].salience_score
            if score_gap < 0.1:  # Top results are very close
                # Check if they're contradictory
                if self._are_contradictory(results[0], results[1]):
                    return AbstentionDecision(
                        abstain=True,
                        confidence=0.4,
                        reason="Retrieved memories appear contradictory",
                        suggestion="Conflicting information found. Please clarify "
                                  "which context applies."
                    )
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(results, query)
        
        # Check 6: Overall confidence too low
        if confidence < self.confidence_threshold:
            return AbstentionDecision(
                abstain=True,
                confidence=confidence,
                reason=f"Overall confidence ({confidence:.2f}) below threshold "
                       f"({self.confidence_threshold})",
                suggestion="The system is uncertain about the relevance of "
                          "retrieved memories."
            )
        
        # All checks passed - don't abstain
        return AbstentionDecision(
            abstain=False,
            confidence=confidence,
            reason="Retrieval results meet quality thresholds"
        )
    
    def _are_contradictory(
        self,
        result1: RetrievedMemory,
        result2: RetrievedMemory,
    ) -> bool:
        """
        Check if two results are contradictory.
        
        Simple heuristic: check for negation words and opposite meanings.
        """
        content1 = result1.memory.content.lower()
        content2 = result2.memory.content.lower()
        
        # List of negation indicators
        negations = ['not', 'no', "don't", "doesn't", "won't", "can't", 
                     'never', 'none', 'nothing', 'nobody', 'neither', 'nowhere']
        
        # Check if one has negation and the other doesn't for same subject
        has_neg1 = any(neg in content1 for neg in negations)
        has_neg2 = any(neg in content2 for neg in negations)
        
        # Simple contradiction: same topic but opposite polarity
        if has_neg1 != has_neg2:
            # Extract key words (simple approach: shared significant words)
            words1 = set(w for w in content1.split() if len(w) > 4)
            words2 = set(w for w in content2.split() if len(w) > 4)
            
            # If they share significant words but have opposite polarity
            if len(words1 & words2) >= 2:
                return True
        
        return False
    
    def _calculate_confidence(
        self,
        results: List[RetrievedMemory],
        query: str,
    ) -> float:
        """
        Calculate overall confidence in the retrieval results.
        
        Considers:
        - Top result salience
        - Score distribution
        - Number of relevant results
        """
        if not results:
            return 0.0
        
        # Base confidence from top result
        top_score = results[0].salience_score
        
        # Boost if multiple high-scoring results agree
        agreement_boost = 0.0
        high_scoring = [r for r in results if r.salience_score > 0.5]
        if len(high_scoring) >= 2:
            agreement_boost = 0.1 * min(len(high_scoring) - 1, 3)
        
        # Penalty for low semantic similarity
        semantic_penalty = 0.0
        if results[0].semantic_similarity < 0.5:
            semantic_penalty = 0.2
        
        confidence = top_score + agreement_boost - semantic_penalty
        return max(0.0, min(1.0, confidence))
    
    def get_abstention_message(
        self,
        decision: AbstentionDecision,
        query: str,
    ) -> str:
        """
        Generate a human-readable abstention message.
        
        Args:
            decision: The abstention decision
            query: The original query
            
        Returns:
            Formatted abstention message
        """
        lines = [
            f"⚠️  ABSTENTION: I'm not confident enough to answer this query.",
            f"",
            f"Query: \"{query}\"",
            f"",
            f"Reason: {decision.reason}",
            f"Confidence: {decision.confidence:.2f}",
        ]
        
        if decision.suggestion:
            lines.extend([
                f"",
                f"Suggestion: {decision.suggestion}"
            ])
        
        return "\n".join(lines)
    
    def filter_with_abstention(
        self,
        query: str,
        results: List[RetrievedMemory],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[List[RetrievedMemory]], AbstentionDecision]:
        """
        Filter results and return abstention decision.
        
        Returns:
            Tuple of (filtered_results or None, abstention_decision)
            If abstaining, filtered_results is None.
        """
        decision = self.should_abstain(query, results, context)
        
        if decision.abstain:
            return None, decision
        
        return results, decision
