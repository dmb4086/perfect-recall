"""
Retrieval pipeline for Perfect Recall.

Implements multi-stage retrieval with salience scoring.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable

from ..models.memory import MemoryNode, MemoryTier
from ..models.retrieval import RetrievedMemory, RetrievalContext
from ..models.session import WorkingMemorySlot
from ..db.connection import DatabaseManager
from ..db.repositories import MemoryRepository


class SalienceScorer:
    """
    Calculates salience (relevance/importance) scores for memories.
    
    Combines multiple factors:
    - Semantic similarity (30%)
    - Superpowers metadata match (20%) - triggers, symptoms, aliases
    - Recency (15%)
    - Importance (15%)
    - Frequency (10%)
    - Contextual match (10%)
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize scorer with custom weights.
        
        Args:
            weights: Custom weights for scoring components.
                    Defaults to balanced weighting with Superpowers support.
        """
        self.weights = weights or {
            'semantic': 0.30,
            'superpowers': 0.20,  # New: triggers, symptoms, aliases matching
            'recency': 0.15,
            'importance': 0.15,
            'frequency': 0.10,
            'contextual': 0.10,
        }
    
    def score(
        self,
        memory: MemoryNode,
        query: str = "",
        query_embedding: Optional[List[float]] = None,
        semantic_similarity: float = 0.0,
        context: Optional[RetrievalContext] = None,
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate salience score for a memory.
        
        Args:
            memory: Memory to score
            query: Original query text (for Superpowers matching)
            query_embedding: Query embedding for semantic similarity
            semantic_similarity: Pre-computed semantic similarity
            context: Current retrieval context
            
        Returns:
            Tuple of (total_score, component_scores)
        """
        components = {}
        
        # Semantic similarity
        components['semantic'] = semantic_similarity
        
        # Superpowers metadata match (triggers, symptoms, aliases)
        components['superpowers'] = self._superpowers_score(memory, query)
        
        # Recency score
        components['recency'] = self._recency_score(memory)
        
        # Importance score
        components['importance'] = self._importance_score(memory)
        
        # Frequency score
        components['frequency'] = self._frequency_score(memory)
        
        # Contextual score
        components['contextual'] = self._contextual_score(memory, context)
        
        # Weighted sum
        total = sum(
            components[k] * self.weights[k]
            for k in components
        )
        
        return total, components
    
    def _superpowers_score(self, memory: MemoryNode, query: str) -> float:
        """
        Calculate Superpowers metadata match score.
        
        Scores based on overlap between query and:
        - triggers (when to recall)
        - symptoms (error patterns)
        - aliases (synonyms)
        """
        if not query:
            return 0.5
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        scores = []
        
        # Check triggers (highest weight - direct activation)
        if memory.triggers:
            trigger_hits = sum(
                1 for t in memory.triggers 
                if any(term in t.lower() for term in query_terms)
            )
            scores.append(min(trigger_hits / max(len(memory.triggers) * 0.3, 1.0), 1.0) * 1.0)
        
        # Check symptoms (high weight for error contexts)
        if memory.symptoms:
            symptom_hits = sum(
                1 for s in memory.symptoms 
                if s.lower() in query_lower or any(term in s.lower() for term in query_terms)
            )
            scores.append(min(symptom_hits / max(len(memory.symptoms) * 0.3, 1.0), 1.0) * 0.9)
        
        # Check aliases (medium weight - synonym matching)
        if memory.aliases:
            alias_hits = sum(
                1 for a in memory.aliases 
                if a.lower() in query_lower
            )
            scores.append(min(alias_hits / max(len(memory.aliases) * 0.3, 1.0), 1.0) * 0.8)
        
        # Check anti-triggers (negative score if matched)
        if memory.anti_triggers:
            anti_hits = sum(
                1 for at in memory.anti_triggers 
                if at.lower() in query_lower or any(term in at.lower() for term in query_terms)
            )
            if anti_hits > 0:
                # Reduce score if anti-triggers match
                return max(0.0, (sum(scores) / len(scores) if scores else 0.5) - 0.3)
        
        if not scores:
            return 0.5  # Neutral if no Superpowers metadata
        
        return sum(scores) / len(scores)
    
    def _recency_score(self, memory: MemoryNode) -> float:
        """
        Calculate recency score with exponential decay.
        
        Half-life of 1 week (168 hours).
        """
        if memory.last_accessed:
            reference_time = memory.last_accessed
        else:
            reference_time = memory.created_at
        
        if reference_time is None:
            return 0.5  # Default score if no time info
        
        # Handle timezone-aware vs naive datetime comparison
        now = datetime.now(timezone.utc)
        if reference_time.tzinfo is None:
            # reference_time is timezone-naive, make it timezone-aware
            reference_time = reference_time.replace(tzinfo=timezone.utc)
        
        age_hours = (now - reference_time).total_seconds() / 3600
        return math.exp(-age_hours / 168)
    
    def _importance_score(self, memory: MemoryNode) -> float:
        """
        Calculate importance score with access boost.
        """
        # Small boost for frequently accessed memories
        access_boost = min(memory.access_count / 20, 0.1)
        return min(memory.importance_score + access_boost, 1.0)
    
    def _frequency_score(self, memory: MemoryNode) -> float:
        """
        Calculate frequency score.
        
        Saturates at ~15 accesses.
        """
        if memory.access_count == 0:
            return 0.0
        return 1 - math.exp(-memory.access_count / 5)
    
    def _contextual_score(
        self,
        memory: MemoryNode,
        context: Optional[RetrievalContext]
    ) -> float:
        """
        Calculate contextual overlap score.
        
        Based on tag/topic overlap between memory and context.
        """
        if not context:
            return 0.5  # Neutral
        
        memory_tags = set(memory.metadata.get('tags', []))
        context_topics = set(context.active_topics)
        
        if not memory_tags or not context_topics:
            return 0.5
        
        overlap = len(memory_tags & context_topics)
        return overlap / max(len(memory_tags), len(context_topics))


class RetrievalPipeline:
    """
    Multi-stage retrieval pipeline for memories.
    
    Stages:
    1. Vector similarity search (broad recall)
    2. Temporal filtering
    3. Metadata filtering
    4. Salience scoring
    5. Ranking and selection
    6. Access logging
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        scorer: Optional[SalienceScorer] = None,
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            db_manager: Database connection manager
            embedding_func: Function to generate embeddings
            scorer: Custom salience scorer
        """
        self.db_manager = db_manager
        self.embedding_func = embedding_func
        self.scorer = scorer or SalienceScorer()
    
    async def retrieve(
        self,
        query: str,
        context: Optional[RetrievalContext] = None,
        limit: int = 10,
        memory_tiers: Optional[List[MemoryTier]] = None,
        threshold: float = 0.5,
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Query text
            context: Retrieval context
            limit: Maximum results
            memory_tiers: Which memory tiers to search
            threshold: Minimum similarity threshold
            
        Returns:
            List of retrieved memories with scores
        """
        # Generate query embedding
        query_embedding = None
        if self.embedding_func:
            try:
                query_embedding = self.embedding_func(query)
            except Exception as e:
                print(f"Embedding generation failed: {e}")
        
        # Stage 1: Vector similarity search
        candidates = await self._vector_search(
            query_embedding=query_embedding,
            k=limit * 3,  # Over-fetch for reranking
            memory_tiers=memory_tiers,
            threshold=threshold,
        )
        
        # Stage 2-3: Temporal and metadata filtering
        filtered = self._apply_filters(candidates, context)
        
        # Stage 3.5: Anti-trigger filtering (Superpowers enhancement)
        filtered = self._apply_anti_trigger_filter(filtered, query)
        
        # Stage 4-5: Salience scoring and ranking
        scored = []
        for memory, semantic_sim in filtered:
            salience, components = self.scorer.score(
                memory=memory,
                query=query,
                query_embedding=query_embedding,
                semantic_similarity=semantic_sim,
                context=context,
            )
            
            scored.append(RetrievedMemory(
                memory=memory,
                salience_score=salience,
                semantic_similarity=semantic_sim,
                score_components=components,
                query_text=query,
            ))
        
        # Sort by salience score
        scored.sort(key=lambda x: x.salience_score, reverse=True)
        
        # Select top results
        results = scored[:limit]
        
        # Stage 6: Log access
        await self._log_access(results, context)
        
        return results
    
    async def retrieve_for_session(
        self,
        session_id: str,
        query: str,
        include_working_memory: bool = True,
        limit: int = 10,
    ) -> List[RetrievedMemory]:
        """
        Retrieve memories for a specific session.
        
        Prioritizes working memory, then searches long-term memory.
        
        Args:
            session_id: Session ID
            query: Query text
            include_working_memory: Include current working memory
            limit: Maximum results
            
        Returns:
            List of retrieved memories
        """
        results = []
        
        # First, get working memory
        if include_working_memory:
            from ..db.repositories import SessionRepository
            
            async with self.db_manager.session() as db_session:
                repo = SessionRepository(db_session)
                slots = await repo.get_working_memory(session_id)
            
            # Load memory nodes for slots
            async with self.db_manager.session() as db_session:
                memory_repo = MemoryRepository(db_session)
                
                for slot in slots:
                    if slot.is_expired():
                        continue
                    
                    memory = await memory_repo.get_by_id(slot.memory_id)
                    if memory:
                        results.append(RetrievedMemory(
                            memory=memory,
                            salience_score=slot.priority,
                            semantic_similarity=0.0,  # Not computed for working memory
                            query_text=query,
                        ))
        
        # Then search long-term memory
        long_term_limit = max(limit - len(results), 0)
        if long_term_limit > 0:
            long_term = await self.retrieve(
                query=query,
                limit=long_term_limit,
                memory_tiers=[MemoryTier.EPISODIC, MemoryTier.SEMANTIC, MemoryTier.PROCEDURAL],
            )
            results.extend(long_term)
        
        return results[:limit]
    
    async def _vector_search(
        self,
        query_embedding: Optional[List[float]],
        k: int,
        memory_tiers: Optional[List[MemoryTier]] = None,
        threshold: float = 0.5,
    ) -> List[tuple[MemoryNode, float]]:
        """
        Perform vector similarity search.
        
        Returns list of (memory, similarity_score) tuples.
        """
        if query_embedding is None:
            # Fallback: return most recent memories
            return await self._recent_memories(k, memory_tiers)
        
        async with self.db_manager.session() as db_session:
            repo = MemoryRepository(db_session)
            return await repo.search_similar(
                embedding=query_embedding,
                limit=k,
                threshold=threshold,
                memory_tiers=memory_tiers,
            )
    
    async def _recent_memories(
        self,
        k: int,
        memory_tiers: Optional[List[MemoryTier]] = None,
    ) -> List[tuple[MemoryNode, float]]:
        """Fallback: Get recent memories when no embedding available."""
        from sqlalchemy import select, desc
        from ..db.sqlalchemy_models import MemoryNodeORM
        
        async with self.db_manager.session() as db_session:
            query = select(MemoryNodeORM).order_by(desc(MemoryNodeORM.created_at)).limit(k)
            
            if memory_tiers:
                tier_values = [t.value for t in memory_tiers]
                query = query.where(MemoryNodeORM.memory_tier.in_(tier_values))
            
            result = await db_session.execute(query)
            orms = result.scalars().all()
            
            # Return with neutral similarity
            return [(MemoryNode.model_validate(orm), 0.5) for orm in orms]
    
    def _apply_filters(
        self,
        candidates: List[tuple[MemoryNode, float]],
        context: Optional[RetrievalContext],
    ) -> List[tuple[MemoryNode, float]]:
        """Apply temporal and metadata filters."""
        if not context:
            return candidates
        
        filtered = []
        for memory, score in candidates:
            # Temporal filtering
            if context.temporal_at:
                # Check if memory was valid at this time
                if memory.valid_from > context.temporal_at:
                    continue
                if memory.valid_until and memory.valid_until < context.temporal_at:
                    continue
            
            if context.temporal_from and memory.valid_until:
                if memory.valid_until < context.temporal_from:
                    continue
            
            if context.temporal_to and memory.valid_from > context.temporal_to:
                continue
            
            # Tier filtering
            if context.include_tiers and memory.memory_tier.value not in context.include_tiers:
                continue
            
            if context.exclude_tiers and memory.memory_tier.value in context.exclude_tiers:
                continue
            
            filtered.append((memory, score))
        
        return filtered
    
    def _apply_anti_trigger_filter(
        self,
        candidates: List[tuple[MemoryNode, float]],
        query: str,
    ) -> List[tuple[MemoryNode, float]]:
        """
        Filter out memories whose anti-triggers match the query.
        
        Superpowers-inspired: anti_triggers tell us when NOT to use a memory.
        """
        if not query:
            return candidates
        
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        filtered = []
        for memory, score in candidates:
            # Skip if memory has no anti_triggers
            if not memory.anti_triggers:
                filtered.append((memory, score))
                continue
            
            # Check if query matches any anti-trigger
            anti_trigger_match = False
            for anti in memory.anti_triggers:
                anti_lower = anti.lower()
                # Direct match
                if anti_lower in query_lower:
                    anti_trigger_match = True
                    break
                # Term overlap match
                anti_terms = set(anti_lower.split())
                if anti_terms & query_terms:  # Intersection
                    # Strong overlap (>50% of anti-trigger terms)
                    if len(anti_terms & query_terms) / len(anti_terms) > 0.5:
                        anti_trigger_match = True
                        break
            
            # Only keep if no anti-trigger matches
            if not anti_trigger_match:
                filtered.append((memory, score))
        
        return filtered
    
    async def _log_access(
        self,
        results: List[RetrievedMemory],
        context: Optional[RetrievalContext],
    ):
        """Log access for retrieved memories."""
        from uuid import UUID
        
        session_id = None
        if context and context.session_id:
            try:
                session_id = UUID(context.session_id)
            except ValueError:
                pass
        
        async with self.db_manager.session() as db_session:
            repo = MemoryRepository(db_session)
            
            for retrieved in results:
                await repo.log_access(
                    memory_id=retrieved.memory.id,
                    access_type='recall',
                    session_id=session_id,
                    query_text=retrieved.query_text,
                )
