"""
Memory Writer Module

Captures session events and decides what to store in memory.
Implements the Write Gate pattern for intelligent memory filtering.
"""

import re
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from uuid import UUID

from ..models.memory import (
    MemoryNode, MemoryTier, EpisodeType, SourceType, Episode
)
from ..models.retrieval import WriteDecision
from ..db.connection import DatabaseManager
from ..db.repositories import MemoryRepository, EpisodeRepository


class MemoryWriter:
    """
    Captures session events and writes them to appropriate memory tiers.
    
    The MemoryWriter implements:
    1. Write Gate - Decides what to store vs. discard
    2. Tier Routing - Routes memories to appropriate tier (episodic, semantic, procedural)
    3. Fact Extraction - Extracts semantic facts from episodic content
    4. Embedding Generation - Creates vector embeddings for retrieval
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        write_threshold: float = 0.6,
    ):
        """
        Initialize MemoryWriter.
        
        Args:
            db_manager: Database connection manager
            embedding_func: Function to generate embeddings from text
            write_threshold: Minimum score to write a memory (0-1)
        """
        self.db_manager = db_manager
        self.embedding_func = embedding_func
        self.write_threshold = write_threshold
        
        # Track recent writes for rate-aware thresholding
        self._recent_writes: deque[datetime] = deque(maxlen=100)
        
        # High-value keywords for utility scoring
        self._high_utility_patterns = {
            'preference': 0.9,
            'prefer': 0.9,
            'like': 0.8,
            'dislike': 0.8,
            'decided': 0.8,
            'decision': 0.8,
            'choose': 0.8,
            'chose': 0.8,
            'project': 0.7,
            'task': 0.7,
            'goal': 0.7,
            'important': 1.0,
            'remember': 1.0,
            'don\'t forget': 1.0,
        }
        
        # Explicit markers
        self._explicit_markers = [
            'remember this',
            "don't forget",
            'important:',
            'note that',
            'for future reference',
        ]
        
        # Superpowers-inspired: Error/symptom patterns for memory classification
        self._error_patterns = [
            r'error[:\s]+([^\n.]+)',
            r'exception[:\s]+([^\n.]+)',
            r'failed?(?:\s+to)?[:\s]+([^\n.]+)',
            r'(?:bug|issue|problem)[:\s]+(?:is|with)?\s*([^\n.]+)',
            r'(?:traceback|stack trace)',
            r'(?:timeout|timed out)',
            r'(?:crash|crashed)',
            r'(?:undefined|not found|not defined)',
        ]
        
        # Anti-trigger patterns (when NOT to apply a memory)
        self._anti_trigger_patterns = [
            r'(?:not|don\'t|never)\s+(?:use|apply|for)',
            r'(?:deprecated|obsolete|replaced by)',
            r'(?:outdated|old version)',
        ]
    
    # ========================================================================
    # Public API
    # ========================================================================
    
    async def record_episode(
        self,
        content: str,
        episode_type: EpisodeType,
        session_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        triggers: Optional[List[str]] = None,
        symptoms: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        anti_triggers: Optional[List[str]] = None,
    ) -> Optional[MemoryNode]:
        """
        Record a new episode to episodic memory.
        
        Runs content through write gate before storage.
        Automatically extracts facts for semantic memory.
        
        Args:
            content: The content to remember
            episode_type: Type of episode (message, action, decision, etc.)
            session_id: Associated session ID
            metadata: Additional metadata
            embedding: Pre-computed embedding (optional)
            triggers: When to recall this memory (auto-extracted if not provided)
            symptoms: Error phrases/failure patterns (auto-extracted if not provided)
            aliases: Synonyms/related terms (auto-extracted if not provided)
            anti_triggers: When NOT to use this memory
            
        Returns:
            MemoryNode if stored, None if rejected by write gate
        """
        # Check write gate
        decision = self._should_write(content, episode_type.value, {'session_id': session_id})
        
        if not decision.write:
            return None
        
        # Auto-extract Superpowers metadata if not provided
        extracted_triggers = triggers or self._extract_triggers(content, episode_type)
        extracted_symptoms = symptoms or self._extract_symptoms(content)
        extracted_aliases = aliases or self._extract_aliases(content)
        extracted_anti_triggers = anti_triggers or self._extract_anti_triggers(content)
        
        # Generate embedding if not provided
        if embedding is None and self.embedding_func:
            try:
                embedding = self.embedding_func(content)
            except Exception as e:
                # Log error but continue without embedding
                print(f"Embedding generation failed: {e}")
                embedding = None
        
        # Create memory node
        memory = MemoryNode(
            memory_tier=MemoryTier.EPISODIC,
            content=content,
            embedding=embedding,
            importance_score=decision.importance,
            source_type=SourceType.DIRECT,
            triggers=extracted_triggers,
            symptoms=extracted_symptoms,
            aliases=extracted_aliases,
            anti_triggers=extracted_anti_triggers,
            metadata={
                **(metadata or {}),
                'episode_type': episode_type.value,
                'write_decision_factors': decision.factors,
            }
        )
        
        # Store in database
        async with self.db_manager.session() as db_session:
            repo = MemoryRepository(db_session)
            await repo.create(memory)
        
        # Track this write
        self._recent_writes.append(datetime.utcnow())
        
        # Extract facts asynchronously (background task)
        # For now, synchronous execution
        await self._extract_facts(memory)
        
        return memory
    
    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source_episode_id: Optional[UUID] = None,
        embedding: Optional[List[float]] = None,
    ) -> MemoryNode:
        """
        Store a semantic fact.
        
        Args:
            subject: Fact subject (e.g., "user")
            predicate: Fact predicate (e.g., "prefers")
            object: Fact object (e.g., "Python")
            confidence: Confidence in fact (0-1)
            source_episode_id: Source episode if extracted
            embedding: Pre-computed embedding
            
        Returns:
            Created memory node
        """
        content = f"{subject} {predicate} {object}"
        
        # Generate embedding
        if embedding is None and self.embedding_func:
            try:
                embedding = self.embedding_func(content)
            except Exception as e:
                print(f"Embedding generation failed: {e}")
                embedding = None
        
        memory = MemoryNode(
            memory_tier=MemoryTier.SEMANTIC,
            content=content,
            embedding=embedding,
            confidence=confidence,
            source_type=SourceType.INFERRED if source_episode_id else SourceType.DIRECT,
            source_episode_id=source_episode_id,
            metadata={
                'extracted_facts': [{
                    'subject': subject,
                    'predicate': predicate,
                    'object': object,
                    'confidence': confidence,
                }]
            }
        )
        
        async with self.db_manager.session() as db_session:
            repo = MemoryRepository(db_session)
            await repo.create(memory)
        
        return memory
    
    async def store_procedural(
        self,
        pattern_name: str,
        description: str,
        trigger_patterns: List[str],
        applicable_contexts: List[str],
        embedding: Optional[List[float]] = None,
        symptoms: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        anti_triggers: Optional[List[str]] = None,
    ) -> MemoryNode:
        """
        Store a procedural memory (skill/pattern).
        
        Args:
            pattern_name: Name of the pattern
            description: Description of what this pattern does
            trigger_patterns: Keywords/phrases that trigger this pattern
            applicable_contexts: Contexts where pattern applies
            embedding: Pre-computed embedding
            symptoms: Error phrases this pattern solves (auto-extracted if not provided)
            aliases: Synonyms for this pattern name (auto-extracted if not provided)
            anti_triggers: When NOT to use this pattern
            
        Returns:
            Created memory node
        """
        content = f"{pattern_name}: {description}"
        
        # Generate embedding
        if embedding is None and self.embedding_func:
            try:
                embedding = self.embedding_func(content)
            except Exception as e:
                print(f"Embedding generation failed: {e}")
                embedding = None
        
        # Auto-extract symptoms if not provided
        extracted_symptoms = symptoms or self._extract_symptoms(description)
        
        # Auto-extract aliases from pattern name if not provided
        extracted_aliases = aliases or []
        if not extracted_aliases:
            # Add common variations of pattern name
            name_parts = pattern_name.lower().replace('-', ' ').replace('_', ' ').split()
            extracted_aliases.append(' '.join(name_parts))
            if len(name_parts) > 1:
                extracted_aliases.append(name_parts[0])  # First word
                extracted_aliases.append(name_parts[-1])  # Last word
        
        # Auto-extract anti-triggers if not provided
        extracted_anti_triggers = anti_triggers or self._extract_anti_triggers(description)
        
        memory = MemoryNode(
            memory_tier=MemoryTier.PROCEDURAL,
            content=content,
            embedding=embedding,
            triggers=trigger_patterns,  # trigger_patterns map directly to triggers
            symptoms=extracted_symptoms,
            aliases=extracted_aliases,
            anti_triggers=extracted_anti_triggers,
            metadata={
                'pattern_name': pattern_name,
                'trigger_patterns': trigger_patterns,
                'applicable_contexts': applicable_contexts,
                'success_count': 0,
                'failure_count': 0,
            }
        )
        
        async with self.db_manager.session() as db_session:
            repo = MemoryRepository(db_session)
            await repo.create(memory)
        
        return memory
    
    async def add_to_working_memory(
        self,
        session_id: UUID,
        content: str,
        slot_type: str = "context",
        priority: float = 0.5,
        expires_in_minutes: Optional[int] = None,
        embedding: Optional[List[float]] = None,
    ) -> MemoryNode:
        """
        Add content directly to working memory for a session.
        
        Working memory is temporary and session-scoped.
        
        Args:
            session_id: Session to add to
            content: Content to remember
            slot_type: Type of working memory slot
            priority: Priority for retention
            expires_in_minutes: When to expire (None = session end)
            embedding: Pre-computed embedding
            
        Returns:
            Created memory node
        """
        # Generate embedding
        if embedding is None and self.embedding_func:
            try:
                embedding = self.embedding_func(content)
            except Exception as e:
                print(f"Embedding generation failed: {e}")
                embedding = None
        
        expires_at = None
        if expires_in_minutes:
            expires_at = datetime.utcnow() + timedelta(minutes=expires_in_minutes)
        
        memory = MemoryNode(
            memory_tier=MemoryTier.WORKING,
            content=content,
            embedding=embedding,
            importance_score=priority,
            valid_until=expires_at,
            metadata={
                'slot_type': slot_type,
                'session_id': str(session_id),
            }
        )
        
        async with self.db_manager.session() as db_session:
            repo = MemoryRepository(db_session)
            await repo.create(memory)
        
        return memory
    
    # ========================================================================
    # Write Gate
    # ========================================================================
    
    def _should_write(
        self,
        content: str,
        content_type: str,
        context: Dict[str, Any]
    ) -> WriteDecision:
        """
        Decide whether to write this content to memory.
        
        Uses multi-factor scoring with dynamic threshold based on recent write rate.
        """
        # Calculate component scores
        scores = {
            'novelty': self._score_novelty(content),
            'utility': self._score_utility(content),
            'explicit': self._score_explicit_marking(content),
            'density': self._score_information_density(content),
        }
        
        # Dynamic threshold based on recent write rate
        recent_rate = len([
            w for w in self._recent_writes
            if w > datetime.utcnow() - timedelta(minutes=5)
        ])
        dynamic_threshold = self.write_threshold + (recent_rate / 100) * 0.2
        
        # Weighted average
        weights = {
            'novelty': 0.30,
            'utility': 0.30,
            'explicit': 0.25,
            'density': 0.15
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores)
        
        if final_score >= dynamic_threshold:
            return WriteDecision(
                write=True,
                importance=final_score,
                factors=scores,
                reason="Passed write gate threshold"
            )
        
        return WriteDecision(
            write=False,
            importance=final_score,
            factors=scores,
            reason="Below write gate threshold"
        )
    
    def _score_novelty(self, content: str) -> float:
        """
        Score how novel this content is.
        
        Without access to existing memories, uses heuristics:
        - Unique entities = higher novelty
        - Specific details = higher novelty
        """
        score = 0.5  # Neutral default
        
        # Check for specific entities (capitalized words)
        has_entities = bool(re.search(r'\b[A-Z][a-z]+\b', content))
        if has_entities:
            score += 0.25
        
        # Check for specific details (numbers, codes, dates)
        has_specifics = bool(re.search(r'\d{4}|\b[A-Z]{2,}\b', content))
        if has_specifics:
            score += 0.25
        
        return min(score, 1.0)
    
    def _score_utility(self, content: str) -> float:
        """
        Score likely future utility.
        
        Uses keyword heuristics to detect high-value information.
        """
        lower = content.lower()
        
        for keyword, score in self._high_utility_patterns.items():
            if keyword in lower:
                return score
        
        return 0.3  # Default low utility
    
    def _score_explicit_marking(self, content: str) -> float:
        """Score explicit user marking of importance."""
        lower = content.lower()
        
        for marker in self._explicit_markers:
            if marker in lower:
                return 1.0
        
        return 0.0
    
    def _score_information_density(self, content: str) -> float:
        """Score information density vs fluff."""
        score = 0.5
        
        # Check for entities
        has_entities = bool(re.search(r'\b[A-Z][a-z]+\b', content))
        if has_entities:
            score += 0.25
        
        # Check for specifics
        has_specifics = bool(re.search(r'\d{4}|\b[A-Z]{2,}\b', content))
        if has_specifics:
            score += 0.25
        
        return min(score, 1.0)
    
    # ========================================================================
    # Superpowers Metadata Extraction
    # ========================================================================
    
    def _extract_triggers(self, content: str, episode_type: EpisodeType) -> List[str]:
        """
        Extract trigger conditions from content.
        
        Triggers are phrases that indicate when this memory should be recalled.
        """
        triggers = []
        lower_content = content.lower()
        
        # Extract context-based triggers
        trigger_patterns = [
            (r'when\s+([^,\n.]+)', 'when'),
            (r'if\s+([^,\n.]+)', 'if'),
            (r'use\s+when\s+([^,\n.]+)', 'use when'),
            (r'applies?\s+to\s+([^,\n.]+)', 'applies to'),
            (r'for\s+([^,\n.]+?)\s+(?:use|try|check)', 'for'),
        ]
        
        for pattern, prefix in trigger_patterns:
            matches = re.findall(pattern, lower_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                trigger = f"{prefix} {match.strip()}".strip()
                if len(trigger) > 5 and trigger not in triggers:
                    triggers.append(trigger)
        
        # Add episode-type specific triggers
        if episode_type == EpisodeType.DECISION:
            triggers.append('making decisions')
            triggers.append('deciding')
        elif episode_type == EpisodeType.ACTION:
            triggers.append('taking action')
            triggers.append('performing tasks')
        elif episode_type == EpisodeType.REFLECTION:
            triggers.append('reflecting')
            triggers.append('reviewing')
        
        # Extract key entities as triggers
        entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
        for entity in set(entities):
            entity_lower = entity.lower()
            if len(entity_lower) > 3 and entity_lower not in ['the', 'this', 'that', 'user']:
                triggers.append(f"mentions {entity_lower}")
        
        return triggers[:10]  # Limit to top 10 triggers
    
    def _extract_symptoms(self, content: str) -> List[str]:
        """
        Extract error symptoms and failure patterns from content.
        
        Symptoms help match memories to error contexts.
        """
        symptoms = []
        
        for pattern in self._error_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else match
                symptom = match.strip() if isinstance(match, str) else pattern.replace(r'\b', '').replace(r'(?:', '(').replace(r')?', ')')
                if symptom and len(symptom) > 3:
                    symptoms.append(symptom.lower())
        
        # Extract specific error messages
        error_lines = re.findall(r'[A-Z][a-zA-Z]*Error:\s*[^\n]+', content)
        for error in error_lines:
            symptoms.append(error.strip().lower())
        
        # Extract quoted error messages
        quoted_errors = re.findall(r'["\']([^"\']*(?:error|fail|exception|bug)[^"\']*)["\']', content, re.IGNORECASE)
        for error in quoted_errors:
            symptoms.append(error.strip().lower())
        
        return list(set(symptoms))[:10]  # Deduplicate and limit
    
    def _extract_aliases(self, content: str) -> List[str]:
        """
        Extract synonyms and related terms for flexible matching.
        
        Aliases help match queries that use different terminology.
        """
        aliases = []
        lower_content = content.lower()
        
        # Extract aliases from "also known as", "aka", "or" patterns
        aka_patterns = [
            r'(?:also known as|aka)\s+["\']?([^,"\n.]+)',
            r'\(([^(]+)\)\s+(?:is|are|refers to)',
            r'([^,\n]+)(?:\s+or\s+)([^,\n.]+)',
        ]
        
        for pattern in aka_patterns:
            matches = re.findall(pattern, lower_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        alias = m.strip()
                        if len(alias) > 2 and alias not in aliases:
                            aliases.append(alias)
                else:
                    alias = match.strip()
                    if len(alias) > 2 and alias not in aliases:
                        aliases.append(alias)
        
        # Extract technical terms and acronyms
        tech_terms = re.findall(r'\b([A-Z]{2,})\b', content)
        for term in set(tech_terms):
            term_lower = term.lower()
            if term_lower not in aliases:
                aliases.append(term_lower)
        
        # Extract important nouns/phrases (capitalized in middle of sentence)
        noun_phrases = re.findall(r'[a-z]\s+([A-Z][a-z]+(?:\s+[a-z]+){0,2})', content)
        for phrase in set(noun_phrases):
            phrase_lower = phrase.lower().strip()
            if len(phrase_lower) > 3 and phrase_lower not in aliases:
                aliases.append(phrase_lower)
        
        return aliases[:10]
    
    def _extract_anti_triggers(self, content: str) -> List[str]:
        """
        Extract anti-triggers - when NOT to use this memory.
        
        Anti-triggers prevent irrelevant memory recall.
        """
        anti_triggers = []
        lower_content = content.lower()
        
        # Extract negations
        for pattern in self._anti_trigger_patterns:
            matches = re.findall(pattern, lower_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, str):
                    anti_trigger = match.strip().lower()
                    if anti_trigger and len(anti_trigger) > 3:
                        anti_triggers.append(anti_trigger)
        
        # Look for explicit "not for" patterns
        not_for_patterns = [
            r'(?:not|don\'t|never)\s+(?:for|use|apply)\s+([^,\n.]+)',
            r'(?:does not|doesn\'t)\s+(?:apply|work)\s+(?:for|with|in)\s+([^,\n.]+)',
        ]
        
        for pattern in not_for_patterns:
            matches = re.findall(pattern, lower_content, re.IGNORECASE)
            for match in matches:
                anti_trigger = match.strip().lower()
                if anti_trigger and len(anti_trigger) > 3:
                    anti_triggers.append(f"not for {anti_trigger}")
        
        return anti_triggers[:5]  # Limit to top 5
    
    # ========================================================================
    # Fact Extraction
    # ========================================================================
    
    async def _extract_facts(self, memory: MemoryNode) -> List[MemoryNode]:
        """
        Extract semantic facts from an episodic memory.
        
        This is a simple rule-based extractor. In production,
        this would use an LLM for sophisticated extraction.
        
        Returns:
            List of extracted fact memories
        """
        extracted = []
        content = memory.content.lower()
        
        # Simple pattern: "I/my [like|prefer|use|work at|live in] X"
        patterns = [
            (r'\bi\s+(like|love|enjoy|prefer)\s+(\w+)', 'preference', 'likes'),
            (r'\bi\s+(use|work\s+with)\s+(\w+)', 'tool', 'uses'),
            (r'\bi\s+(work\s+at|work\s+for)\s+([\w\s]+?)(?:\s|$|\.|:)', 'work', 'works_at'),
            (r'\bi\s+(live\s+in|am\s+from)\s+([\w\s]+?)(?:\s|$|\.|:)', 'location', 'lives_in'),
            (r'\bmy\s+(name\s+is|name\'s)\s+(\w+)', 'name', 'name_is'),
        ]
        
        for pattern, fact_type, predicate in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    value = match[-1].strip()
                else:
                    value = match.strip()
                
                if value:
                    try:
                        fact = await self.store_fact(
                            subject="user",
                            predicate=predicate,
                            object=value,
                            confidence=0.7,
                            source_episode_id=memory.id,
                        )
                        extracted.append(fact)
                    except Exception as e:
                        print(f"Fact extraction failed: {e}")
        
        return extracted
