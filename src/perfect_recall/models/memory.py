"""Memory node models for the four-tier memory system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class MemoryTier(str, Enum):
    """The four tiers of the memory continuum."""
    WORKING = "working"      # Active context (conscious awareness)
    EPISODIC = "episodic"    # Event sequences (autobiographical memory)
    SEMANTIC = "semantic"    # Facts & knowledge (general knowledge)
    PROCEDURAL = "procedural" # Skills & patterns (muscle memory)


class EpisodeType(str, Enum):
    """Types of episodic memories."""
    MESSAGE = "message"
    ACTION = "action"
    DECISION = "decision"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    INTERACTION = "interaction"


class SourceType(str, Enum):
    """Provenance of memory content."""
    DIRECT = "direct"
    INFERRED = "inferred"
    IMPORTED = "imported"
    CORRECTED = "corrected"
    COMPOUND = "compound"


class MemoryNode(BaseModel):
    """
    A single memory node in the Perfect Recall system.
    
    Memory nodes form the foundation of all four memory tiers,
    with tier-specific metadata stored in the metadata JSONB field.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(default_factory=uuid4)
    memory_tier: MemoryTier
    content: str
    
    # Embedding for vector similarity search
    embedding: Optional[List[float]] = None
    
    # Temporal metadata (bi-temporal model)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None
    
    # Salience metadata
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    emotional_valence: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    
    # Provenance
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_type: SourceType = SourceType.DIRECT
    source_episode_id: Optional[UUID] = None
    
    # Versioning
    version: int = Field(default=1)
    supersedes_id: Optional[UUID] = None
    superseded_by_id: Optional[UUID] = None
    
    # Flexible metadata for tier-specific data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if this memory is still valid."""
        if self.valid_until is None:
            return True
        return datetime.utcnow() < self.valid_until
    
    def calculate_salience(self) -> float:
        """
        Calculate salience score for this memory.
        Higher = more relevant/important.
        """
        import math
        
        # Importance component (40%)
        importance = self.importance_score * 0.4
        
        # Frequency component (30%) - saturates at 20 accesses
        frequency = min(self.access_count / 20.0, 1.0) * 0.3
        
        # Recency component (30%) - exponential decay with 1 week half-life
        if self.last_accessed is None:
            recency = 0.1
        else:
            age_hours = (datetime.utcnow() - self.last_accessed).total_seconds() / 3600
            recency = math.exp(-age_hours / 168) * 0.3  # 168 hours = 1 week
        
        return importance + frequency + recency
    
    def to_context_string(self) -> str:
        """Format this memory as a context string for LLM injection."""
        tier_prefix = {
            MemoryTier.WORKING: "[Current Context]",
            MemoryTier.EPISODIC: "[Previous Interaction]",
            MemoryTier.SEMANTIC: "[Known Fact]",
            MemoryTier.PROCEDURAL: "[Known Pattern]"
        }
        prefix = tier_prefix.get(self.memory_tier, "[Memory]")
        return f"{prefix} {self.content}"


class MemoryRelationship(BaseModel):
    """Relationship between two memory nodes (lightweight graph)."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(default_factory=uuid4)
    source_memory_id: UUID
    target_memory_id: UUID
    relationship_type: str  # 'precedes', 'causes', 'relates_to', 'supersedes', 'mentions'
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    """An episode groups related memories temporally."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(default_factory=uuid4)
    session_id: Optional[UUID] = None
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    episode_type: EpisodeType = EpisodeType.INTERACTION
    summary: Optional[str] = None
    summary_embedding: Optional[List[float]] = None
    
    participant_ids: List[str] = Field(default_factory=list)
    topic_tags: List[str] = Field(default_factory=list)
    
    parent_episode_id: Optional[UUID] = None
    memory_count: int = Field(default=0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProceduralPattern(BaseModel):
    """Extended metadata for procedural memory patterns."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(default_factory=uuid4)
    memory_id: UUID
    
    pattern_name: Optional[str] = None
    trigger_patterns: List[str] = Field(default_factory=list)
    
    success_count: int = Field(default=0)
    failure_count: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    
    avg_execution_time_ms: Optional[float] = None
    last_executed_at: Optional[datetime] = None
    
    applicable_contexts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def record_execution(self, success: bool, execution_time_ms: Optional[float] = None):
        """Record an execution of this pattern."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        
        total = self.success_count + self.failure_count
        self.success_rate = self.success_count / total if total > 0 else 0.0
        
        if execution_time_ms is not None:
            if self.avg_execution_time_ms is None:
                self.avg_execution_time_ms = execution_time_ms
            else:
                # Running average
                self.avg_execution_time_ms = (
                    (self.avg_execution_time_ms * (total - 1) + execution_time_ms) / total
                )
        
        self.last_executed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
