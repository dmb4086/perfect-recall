"""Retrieval-related models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from .memory import MemoryNode


class RetrievalContext(BaseModel):
    """Context for memory retrieval operations."""
    model_config = ConfigDict(from_attributes=True)
    
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Temporal constraints
    temporal_at: Optional[datetime] = None  # Query as-of this time
    temporal_from: Optional[datetime] = None
    temporal_to: Optional[datetime] = None
    
    # Context filters
    active_topics: List[str] = Field(default_factory=list)
    participant_ids: List[str] = Field(default_factory=list)
    
    # Tier preferences
    include_tiers: List[str] = Field(default_factory=lambda: ["working", "episodic", "semantic", "procedural"])
    exclude_tiers: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedMemory(BaseModel):
    """A memory with its retrieval score and metadata."""
    model_config = ConfigDict(from_attributes=True)
    
    memory: MemoryNode
    
    # Scoring
    salience_score: float = Field(default=0.0)
    semantic_similarity: float = Field(default=0.0)
    
    # Component scores (for debugging/analysis)
    score_components: Dict[str, float] = Field(default_factory=dict)
    
    # Retrieval metadata
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    query_text: Optional[str] = None
    
    def to_context_string(self, include_score: bool = False) -> str:
        """Format for LLM context injection."""
        base = self.memory.to_context_string()
        if include_score:
            return f"{base} (relevance: {self.salience_score:.2f})"
        return base


class WriteDecision(BaseModel):
    """Decision from the write gate on whether to store a memory."""
    model_config = ConfigDict(from_attributes=True)
    
    write: bool
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    factors: Dict[str, float] = Field(default_factory=dict)
    reason: Optional[str] = None
    
    def __bool__(self):
        return self.write


class ConflictInfo(BaseModel):
    """Information about a memory conflict."""
    model_config = ConfigDict(from_attributes=True)
    
    memory_a_id: UUID
    memory_b_id: UUID
    conflict_type: str  # 'contradiction', 'temporal', 'exclusive'
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    explanation: Optional[str] = None
