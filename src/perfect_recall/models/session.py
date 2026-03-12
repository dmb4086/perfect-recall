"""Session and working memory models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


class Session(BaseModel):
    """
    A session represents a continuous interaction context.
    
    Sessions provide the container for working memory and
    track the lifecycle of agent-user interactions.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    agent_id: Optional[str] = None
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    # Working memory snapshot (for resumption)
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics
    message_count: int = Field(default=0)
    token_usage: int = Field(default=0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def is_active(self) -> bool:
        """Check if session is still active."""
        return self.ended_at is None
    
    def duration_seconds(self) -> Optional[float]:
        """Get session duration in seconds."""
        end = self.ended_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()
    
    def to_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of session state for persistence."""
        return {
            "session_id": str(self.id),
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "ended_at": datetime.utcnow().isoformat(),
            "message_count": self.message_count,
            "token_usage": self.token_usage,
        }


class WorkingMemorySlot(BaseModel):
    """
    A slot in working memory linking a session to an active memory.
    
    Working memory is limited and holds only the most relevant
    memories for the current context.
    """
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    memory_id: UUID
    
    # Slot metadata
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    slot_type: str = Field(default="context")  # 'context', 'goal', 'task', 'scratchpad'
    
    # Temporal bounds in working memory
    added_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Position in the working memory stack
    position: int = Field(default=0)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this slot has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class SessionManagerState(BaseModel):
    """Internal state tracking for session management."""
    model_config = ConfigDict(from_attributes=True)
    
    active_sessions: Dict[str, Session] = Field(default_factory=dict)
    working_memory_cache: Dict[str, List[WorkingMemorySlot]] = Field(default_factory=dict)
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an active session by ID."""
        return self.active_sessions.get(session_id)
    
    def add_session(self, session: Session):
        """Add a session to active sessions."""
        self.active_sessions[str(session.id)] = session
        self.working_memory_cache[str(session.id)] = []
    
    def end_session(self, session_id: str) -> Optional[Session]:
        """End a session and remove from active."""
        session = self.active_sessions.pop(session_id, None)
        if session:
            session.ended_at = datetime.utcnow()
            self.working_memory_cache.pop(session_id, None)
        return session
