"""SQLite-compatible ORM models for in-memory testing.

These models use JSON instead of ARRAY for list storage, which is
compatible with SQLite. PostgreSQL models use ARRAY for better type safety.
"""

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, 
    ForeignKey, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


def utc_now():
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class SessionORM(Base):
    """ORM model for sessions table."""
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String(100), nullable=False)
    agent_id = Column(String(100))
    
    started_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    ended_at = Column(DateTime(timezone=True))
    
    context_snapshot = Column(JSON, default=dict)
    
    message_count = Column(Integer, default=0)
    token_usage = Column(Integer, default=0)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    
    # Relationships
    episodes = relationship("EpisodeORM", back_populates="session", cascade="all, delete-orphan")
    working_memory_slots = relationship("WorkingMemoryORM", back_populates="session", cascade="all, delete-orphan")


class EpisodeORM(Base):
    """ORM model for episodes table."""
    __tablename__ = "episodes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"))
    
    started_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    ended_at = Column(DateTime(timezone=True))
    
    episode_type = Column(String(50), nullable=False, default="interaction")
    summary = Column(Text)
    summary_embedding = Column(JSON)  # Stored as JSON list for SQLite
    
    participant_ids = Column(JSON, default=list)  # Stored as JSON list
    topic_tags = Column(JSON, default=list)  # Stored as JSON list
    
    parent_episode_id = Column(String(36), ForeignKey("episodes.id"))
    memory_count = Column(Integer, default=0)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    
    # Relationships
    session = relationship("SessionORM", back_populates="episodes")
    memories = relationship("MemoryNodeORM", back_populates="source_episode")


class MemoryNodeORM(Base):
    """ORM model for memory_nodes table."""
    __tablename__ = "memory_nodes"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    memory_tier = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSON)  # Stored as JSON list for SQLite
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    valid_from = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    valid_until = Column(DateTime(timezone=True))
    
    importance_score = Column(Float, nullable=False, default=0.5)
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime(timezone=True))
    emotional_valence = Column(Float)
    
    confidence = Column(Float, nullable=False, default=1.0)
    source_type = Column(String(20), nullable=False, default="direct")
    source_episode_id = Column(String(36), ForeignKey("episodes.id", ondelete="SET NULL"))
    
    version = Column(Integer, nullable=False, default=1)
    supersedes_id = Column(String(36), ForeignKey("memory_nodes.id"))
    superseded_by_id = Column(String(36), ForeignKey("memory_nodes.id"))
    
    # Superpowers-inspired metadata fields - stored as JSON for SQLite
    triggers = Column(JSON, default=list)
    symptoms = Column(JSON, default=list)
    aliases = Column(JSON, default=list)
    anti_triggers = Column(JSON, default=list)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    source_episode = relationship("EpisodeORM", back_populates="memories")
    working_memory_slots = relationship("WorkingMemoryORM", back_populates="memory", cascade="all, delete-orphan")


class WorkingMemoryORM(Base):
    """ORM model for working_memory table."""
    __tablename__ = "working_memory"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    memory_id = Column(String(36), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    
    priority = Column(Float, nullable=False, default=0.5)
    slot_type = Column(String(20), nullable=False, default="context")
    
    added_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    expires_at = Column(DateTime(timezone=True))
    position = Column(Integer, nullable=False, default=0)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    
    # Relationships
    session = relationship("SessionORM", back_populates="working_memory_slots")
    memory = relationship("MemoryNodeORM", back_populates="working_memory_slots")
    
    __table_args__ = (
        UniqueConstraint("session_id", "memory_id"),
    )


class MemoryRelationshipORM(Base):
    """ORM model for memory_relationships table."""
    __tablename__ = "memory_relationships"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    source_memory_id = Column(String(36), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    target_memory_id = Column(String(36), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    strength = Column(Float, nullable=False, default=1.0)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    extra_metadata = Column(JSON, nullable=False, default=dict)
    
    __table_args__ = (
        UniqueConstraint("source_memory_id", "target_memory_id", "relationship_type"),
    )


class MemoryAccessLogORM(Base):
    """ORM model for memory_access_log table."""
    __tablename__ = "memory_access_log"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    memory_id = Column(String(36), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="SET NULL"))
    
    access_type = Column(String(20), nullable=False)
    query_text = Column(Text)
    accessed_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class MemoryConflictORM(Base):
    """ORM model for memory_conflicts table."""
    __tablename__ = "memory_conflicts"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    memory_a_id = Column(String(36), ForeignKey("memory_nodes.id"), nullable=False)
    memory_b_id = Column(String(36), ForeignKey("memory_nodes.id"), nullable=False)
    
    conflict_type = Column(String(50), nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    
    resolution_status = Column(String(20), default="open")
    resolution_notes = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        UniqueConstraint("memory_a_id", "memory_b_id"),
    )


class EpisodeMemoryORM(Base):
    """ORM model for episode_memories junction table."""
    __tablename__ = "episode_memories"
    
    episode_id = Column(String(36), ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True)
    memory_id = Column(String(36), ForeignKey("memory_nodes.id", ondelete="CASCADE"), primary_key=True)
    position = Column(Integer, nullable=False, default=0)
