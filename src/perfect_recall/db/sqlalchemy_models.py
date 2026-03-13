"""SQLAlchemy ORM models mapping to database schema."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer, 
    ForeignKey, JSON, ARRAY, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, TSVECTOR
try:
    from pgvector.sqlalchemy import Vector as VECTOR
except ImportError:
    # Fallback for older pgvector versions
    from sqlalchemy.dialects.postgresql import ARRAY
    # VECTOR will be handled as ARRAY(FLOAT) in fallback
    VECTOR = None
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class SessionORM(Base):
    """ORM model for sessions table."""
    __tablename__ = "sessions"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    user_id = Column(String(100), nullable=False)
    agent_id = Column(String(100))
    
    started_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime(timezone=True))
    
    context_snapshot = Column(JSON, default=dict)
    
    message_count = Column(Integer, default=0)
    token_usage = Column(Integer, default=0)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Relationships
    episodes = relationship("EpisodeORM", back_populates="session", cascade="all, delete-orphan")
    working_memory_slots = relationship("WorkingMemoryORM", back_populates="session", cascade="all, delete-orphan")


class EpisodeORM(Base):
    """ORM model for episodes table."""
    __tablename__ = "episodes"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"))
    
    started_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime(timezone=True))
    
    episode_type = Column(String(50), nullable=False, default="interaction")
    summary = Column(Text)
    summary_embedding = Column(VECTOR(1536) if VECTOR else ARRAY(Float))
    
    participant_ids = Column(ARRAY(String), default=list)
    topic_tags = Column(ARRAY(String), default=list)
    
    parent_episode_id = Column(PGUUID(as_uuid=True), ForeignKey("episodes.id"))
    memory_count = Column(Integer, default=0)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Relationships
    session = relationship("SessionORM", back_populates="episodes")
    memories = relationship("MemoryNodeORM", back_populates="source_episode")


class MemoryNodeORM(Base):
    """ORM model for memory_nodes table."""
    __tablename__ = "memory_nodes"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    memory_tier = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(VECTOR(1536) if VECTOR else ARRAY(Float))
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    valid_from = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    valid_until = Column(DateTime(timezone=True))
    
    importance_score = Column(Float, nullable=False, default=0.5)
    access_count = Column(Integer, nullable=False, default=0)
    last_accessed = Column(DateTime(timezone=True))
    emotional_valence = Column(Float)
    
    confidence = Column(Float, nullable=False, default=1.0)
    source_type = Column(String(20), nullable=False, default="direct")
    source_episode_id = Column(PGUUID(as_uuid=True), ForeignKey("episodes.id", ondelete="SET NULL"))
    
    version = Column(Integer, nullable=False, default=1)
    supersedes_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id"))
    superseded_by_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id"))
    
    # Superpowers-inspired metadata fields
    triggers = Column(ARRAY(String), default=list)
    symptoms = Column(ARRAY(String), default=list)
    aliases = Column(ARRAY(String), default=list)
    anti_triggers = Column(ARRAY(String), default=list)
    
    extra_metadata = Column(JSON, nullable=False, default=dict)
    search_vector = Column(TSVECTOR)
    
    # Relationships
    source_episode = relationship("EpisodeORM", back_populates="memories")
    working_memory_slots = relationship("WorkingMemoryORM", back_populates="memory", cascade="all, delete-orphan")
    
    # Table args for indexes (only if VECTOR is available)
    if VECTOR:
        __table_args__ = (
            Index("idx_memory_embedding", "embedding", postgresql_using="ivfflat", 
                  postgresql_ops={"embedding": "vector_cosine_ops"}, postgresql_with={"lists": 100}),
        )


class WorkingMemoryORM(Base):
    """ORM model for working_memory table."""
    __tablename__ = "working_memory"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    memory_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    
    priority = Column(Float, nullable=False, default=0.5)
    slot_type = Column(String(20), nullable=False, default="context")
    
    added_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
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
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    source_memory_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    target_memory_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    relationship_type = Column(String(50), nullable=False)
    strength = Column(Float, nullable=False, default=1.0)
    
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    extra_metadata = Column(JSON, nullable=False, default=dict)
    
    __table_args__ = (
        UniqueConstraint("source_memory_id", "target_memory_id", "relationship_type"),
    )


class MemoryAccessLogORM(Base):
    """ORM model for memory_access_log table."""
    __tablename__ = "memory_access_log"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    memory_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(PGUUID(as_uuid=True), ForeignKey("sessions.id", ondelete="SET NULL"))
    
    access_type = Column(String(20), nullable=False)
    query_text = Column(Text)
    accessed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class MemoryConflictORM(Base):
    """ORM model for memory_conflicts table."""
    __tablename__ = "memory_conflicts"
    
    id = Column(PGUUID(as_uuid=True), primary_key=True, default="uuid_generate_v4()")
    memory_a_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id"), nullable=False)
    memory_b_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id"), nullable=False)
    
    conflict_type = Column(String(50), nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    resolution_status = Column(String(20), default="open")
    resolution_notes = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        UniqueConstraint("memory_a_id", "memory_b_id"),
    )


class EpisodeMemoryORM(Base):
    """ORM model for episode_memories junction table."""
    __tablename__ = "episode_memories"
    
    episode_id = Column(PGUUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), primary_key=True)
    memory_id = Column(PGUUID(as_uuid=True), ForeignKey("memory_nodes.id", ondelete="CASCADE"), primary_key=True)
    position = Column(Integer, nullable=False, default=0)
