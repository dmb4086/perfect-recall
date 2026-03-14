"""
Enhanced In-Memory Database for Perfect Recall Testing

Provides SQLite-based in-memory storage with text search fallback.
For development and testing without PostgreSQL.
"""

import os
import re
import hashlib
import numpy as np
from typing import AsyncGenerator, Optional, List, Tuple, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import select, desc, func, text

# Import SQLite-compatible models
from .sqlite_models import Base
from .sqlite_models import SessionORM, EpisodeORM, MemoryNodeORM, WorkingMemoryORM, MemoryAccessLogORM

# Re-export for repositories to use
__all__ = [
    'InMemoryDatabaseManager', 'InMemoryMemoryRepository', 'InMemorySessionRepository',
    'mock_embedding', 'cosine_similarity', 'create_in_memory_perfect_recall',
    # Re-export ORM classes so repositories can use them
    'SessionORM', 'EpisodeORM', 'MemoryNodeORM', 'WorkingMemoryORM', 'MemoryAccessLogORM', 'Base'
]


# ============================================================================
# Mock Vector Operations (for SQLite without pgvector)
# ============================================================================

def mock_embedding(text: str, dim: int = 1536) -> List[float]:
    """Generate deterministic mock embedding for testing."""
    hash_bytes = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
    embedding = np.random.randn(dim).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    a_arr = np.array(a)
    b_arr = np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


# ============================================================================
# In-Memory Database Manager
# ============================================================================

class InMemoryDatabaseManager:
    """
    In-memory SQLite database for testing Perfect Recall.
    
    Features:
    - Full SQLAlchemy ORM compatibility
    - Mock vector similarity search
    - Fast, isolated testing
    """
    
    def __init__(self):
        # Use shared cache for in-memory database to persist across connections
        self.async_url = "sqlite+aiosqlite:///file::memory:?cache=shared"
        self.engine = None
        self.async_session = None
    
    async def initialize(self):
        """Initialize the in-memory database."""
        self.engine = create_async_engine(
            self.async_url,
            echo=False,
            poolclass=NullPool,
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        
        await self._create_tables()
    
    async def _create_tables(self):
        """Create all tables from SQLAlchemy metadata."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """Close the database."""
        if self.engine:
            await self.engine.dispose()
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        if self.async_session is None:
            raise RuntimeError("Database not initialized")
        
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            async with self.session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception:
            return False
    
    def get_memory_repository(self, session: AsyncSession):
        """Get a memory repository for this database."""
        return InMemoryMemoryRepository(session)
    
    def get_session_repository(self, session: AsyncSession):
        """Get a session repository for this database."""
        return InMemorySessionRepository(session)


# ============================================================================
# In-Memory Repositories with Vector Fallback
# ============================================================================

class InMemoryMemoryRepository:
    """Memory repository with in-memory vector similarity."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, memory) -> Any:
        """Create a memory node."""
        from ..models.memory import MemoryTier
        
        orm = MemoryNodeORM(
            id=str(memory.id) if memory.id else str(uuid4()),
            memory_tier=memory.memory_tier.value if isinstance(memory.memory_tier, MemoryTier) else memory.memory_tier,
            content=memory.content,
            embedding=memory.embedding,  # Already a list, will be stored as JSON
            valid_from=memory.valid_from,
            valid_until=memory.valid_until,
            importance_score=memory.importance_score,
            access_count=memory.access_count,
            last_accessed=memory.last_accessed,
            emotional_valence=memory.emotional_valence,
            confidence=memory.confidence,
            source_type=memory.source_type.value if hasattr(memory.source_type, 'value') else memory.source_type,
            source_episode_id=str(memory.source_episode_id) if memory.source_episode_id else None,
            version=memory.version,
            supersedes_id=str(memory.supersedes_id) if memory.supersedes_id else None,
            superseded_by_id=str(memory.superseded_by_id) if memory.superseded_by_id else None,
            triggers=memory.triggers,  # Stored as JSON
            symptoms=memory.symptoms,  # Stored as JSON
            aliases=memory.aliases,  # Stored as JSON
            anti_triggers=memory.anti_triggers,  # Stored as JSON
            extra_metadata=memory.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return memory
    
    async def get_by_id(self, memory_id: UUID) -> Optional[Any]:
        """Get memory by ID."""
        from ..models.memory import MemoryNode, MemoryTier, SourceType
        
        result = await self.session.execute(
            select(MemoryNodeORM).where(MemoryNodeORM.id == str(memory_id))
        )
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        
        return self._to_model(orm)
    
    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5,
        memory_tiers: Optional[List] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Search for similar memories using cosine similarity.
        
        Falls back to text search for memories without embeddings.
        """
        from ..models.memory import MemoryNode, MemoryTier
        
        now = datetime.now(timezone.utc)
        
        # Get all active memories
        query = select(MemoryNodeORM).where(
            (MemoryNodeORM.valid_until == None) | (MemoryNodeORM.valid_until > now)
        )
        
        if memory_tiers:
            tier_values = [t.value if hasattr(t, 'value') else t for t in memory_tiers]
            query = query.where(MemoryNodeORM.memory_tier.in_(tier_values))
        
        result = await self.session.execute(query)
        memories = result.scalars().all()
        
        # Calculate similarity for each
        scored = []
        for orm in memories:
            if orm.embedding:
                sim = cosine_similarity(embedding, orm.embedding)
            else:
                # Fallback: neutral similarity
                sim = 0.5
            
            if sim >= threshold:
                scored.append((self._to_model(orm), sim))
        
        # If no results from similarity search, fall back to recent memories
        if not scored:
            from sqlalchemy import desc
            recent_query = select(MemoryNodeORM).order_by(desc(MemoryNodeORM.created_at)).limit(limit)
            if memory_tiers:
                tier_values = [t.value if hasattr(t, 'value') else t for t in memory_tiers]
                recent_query = recent_query.where(MemoryNodeORM.memory_tier.in_(tier_values))
            
            result = await self.session.execute(recent_query)
            recent_memories = result.scalars().all()
            scored = [(self._to_model(orm), 0.5) for orm in recent_memories]
        
        # Sort by similarity and limit
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
    
    async def search_by_text(self, query_text: str, limit: int = 10) -> List[Any]:
        """Simple text search for memories."""
        # Simple case-insensitive substring search
        pattern = f"%{query_text.lower()}%"
        result = await self.session.execute(
            select(MemoryNodeORM)
            .where(func.lower(MemoryNodeORM.content).like(pattern))
            .limit(limit)
        )
        return [self._to_model(orm) for orm in result.scalars().all()]
    
    async def get_recent_memories(
        self,
        limit: int = 10,
        memory_tiers: Optional[List] = None,
    ) -> List[Any]:
        """Get recent memories ordered by creation time."""
        from sqlalchemy import desc
        
        query = select(MemoryNodeORM).order_by(desc(MemoryNodeORM.created_at)).limit(limit)
        
        if memory_tiers:
            tier_values = [t.value if hasattr(t, 'value') else t for t in memory_tiers]
            query = query.where(MemoryNodeORM.memory_tier.in_(tier_values))
        
        result = await self.session.execute(query)
        return [self._to_model(orm) for orm in result.scalars().all()]
    
    async def log_access(
        self,
        memory_id: UUID,
        access_type: str,
        session_id: Optional[UUID] = None,
        query_text: Optional[str] = None,
    ):
        """Log a memory access."""
        log = MemoryAccessLogORM(
            memory_id=str(memory_id),
            session_id=str(session_id) if session_id else None,
            access_type=access_type,
            query_text=query_text,
        )
        self.session.add(log)
        
        # Update access count
        result = await self.session.execute(
            select(MemoryNodeORM).where(MemoryNodeORM.id == str(memory_id))
        )
        memory = result.scalar_one_or_none()
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.now(timezone.utc)
        
        await self.session.flush()
    
    def _to_model(self, orm):
        """Convert ORM to Pydantic model."""
        from ..models.memory import MemoryNode, MemoryTier, SourceType
        
        return MemoryNode(
            id=UUID(orm.id),
            memory_tier=MemoryTier(orm.memory_tier),
            content=orm.content,
            embedding=orm.embedding,
            created_at=orm.created_at,
            valid_from=orm.valid_from,
            valid_until=orm.valid_until,
            importance_score=orm.importance_score,
            access_count=orm.access_count,
            last_accessed=orm.last_accessed,
            emotional_valence=orm.emotional_valence,
            confidence=orm.confidence,
            source_type=SourceType(orm.source_type),
            source_episode_id=UUID(orm.source_episode_id) if orm.source_episode_id else None,
            version=orm.version,
            supersedes_id=UUID(orm.supersedes_id) if orm.supersedes_id else None,
            superseded_by_id=UUID(orm.superseded_by_id) if orm.superseded_by_id else None,
            triggers=list(orm.triggers) if orm.triggers else [],
            symptoms=list(orm.symptoms) if orm.symptoms else [],
            aliases=list(orm.aliases) if orm.aliases else [],
            anti_triggers=list(orm.anti_triggers) if orm.anti_triggers else [],
            metadata=orm.extra_metadata or {},
        )


class InMemorySessionRepository:
    """Session repository for in-memory database."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, session_obj) -> Any:
        """Create a session."""
        orm = SessionORM(
            id=str(session_obj.id) if session_obj.id else str(uuid4()),
            user_id=session_obj.user_id,
            agent_id=session_obj.agent_id,
            started_at=session_obj.started_at,
            context_snapshot=session_obj.context_snapshot,
            message_count=session_obj.message_count,
            token_usage=session_obj.token_usage,
            extra_metadata=session_obj.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return session_obj
    
    async def get_by_id(self, session_id: UUID) -> Optional[Any]:
        """Get session by ID."""
        from ..models.session import Session
        
        result = await self.session.execute(
            select(SessionORM).where(SessionORM.id == str(session_id))
        )
        orm = result.scalar_one_or_none()
        if not orm:
            return None
        
        return Session(
            id=UUID(orm.id),
            user_id=orm.user_id,
            agent_id=orm.agent_id,
            started_at=orm.started_at,
            ended_at=orm.ended_at,
            context_snapshot=orm.context_snapshot or {},
            message_count=orm.message_count,
            token_usage=orm.token_usage,
            metadata=orm.extra_metadata or {},
            created_at=orm.created_at,
        )
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = False,
    ) -> List[Any]:
        """Get sessions for a user."""
        from ..models.session import Session
        
        query = select(SessionORM).where(SessionORM.user_id == user_id)
        
        if active_only:
            query = query.where(SessionORM.ended_at == None)
        
        query = query.order_by(desc(SessionORM.started_at)).limit(limit)
        
        result = await self.session.execute(query)
        return [
            Session(
                id=UUID(orm.id),
                user_id=orm.user_id,
                agent_id=orm.agent_id,
                started_at=orm.started_at,
                ended_at=orm.ended_at,
                context_snapshot=orm.context_snapshot or {},
                message_count=orm.message_count,
                token_usage=orm.token_usage,
                metadata=orm.extra_metadata or {},
                created_at=orm.created_at,
            )
            for orm in result.scalars().all()
        ]
    
    async def add_working_memory_slot(self, slot) -> Any:
        """Add a working memory slot."""
        orm = WorkingMemoryORM(
            id=str(slot.id) if slot.id else str(uuid4()),
            session_id=str(slot.session_id),
            memory_id=str(slot.memory_id),
            priority=slot.priority,
            slot_type=slot.slot_type,
            added_at=slot.added_at,
            expires_at=slot.expires_at,
            position=slot.position,
            extra_metadata=slot.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return slot
    
    async def get_working_memory(self, session_id: UUID) -> List[Any]:
        """Get working memory slots for a session."""
        from ..models.session import WorkingMemorySlot
        
        result = await self.session.execute(
            select(WorkingMemoryORM)
            .where(WorkingMemoryORM.session_id == str(session_id))
            .order_by(WorkingMemoryORM.position)
        )
        
        return [
            WorkingMemorySlot(
                id=UUID(orm.id),
                session_id=UUID(orm.session_id),
                memory_id=UUID(orm.memory_id),
                priority=orm.priority,
                slot_type=orm.slot_type,
                added_at=orm.added_at,
                expires_at=orm.expires_at,
                position=orm.position,
                metadata=orm.extra_metadata or {},
            )
            for orm in result.scalars().all()
        ]


# ============================================================================
# Factory Functions
# ============================================================================

async def create_in_memory_perfect_recall() -> Any:
    """
    Create a PerfectRecall instance with in-memory database.
    
    Usage:
        pr = await create_in_memory_perfect_recall()
        # Use pr normally for testing
    """
    from ..core.perfect_recall import PerfectRecall
    from ..core.memory_writer import MemoryWriter
    from ..core.session_manager import SessionManager
    from ..core.retrieval import RetrievalPipeline, SalienceScorer
    
    # Create in-memory database
    db_manager = InMemoryDatabaseManager()
    await db_manager.initialize()
    
    # Verify tables were created
    async with db_manager.session() as session:
        result = await session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]
        if 'sessions' not in tables:
            # Force create tables if not present
            await db_manager._create_tables()
    
    # Create components with mock embedding
    writer = MemoryWriter(
        db_manager=db_manager,
        embedding_func=mock_embedding,
    )
    
    session_manager = SessionManager(db_manager)
    
    retrieval = RetrievalPipeline(
        db_manager=db_manager,
        embedding_func=mock_embedding,
        scorer=SalienceScorer(),
    )
    
    return PerfectRecall(
        db_manager=db_manager,
        memory_writer=writer,
        session_manager=session_manager,
        retrieval_pipeline=retrieval,
    )
