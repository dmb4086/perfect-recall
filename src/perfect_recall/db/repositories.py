"""Repository classes for database operations."""

from typing import List, Optional, Any, Dict
from uuid import UUID

from sqlalchemy import select, and_, or_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from .sqlalchemy_models import (
    MemoryNodeORM, SessionORM, EpisodeORM, WorkingMemoryORM,
    MemoryRelationshipORM, MemoryAccessLogORM
)
from ..models.memory import MemoryNode, Episode, MemoryRelationship, MemoryTier
from ..models.session import Session, WorkingMemorySlot


class MemoryRepository:
    """Repository for memory node operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, memory: MemoryNode) -> MemoryNode:
        """Create a new memory node."""
        orm = MemoryNodeORM(
            id=memory.id,
            memory_tier=memory.memory_tier.value,
            content=memory.content,
            embedding=memory.embedding,
            valid_from=memory.valid_from,
            valid_until=memory.valid_until,
            importance_score=memory.importance_score,
            access_count=memory.access_count,
            last_accessed=memory.last_accessed,
            emotional_valence=memory.emotional_valence,
            confidence=memory.confidence,
            source_type=memory.source_type.value,
            source_episode_id=memory.source_episode_id,
            version=memory.version,
            supersedes_id=memory.supersedes_id,
            superseded_by_id=memory.superseded_by_id,
            metadata=memory.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return memory
    
    async def get_by_id(self, memory_id: UUID) -> Optional[MemoryNode]:
        """Get a memory by ID."""
        result = await self.session.execute(
            select(MemoryNodeORM).where(MemoryNodeORM.id == memory_id)
        )
        orm = result.scalar_one_or_none()
        return self._to_model(orm) if orm else None
    
    async def update(self, memory: MemoryNode) -> MemoryNode:
        """Update a memory node."""
        result = await self.session.execute(
            select(MemoryNodeORM).where(MemoryNodeORM.id == memory.id)
        )
        orm = result.scalar_one_or_none()
        if orm:
            orm.content = memory.content
            orm.embedding = memory.embedding
            orm.importance_score = memory.importance_score
            orm.access_count = memory.access_count
            orm.last_accessed = memory.last_accessed
            orm.valid_until = memory.valid_until
            orm.metadata = memory.metadata
            await self.session.flush()
        return memory
    
    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5,
        memory_tiers: Optional[List[MemoryTier]] = None,
    ) -> List[tuple[MemoryNode, float]]:
        """
        Search for similar memories using vector similarity.
        
        Returns list of (memory, similarity_score) tuples.
        """
        # Use raw SQL with proper vector casting for pgvector
        embedding_str = f"'[{','.join(str(x) for x in embedding)}]'::vector"
        
        sql = f"""
            SELECT 
                id, memory_tier, content, embedding, created_at, valid_from, 
                valid_until, importance_score, access_count, last_accessed,
                emotional_valence, confidence, source_type, source_episode_id,
                version, supersedes_id, superseded_by_id, triggers, symptoms,
                aliases, anti_triggers, extra_metadata,
                1 - (embedding <=> {embedding_str}) as similarity
            FROM memory_nodes
            WHERE embedding IS NOT NULL
              AND 1 - (embedding <=> {embedding_str}) >= :threshold
              AND (valid_until IS NULL OR valid_until > NOW())
        """
        
        params: Dict[str, Any] = {"threshold": threshold}
        
        if memory_tiers:
            tier_values = [t.value for t in memory_tiers]
            placeholders = ", ".join([f":tier_{i}" for i in range(len(tier_values))])
            sql += f" AND memory_tier IN ({placeholders})"
            for i, tier in enumerate(tier_values):
                params[f"tier_{i}"] = tier
        
        sql += f" ORDER BY embedding <=> {embedding_str} LIMIT :limit"
        params["limit"] = limit
        
        result = await self.session.execute(text(sql), params)
        
        memories: List[tuple[MemoryNode, float]] = []
        for row in result:
            memory = MemoryNode(
                id=row.id,
                memory_tier=MemoryTier(row.memory_tier),
                content=row.content,
                embedding=list(row.embedding) if row.embedding else None,
                created_at=row.created_at,
                valid_from=row.valid_from,
                valid_until=row.valid_until,
                importance_score=row.importance_score,
                access_count=row.access_count,
                last_accessed=row.last_accessed,
                emotional_valence=row.emotional_valence,
                confidence=row.confidence,
                source_type=row.source_type,
                source_episode_id=row.source_episode_id,
                version=row.version,
                supersedes_id=row.supersedes_id,
                superseded_by_id=row.superseded_by_id,
                triggers=list(row.triggers) if row.triggers else [],
                symptoms=list(row.symptoms) if row.symptoms else [],
                aliases=list(row.aliases) if row.aliases else [],
                anti_triggers=list(row.anti_triggers) if row.anti_triggers else [],
                metadata=row.extra_metadata or {},
            )
            memories.append((memory, float(row.similarity)))
        
        return memories
    
    async def log_access(
        self,
        memory_id: UUID,
        access_type: str,
        session_id: Optional[UUID] = None,
        query_text: Optional[str] = None,
    ):
        """Log a memory access."""
        log = MemoryAccessLogORM(
            memory_id=memory_id,
            session_id=session_id,
            access_type=access_type,
            query_text=query_text,
        )
        self.session.add(log)
        
        # Update access count
        await self.session.execute(
            text("""
                UPDATE memory_nodes 
                SET access_count = access_count + 1, last_accessed = NOW()
                WHERE id = :memory_id
            """),
            {"memory_id": memory_id}
        )
        await self.session.flush()
    
    def _to_model(self, orm: MemoryNodeORM) -> MemoryNode:
        """Convert ORM to Pydantic model."""
        return MemoryNode(
            id=orm.id,
            memory_tier=MemoryTier(orm.memory_tier),
            content=orm.content,
            embedding=orm.embedding.tolist() if orm.embedding else None,
            created_at=orm.created_at,
            valid_from=orm.valid_from,
            valid_until=orm.valid_until,
            importance_score=orm.importance_score,
            access_count=orm.access_count,
            last_accessed=orm.last_accessed,
            emotional_valence=orm.emotional_valence,
            confidence=orm.confidence,
            source_type=orm.source_type,
            source_episode_id=orm.source_episode_id,
            version=orm.version,
            supersedes_id=orm.supersedes_id,
            superseded_by_id=orm.superseded_by_id,
            metadata=orm.metadata or {},
        )


class SessionRepository:
    """Repository for session operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, session_obj: Session) -> Session:
        """Create a new session."""
        orm = SessionORM(
            id=session_obj.id,
            user_id=session_obj.user_id,
            agent_id=session_obj.agent_id,
            started_at=session_obj.started_at,
            context_snapshot=session_obj.context_snapshot,
            message_count=session_obj.message_count,
            token_usage=session_obj.token_usage,
            metadata=session_obj.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return session_obj
    
    async def get_by_id(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID."""
        result = await self.session.execute(
            select(SessionORM).where(SessionORM.id == session_id)
        )
        orm = result.scalar_one_or_none()
        return self._to_model(orm) if orm else None
    
    async def update(self, session_obj: Session) -> Session:
        """Update a session."""
        result = await self.session.execute(
            select(SessionORM).where(SessionORM.id == session_obj.id)
        )
        orm = result.scalar_one_or_none()
        if orm:
            orm.ended_at = session_obj.ended_at
            orm.context_snapshot = session_obj.context_snapshot
            orm.message_count = session_obj.message_count
            orm.token_usage = session_obj.token_usage
            await self.session.flush()
        return session_obj
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = False,
    ) -> List[Session]:
        """Get sessions for a user."""
        query = select(SessionORM).where(SessionORM.user_id == user_id)
        
        if active_only:
            query = query.where(SessionORM.ended_at.is_(None))
        
        query = query.order_by(desc(SessionORM.started_at)).limit(limit)
        
        result = await self.session.execute(query)
        return [self._to_model(orm) for orm in result.scalars().all()]
    
    async def add_working_memory_slot(self, slot: WorkingMemorySlot) -> WorkingMemorySlot:
        """Add a memory to working memory."""
        orm = WorkingMemoryORM(
            id=slot.id,
            session_id=slot.session_id,
            memory_id=slot.memory_id,
            priority=slot.priority,
            slot_type=slot.slot_type,
            added_at=slot.added_at,
            expires_at=slot.expires_at,
            position=slot.position,
            metadata=slot.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return slot
    
    async def get_working_memory(self, session_id: UUID) -> List[WorkingMemorySlot]:
        """Get all working memory slots for a session."""
        result = await self.session.execute(
            select(WorkingMemoryORM)
            .where(WorkingMemoryORM.session_id == session_id)
            .order_by(WorkingMemoryORM.position)
        )
        return [self._to_slot_model(orm) for orm in result.scalars().all()]
    
    def _to_model(self, orm: SessionORM) -> Session:
        """Convert ORM to Pydantic model."""
        return Session(
            id=orm.id,
            user_id=orm.user_id,
            agent_id=orm.agent_id,
            started_at=orm.started_at,
            ended_at=orm.ended_at,
            context_snapshot=orm.context_snapshot or {},
            message_count=orm.message_count,
            token_usage=orm.token_usage,
            metadata=orm.metadata or {},
            created_at=orm.created_at,
        )
    
    def _to_slot_model(self, orm: WorkingMemoryORM) -> WorkingMemorySlot:
        """Convert WorkingMemoryORM to Pydantic model."""
        return WorkingMemorySlot(
            id=orm.id,
            session_id=orm.session_id,
            memory_id=orm.memory_id,
            priority=orm.priority,
            slot_type=orm.slot_type,
            added_at=orm.added_at,
            expires_at=orm.expires_at,
            position=orm.position,
            metadata=orm.metadata or {},
        )


class EpisodeRepository:
    """Repository for episode operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, episode: Episode) -> Episode:
        """Create a new episode."""
        orm = EpisodeORM(
            id=episode.id,
            session_id=episode.session_id,
            started_at=episode.started_at,
            ended_at=episode.ended_at,
            episode_type=episode.episode_type.value,
            summary=episode.summary,
            summary_embedding=episode.summary_embedding,
            participant_ids=episode.participant_ids,
            topic_tags=episode.topic_tags,
            parent_episode_id=episode.parent_episode_id,
            memory_count=episode.memory_count,
            metadata=episode.metadata,
        )
        self.session.add(orm)
        await self.session.flush()
        return episode
    
    async def get_by_id(self, episode_id: UUID) -> Optional[Episode]:
        """Get an episode by ID."""
        result = await self.session.execute(
            select(EpisodeORM).where(EpisodeORM.id == episode_id)
        )
        orm = result.scalar_one_or_none()
        return self._to_model(orm) if orm else None
    
    async def get_session_episodes(
        self,
        session_id: UUID,
        limit: int = 50,
    ) -> List[Episode]:
        """Get episodes for a session."""
        result = await self.session.execute(
            select(EpisodeORM)
            .where(EpisodeORM.session_id == session_id)
            .order_by(EpisodeORM.started_at)
            .limit(limit)
        )
        return [self._to_model(orm) for orm in result.scalars().all()]
    
    def _to_model(self, orm: EpisodeORM) -> Episode:
        """Convert ORM to Pydantic model."""
        from ..models.memory import EpisodeType
        return Episode(
            id=orm.id,
            session_id=orm.session_id,
            started_at=orm.started_at,
            ended_at=orm.ended_at,
            episode_type=EpisodeType(orm.episode_type),
            summary=orm.summary,
            summary_embedding=orm.summary_embedding.tolist() if orm.summary_embedding else None,
            participant_ids=orm.participant_ids or [],
            topic_tags=orm.topic_tags or [],
            parent_episode_id=orm.parent_episode_id,
            memory_count=orm.memory_count,
            metadata=orm.metadata or {},
            created_at=orm.created_at,
        )
