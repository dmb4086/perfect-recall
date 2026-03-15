"""Session management for Perfect Recall."""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from uuid import UUID

from ..models.session import Session, WorkingMemorySlot, SessionManagerState
from ..models.memory import MemoryNode
from ..db.connection import DatabaseManager
from ..db.repositories import SessionRepository, MemoryRepository


class SessionManager:
    """
    Manages session lifecycle and working memory.
    
    Sessions provide:
    - Container for working memory
    - Context tracking across interactions
    - Session resumption from snapshots
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._state = SessionManagerState()
    
    async def start_session(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        resume_from: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Start a new session, optionally resuming from a previous one.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            resume_from: Previous session ID to resume from
            metadata: Additional session metadata
            
        Returns:
            New session object
        """
        # Create new session
        session = Session(
            user_id=user_id,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        
        # Store in database
        async with self.db_manager.session() as db_session:
            if hasattr(self.db_manager, 'get_session_repository'):
                repo = self.db_manager.get_session_repository(db_session)
            else:
                repo = SessionRepository(db_session)
            await repo.create(session)
        
        # If resuming, hydrate working memory from previous session
        if resume_from:
            await self._hydrate_working_memory(session, resume_from)
        
        # Add to active sessions
        self._state.add_session(session)
        
        return session
    
    async def end_session(
        self,
        session_id: UUID,
        save_snapshot: bool = True,
    ) -> Optional[Session]:
        """
        End a session and optionally save a snapshot.
        
        Args:
            session_id: Session to end
            save_snapshot: Whether to save working memory snapshot
            
        Returns:
            Ended session or None if not found
        """
        session = self._state.end_session(str(session_id))
        
        if session and save_snapshot:
            # Get current working memory
            working_memories = await self.get_working_memory(session_id)
            
            # Create snapshot
            snapshot = {
                'working_memory_ids': [str(m.id) for m in working_memories],
                'working_memory_count': len(working_memories),
                'ended_at': datetime.now(timezone.utc).isoformat(),
            }
            
            session.context_snapshot = snapshot
        
        if session:
            async with self.db_manager.session() as db_session:
                if hasattr(self.db_manager, 'get_session_repository'):
                    repo = self.db_manager.get_session_repository(db_session)
                else:
                    repo = SessionRepository(db_session)
                await repo.update(session)
        
        return session
    
    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """
        Get a session by ID.
        
        Checks active cache first, then database.
        """
        # Check cache
        cached = self._state.get_session(str(session_id))
        if cached:
            return cached
        
        # Fetch from database
        async with self.db_manager.session() as db_session:
            if hasattr(self.db_manager, 'get_session_repository'):
                repo = self.db_manager.get_session_repository(db_session)
            else:
                repo = SessionRepository(db_session)
            return await repo.get_by_id(session_id)
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = False,
    ) -> List[Session]:
        """Get sessions for a user."""
        async with self.db_manager.session() as db_session:
            if hasattr(self.db_manager, 'get_session_repository'):
                repo = self.db_manager.get_session_repository(db_session)
            else:
                repo = SessionRepository(db_session)
            return await repo.get_user_sessions(user_id, limit, active_only)
    
    async def add_to_working_memory(
        self,
        session_id: UUID,
        memory: MemoryNode,
        slot_type: str = "context",
        priority: float = 0.5,
    ) -> WorkingMemorySlot:
        """
        Add a memory to a session's working memory.
        
        Args:
            session_id: Session to add to
            memory: Memory to add
            slot_type: Type of working memory slot
            priority: Priority for this slot
            
        Returns:
            Created working memory slot
        """
        # Get current position
        current = await self.get_working_memory(session_id)
        position = len(current)
        
        slot = WorkingMemorySlot(
            session_id=session_id,
            memory_id=memory.id,
            priority=priority,
            slot_type=slot_type,
            position=position,
        )
        
        async with self.db_manager.session() as db_session:
            if hasattr(self.db_manager, 'get_session_repository'):
                repo = self.db_manager.get_session_repository(db_session)
            else:
                repo = SessionRepository(db_session)
            await repo.add_working_memory_slot(slot)
        
        # Update cache
        if str(session_id) in self._state.working_memory_cache:
            self._state.working_memory_cache[str(session_id)].append(slot)
        
        return slot
    
    async def get_working_memory(
        self,
        session_id: UUID,
        include_expired: bool = False,
    ) -> List[MemoryNode]:
        """
        Get all memories in a session's working memory.
        
        Args:
            session_id: Session to get working memory for
            include_expired: Whether to include expired slots
            
        Returns:
            List of memory nodes in working memory
        """
        async with self.db_manager.session() as db_session:
            if hasattr(self.db_manager, 'get_session_repository'):
                session_repo = self.db_manager.get_session_repository(db_session)
            else:
                session_repo = SessionRepository(db_session)
            if hasattr(self.db_manager, 'get_memory_repository'):
                memory_repo = self.db_manager.get_memory_repository(db_session)
            else:
                memory_repo = MemoryRepository(db_session)
            
            # Get working memory slots
            slots = await session_repo.get_working_memory(session_id)
            
            # Filter expired and fetch memories
            memories = []
            for slot in slots:
                if not include_expired and slot.is_expired():
                    continue
                
                memory = await memory_repo.get_by_id(slot.memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
    
    async def clear_working_memory(self, session_id: UUID) -> int:
        """
        Clear all working memory for a session.
        
        Returns:
            Number of slots cleared
        """
        from sqlalchemy import delete
        from ..db.sqlalchemy_models import WorkingMemoryORM
        
        async with self.db_manager.session() as db_session:
            result = await db_session.execute(
                delete(WorkingMemoryORM).where(WorkingMemoryORM.session_id == session_id)
            )
            await db_session.flush()
            
            # Clear cache
            if str(session_id) in self._state.working_memory_cache:
                self._state.working_memory_cache[str(session_id)] = []
            
            return result.rowcount
    
    async def update_session_metrics(
        self,
        session_id: UUID,
        message_count: Optional[int] = None,
        token_usage: Optional[int] = None,
    ):
        """Update session metrics."""
        session = await self.get_session(session_id)
        if session:
            if message_count is not None:
                session.message_count = message_count
            if token_usage is not None:
                session.token_usage = token_usage
            
            async with self.db_manager.session() as db_session:
                if hasattr(self.db_manager, 'get_session_repository'):
                    repo = self.db_manager.get_session_repository(db_session)
                else:
                    repo = SessionRepository(db_session)
                await repo.update(session)
    
    async def _hydrate_working_memory(
        self,
        new_session: Session,
        previous_session_id: UUID,
    ):
        """
        Hydrate working memory from a previous session.
        
        Loads the most salient memories from the previous session's
        working memory into the new session.
        """
        # Get previous session
        async with self.db_manager.session() as db_session:
            if hasattr(self.db_manager, 'get_session_repository'):
                repo = self.db_manager.get_session_repository(db_session)
            else:
                repo = SessionRepository(db_session)
            prev_session = await repo.get_by_id(previous_session_id)
        
        if not prev_session or not prev_session.context_snapshot:
            return
        
        # Get working memory IDs from snapshot
        memory_ids = prev_session.context_snapshot.get('working_memory_ids', [])
        
        # Load and copy most salient memories
        async with self.db_manager.session() as db_session:
            memory_repo = MemoryRepository(db_session)
            
            loaded_memories = []
            for memory_id_str in memory_ids[:5]:  # Top 5
                try:
                    from uuid import UUID
                    memory = await memory_repo.get_by_id(UUID(memory_id_str))
                    if memory and memory.is_active():
                        loaded_memories.append(memory)
                except Exception:
                    continue
            
            # Add to new session's working memory
            session_repo = SessionRepository(db_session)
            for i, memory in enumerate(loaded_memories):
                slot = WorkingMemorySlot(
                    session_id=new_session.id,
                    memory_id=memory.id,
                    priority=memory.importance_score,
                    slot_type="context",
                    position=i,
                )
                await session_repo.add_working_memory_slot(slot)
