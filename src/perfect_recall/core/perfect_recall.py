"""
Perfect Recall - Main API

The unified interface for the four-tier memory system.
"""

from typing import List, Optional, Dict, Any, Callable
from uuid import UUID

from ..models.memory import MemoryNode, MemoryTier, EpisodeType
from ..models.session import Session
from ..models.retrieval import RetrievedMemory, RetrievalContext
from ..db.connection import DatabaseManager
from .memory_writer import MemoryWriter
from .session_manager import SessionManager
from .retrieval import RetrievalPipeline


class PerfectRecall:
    """
    Main API for Perfect Recall memory system.
    
    Provides a unified interface for:
    - Session management
    - Memory writing (all four tiers)
    - Memory retrieval
    - Context formatting
    
    Example:
        ```python
        pr = await PerfectRecall.create()
        
        # Start session
        session = await pr.start_session(user_id="user_123")
        
        # Record interaction
        await pr.record_episode(
            content="User: I need help with Python",
            episode_type=EpisodeType.MESSAGE,
            session_id=session.id
        )
        
        # Retrieve memories
        memories = await pr.recall("What does the user need help with?")
        ```
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        memory_writer: MemoryWriter,
        session_manager: SessionManager,
        retrieval_pipeline: RetrievalPipeline,
    ):
        self.db = db_manager
        self.writer = memory_writer
        self.sessions = session_manager
        self.retrieval = retrieval_pipeline
    
    @classmethod
    async def create(
        cls,
        database_url: Optional[str] = None,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
    ) -> "PerfectRecall":
        """
        Create and initialize a PerfectRecall instance.
        
        Args:
            database_url: PostgreSQL connection URL
            embedding_func: Function to generate embeddings from text
            
        Returns:
            Initialized PerfectRecall instance
        """
        # Initialize database
        db_manager = DatabaseManager(database_url)
        await db_manager.initialize()
        
        # Create components
        writer = MemoryWriter(
            db_manager=db_manager,
            embedding_func=embedding_func,
        )
        
        session_manager = SessionManager(db_manager)
        
        retrieval = RetrievalPipeline(
            db_manager=db_manager,
            embedding_func=embedding_func,
        )
        
        return cls(
            db_manager=db_manager,
            memory_writer=writer,
            session_manager=session_manager,
            retrieval_pipeline=retrieval,
        )
    
    async def close(self):
        """Close all connections."""
        await self.db.close()
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    async def start_session(
        self,
        user_id: str,
        agent_id: Optional[str] = None,
        resume_from: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Start a new session.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            resume_from: Previous session ID to resume from
            metadata: Additional session metadata
            
        Returns:
            New session
        """
        return await self.sessions.start_session(
            user_id=user_id,
            agent_id=agent_id,
            resume_from=resume_from,
            metadata=metadata,
        )
    
    async def end_session(
        self,
        session_id: UUID,
        save_snapshot: bool = True,
    ) -> Optional[Session]:
        """
        End a session.
        
        Args:
            session_id: Session to end
            save_snapshot: Whether to save working memory snapshot
            
        Returns:
            Ended session or None if not found
        """
        return await self.sessions.end_session(session_id, save_snapshot)
    
    async def get_session(self, session_id: UUID) -> Optional[Session]:
        """Get a session by ID."""
        return await self.sessions.get_session(session_id)
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
        active_only: bool = False,
    ) -> List[Session]:
        """Get sessions for a user."""
        return await self.sessions.get_user_sessions(user_id, limit, active_only)
    
    # ========================================================================
    # Memory Writing
    # ========================================================================
    
    async def record_episode(
        self,
        content: str,
        episode_type: EpisodeType,
        session_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[MemoryNode]:
        """
        Record an episode to episodic memory.
        
        Runs through write gate for intelligent filtering.
        
        Args:
            content: Episode content
            episode_type: Type of episode
            session_id: Associated session
            metadata: Additional metadata
            
        Returns:
            Memory node if stored, None if rejected
        """
        return await self.writer.record_episode(
            content=content,
            episode_type=episode_type,
            session_id=session_id,
            metadata=metadata,
        )
    
    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        source_episode_id: Optional[UUID] = None,
    ) -> MemoryNode:
        """
        Store a semantic fact.
        
        Args:
            subject: Fact subject
            predicate: Fact predicate
            object: Fact object
            confidence: Confidence level (0-1)
            source_episode_id: Source episode if extracted
            
        Returns:
            Created memory node
        """
        return await self.writer.store_fact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=confidence,
            source_episode_id=source_episode_id,
        )
    
    async def store_procedural(
        self,
        pattern_name: str,
        description: str,
        trigger_patterns: List[str],
        applicable_contexts: List[str],
    ) -> MemoryNode:
        """
        Store a procedural memory (skill/pattern).
        
        Args:
            pattern_name: Name of the pattern
            description: Description of the pattern
            trigger_patterns: Keywords that trigger this pattern
            applicable_contexts: Contexts where pattern applies
            
        Returns:
            Created memory node
        """
        return await self.writer.store_procedural(
            pattern_name=pattern_name,
            description=description,
            trigger_patterns=trigger_patterns,
            applicable_contexts=applicable_contexts,
        )
    
    async def add_to_working_memory(
        self,
        session_id: UUID,
        content: str,
        slot_type: str = "context",
        priority: float = 0.5,
        expires_in_minutes: Optional[int] = None,
    ) -> MemoryNode:
        """
        Add content to working memory for a session.
        
        Args:
            session_id: Session to add to
            content: Content to remember
            slot_type: Type of slot (context, goal, task, scratchpad)
            priority: Priority for retention
            expires_in_minutes: When to expire (None = session end)
            
        Returns:
            Created memory node
        """
        return await self.writer.add_to_working_memory(
            session_id=session_id,
            content=content,
            slot_type=slot_type,
            priority=priority,
            expires_in_minutes=expires_in_minutes,
        )
    
    # ========================================================================
    # Memory Retrieval
    # ========================================================================
    
    async def recall(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        memory_tiers: Optional[List[MemoryTier]] = None,
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Query text
            session_id: Optional session context
            user_id: Optional user context
            limit: Maximum results
            memory_tiers: Which tiers to search
            
        Returns:
            List of retrieved memories with salience scores
        """
        context = RetrievalContext(
            session_id=session_id,
            user_id=user_id,
        )
        
        return await self.retrieval.retrieve(
            query=query,
            context=context,
            limit=limit,
            memory_tiers=memory_tiers,
        )
    
    async def recall_for_session(
        self,
        session_id: str,
        query: str,
        include_working_memory: bool = True,
        limit: int = 10,
    ) -> List[RetrievedMemory]:
        """
        Retrieve memories for a specific session.
        
        Prioritizes working memory, then searches long-term.
        
        Args:
            session_id: Session ID
            query: Query text
            include_working_memory: Include current working memory
            limit: Maximum results
            
        Returns:
            List of retrieved memories
        """
        return await self.retrieval.retrieve_for_session(
            session_id=session_id,
            query=query,
            include_working_memory=include_working_memory,
            limit=limit,
        )
    
    async def get_working_memory(
        self,
        session_id: UUID,
        include_expired: bool = False,
    ) -> List[MemoryNode]:
        """Get all memories in session's working memory."""
        return await self.sessions.get_working_memory(session_id, include_expired)
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def format_memories_for_context(
        self,
        memories: List[RetrievedMemory],
        max_items: int = 5,
        include_scores: bool = False,
    ) -> str:
        """
        Format memories for LLM context injection.
        
        Args:
            memories: Retrieved memories
            max_items: Maximum items to include
            include_scores: Include relevance scores
            
        Returns:
            Formatted context string
        """
        if not memories:
            return ""
        
        lines = ["## Relevant Context from Memory"]
        
        for i, mem in enumerate(memories[:max_items], 1):
            prefix = f"{i}. [{mem.memory.memory_tier.value.upper()}]"
            content = mem.memory.content
            
            if include_scores:
                content += f" (relevance: {mem.salience_score:.2f})"
            
            lines.append(f"{prefix} {content}")
        
        return "\n".join(lines)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        db_healthy = await self.db.health_check()
        return {
            "database": db_healthy,
            "status": "healthy" if db_healthy else "unhealthy",
        }
