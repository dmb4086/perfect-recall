"""
In-Memory Database Adapter for Perfect Recall

Provides a SQLite-based in-memory mode for testing without PostgreSQL.
This is useful for CI/CD and development environments where Postgres isn't available.
"""

import os
import json
from typing import AsyncGenerator, Optional, List, Tuple, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import select, desc, func, and_, or_, text

# Import Base from sqlalchemy_models to ensure shared metadata
from .sqlalchemy_models import Base


class InMemoryDatabaseManager:
    """
    In-memory database manager using SQLite.
    
    For testing only - does not support pgvector operations.
    Falls back to text-based search and simple scoring.
    """
    
    def __init__(self):
        """Initialize in-memory database manager."""
        self.async_url = "sqlite+aiosqlite:///:memory:"
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
        
        # Create tables using SQLAlchemy metadata
        await self._create_tables()
    
    async def _create_tables(self):
        """Create tables using SQLAlchemy metadata for ORM compatibility."""
        # Import all models to ensure they're registered with Base.metadata
        from .sqlalchemy_models import (
            MemoryNodeORM, SessionORM, EpisodeORM, 
            WorkingMemoryORM, MemoryAccessLogORM
        )
        
        async with self.engine.begin() as conn:
            # Use SQLAlchemy's create_all for proper ORM table creation
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """Close the database engine."""
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


class InMemoryMemoryRepository:
    """In-memory repository for memory operations (legacy - tests now use SQLAlchemy ORM)."""
    
    def __init__(self, db_manager: InMemoryDatabaseManager):
        self.db = db_manager
    
    async def create(self, memory) -> Any:
        """Create a memory node via ORM."""
        # This is a shim for compatibility - actual storage goes through SQLAlchemy ORM
        from ..db.repositories import MemoryRepository
        async with self.db.session() as session:
            repo = MemoryRepository(session)
            return await repo.create(memory)
    
    async def get_by_id(self, memory_id: UUID) -> Optional[Any]:
        """Get memory by ID."""
        from ..db.repositories import MemoryRepository
        async with self.db.session() as session:
            repo = MemoryRepository(session)
            return await repo.get_by_id(memory_id)
    
    async def search_by_text(self, query: str, limit: int = 10) -> List[Any]:
        """Simple text search."""
        from ..db.repositories import MemoryRepository
        async with self.db.session() as session:
            repo = MemoryRepository(session)
            return await repo.search_by_text(query, limit)
