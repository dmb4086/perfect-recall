"""Database connection management."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

Base = declarative_base()


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL. If not provided,
                         uses DATABASE_URL environment variable.
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql+asyncpg://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall"
        )
        
        # Convert sync URL to async if needed
        if self.database_url.startswith("postgresql://"):
            self.async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        else:
            self.async_url = self.database_url
        
        self.engine = None
        self.async_session = None
    
    async def initialize(self):
        """Initialize the database engine and session factory."""
        self.engine = create_async_engine(
            self.async_url,
            echo=False,
            poolclass=NullPool,  # For async simplicity
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def close(self):
        """Close the database engine."""
        if self.engine:
            await self.engine.dispose()
    
    async def create_tables(self):
        """Create all database tables."""
        async with self.engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
            
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session as async context manager."""
        if self.async_session is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
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
        from sqlalchemy import text
        try:
            async with self.session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            print(f"Health check error: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def reset_db_manager():
    """Reset the global database manager (for testing)."""
    global _db_manager
    if _db_manager:
        await _db_manager.close()
    _db_manager = None
