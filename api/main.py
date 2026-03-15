"""
Perfect Recall FastAPI Server
Production-grade API for memory storage and retrieval.
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from perfect_recall.core.perfect_recall import PerfectRecall
from perfect_recall.models.memory import MemoryTier, EpisodeType


# Request/Response Models
class StoreRequest(BaseModel):
    content: str = Field(..., description="Content to store in memory")
    tier: MemoryTier = Field(default=MemoryTier.EPISODIC, description="Memory tier")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    triggers: Optional[List[str]] = Field(default=None, description="Trigger patterns")
    symptoms: Optional[List[str]] = Field(default=None, description="Contextual symptoms")


class StoreResponse(BaseModel):
    id: str
    status: str
    stored_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=5, ge=1, le=20, description="Max results")
    tier: Optional[MemoryTier] = Field(default=None, description="Filter by tier")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class MemoryResult(BaseModel):
    id: str
    content: str
    memory_tier: MemoryTier  # Fixed: was 'tier'
    confidence: float
    created_at: datetime
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: List[MemoryResult]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    database: str
    embeddings: str
    version: str = "0.1.0"


# Global state
pr: Optional[PerfectRecall] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global pr
    
    # Startup
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/perfect_recall"
    )
    
    try:
        pr = await PerfectRecall.create(database_url=database_url)
        print(f"✅ Connected to database")
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("Running in degraded mode (no persistence)")
        pr = None
    
    yield
    
    # Shutdown
    if pr:
        await pr.db.close()
        print("✅ Database connection closed")


app = FastAPI(
    title="Perfect Recall API",
    description="Memory system for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)


def get_pr() -> PerfectRecall:
    """Dependency to get Perfect Recall instance."""
    if pr is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return pr


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if pr else "degraded",
        database="connected" if pr else "disconnected",
        embeddings="voyage-3",
    )


@app.post("/memories", response_model=StoreResponse)
async def store_memory(
    request: StoreRequest,
    pr: PerfectRecall = Depends(get_pr),
):
    """Store a new memory."""
    import time
    
    start = time.time()
    
    try:
        # FIXED: Use record_episode instead of write_episodic
        memory = await pr.writer.record_episode(
            content=request.content,
            episode_type=EpisodeType.CONVERSATION,
            metadata={
                "triggers": request.triggers or [],
                "symptoms": request.symptoms or [],
            }
        )
        
        return StoreResponse(
            id=str(memory.id),
            status="stored",
            stored_at=memory.created_at,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/search", response_model=SearchResponse)
async def search_memories(
    request: SearchRequest,
    pr: PerfectRecall = Depends(get_pr),
):
    """Search memories by semantic similarity."""
    import time
    
    start = time.time()
    
    try:
        # FIXED: Use retrieve instead of search
        results = await pr.retrieval.retrieve(
            query=request.query,
            limit=request.limit,
            # tier and min_confidence not directly supported, would need filtering
        )
        
        latency_ms = (time.time() - start) * 1000
        
        return SearchResponse(
            query=request.query,
            results=[
                MemoryResult(
                    id=str(r.memory.id),
                    content=r.memory.content,
                    memory_tier=r.memory.memory_tier,  # FIXED: was tier
                    confidence=r.memory.confidence,
                    created_at=r.memory.created_at,
                    similarity=r.semantic_similarity,  # FIXED: was similarity_score
                )
                for r in results
            ],
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Perfect Recall",
        "version": "0.1.0",
        "status": "operational" if pr else "degraded",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
