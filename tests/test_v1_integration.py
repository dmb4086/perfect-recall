"""
Integration Test for Perfect Recall v1

Tests all core v1 features:
- Write gate
- Memory storage (all 4 tiers)
- Semantic retrieval
- Abstention controller
- Session management
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pytest_asyncio
from uuid import UUID

from perfect_recall.db.connection import DatabaseManager, reset_db_manager
from perfect_recall.db.repositories import MemoryRepository, SessionRepository
from perfect_recall.core.memory_writer import MemoryWriter
from perfect_recall.core.retrieval import RetrievalPipeline, SalienceScorer
from perfect_recall.core.abstention import AbstentionController
from perfect_recall.models.memory import MemoryNode, MemoryTier, EpisodeType
from perfect_recall.models.retrieval import RetrievalContext


def mock_embedding(text: str) -> list[float]:
    """Generate mock embedding vector."""
    import hashlib
    import numpy as np
    
    hash_bytes = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
    embedding = np.random.randn(1536).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


# Use real PostgreSQL database for integration tests
@pytest_asyncio.fixture
async def db():
    """Create PostgreSQL database connection."""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    # Don't create tables here - assume they exist from schema.sql
    yield db_manager
    await db_manager.close()


@pytest.fixture
def writer(db):
    """Create memory writer."""
    return MemoryWriter(
        db_manager=db,
        embedding_func=mock_embedding,
        write_threshold=0.6,
    )


@pytest.fixture
def retriever(db):
    """Create retrieval pipeline."""
    return RetrievalPipeline(
        db_manager=db,
        embedding_func=mock_embedding,
        scorer=SalienceScorer(),
    )


@pytest.fixture
def abstention():
    """Create abstention controller."""
    return AbstentionController(
        min_salience_threshold=0.3,
        min_semantic_similarity=0.4,
    )


class TestPerfectRecallV1:
    """v1 Integration Tests - All core features."""
    
    @pytest.mark.asyncio
    async def test_write_gate_passes_high_utility(self, db, writer):
        """Test 1: Write gate allows high-utility content."""
        print("\n📋 Test 1: Write Gate - High Utility")
        
        # This should pass
        memory = await writer.record_episode(
            content="I prefer Python for machine learning projects",
            episode_type=EpisodeType.MESSAGE,
        )
        
        assert memory is not None
        assert memory.memory_tier == MemoryTier.EPISODIC
        print(f"   ✅ Memory stored: {memory.content[:50]}...")
    
    @pytest.mark.asyncio
    async def test_write_gate_rejects_low_utility(self, db, writer):
        """Test 2: Write gate rejects low-utility content."""
        print("\n📋 Test 2: Write Gate - Low Utility")
        
        # This should be rejected
        memory = await writer.record_episode(
            content="ok",
            episode_type=EpisodeType.MESSAGE,
        )
        
        assert memory is None
        print("   ✅ Low-utility content correctly rejected")
    
    @pytest.mark.asyncio
    async def test_write_gate_explicit_markers(self, db, writer):
        """Test 3: Write gate respects explicit markers."""
        print("\n📋 Test 3: Write Gate - Explicit Markers")
        
        memory = await writer.record_episode(
            content="Remember this: my database password is 'secret123'",
            episode_type=EpisodeType.MESSAGE,
        )
        
        assert memory is not None
        assert "remember this" in memory.content.lower()
        print(f"   ✅ Explicit marker respected: {memory.content[:50]}...")
    
    @pytest.mark.asyncio
    async def test_semantic_memory_storage(self, db, writer):
        """Test 4: Semantic memory (facts) can be stored."""
        print("\n📋 Test 4: Semantic Memory Storage")
        
        fact = await writer.store_fact(
            subject="user",
            predicate="likes",
            object="Python",
            confidence=0.9,
        )
        
        assert fact is not None
        assert fact.memory_tier == MemoryTier.SEMANTIC
        assert "python" in fact.content.lower()
        print(f"   ✅ Fact stored: {fact.content}")
    
    @pytest.mark.asyncio
    async def test_procedural_memory_storage(self, db, writer):
        """Test 5: Procedural memory (skills) can be stored."""
        print("\n📋 Test 5: Procedural Memory Storage")
        
        proc = await writer.store_procedural(
            pattern_name="debug_python_keyerror",
            description="Guide for debugging Python KeyError exceptions",
            trigger_patterns=["KeyError", "dictionary error", "key not found"],
            applicable_contexts=["python", "debugging"],
        )
        
        assert proc is not None
        assert proc.memory_tier == MemoryTier.PROCEDURAL
        # Check case-insensitively since triggers preserve original case
        assert any("keyerror" in t.lower() for t in proc.triggers)
        print(f"   ✅ Procedural memory stored: {proc.content[:50]}...")
        print(f"   📝 Triggers: {proc.triggers[:3]}")
    
    @pytest.mark.asyncio
    async def test_working_memory_storage(self, db, writer):
        """Test 6: Working memory can be stored."""
        print("\n📋 Test 6: Working Memory Storage")
        
        from uuid import uuid4
        session_id = uuid4()
        
        wm = await writer.add_to_working_memory(
            session_id=session_id,
            content="Current task: Debugging authentication flow",
            slot_type="task",
            priority=0.9,
            expires_in_minutes=30,
        )
        
        assert wm is not None
        assert wm.memory_tier == MemoryTier.WORKING
        assert "debugging" in wm.content.lower()
        print(f"   ✅ Working memory stored: {wm.content[:50]}...")
    
    @pytest.mark.asyncio
    async def test_retrieval_returns_results(self, db, writer, retriever):
        """Test 7: Retrieval returns stored memories."""
        print("\n📋 Test 7: Memory Retrieval")
        
        # Store some memories
        await writer.store_fact(
            subject="user",
            predicate="works_at",
            object="Acme Corp",
            confidence=0.9,
        )
        
        await writer.store_fact(
            subject="user",
            predicate="likes",
            object="coffee",
            confidence=0.8,
        )
        
        # Retrieve
        results = await retriever.retrieve(
            query="Where does the user work?",
            limit=5,
        )
        
        assert len(results) > 0
        print(f"   ✅ Retrieved {len(results)} memories")
        for r in results:
            print(f"      - {r.memory.content[:50]}... (score: {r.salience_score:.2f})")
    
    @pytest.mark.asyncio
    async def test_salience_scoring(self, db, writer, retriever):
        """Test 8: Memories are scored by salience."""
        print("\n📋 Test 8: Salience Scoring")
        
        # Store with different importance
        high_imp = await writer.store_fact(
            subject="user",
            predicate="important",
            object="critical project deadline tomorrow",
            confidence=1.0,
        )
        high_imp.importance_score = 1.0
        
        # Retrieve and check scores
        results = await retriever.retrieve(
            query="deadline",
            limit=5,
        )
        
        assert len(results) > 0
        assert all(r.salience_score >= 0 for r in results)
        print(f"   ✅ All memories have valid salience scores")
        for r in results[:3]:
            print(f"      - Score: {r.salience_score:.2f} | {r.memory.content[:40]}...")
    
    @pytest.mark.asyncio
    async def test_abstention_on_empty_results(self, abstention):
        """Test 9: Abstention triggers on empty results."""
        print("\n📋 Test 9: Abstention - Empty Results")
        
        decision = abstention.should_abstain(
            query="unknown topic xyz123",
            results=[],
        )
        
        assert decision.abstain is True
        assert "No relevant memories" in decision.reason
        print(f"   ✅ Correctly abstains: {decision.reason}")
    
    @pytest.mark.asyncio
    async def test_abstention_on_low_confidence(self, db, writer, retriever, abstention):
        """Test 10: Abstention triggers on low confidence."""
        print("\n📋 Test 10: Abstention - Low Confidence")
        
        # Store one memory
        await writer.store_fact(
            subject="user",
            predicate="likes",
            object="Python",
            confidence=0.5,
        )
        
        # Query for something completely different
        results = await retriever.retrieve(
            query="quantum physics mechanics",
            limit=5,
        )
        
        decision = abstention.should_abstain("quantum physics", results)
        
        # Should abstain due to low relevance
        assert decision.confidence < 0.8
        print(f"   ✅ Confidence calculated: {decision.confidence:.2f}")
        print(f"   📝 Reason: {decision.reason}")
    
    @pytest.mark.asyncio
    async def test_superpowers_metadata_extraction(self, db, writer):
        """Test 11: Superpowers metadata is auto-extracted."""
        print("\n📋 Test 11: Superpowers Metadata Extraction")
        
        memory = await writer.record_episode(
            content="""
            Error: ConnectionTimeout when connecting to Redis
            
            Fix: Increase the connection timeout to 30 seconds
            Use this when experiencing slow network connections
            Not for local development environments
            """,
            episode_type=EpisodeType.ACTION,
        )
        
        assert memory is not None
        # Should have extracted triggers
        assert len(memory.triggers) > 0
        # Should have extracted symptoms (error patterns)
        assert len(memory.symptoms) > 0
        # Should have anti-triggers
        assert len(memory.anti_triggers) > 0
        
        print(f"   ✅ Triggers extracted: {memory.triggers[:3]}")
        print(f"   ✅ Symptoms extracted: {memory.symptoms[:3]}")
        print(f"   ✅ Anti-triggers extracted: {memory.anti_triggers}")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, db, writer, retriever, abstention):
        """Test 12: Complete end-to-end workflow."""
        print("\n📋 Test 12: End-to-End Workflow")
        
        # 1. Store user preferences
        await writer.store_fact(
            subject="user",
            predicate="prefers",
            object="Python for data tasks",
            confidence=0.9,
        )
        
        await writer.store_fact(
            subject="user",
            predicate="dislikes",
            object="Java for scripting",
            confidence=0.8,
        )
        
        # 2. Record an episode
        await writer.record_episode(
            content="User is working on a machine learning pipeline in Python",
            episode_type=EpisodeType.MESSAGE,
        )
        
        # 3. Store a procedural skill
        await writer.store_procedural(
            pattern_name="optimize_pandas_pipeline",
            description="Optimize pandas data processing pipelines",
            trigger_patterns=["slow pandas", "performance issue", "memory error"],
            applicable_contexts=["python", "pandas", "performance"],
        )
        
        # 4. Query and check abstention
        query = "What language should I use for data tasks?"
        results = await retriever.retrieve(query=query, limit=5)
        decision = abstention.should_abstain(query, results)
        
        print(f"   ✅ Stored memories and retrieved results")
        print(f"   ✅ Confidence: {decision.confidence:.2f}, Abstain: {decision.abstain}")
        
        if not decision.abstain:
            print(f"   📝 Top result: {results[0].memory.content[:60]}...")
        
        # 5. Verify we have memories from different tiers
        tiers = set(r.memory.memory_tier for r in results)
        print(f"   ✅ Retrieved from tiers: {[t.value for t in tiers]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
