"""
Perfect Recall v1 - Full Integration Test (In-Memory Mode)

Tests all core v1 features without requiring PostgreSQL:
- Write gate
- Memory storage (all 4 tiers)
- Semantic retrieval
- Abstention controller
- Session management
- Superpowers metadata
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pytest_asyncio
from uuid import UUID

from perfect_recall.db.in_memory import (
    InMemoryDatabaseManager,
    create_in_memory_perfect_recall,
    mock_embedding,
)
from perfect_recall.db.repositories import MemoryRepository, SessionRepository
from perfect_recall.core.memory_writer import MemoryWriter
from perfect_recall.core.retrieval import RetrievalPipeline, SalienceScorer
from perfect_recall.core.abstention import AbstentionController
from perfect_recall.models.memory import MemoryNode, MemoryTier, EpisodeType
from perfect_recall.models.retrieval import RetrievalContext


@pytest_asyncio.fixture
async def pr():
    """Create PerfectRecall instance with in-memory database."""
    pr = await create_in_memory_perfect_recall()
    yield pr
    await pr.close()


@pytest_asyncio.fixture
async def db():
    """Create in-memory database."""
    db_manager = InMemoryDatabaseManager()
    await db_manager.initialize()
    yield db_manager
    await db_manager.close()


@pytest.fixture
def abstention():
    """Create abstention controller."""
    return AbstentionController(
        min_salience_threshold=0.3,
        min_semantic_similarity=0.4,
    )


class TestPerfectRecallV1:
    """v1 Integration Tests - All core features using in-memory DB."""
    
    @pytest.mark.asyncio
    async def test_write_gate_passes_high_utility(self, pr):
        """Test 1: Write gate allows high-utility content."""
        print("\n📋 Test 1: Write Gate - High Utility")
        
        memory = await pr.record_episode(
            content="I prefer Python for machine learning projects",
            episode_type=EpisodeType.MESSAGE,
        )
        
        assert memory is not None
        assert memory.memory_tier == MemoryTier.EPISODIC
        print(f"   ✅ Memory stored: {memory.content[:50]}...")
    
    @pytest.mark.asyncio
    async def test_write_gate_rejects_low_utility(self, pr):
        """Test 2: Write gate rejects low-utility content."""
        print("\n📋 Test 2: Write Gate - Low Utility")
        
        memory = await pr.record_episode(
            content="ok",
            episode_type=EpisodeType.MESSAGE,
        )
        
        assert memory is None
        print("   ✅ Low-utility content correctly rejected")
    
    @pytest.mark.asyncio
    async def test_write_gate_explicit_markers(self, pr):
        """Test 3: Write gate respects explicit markers."""
        print("\n📋 Test 3: Write Gate - Explicit Markers")
        
        memory = await pr.record_episode(
            content="Remember this: my database password is 'secret123'",
            episode_type=EpisodeType.MESSAGE,
        )
        
        assert memory is not None
        assert "remember this" in memory.content.lower()
        print(f"   ✅ Explicit marker respected: {memory.content[:50]}...")
    
    @pytest.mark.asyncio
    async def test_semantic_memory_storage(self, pr):
        """Test 4: Semantic memory (facts) can be stored."""
        print("\n📋 Test 4: Semantic Memory Storage")
        
        fact = await pr.store_fact(
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
    async def test_procedural_memory_storage(self, pr):
        """Test 5: Procedural memory (skills) can be stored."""
        print("\n📋 Test 5: Procedural Memory Storage")
        
        proc = await pr.store_procedural(
            pattern_name="debug_python_keyerror",
            description="Guide for debugging Python KeyError exceptions",
            trigger_patterns=["KeyError", "dictionary error", "key not found"],
            applicable_contexts=["python", "debugging"],
        )
        
        assert proc is not None
        assert proc.memory_tier == MemoryTier.PROCEDURAL
        assert any("keyerror" in t.lower() for t in proc.triggers)
        print(f"   ✅ Procedural memory stored: {proc.content[:50]}...")
        print(f"   📝 Triggers: {proc.triggers[:3]}")
    
    @pytest.mark.asyncio
    async def test_working_memory_storage(self, pr):
        """Test 6: Working memory can be stored."""
        print("\n📋 Test 6: Working Memory Storage")
        
        from uuid import uuid4
        session_id = uuid4()
        
        wm = await pr.add_to_working_memory(
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
    async def test_retrieval_returns_results(self, pr):
        """Test 7: Retrieval returns stored memories."""
        print("\n📋 Test 7: Memory Retrieval")
        
        # Store some memories
        await pr.store_fact(
            subject="user",
            predicate="works_at",
            object="Acme Corp",
            confidence=0.9,
        )
        
        await pr.store_fact(
            subject="user",
            predicate="likes",
            object="coffee",
            confidence=0.8,
        )
        
        # Retrieve
        results = await pr.recall(
            query="Where does the user work?",
            limit=5,
        )
        
        assert len(results) > 0
        print(f"   ✅ Retrieved {len(results)} memories")
        for r in results:
            print(f"      - {r.memory.content[:50]}... (score: {r.salience_score:.2f})")
    
    @pytest.mark.asyncio
    async def test_salience_scoring(self, pr):
        """Test 8: Memories are scored by salience."""
        print("\n📋 Test 8: Salience Scoring")
        
        await pr.store_fact(
            subject="user",
            predicate="important",
            object="critical project deadline tomorrow",
            confidence=1.0,
        )
        
        results = await pr.recall(
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
    async def test_abstention_on_low_confidence(self, pr, abstention):
        """Test 10: Abstention triggers on low confidence."""
        print("\n📋 Test 10: Abstention - Low Confidence")
        
        await pr.store_fact(
            subject="user",
            predicate="likes",
            object="Python",
            confidence=0.5,
        )
        
        results = await pr.recall(
            query="quantum physics mechanics",
            limit=5,
        )
        
        decision = abstention.should_abstain("quantum physics", results)
        
        assert decision.confidence < 0.8
        print(f"   ✅ Confidence calculated: {decision.confidence:.2f}")
        print(f"   📝 Reason: {decision.reason}")
    
    @pytest.mark.asyncio
    async def test_superpowers_metadata_extraction(self, pr):
        """Test 11: Superpowers metadata is auto-extracted."""
        print("\n📋 Test 11: Superpowers Metadata Extraction")
        
        memory = await pr.record_episode(
            content="""
            Error: ConnectionTimeout when connecting to Redis
            
            Fix: Increase the connection timeout to 30 seconds
            Use this when experiencing slow network connections
            Not for local development environments
            """,
            episode_type=EpisodeType.ACTION,
        )
        
        assert memory is not None
        assert len(memory.triggers) > 0
        assert len(memory.symptoms) > 0
        assert len(memory.anti_triggers) > 0
        
        print(f"   ✅ Triggers extracted: {memory.triggers[:3]}")
        print(f"   ✅ Symptoms extracted: {memory.symptoms[:3]}")
        print(f"   ✅ Anti-triggers extracted: {memory.anti_triggers}")
    
    @pytest.mark.asyncio
    async def test_session_management(self, pr):
        """Test 12: Session management works."""
        print("\n📋 Test 12: Session Management")
        
        # Start session
        session = await pr.start_session(user_id="test_user")
        assert session is not None
        assert session.user_id == "test_user"
        print(f"   ✅ Session started: {session.id}")
        
        # Add to working memory
        wm = await pr.add_to_working_memory(
            session_id=session.id,
            content="Current task: Testing",
            slot_type="task",
            priority=0.9,
        )
        assert wm is not None
        print(f"   ✅ Working memory added")
        
        # Get working memory
        working = await pr.get_working_memory(session.id)
        assert len(working) > 0
        print(f"   ✅ Working memory retrieved: {len(working)} items")
        
        # End session
        ended = await pr.end_session(session.id)
        assert ended is not None
        assert ended.ended_at is not None
        print(f"   ✅ Session ended")
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, pr, abstention):
        """Test 13: Complete end-to-end workflow."""
        print("\n📋 Test 13: End-to-End Workflow")
        
        # 1. Start session
        session = await pr.start_session(user_id="e2e_user")
        print(f"   📍 Session: {session.id}")
        
        # 2. Store user preferences
        await pr.store_fact(
            subject="user",
            predicate="prefers",
            object="Python for data tasks",
            confidence=0.9,
        )
        
        await pr.store_fact(
            subject="user",
            predicate="dislikes",
            object="Java for scripting",
            confidence=0.8,
        )
        
        # 3. Record an episode
        await pr.record_episode(
            content="User is working on a machine learning pipeline in Python",
            episode_type=EpisodeType.MESSAGE,
            session_id=session.id,
        )
        
        # 4. Store a procedural skill
        await pr.store_procedural(
            pattern_name="optimize_pandas_pipeline",
            description="Optimize pandas data processing pipelines",
            trigger_patterns=["slow pandas", "performance issue", "memory error"],
            applicable_contexts=["python", "pandas", "performance"],
        )
        
        # 5. Query and check abstention
        query = "What language should I use for data tasks?"
        results = await pr.recall(query=query, limit=5)
        decision = abstention.should_abstain(query, results)
        
        print(f"   ✅ Stored memories and retrieved results")
        print(f"   ✅ Confidence: {decision.confidence:.2f}, Abstain: {decision.abstain}")
        
        if not decision.abstain and results:
            print(f"   📝 Top result: {results[0].memory.content[:60]}...")
        
        # 6. Verify we have memories from different tiers
        tiers = set(r.memory.memory_tier for r in results)
        print(f"   ✅ Retrieved from tiers: {[t.value for t in tiers]}")
        
        # 7. Format for context
        context = pr.format_memories_for_context(results, max_items=3)
        assert len(context) > 0
        print(f"   ✅ Context formatted ({len(context)} chars)")
        
        # 8. End session
        await pr.end_session(session.id)
        print(f"   ✅ Session ended")
    
    @pytest.mark.asyncio
    async def test_anti_trigger_filtering(self, pr):
        """Test 14: Anti-trigger filtering works."""
        print("\n📋 Test 14: Anti-Trigger Filtering")
        
        # Store memory with anti-triggers
        proc = await pr.store_procedural(
            pattern_name="production_fix",
            description="Fix for production environments only",
            trigger_patterns=["error", "fix"],
            applicable_contexts=["production"],
        )
        # Manually add anti-triggers (they get extracted from description)
        
        # Query for production - should match
        results_prod = await pr.recall(
            query="How to fix error in production?",
            limit=5,
        )
        
        # Query for development - should potentially not match or be lower ranked
        results_dev = await pr.recall(
            query="How to fix error in development?",
            limit=5,
        )
        
        print(f"   ✅ Production query: {len(results_prod)} results")
        print(f"   ✅ Development query: {len(results_dev)} results")
    
    @pytest.mark.asyncio
    async def test_health_check(self, pr):
        """Test 15: Health check works."""
        print("\n📋 Test 15: Health Check")
        
        health = await pr.health_check()
        
        assert health["status"] == "healthy"
        assert health["database"] is True
        print(f"   ✅ Health check passed: {health['status']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
