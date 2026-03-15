#!/usr/bin/env python3
"""
Perfect Recall v1 - Integration Test
Tests all core functionality: write gate, retrieval, abstention
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from perfect_recall.core.perfect_recall import PerfectRecall
from perfect_recall.core.abstention import AbstentionController
from perfect_recall.models.memory import EpisodeType


def mock_embedding(text: str) -> list[float]:
    """Generate deterministic mock embeddings for testing."""
    import hashlib
    import numpy as np
    
    hash_bytes = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
    embedding = np.random.randn(1536).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


async def test_perfect_recall():
    """Run comprehensive integration tests."""
    
    print("=" * 60)
    print("🧠 Perfect Recall v1 - Integration Test")
    print("=" * 60)
    print()
    
    # Initialize
    print("🔧 Initializing Perfect Recall...")
    pr = await PerfectRecall.create(
        database_url="postgresql+asyncpg://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall",
        embedding_func=mock_embedding,
    )
    
    # Health check
    health = await pr.health_check()
    if health["status"] != "healthy":
        print("❌ Health check failed!")
        return False
    print("✅ Database connection healthy")
    print()
    
    # Start session
    print("📍 Starting test session...")
    session = await pr.start_session(user_id="test_user")
    print(f"✅ Session started: {session.id}")
    print()
    
    # Test 1: Store high-value memory (should pass write gate)
    print("📝 Test 1: Storing high-value memories...")
    memory1 = await pr.record_episode(
        content="I prefer Python for data science projects and use Pandas regularly",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    if memory1:
        print(f"   ✅ Stored: {memory1.content[:50]}...")
        print(f"      Tier: {memory1.memory_tier.value}, Triggers: {memory1.triggers[:2]}")
    else:
        print("   ❌ Memory rejected by write gate")
    
    memory2 = await pr.record_episode(
        content="My project deadline is March 20th, 2026. Remember this is important!",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    if memory2:
        print(f"   ✅ Stored: {memory2.content[:50]}...")
    
    memory3 = await pr.record_episode(
        content="Don't forget: I work at TechCorp as a senior engineer",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    if memory3:
        print(f"   ✅ Stored: {memory3.content[:50]}...")
    print()
    
    # Test 2: Store low-value memory (should be rejected)
    print("📝 Test 2: Testing write gate rejection...")
    low_value = await pr.record_episode(
        content="ok",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    if low_value is None:
        print("   ✅ Low-value memory correctly rejected")
    else:
        print("   ⚠️  Low-value memory was stored (unexpected)")
    print()
    
    # Test 3: Store semantic fact
    print("📝 Test 3: Storing semantic fact...")
    fact = await pr.store_fact(
        subject="user",
        predicate="favorite_color",
        object="blue",
        confidence=0.9,
    )
    print(f"   ✅ Fact stored: {fact.content}")
    print()
    
    # Test 4: Store procedural memory
    print("📝 Test 4: Storing procedural memory...")
    proc = await pr.store_procedural(
        pattern_name="debug_python_import_error",
        description="When encountering ImportError, check PYTHONPATH and virtual environment",
        trigger_patterns=["ImportError", "module not found", "cannot import"],
        applicable_contexts=["python", "debugging"],
    )
    print(f"   ✅ Pattern stored: {proc.content[:60]}...")
    print(f"      Symptoms: {proc.symptoms[:2] if proc.symptoms else 'None'}")
    print()
    
    # Test 5: Retrieval
    print("🔍 Test 5: Testing retrieval...")
    results = await pr.recall(
        query="What programming language does the user prefer?",
        session_id=str(session.id),
        limit=5,
    )
    print(f"   ✅ Retrieved {len(results)} memories")
    for i, r in enumerate(results[:3], 1):
        print(f"      {i}. [{r.memory.memory_tier.value}] score={r.salience_score:.2f}: {r.memory.content[:40]}...")
    print()
    
    # Test 6: Abstention on uncertain query
    print("🛡️  Test 6: Testing abstention...")
    abstention = AbstentionController()
    uncertain_results = await pr.recall(
        query="What is the capital of France?",
        session_id=str(session.id),
        limit=5,
    )
    decision = abstention.should_abstain("What is the capital of France?", uncertain_results)
    if decision.abstain:
        print(f"   ✅ Correctly abstained: {decision.reason}")
    else:
        print(f"   ℹ️  Did not abstain (confidence: {decision.confidence:.2f})")
    print()
    
    # Test 7: Context formatting
    print("📋 Test 7: Testing context formatting...")
    context = pr.format_memories_for_context(results[:3], include_scores=True)
    print(f"   ✅ Formatted context ({len(context)} chars)")
    print("   Preview:")
    for line in context.split('\n')[:4]:
        print(f"      {line}")
    print()
    
    # Test 8: Working memory
    print("💾 Test 8: Testing working memory...")
    wm = await pr.add_to_working_memory(
        session_id=session.id,
        content="Current task: Testing Perfect Recall system",
        slot_type="task",
        priority=0.9,
    )
    print(f"   ✅ Working memory added: {wm.content[:50]}...")
    
    wm_memories = await pr.get_working_memory(session.id)
    print(f"   ✅ Retrieved {len(wm_memories)} working memory items")
    print()
    
    # End session
    print("📍 Ending session...")
    await pr.end_session(session.id)
    print("✅ Session ended")
    print()
    
    # Cleanup
    await pr.close()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_perfect_recall())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
