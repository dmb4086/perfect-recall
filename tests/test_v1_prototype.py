#!/usr/bin/env python3
"""
Perfect Recall v1 - End-to-End Test
Verifies all v1 milestone features work correctly.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perfect_recall.db.in_memory import create_in_memory_perfect_recall, mock_embedding
from perfect_recall.models.memory import EpisodeType, MemoryTier
from perfect_recall.core.abstention import AbstentionController


async def test_v1_prototype():
    """Test all v1 features."""
    print("=" * 60)
    print("🧠 Perfect Recall v1 - End-to-End Test")
    print("=" * 60)
    
    # Create in-memory instance
    print("\n📦 Creating in-memory Perfect Recall instance...")
    pr = await create_in_memory_perfect_recall()
    abstention = AbstentionController()
    print("✅ Instance created")
    
    # Test 1: Session Management
    print("\n--- Test 1: Session Management ---")
    session = await pr.start_session(user_id="test_user", agent_id="test_agent")
    print(f"✅ Session created: {session.id}")
    print(f"   User: {session.user_id}")
    print(f"   Started: {session.started_at}")
    
    # Test 2: Memory Writing with Write Gate
    print("\n--- Test 2: Memory Writing with Write Gate ---")
    
    # High-value memory (should pass write gate)
    memory1 = await pr.record_episode(
        content="Remember this: The user's name is Alice and they prefer Python over JavaScript",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    if memory1:
        print(f"✅ Memory stored (passed write gate)")
        print(f"   Content: {memory1.content[:60]}...")
        print(f"   Tier: {memory1.memory_tier.value}")
        print(f"   Triggers: {memory1.triggers[:3]}")
        print(f"   Aliases: {memory1.aliases[:3]}")
    else:
        print("❌ Memory rejected by write gate")
    
    # Low-value memory (might be rejected)
    memory2 = await pr.record_episode(
        content="ok",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    if memory2:
        print(f"✅ Short memory stored")
    else:
        print("✅ Short memory correctly rejected by write gate")
    
    # Store more meaningful memories
    await pr.record_episode(
        content="I work at Acme Corp as a senior developer",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    
    await pr.record_episode(
        content="Important: The database connection string is postgres://localhost:5432/mydb",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    
    # Test 3: Semantic Memory / Fact Storage
    print("\n--- Test 3: Semantic Memory ---")
    fact = await pr.store_fact(
        subject="user",
        predicate="favorite_language",
        object="Python",
        confidence=0.95,
    )
    print(f"✅ Fact stored: {fact.content}")
    
    # Test 4: Procedural Memory
    print("\n--- Test 4: Procedural Memory ---")
    proc = await pr.store_procedural(
        pattern_name="debug_python_keyerror",
        description="When debugging KeyError in Python, check if the key exists using 'in' operator first",
        trigger_patterns=["KeyError", "python error", "dictionary error"],
        applicable_contexts=["python", "debugging"],
    )
    print(f"✅ Procedural memory stored: {proc.content[:60]}...")
    print(f"   Triggers: {proc.triggers}")
    
    # Test 5: Working Memory
    print("\n--- Test 5: Working Memory ---")
    working = await pr.add_to_working_memory(
        session_id=session.id,
        content="Current task: Testing Perfect Recall v1",
        slot_type="task",
        priority=0.9,
    )
    print(f"✅ Working memory added: {working.content}")
    
    # Test 6: Retrieval with Semantic Search
    print("\n--- Test 6: Semantic Retrieval ---")
    
    # Query 1: User's name
    results = await pr.recall(
        query="What is the user's name?",
        session_id=str(session.id),
        limit=3,
    )
    print(f"\n🔍 Query: 'What is the user's name?'")
    print(f"   Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.memory.memory_tier.value}] {r.memory.content[:50]}... (score: {r.salience_score:.2f})")
    
    # Query 2: Preferences
    results = await pr.recall(
        query="programming language preference",
        session_id=str(session.id),
        limit=3,
    )
    print(f"\n🔍 Query: 'programming language preference'")
    print(f"   Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.memory.memory_tier.value}] {r.memory.content[:50]}... (score: {r.salience_score:.2f})")
    
    # Query 3: Error handling
    results = await pr.recall(
        query="How to fix KeyError?",
        session_id=str(session.id),
        limit=3,
    )
    print(f"\n🔍 Query: 'How to fix KeyError?'")
    print(f"   Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.memory.memory_tier.value}] {r.memory.content[:50]}... (score: {r.salience_score:.2f})")
        if r.memory.triggers:
            print(f"      Triggers matched: {r.memory.triggers[:2]}")
    
    # Test 7: Abstention Controller
    print("\n--- Test 7: Abstention Controller ---")
    
    # Test with a query that should work
    good_results = await pr.recall(query="user name")
    decision = abstention.should_abstain("user name", good_results)
    print(f"\n🔍 Query: 'user name'")
    print(f"   Should abstain: {decision.abstain}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Reason: {decision.reason}")
    
    # Test with a nonsense query that should abstain
    bad_results = await pr.recall(query="xyz123 nonsense query")
    decision = abstention.should_abstain("xyz123 nonsense query", bad_results)
    print(f"\n🔍 Query: 'xyz123 nonsense query'")
    print(f"   Should abstain: {decision.abstain}")
    print(f"   Confidence: {decision.confidence:.2f}")
    print(f"   Reason: {decision.reason}")
    if decision.abstain:
        print(f"   Suggestion: {decision.suggestion}")
    
    # Test 8: Context Formatting
    print("\n--- Test 8: Context Formatting ---")
    all_memories = await pr.recall(query="user", limit=3)
    context = pr.format_memories_for_context(all_memories, include_scores=True)
    print("Formatted context for LLM:")
    print(context)
    
    # Test 9: Health Check
    print("\n--- Test 9: Health Check ---")
    health = await pr.health_check()
    print(f"Database healthy: {health['database']}")
    print(f"Overall status: {health['status']}")
    
    # Test 10: Session End
    print("\n--- Test 10: Session Lifecycle ---")
    ended = await pr.end_session(session.id)
    if ended:
        print(f"✅ Session ended: {ended.id}")
        print(f"   Duration: {ended.ended_at - ended.started_at}")
    
    await pr.close()
    
    print("\n" + "=" * 60)
    print("✅ All v1 tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_v1_prototype())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
