"""
Basic Usage Example for Perfect Recall

This example demonstrates the four-tier memory system:
- Working Memory: Active context (conscious awareness)
- Episodic Memory: Event sequences (autobiographical memory)
- Semantic Memory: Facts & knowledge (general knowledge)
- Procedural Memory: Skills & patterns (muscle memory)
"""

import asyncio
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perfect_recall import PerfectRecall, EpisodeType


# Simple mock embedding function (replace with real one in production)
def mock_embedding(text: str) -> list[float]:
    """Generate a mock embedding vector."""
    # In production, use OpenAI, Cohere, or Ollama embeddings
    # This is just for demonstration
    import hashlib
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    # Generate 1536-dim vector (standard OpenAI dimension)
    return [(hash_val % 1000) / 1000.0 for _ in range(1536)]


async def main():
    print("=" * 60)
    print("Perfect Recall - Basic Usage Example")
    print("=" * 60)
    
    # Initialize Perfect Recall
    print("\n1. Initializing Perfect Recall...")
    pr = await PerfectRecall.create(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall"
        ),
        embedding_func=mock_embedding,
    )
    print("   ✓ Connected to database")
    
    # Check health
    health = await pr.health_check()
    print(f"   ✓ Health check: {health['status']}")
    
    # ========================================================================
    # 2. Start a Session
    # ========================================================================
    print("\n2. Starting a new session...")
    session = await pr.start_session(
        user_id="user_123",
        agent_id="assistant_v1",
        metadata={"platform": "cli", "version": "0.1.0"}
    )
    print(f"   ✓ Session started: {session.id}")
    print(f"   ✓ User: {session.user_id}")
    print(f"   ✓ Started at: {session.started_at}")
    
    # ========================================================================
    # 3. Record Episodes (Episodic Memory)
    # ========================================================================
    print("\n3. Recording episodes to episodic memory...")
    
    episodes = [
        ("User: Hi, I need help debugging a Python script", EpisodeType.MESSAGE),
        ("Assistant: I'd be happy to help! What's the error you're seeing?", EpisodeType.MESSAGE),
        ("User: I'm getting a KeyError on line 42", EpisodeType.MESSAGE),
        ("Assistant: That usually means you're accessing a dictionary key that doesn't exist", EpisodeType.MESSAGE),
        ("User: I prefer using Python for data analysis tasks", EpisodeType.MESSAGE),
        ("User: Remember this: my API key is 'sk-test-12345'", EpisodeType.MESSAGE),
    ]
    
    stored_memories = []
    for content, ep_type in episodes:
        memory = await pr.record_episode(
            content=content,
            episode_type=ep_type,
            session_id=session.id,
        )
        if memory:
            stored_memories.append(memory)
            print(f"   ✓ Stored: {content[:50]}...")
        else:
            print(f"   ✗ Rejected by write gate: {content[:50]}...")
    
    print(f"\n   Total stored: {len(stored_memories)} episodes")
    
    # ========================================================================
    # 4. Add to Working Memory
    # ========================================================================
    print("\n4. Adding active context to working memory...")
    
    working_mem = await pr.add_to_working_memory(
        session_id=session.id,
        content="Current task: Debugging Python KeyError on line 42",
        slot_type="task",
        priority=0.9,
    )
    print(f"   ✓ Added to working memory: {working_mem.id}")
    
    working_mem2 = await pr.add_to_working_memory(
        session_id=session.id,
        content="User preference: Python for data analysis",
        slot_type="context",
        priority=0.8,
    )
    print(f"   ✓ Added to working memory: {working_mem2.id}")
    
    # ========================================================================
    # 5. Store Facts (Semantic Memory)
    # ========================================================================
    print("\n5. Storing semantic facts...")
    
    facts = [
        ("user", "programming_language_preference", "Python", 0.9),
        ("user", "primary_use_case", "data analysis", 0.8),
        ("user", "experience_level", "intermediate", 0.7),
    ]
    
    for subject, predicate, obj, confidence in facts:
        fact = await pr.store_fact(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
        )
        print(f"   ✓ Stored fact: {subject} {predicate} {obj}")
    
    # ========================================================================
    # 6. Store Procedural Memory
    # ========================================================================
    print("\n6. Storing procedural memory (skills/patterns)...")
    
    procedural = await pr.store_procedural(
        pattern_name="debug_python_keyerror",
        description="When user has KeyError in Python, check dictionary key existence first",
        trigger_patterns=["KeyError", "dictionary", "key error"],
        applicable_contexts=["python", "debugging", "error_handling"],
    )
    print(f"   ✓ Stored procedural pattern: {procedural.id}")
    
    # ========================================================================
    # 7. Retrieve Memories
    # ========================================================================
    print("\n7. Retrieving memories...")
    
    # Simple recall
    print("\n   Query: 'What does the user prefer?'")
    memories = await pr.recall(
        query="What does the user prefer?",
        limit=5,
    )
    
    for i, mem in enumerate(memories, 1):
        print(f"   {i}. [{mem.memory.memory_tier.value}] {mem.memory.content[:60]}...")
        print(f"      Salience: {mem.salience_score:.2f}, Similarity: {mem.semantic_similarity:.2f}")
    
    # Session-specific recall
    print("\n   Query (with session context): 'What are we working on?'")
    session_memories = await pr.recall_for_session(
        session_id=str(session.id),
        query="What are we working on?",
        include_working_memory=True,
        limit=5,
    )
    
    for i, mem in enumerate(session_memories, 1):
        print(f"   {i}. [{mem.memory.memory_tier.value}] {mem.memory.content[:60]}...")
    
    # ========================================================================
    # 8. Format for Context
    # ========================================================================
    print("\n8. Formatting memories for LLM context...")
    context = pr.format_memories_for_context(memories, max_items=3)
    print(context)
    
    # ========================================================================
    # 9. Get Working Memory
    # ========================================================================
    print("\n9. Current working memory contents:")
    working = await pr.get_working_memory(session.id)
    for mem in working:
        print(f"   - {mem.content}")
    
    # ========================================================================
    # 10. End Session
    # ========================================================================
    print("\n10. Ending session...")
    ended = await pr.end_session(session.id, save_snapshot=True)
    print(f"   ✓ Session ended at: {ended.ended_at}")
    print(f"   ✓ Snapshot saved with {ended.context_snapshot.get('working_memory_count', 0)} working memories")
    
    # ========================================================================
    # 11. List User Sessions
    # ========================================================================
    print("\n11. User's session history:")
    sessions = await pr.get_user_sessions("user_123", limit=5)
    for s in sessions:
        duration = s.duration_seconds()
        duration_str = f"{duration:.1f}s" if duration else "active"
        print(f"   - {s.id} | Started: {s.started_at} | Duration: {duration_str}")
    
    # Cleanup
    await pr.close()
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Check if we should skip DB operations (for testing without DB)
    if os.getenv("SKIP_DB"):
        print("SKIP_DB set - skipping database operations")
        sys.exit(0)
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure PostgreSQL is running with:")
        print("  docker-compose up -d")
        sys.exit(1)
