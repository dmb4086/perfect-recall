"""
Minimal test of Perfect Recall with mocked database.
Validates the API structure without requiring Postgres.
"""

import asyncio
import pytest
from datetime import datetime
from typing import List, Optional


class MockMemory:
    """Mock memory for testing."""
    
    def __init__(self, content: str, tier: str = "episodic", confidence: float = 0.8):
        self.id = f"mock_{hash(content) & 0xFFFFFFFF}"
        self.content = content
        self.tier = tier
        self.confidence = confidence
        self.created_at = datetime.now()


class MockRetrievalPipeline:
    """Mock retrieval for testing."""
    
    def __init__(self):
        self.memories: List[MockMemory] = []
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        tier: Optional[str] = None,
        min_confidence: float = 0.0,
    ):
        """Simple keyword search for testing."""
        results = []
        query_lower = query.lower()
        
        for memory in self.memories:
            # Simple relevance scoring
            score = 0.0
            if query_lower in memory.content.lower():
                score = 0.8
            else:
                # Word overlap
                query_words = set(query_lower.split())
                content_words = set(memory.content.lower().split())
                overlap = len(query_words & content_words)
                score = overlap / max(len(query_words), 1) * 0.5
            
            if score >= 0.3 and memory.confidence >= min_confidence:
                results.append({
                    "memory": memory,
                    "similarity_score": score,
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:limit]


class MockMemoryWriter:
    """Mock writer for testing."""
    
    def __init__(self, retrieval: MockRetrievalPipeline):
        self.retrieval = retrieval
    
    async def write_episodic(self, content: str, confidence: float = 0.8, metadata: Optional[dict] = None):
        """Store a mock memory."""
        memory = MockMemory(content=content, confidence=confidence)
        self.retrieval.memories.append(memory)
        print(f"📝 Stored: {content[:50]}...")
        return memory


@pytest.mark.asyncio
async def test_perfect_recall():
    """Test the mock Perfect Recall system."""
    print("=" * 60)
    print("🧠 Perfect Recall - Mock Test")
    print("=" * 60)
    
    # Setup
    retrieval = MockRetrievalPipeline()
    writer = MockMemoryWriter(retrieval)
    
    # Store some memories
    print("\n📥 Storing memories...")
    await writer.write_episodic(
        content="dev prefers minimal greetings - no 'Sure!' or 'No problem!'",
        confidence=0.9,
    )
    await writer.write_episodic(
        content="dev is building Perfect Recall with GCP Postgres + Voyage AI",
        confidence=0.95,
    )
    await writer.write_episodic(
        content="dev stays up late - often texts at 4 AM",
        confidence=0.85,
    )
    
    # Test retrieval
    print("\n🔍 Testing retrieval...")
    
    queries = [
        "How should I greet dev?",
        "What is dev building?",
        "When does dev usually message?",
        "greeting preferences",
    ]
    
    for query in queries:
        print(f"\n  Query: '{query}'")
        results = await retrieval.search(query, limit=2)
        
        if results:
            for r in results:
                print(f"    → {r['memory'].content[:60]}... (score: {r['similarity_score']:.2f})")
        else:
            print("    → No results")
    
    print("\n" + "=" * 60)
    print("✅ Mock test complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_perfect_recall())
