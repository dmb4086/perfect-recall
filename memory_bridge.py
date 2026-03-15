#!/usr/bin/env python3
"""
Perfect Recall Memory Bridge

This script allows the assistant to store and retrieve memories from
PostgreSQL with pgvector. Run this to activate the memory system.
"""

import asyncio
import sys
import os

# Add the source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from perfect_recall.db.connection import DatabaseManager
from perfect_recall.core.memory_writer import MemoryWriter
from perfect_recall.core.retrieval import RetrievalPipeline, SalienceScorer
from perfect_recall.core.abstention import AbstentionController


class PerfectRecallBridge:
    """Bridge between assistant and Perfect Recall memory system."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.writer = None
        self.retriever = None
        self.abstention = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the memory system."""
        if self._initialized:
            return
        
        await self.db.initialize()
        
        # Use a simple embedding function for now (identity-based)
        # In production, this would call an embedding API
        def simple_embedding(text: str) -> list[float]:
            import hashlib
            import numpy as np
            # Deterministic embedding from text hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            np.random.seed(int.from_bytes(hash_bytes[:4], 'big'))
            embedding = np.random.randn(1536).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        
        self.writer = MemoryWriter(
            db_manager=self.db,
            embedding_func=simple_embedding,
            write_threshold=0.5,  # Lower threshold for testing
        )
        
        self.retriever = RetrievalPipeline(
            db_manager=self.db,
            embedding_func=simple_embedding,
            scorer=SalienceScorer(),
        )
        
        self.abstention = AbstentionController()
        
        self._initialized = True
        print("✅ Perfect Recall memory system activated")
    
    async def store_fact(self, subject: str, predicate: str, object: str) -> bool:
        """Store a semantic fact."""
        if not self._initialized:
            await self.initialize()
        
        memory = await self.writer.store_fact(
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=0.8,
        )
        return memory is not None
    
    async def store_skill(self, pattern: str, description: str, triggers: list[str]) -> bool:
        """Store a procedural skill."""
        if not self._initialized:
            await self.initialize()
        
        memory = await self.writer.store_procedural(
            pattern_name=pattern,
            description=description,
            trigger_patterns=triggers,
            applicable_contexts=["general"],
        )
        return memory is not None
    
    async def retrieve(self, query: str, limit: int = 5) -> list[dict]:
        """Retrieve relevant memories."""
        if not self._initialized:
            await self.initialize()
        
        results = await self.retriever.retrieve(query, limit=limit)
        
        # Check if we should abstain
        decision = self.abstention.should_abstain(query, results)
        if decision.abstain:
            return []
        
        # Format results
        memories = []
        for r in results:
            memories.append({
                'content': r.memory.content,
                'tier': r.memory.memory_tier.value,
                'score': r.salience_score,
                'created_at': r.memory.created_at.isoformat() if r.memory.created_at else None,
            })
        
        return memories
    
    async def status(self) -> dict:
        """Get memory system status."""
        if not self._initialized:
            await self.initialize()
        
        async with self.db.session() as session:
            from sqlalchemy import text
            result = await session.execute(text("""
                SELECT 
                    (SELECT COUNT(*) FROM memory_nodes) as total_memories,
                    (SELECT COUNT(*) FROM memory_nodes WHERE memory_tier = 'episodic') as episodic,
                    (SELECT COUNT(*) FROM memory_nodes WHERE memory_tier = 'semantic') as semantic,
                    (SELECT COUNT(*) FROM memory_nodes WHERE memory_tier = 'procedural') as procedural,
                    (SELECT COUNT(*) FROM memory_nodes WHERE memory_tier = 'working') as working
            """))
            row = result.fetchone()
            
        return {
            'total_memories': row.total_memories,
            'by_tier': {
                'episodic': row.episodic,
                'semantic': row.semantic,
                'procedural': row.procedural,
                'working': row.working,
            },
            'status': 'active',
        }


# Global bridge instance
_bridge = None

async def get_bridge() -> PerfectRecallBridge:
    """Get or create the global bridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = PerfectRecallBridge()
        await _bridge.initialize()
    return _bridge


# CLI for testing
if __name__ == "__main__":
    async def main():
        bridge = await get_bridge()
        
        print("\n" + "="*50)
        print("Perfect Recall Memory Bridge")
        print("="*50)
        
        # Show status
        status = await bridge.status()
        print(f"\n📊 Memory Status:")
        print(f"   Total memories: {status['total_memories']}")
        print(f"   Episodic: {status['by_tier']['episodic']}")
        print(f"   Semantic: {status['by_tier']['semantic']}")
        print(f"   Procedural: {status['by_tier']['procedural']}")
        
        # Test store
        print("\n📝 Testing memory storage...")
        
        success = await bridge.store_fact("dev", "wants", "Perfect Recall activated")
        print(f"   Store fact: {'✅' if success else '❌'}")
        
        # Test retrieve
        print("\n🔍 Testing memory retrieval...")
        results = await bridge.retrieve("memory system", limit=3)
        if results:
            print(f"   Found {len(results)} memories:")
            for r in results:
                print(f"      - {r['content'][:50]}... ({r['tier']}, score: {r['score']:.2f})")
        else:
            print("   No memories found (may need real embeddings)")
        
        print("\n✅ Perfect Recall is ready to use!")
        print("="*50)
    
    asyncio.run(main())
