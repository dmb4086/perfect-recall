#!/usr/bin/env python3
"""
Memory Orchestrator — Integrated Perfect Recall for conversations.

This is the interface the assistant actually uses. It:
1. Logs everything
2. Decides when to write
3. Retrieves before responding
4. Tracks conversation turns
"""

import asyncio
import sys
import os
import hashlib
import time
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from perfect_recall.db.connection import DatabaseManager
from perfect_recall.core.memory_writer import MemoryWriter
from perfect_recall.core.retrieval import RetrievalPipeline, SalienceScorer
from perfect_recall.core.abstention import AbstentionController
from perfect_recall.models.memory import MemoryTier

from memory_logger import get_logger
from local_embeddings import get_embedding_func


class MemoryOrchestrator:
    """
    Orchestrates memory operations during conversations.
    
    Usage:
        orch = await get_orchestrator()
        
        # Before responding, check for context
        context = await orch.get_context("user's question")
        
        # After responding, store what we learned
        await orch.store_fact("dev", "wants", "automatic memory writes")
        
        # At end of conversation
        await orch.end_conversation()
    """
    
    def __init__(self):
        self.db = DatabaseManager()
        self.writer: Optional[MemoryWriter] = None
        self.retriever: Optional[RetrievalPipeline] = None
        self.abstention: Optional[AbstentionController] = None
        self.logger = get_logger()
        
        self._initialized = False
        self._turn_count = 0
        self._facts_this_turn: List[str] = []
        self._memories_used_this_turn: List[str] = []
    
    async def initialize(self):
        """Initialize the memory system."""
        if self._initialized:
            return
        
        await self.db.initialize()
        
        # Use local embeddings
        local_embed = get_embedding_func()
        
        # Embedding function with logging
        def embedding_with_logs(text: str) -> list[float]:
            start = time.time()
            
            # Generate embedding using local embedder
            result = local_embed(text)
            
            latency = (time.time() - start) * 1000
            embedding_hash = hashlib.sha256(str(result).encode()).hexdigest()[:16]
            
            self.logger.log_embedding(
                text=text,
                embedding_hash=embedding_hash,
                model="local_hash_v1",
                latency_ms=latency,
            )
            
            return result
        
        self.writer = MemoryWriter(
            db_manager=self.db,
            embedding_func=embedding_with_logs,
            write_threshold=0.5,
        )
        
        self.retriever = RetrievalPipeline(
            db_manager=self.db,
            embedding_func=embedding_with_logs,
            scorer=SalienceScorer(),
        )
        
        self.abstention = AbstentionController()
        
        self._initialized = True
        print("✅ Memory Orchestrator initialized with local embeddings")
    
    async def get_context(self, query: str, limit: int = 5) -> dict:
        """
        Get relevant context before responding to a query.
        
        Returns:
            {
                'memories': [...],
                'should_use': bool,
                'confidence': float,
            }
        """
        if not self._initialized:
            await self.initialize()
        
        start = time.time()
        
        # Retrieve
        results = await self.retriever.retrieve(query, limit=limit)
        
        # Check abstention
        decision = self.abstention.should_abstain(query, results)
        
        latency = (time.time() - start) * 1000
        
        # Format for logging
        log_results = [
            {
                'id': str(r.memory.id),
                'score': r.salience_score,
                'tier': r.memory.memory_tier.value,
                'content': r.memory.content,
            }
            for r in results
        ]
        
        query_embedding = self.writer.embedding_func(query) if self.writer else []
        query_hash = hashlib.sha256(str(query_embedding).encode()).hexdigest()[:16]
        
        self.logger.log_retrieve(
            query=query,
            query_embedding_hash=query_hash,
            results=log_results,
            abstention={
                'abstain': decision.abstain,
                'reason': decision.reason if decision.abstain else None,
            },
            latency_ms=latency,
        )
        
        # Track for turn summary
        if not decision.abstain:
            self._memories_used_this_turn = [r.memory.content[:100] for r in results[:3]]
        
        return {
            'memories': [
                {
                    'content': r.memory.content,
                    'tier': r.memory.memory_tier.value,
                    'score': r.salience_score,
                }
                for r in results
            ],
            'should_use': not decision.abstain,
            'confidence': decision.confidence,
        }
    
    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.8,
    ) -> bool:
        """Store a semantic fact."""
        if not self._initialized:
            await self.initialize()
        
        start = time.time()
        content = f"{subject} {predicate} {object}"
        
        try:
            memory = await self.writer.store_fact(
                subject=subject,
                predicate=predicate,
                object=object,
                confidence=confidence,
            )
            
            latency = (time.time() - start) * 1000
            
            # Log the write
            self.logger.log_write(
                operation="store_fact",
                content=content,
                tier="semantic",
                confidence=confidence,
                gate_scores={"final": confidence},  # Simplified
                memory_id=str(memory.id) if memory else None,
                latency_ms=latency,
            )
            
            if memory:
                self._facts_this_turn.append(content)
                return True
            return False
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            self.logger.log_write(
                operation="store_fact",
                content=content,
                tier="semantic",
                confidence=confidence,
                gate_scores={},
                latency_ms=latency,
                error=str(e),
            )
            return False
    
    async def store_episode(
        self,
        content: str,
        importance: float = 0.7,
    ) -> bool:
        """Store an episodic memory."""
        if not self._initialized:
            await self.initialize()
        
        start = time.time()
        
        try:
            from perfect_recall.models.memory import EpisodeType
            
            memory = await self.writer.record_episode(
                content=content,
                episode_type=EpisodeType.CONVERSATION,
            )
            
            latency = (time.time() - start) * 1000
            
            self.logger.log_write(
                operation="record_episode",
                content=content,
                tier="episodic",
                confidence=importance,
                gate_scores={"final": importance},
                memory_id=str(memory.id) if memory else None,
                latency_ms=latency,
            )
            
            if memory:
                self._facts_this_turn.append(content)
                return True
            return False
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            self.logger.log_write(
                operation="record_episode",
                content=content,
                tier="episodic",
                confidence=importance,
                gate_scores={},
                latency_ms=latency,
                error=str(e),
            )
            return False
    
    def start_turn(self):
        """Call at the start of each conversation turn."""
        self._turn_count += 1
        self._facts_this_turn = []
        self._memories_used_this_turn = []
    
    async def end_turn(self, user_message: str):
        """Call at the end of each turn to log the summary."""
        self.logger.log_conversation_turn(
            turn_number=self._turn_count,
            user_message=user_message,
            facts_extracted=self._facts_this_turn,
            memories_retrieved=self._memories_used_this_turn,
        )
    
    def get_stats(self) -> dict:
        """Get current session stats."""
        return {
            **self.logger.get_stats(),
            'turns': self._turn_count,
        }
    
    def get_recent_activity(self, n: int = 5) -> dict:
        """Get recent activity for inspection."""
        return {
            'recent_writes': self.logger.get_recent_writes(n),
            'recent_retrieves': self.logger.get_recent_retrieves(n),
        }


# Global orchestrator
_orchestrator: Optional[MemoryOrchestrator] = None


async def get_orchestrator() -> MemoryOrchestrator:
    """Get or create the global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MemoryOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator


# Convenience functions for direct use
async def remember_fact(subject: str, predicate: str, object: str) -> bool:
    """Quick store a fact."""
    orch = await get_orchestrator()
    return await orch.store_fact(subject, predicate, object)


async def recall(query: str) -> list[dict]:
    """Quick retrieve."""
    orch = await get_orchestrator()
    context = await orch.get_context(query)
    return context['memories'] if context['should_use'] else []


if __name__ == "__main__":
    async def test():
        orch = await get_orchestrator()
        
        print("\n" + "="*60)
        print("Memory Orchestrator Test")
        print("="*60)
        
        # Simulate a conversation turn
        orch.start_turn()
        
        # User asks something
        user_msg = "are you actually logging everything now?"
        print(f"\n👤 User: {user_msg}")
        
        # Check context before responding
        context = await orch.get_context(user_msg)
        print(f"\n🔍 Retrieved {len(context['memories'])} memories")
        print(f"   Should use: {context['should_use']}")
        print(f"   Confidence: {context['confidence']:.2f}")
        
        # Store what we learned
        await orch.store_fact("dev", "wants", "comprehensive logging", confidence=0.95)
        await orch.store_fact("dev", "asked about", "memory system integration", confidence=0.9)
        
        # End turn
        await orch.end_turn(user_msg)
        
        # Show stats
        print(f"\n📊 Stats: {orch.get_stats()}")
        
        # Show recent activity
        activity = orch.get_recent_activity(3)
        print(f"\n📝 Recent writes: {len(activity['recent_writes'])}")
        print(f"🔍 Recent retrieves: {len(activity['recent_retrieves'])}")
        
        print("\n✅ Orchestrator test complete!")
        print("="*60)
    
    asyncio.run(test())
