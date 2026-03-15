#!/usr/bin/env python3
"""
Auto-Memory Integration for Assistant

This module integrates Perfect Recall directly into the assistant's
response flow:
- ALWAYS retrieves before responding (context awareness)
- Stores based on confidence tiers (high = auto, medium = flag, low = skip)
- Logs everything for review
"""

import asyncio
import sys
import os
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from memory_orchestrator import get_orchestrator, MemoryOrchestrator
from memory_logger import get_logger
from local_embeddings import get_embedding_func


@dataclass
class ExtractionResult:
    """A fact extracted from conversation."""
    subject: str
    predicate: str
    object: str
    confidence: float  # 0-1
    reason: str
    quote: str  # Original text


class MemoryIntegration:
    """
    Integrates memory into assistant workflow.
    
    Usage:
        mem = await get_memory_integration()
        
        # Before responding
        context = await mem.get_context("user message")
        
        # After responding  
        await mem.process_turn("user message", "assistant response")
    """
    
    def __init__(self):
        self.orch: Optional[MemoryOrchestrator] = None
        self.logger = get_logger()
        self._turn_count = 0
        
        # High-confidence patterns (auto-store)
        self.high_confidence_patterns = {
            r'\b(?:my name is|i am|i\'m)\s+(\w+)': ('dev', 'name is', 0.95),
            r'\b(i|we)\s+(?:prefer|like|love|enjoy)\s+(.+?)(?:\.|$|,)': ('dev', 'prefers', 0.9),
            r'\b(i|we)\s+(?:hate|dislike|can\'t stand)\s+(.+?)(?:\.|$|,)': ('dev', 'dislikes', 0.9),
            r'\b(i|we)\s+(?:decided|chose|picked|went with)\s+(.+?)(?:\.|$|,)': ('dev', 'decided', 0.9),
            r'\b(?:remember this|important|don\'t forget|note that)\s*:?\s*(.+?)(?:\.|$)': ('dev', 'noted', 1.0),
            r'\b(my|our)\s+(?:goal|priority|focus)\s+(?:is|should be)\s+(.+?)(?:\.|$|,)': ('dev', 'goal is', 0.85),
            r'\b(?:from now on|going forward|in the future)\s*,?\s*(.+?)(?:\.|$)': ('dev', 'wants', 0.85),
            r'\b(i|we)\s+(?:need|want|require)\s+(.+?)(?:\.|$|,)': ('dev', 'wants', 0.8),
            r'\b(?:i work at|my company is|i\'m at)\s+(.+?)(?:\.|$|,)': ('dev', 'works at', 0.9),
            r'\b(?:i live in|my home is|based in)\s+(.+?)(?:\.|$|,)': ('dev', 'location is', 0.9),
        }
        
        # Medium-confidence (flag for review)
        self.medium_confidence_patterns = {
            r'\b(?:usually|typically|generally|often)\s+(.+?)(?:\.|$)': ('dev', 'usually', 0.6),
            r'\b(?:sometimes|occasionally)\s+(.+?)(?:\.|$)': ('dev', 'sometimes', 0.5),
            r'\b(i think|i believe|in my opinion)\s+(.+?)(?:\.|$)': ('dev', 'thinks', 0.55),
        }
    
    async def initialize(self):
        """Initialize the memory system."""
        if self.orch is None:
            self.orch = await get_orchestrator()
        return self
    
    async def get_context(self, user_message: str) -> Dict:
        """
        Get relevant context before responding.
        ALWAYS call this before generating a response.
        """
        if not self.orch:
            await self.initialize()
        
        # Start turn tracking
        self.orch.start_turn()
        self._turn_count += 1
        
        # Retrieve relevant memories
        context = await self.orch.get_context(user_message, limit=5)
        
        # Log decision
        self.logger.log_decision(
            decision="retrieved" if context['should_use'] else "abstained",
            context={
                'query': user_message[:200],
                'results_count': len(context['memories']),
                'confidence': context['confidence'],
            },
            memories_used=[m['content'][:100] for m in context['memories'][:3]],
        )
        
        return context
    
    def extract_facts(self, user_message: str) -> List[ExtractionResult]:
        """
        Extract facts from user message using patterns.
        
        Returns facts with confidence scores.
        """
        facts = []
        
        # High confidence patterns
        for pattern, (subject, predicate, conf) in self.high_confidence_patterns.items():
            for match in re.finditer(pattern, user_message, re.IGNORECASE):
                groups = match.groups()
                if groups:
                    obj = groups[-1] if len(groups) > 1 else groups[0]
                    facts.append(ExtractionResult(
                        subject=subject,
                        predicate=predicate,
                        object=obj.strip(),
                        confidence=conf,
                        reason=f"High-confidence pattern: {pattern[:30]}...",
                        quote=match.group(0),
                    ))
        
        # Medium confidence patterns
        for pattern, (subject, predicate, conf) in self.medium_confidence_patterns.items():
            for match in re.finditer(pattern, user_message, re.IGNORECASE):
                groups = match.groups()
                if groups:
                    obj = groups[-1] if len(groups) > 1 else groups[0]
                    facts.append(ExtractionResult(
                        subject=subject,
                        predicate=predicate,
                        object=obj.strip(),
                        confidence=conf,
                        reason=f"Medium-confidence pattern: {pattern[:30]}...",
                        quote=match.group(0),
                    ))
        
        return facts
    
    async def process_turn(
        self,
        user_message: str,
        assistant_response: str,
        auto_store: bool = True,
    ) -> Dict:
        """
        Process a complete conversation turn.
        
        1. Extract facts from user message
        2. Store high-confidence facts (if auto_store=True)
        3. Flag medium-confidence facts for review
        4. Log the turn
        
        Returns:
            {
                'stored': [...],  # What was auto-stored
                'flagged': [...], # What needs review
                'skipped': [...], # What was too low confidence
            }
        """
        if not self.orch:
            await self.initialize()
        
        # Extract facts
        facts = self.extract_facts(user_message)
        
        stored = []
        flagged = []
        skipped = []
        
        for fact in facts:
            if fact.confidence >= 0.8 and auto_store:
                # Auto-store high confidence
                success = await self.orch.store_fact(
                    fact.subject,
                    fact.predicate,
                    fact.object,
                    confidence=fact.confidence,
                )
                if success:
                    stored.append(fact)
                    
            elif fact.confidence >= 0.5:
                # Flag for review
                flagged.append(fact)
                
            else:
                # Skip low confidence
                skipped.append(fact)
        
        # End turn logging
        await self.orch.end_turn(user_message)
        
        return {
            'stored': stored,
            'flagged': flagged,
            'skipped': skipped,
            'stats': self.orch.get_stats(),
        }
    
    async def store_episode(self, content: str, importance: float = 0.7) -> bool:
        """Manually store an episodic memory."""
        if not self.orch:
            await self.initialize()
        return await self.orch.store_episode(content, importance)
    
    async def store_fact(self, subject: str, predicate: str, obj: str) -> bool:
        """Manually store a fact."""
        if not self.orch:
            await self.initialize()
        return await self.orch.store_fact(subject, predicate, obj)
    
    def format_context_for_prompt(self, context: Dict) -> str:
        """
        Format retrieved memories for inclusion in system prompt.
        """
        if not context['memories'] or not context['should_use']:
            return ""
        
        lines = ["\n[Relevant context from memory:]"]
        for i, mem in enumerate(context['memories'][:3], 1):
            lines.append(f"{i}. {mem['content']} (confidence: {mem['score']:.2f})")
        lines.append("")
        
        return "\n".join(lines)
    
    async def get_stats(self) -> Dict:
        """Get memory system stats."""
        if not self.orch:
            await self.initialize()
        return self.orch.get_stats()
    
    async def get_recent_activity(self, n: int = 5) -> Dict:
        """Get recent activity."""
        if not self.orch:
            await self.initialize()
        return self.orch.get_recent_activity(n)


# Global instance
_memory_integration: Optional[MemoryIntegration] = None


async def get_memory_integration() -> MemoryIntegration:
    """Get or create the global memory integration."""
    global _memory_integration
    if _memory_integration is None:
        _memory_integration = MemoryIntegration()
        await _memory_integration.initialize()
    return _memory_integration


# Test
if __name__ == "__main__":
    async def test():
        mem = await get_memory_integration()
        
        print("\n" + "="*60)
        print("Memory Integration Test")
        print("="*60)
        
        # Simulate a conversation
        test_messages = [
            "My name is dev and I prefer Python over JavaScript",
            "I decided to use PostgreSQL for this project",
            "Remember this: I hate waiting for slow builds",
            "Sometimes I work late at night",
        ]
        
        for msg in test_messages:
            print(f"\n👤 User: {msg}")
            
            # Get context (what would happen before I respond)
            context = await mem.get_context(msg)
            if context['memories']:
                print("🔍 Retrieved context:")
                for m in context['memories']:
                    print(f"   - {m['content'][:50]}...")
            
            # Process turn (what happens after I respond)
            result = await mem.process_turn(msg, "Good to know!")
            
            print(f"✅ Stored: {len(result['stored'])}")
            for f in result['stored']:
                print(f"   → {f.subject} {f.predicate} {f.object} ({f.confidence})")
            
            if result['flagged']:
                print(f"⚠️  Flagged for review: {len(result['flagged'])}")
                for f in result['flagged']:
                    print(f"   → {f.subject} {f.predicate} {f.object} ({f.confidence})")
        
        # Show final stats
        print(f"\n📊 Final Stats:")
        stats = await mem.get_stats()
        for k, v in stats.items():
            print(f"   {k}: {v}")
        
        # Show how context would look in prompt
        print("\n📝 Example context formatting:")
        context = await mem.get_context("what do I prefer?")
        formatted = mem.format_context_for_prompt(context)
        if formatted:
            print(formatted)
        else:
            print("(No relevant context found)")
        
        print("\n✅ Test complete!")
        print("="*60)
    
    asyncio.run(test())
