"""Tests for the MemoryWriter module."""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from perfect_recall.models.memory import MemoryNode, MemoryTier, EpisodeType
from perfect_recall.models.retrieval import WriteDecision
from perfect_recall.core.memory_writer import MemoryWriter


def mock_embedding(text: str) -> list[float]:
    """Generate mock embedding vector."""
    return [0.1] * 1536


class TestWriteGate:
    """Test the write gate decision logic."""
    
    def test_high_utility_content_passes(self):
        """Content with high-utility keywords should pass."""
        writer = MemoryWriter(db_manager=None, embedding_func=mock_embedding)
        
        # Content with explicit importance marker
        decision = writer._should_write(
            "Remember this: my password is secret123",
            "message",
            {}
        )
        
        assert decision.write is True
        assert decision.importance > 0.6
    
    def test_low_utility_content_rejected(self):
        """Low utility content should be rejected."""
        writer = MemoryWriter(db_manager=None, embedding_func=mock_embedding)
        
        # Generic content
        decision = writer._should_write(
            "ok",
            "message",
            {}
        )
        
        assert decision.write is False
    
    def test_preference_detection(self):
        """Content with preferences should score high utility."""
        writer = MemoryWriter(db_manager=None, embedding_func=mock_embedding)
        
        decision = writer._should_write(
            "I prefer using Python for data analysis",
            "message",
            {}
        )
        
        assert decision.factors['utility'] >= 0.8


class TestMemoryNode:
    """Test MemoryNode model."""
    
    def test_salience_calculation(self):
        """Test salience score calculation."""
        memory = MemoryNode(
            memory_tier=MemoryTier.EPISODIC,
            content="Test content",
            importance_score=0.8,
            access_count=10,
        )
        
        salience = memory.calculate_salience()
        
        # Should be between 0 and 1
        assert 0 <= salience <= 1
        
        # Higher importance should increase salience
        assert salience > 0.3
    
    def test_active_check(self):
        """Test memory active/expired check."""
        from datetime import datetime, timedelta
        
        # Active memory
        active = MemoryNode(
            memory_tier=MemoryTier.EPISODIC,
            content="Active",
            valid_until=None,
        )
        assert active.is_active() is True
        
        # Expired memory
        expired = MemoryNode(
            memory_tier=MemoryTier.EPISODIC,
            content="Expired",
            valid_until=datetime.utcnow() - timedelta(days=1),
        )
        assert expired.is_active() is False


class TestWriteDecision:
    """Test WriteDecision model."""
    
    def test_boolean_conversion(self):
        """WriteDecision should be usable as boolean."""
        approved = WriteDecision(write=True, importance=0.8)
        rejected = WriteDecision(write=False, importance=0.3)
        
        assert bool(approved) is True
        assert bool(rejected) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
