#!/usr/bin/env python3
"""
Structured logging for Perfect Recall memory system.

Every operation is logged as JSON for later analysis.
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from contextlib import contextmanager


class MemoryLogger:
    """Structured logger for memory operations."""
    
    def __init__(self, log_dir: str = "/root/.openclaw/workspace/perfect-recall/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily log files
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"memory-{self.current_date}.jsonl"
        
        # Session tracking
        self.session_id = self._generate_session_id()
        self.write_count = 0
        self.retrieve_count = 0
        
        # In-memory buffer for recent events (last 100)
        self.recent_events: list[dict] = []
        self.max_buffer = 100
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(timestamp.encode()).hexdigest()[:16]
    
    def _get_timestamp(self) -> str:
        """ISO format timestamp with timezone."""
        return datetime.now().astimezone().isoformat()
    
    def _write_log(self, event: dict):
        """Write event to log file and buffer."""
        # Add to buffer
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_buffer:
            self.recent_events.pop(0)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
    
    def log_write(
        self,
        operation: str,
        content: str,
        tier: str,
        confidence: float,
        gate_scores: dict,
        memory_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ):
        """Log a memory write operation."""
        self.write_count += 1
        
        event = {
            "timestamp": self._get_timestamp(),
            "level": "ERROR" if error else "INFO",
            "component": "memory_writer",
            "operation": operation,
            "session_id": self.session_id,
            "latency_ms": latency_ms,
            "payload": {
                "content": content[:500],  # Truncate long content
                "tier": tier,
                "confidence": confidence,
                "write_gate_scores": gate_scores,
                "memory_id": memory_id,
                "error": error,
                "write_number": self.write_count,
            }
        }
        self._write_log(event)
    
    def log_retrieve(
        self,
        query: str,
        query_embedding_hash: str,
        results: list[dict],
        abstention: dict,
        latency_ms: float,
        raw_sql: Optional[str] = None,
    ):
        """Log a retrieval operation."""
        self.retrieve_count += 1
        
        # Calculate summary stats
        scores = [r.get('score', 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        # Warn if low confidence
        level = "WARN" if max_score < 0.3 and not abstention.get('abstain') else "INFO"
        
        event = {
            "timestamp": self._get_timestamp(),
            "level": level,
            "component": "retrieval",
            "operation": "retrieve",
            "session_id": self.session_id,
            "latency_ms": latency_ms,
            "payload": {
                "query": query[:500],
                "query_embedding_hash": query_embedding_hash,
                "results_count": len(results),
                "results": [
                    {
                        "memory_id": r.get('id'),
                        "score": r.get('score'),
                        "tier": r.get('tier'),
                        "content_preview": r.get('content', '')[:100],
                    }
                    for r in results[:5]  # Top 5 only
                ],
                "score_stats": {
                    "avg": round(avg_score, 3),
                    "max": round(max_score, 3),
                    "min": round(min(scores), 3) if scores else 0,
                },
                "abstention": abstention,
                "retrieve_number": self.retrieve_count,
            }
        }
        self._write_log(event)
    
    def log_embedding(
        self,
        text: str,
        embedding_hash: str,
        model: str,
        latency_ms: float,
        error: Optional[str] = None,
    ):
        """Log embedding generation."""
        level = "ERROR" if error else "DEBUG"
        
        event = {
            "timestamp": self._get_timestamp(),
            "level": level,
            "component": "embedding",
            "operation": "generate",
            "session_id": self.session_id,
            "latency_ms": latency_ms,
            "payload": {
                "text_preview": text[:200],
                "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
                "embedding_hash": embedding_hash,
                "model": model,
                "error": error,
            }
        }
        self._write_log(event)
    
    def log_decision(
        self,
        decision: str,
        context: dict,
        memories_used: list[str],
    ):
        """Log a high-level decision (e.g., whether to use memories in response)."""
        event = {
            "timestamp": self._get_timestamp(),
            "level": "INFO",
            "component": "orchestrator",
            "operation": "decision",
            "session_id": self.session_id,
            "payload": {
                "decision": decision,
                "context": context,
                "memories_used": memories_used,
            }
        }
        self._write_log(event)
    
    def log_conversation_turn(
        self,
        turn_number: int,
        user_message: str,
        facts_extracted: list[str],
        memories_retrieved: list[str],
    ):
        """Log a complete conversation turn summary."""
        event = {
            "timestamp": self._get_timestamp(),
            "level": "INFO",
            "component": "conversation",
            "operation": "turn",
            "session_id": self.session_id,
            "payload": {
                "turn_number": turn_number,
                "user_message_preview": user_message[:500],
                "facts_stored": len(facts_extracted),
                "facts_extracted": facts_extracted[:10],  # First 10
                "memories_retrieved_count": len(memories_retrieved),
                "memories_retrieved": memories_retrieved[:5],  # First 5
            }
        }
        self._write_log(event)
    
    @contextmanager
    def timer(self, operation: str, component: str):
        """Context manager to time operations."""
        start = time.time()
        try:
            yield
            latency = (time.time() - start) * 1000
            self._write_log({
                "timestamp": self._get_timestamp(),
                "level": "DEBUG",
                "component": component,
                "operation": f"{operation}_timing",
                "session_id": self.session_id,
                "latency_ms": round(latency, 2),
                "payload": {"status": "success"}
            })
        except Exception as e:
            latency = (time.time() - start) * 1000
            self._write_log({
                "timestamp": self._get_timestamp(),
                "level": "ERROR",
                "component": component,
                "operation": f"{operation}_timing",
                "session_id": self.session_id,
                "latency_ms": round(latency, 2),
                "payload": {"status": "error", "error": str(e)}
            })
            raise
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "session_id": self.session_id,
            "writes": self.write_count,
            "retrieves": self.retrieve_count,
            "log_file": str(self.log_file),
            "recent_events": len(self.recent_events),
        }
    
    def get_recent_writes(self, n: int = 10) -> list[dict]:
        """Get recent write operations."""
        writes = [e for e in self.recent_events if e.get('component') == 'memory_writer']
        return writes[-n:]
    
    def get_recent_retrieves(self, n: int = 10) -> list[dict]:
        """Get recent retrieval operations."""
        retrieves = [e for e in self.recent_events if e.get('component') == 'retrieval']
        return retrieves[-n:]


# Global logger instance
_logger: Optional[MemoryLogger] = None


def get_logger() -> MemoryLogger:
    """Get or create global logger."""
    global _logger
    if _logger is None:
        _logger = MemoryLogger()
    return _logger


if __name__ == "__main__":
    # Test logging
    logger = get_logger()
    
    logger.log_write(
        operation="store_fact",
        content="dev wants comprehensive logging for memory system",
        tier="semantic",
        confidence=0.95,
        gate_scores={"novelty": 0.9, "utility": 0.95, "confidence": 0.9, "final": 0.92},
        memory_id="test-uuid-123",
        latency_ms=45.2,
    )
    
    logger.log_retrieve(
        query="how does dev want logging",
        query_embedding_hash="a3f2...",
        results=[
            {"id": "test-uuid-123", "score": 0.87, "tier": "semantic", "content": "dev wants comprehensive logging..."},
        ],
        abstention={"abstain": False},
        latency_ms=120.5,
    )
    
    logger.log_conversation_turn(
        turn_number=1,
        user_message="are you actually doing it? when and how do you know when to write to it?",
        facts_extracted=["dev wants comprehensive logging", "dev wants automatic memory writes"],
        memories_retrieved=[],
    )
    
    print(f"Logger stats: {logger.get_stats()}")
    print(f"Recent writes: {len(logger.get_recent_writes())}")
    print(f"Log file: {logger.log_file}")
