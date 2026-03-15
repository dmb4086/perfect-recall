#!/usr/bin/env python3
"""
Memory Manager CLI

Review, manage, and analyze the Perfect Recall memory system.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from perfect_recall.db.connection import DatabaseManager
from sqlalchemy import text


class MemoryManager:
    """CLI for managing Perfect Recall memories."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.log_dir = Path("/root/.openclaw/workspace/perfect-recall/logs")
    
    async def initialize(self):
        await self.db.initialize()
    
    async def get_stats(self):
        """Get memory system statistics."""
        async with self.db.session() as session:
            # Total counts
            result = await session.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE memory_tier = 'episodic') as episodic,
                    COUNT(*) FILTER (WHERE memory_tier = 'semantic') as semantic,
                    COUNT(*) FILTER (WHERE memory_tier = 'procedural') as procedural,
                    COUNT(*) FILTER (WHERE memory_tier = 'working') as working
                FROM memory_nodes
            """))
            row = result.fetchone()
            
            # Recent additions
            result = await session.execute(text("""
                SELECT COUNT(*) 
                FROM memory_nodes 
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """))
            recent = result.scalar()
            
            # Top confidences
            result = await session.execute(text("""
                SELECT content, confidence, memory_tier
                FROM memory_nodes
                ORDER BY confidence DESC
                LIMIT 5
            """))
            top_memories = result.fetchall()
            
        return {
            'total': row.total,
            'by_tier': {
                'episodic': row.episodic,
                'semantic': row.semantic,
                'procedural': row.procedural,
                'working': row.working,
            },
            'recent_24h': recent,
            'top_memories': [
                {'content': m.content[:60], 'confidence': m.confidence, 'tier': m.memory_tier}
                for m in top_memories
            ]
        }
    
    async def list_memories(self, tier: Optional[str] = None, limit: int = 20):
        """List memories with optional tier filter."""
        async with self.db.session() as session:
            if tier:
                result = await session.execute(text("""
                    SELECT id, content, memory_tier, confidence, created_at
                    FROM memory_nodes
                    WHERE memory_tier = :tier
                    ORDER BY created_at DESC
                    LIMIT :limit
                """), {'tier': tier, 'limit': limit})
            else:
                result = await session.execute(text("""
                    SELECT id, content, memory_tier, confidence, created_at
                    FROM memory_nodes
                    ORDER BY created_at DESC
                    LIMIT :limit
                """), {'limit': limit})
            
            return result.fetchall()
    
    async def search_memories(self, query: str):
        """Search memories by content."""
        async with self.db.session() as session:
            result = await session.execute(text("""
                SELECT id, content, memory_tier, confidence, created_at
                FROM memory_nodes
                WHERE content ILIKE :query
                ORDER BY confidence DESC
                LIMIT 10
            """), {'query': f'%{query}%'})
            return result.fetchall()
    
    async def delete_memory(self, memory_id: str):
        """Delete a memory by ID."""
        async with self.db.session() as session:
            await session.execute(text("""
                DELETE FROM memory_nodes WHERE id = :id
            """), {'id': memory_id})
        return True
    
    def analyze_logs(self, days: int = 1):
        """Analyze log files for patterns."""
        log_files = []
        cutoff = datetime.now() - timedelta(days=days)
        
        for log_file in self.log_dir.glob("memory-*.jsonl"):
            try:
                date_str = log_file.stem.replace("memory-", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date >= cutoff:
                    log_files.append(log_file)
            except ValueError:
                continue
        
        if not log_files:
            return None
        
        # Parse logs
        writes = []
        retrieves = []
        turns = []
        
        for log_file in log_files:
            with open(log_file) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        if event.get('component') == 'memory_writer':
                            writes.append(event)
                        elif event.get('component') == 'retrieval':
                            retrieves.append(event)
                        elif event.get('component') == 'conversation':
                            turns.append(event)
                    except json.JSONDecodeError:
                        continue
        
        # Calculate metrics
        retrieval_hit_rate = len([r for r in retrieves if r['payload'].get('results_count', 0) > 0]) / len(retrieves) if retrieves else 0
        
        avg_top_score = sum(
            r['payload'].get('score_stats', {}).get('max', 0) 
            for r in retrieves
        ) / len(retrieves) if retrieves else 0
        
        return {
            'period_days': days,
            'total_writes': len(writes),
            'total_retrieves': len(retrieves),
            'total_turns': len(turns),
            'retrieval_hit_rate': f"{retrieval_hit_rate:.1%}",
            'avg_top_score': f"{avg_top_score:.2f}",
            'writes_per_turn': len(writes) / len(turns) if turns else 0,
        }
    
    def show_recent_logs(self, n: int = 10):
        """Show recent log entries."""
        log_files = sorted(self.log_dir.glob("memory-*.jsonl"), reverse=True)
        if not log_files:
            return []
        
        events = []
        with open(log_files[0]) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        return events[-n:]


async def main():
    """CLI main function."""
    manager = MemoryManager()
    await manager.initialize()
    
    args = sys.argv[1:]
    
    if not args or args[0] == 'status':
        print("\n📊 Memory System Status")
        print("=" * 50)
        stats = await manager.get_stats()
        print(f"Total memories: {stats['total']}")
        print(f"  Episodic: {stats['by_tier']['episodic']}")
        print(f"  Semantic: {stats['by_tier']['semantic']}")
        print(f"  Procedural: {stats['by_tier']['procedural']}")
        print(f"  Working: {stats['by_tier']['working']}")
        print(f"\nAdded in last 24h: {stats['recent_24h']}")
        print("\nTop memories by confidence:")
        for m in stats['top_memories']:
            print(f"  [{m['tier'][:3]}] {m['content'][:50]}... ({m['confidence']})")
    
    elif args[0] == 'list':
        tier = args[1] if len(args) > 1 else None
        limit = int(args[2]) if len(args) > 2 else 20
        memories = await manager.list_memories(tier, limit)
        print(f"\n📋 Recent memories" + (f" ({tier})" if tier else ""))
        print("=" * 50)
        for m in memories:
            print(f"[{m.memory_tier[:3]}] {m.content[:60]}...")
            print(f"      confidence: {m.confidence:.2f} | id: {m.id}")
    
    elif args[0] == 'search':
        if len(args) < 2:
            print("Usage: memory_manager.py search <query>")
            return
        query = args[1]
        results = await manager.search_memories(query)
        print(f"\n🔍 Search results for '{query}':")
        print("=" * 50)
        for r in results:
            print(f"[{r.memory_tier[:3]}] {r.content}")
            print(f"      confidence: {r.confidence:.2f}")
    
    elif args[0] == 'analytics':
        days = int(args[1]) if len(args) > 1 else 1
        analysis = manager.analyze_logs(days)
        if analysis:
            print(f"\n📈 Analytics (last {days} day(s))")
            print("=" * 50)
            for k, v in analysis.items():
                print(f"  {k}: {v}")
        else:
            print("No log data found for the specified period.")
    
    elif args[0] == 'logs':
        n = int(args[1]) if len(args) > 1 else 10
        events = manager.show_recent_logs(n)
        print(f"\n📝 Recent log events (last {len(events)})")
        print("=" * 50)
        for e in events:
            ts = e['timestamp'][:19] if 'timestamp' in e else '?'
            comp = e.get('component', '?')
            op = e.get('operation', '?')
            print(f"{ts} | {comp:15} | {op}")
    
    elif args[0] == 'delete':
        if len(args) < 2:
            print("Usage: memory_manager.py delete <memory_id>")
            return
        memory_id = args[1]
        await manager.delete_memory(memory_id)
        print(f"✅ Deleted memory {memory_id}")
    
    else:
        print("""
Memory Manager CLI

Commands:
  status              Show memory system status
  list [tier] [n]     List recent memories
  search <query>      Search memories by content
  analytics [days]    Show analytics for period
  logs [n]            Show recent log entries
  delete <id>         Delete a memory

Examples:
  python memory_manager.py status
  python memory_manager.py list semantic 10
  python memory_manager.py search "python"
  python memory_manager.py analytics 7
        """)


if __name__ == "__main__":
    asyncio.run(main())
