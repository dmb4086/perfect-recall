#!/usr/bin/env python3
"""Add diverse test memories for shadow mode validation."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perfect_recall import PerfectRecall
from perfect_recall.models.memory import EpisodeType


async def add_test_memories():
    """Add 15+ diverse test memories."""
    pr = await PerfectRecall.create()
    
    # Diverse test memories organized by category
    test_memories = [
        # Preferences (5)
        {
            "content": "Developer prefers to use Vim over Emacs for text editing",
            "category": "preference",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "User enjoys listening to lo-fi music while coding",
            "category": "preference",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "User prefers light theme during daytime and dark theme at night",
            "category": "preference",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "User likes their coffee with oat milk and no sugar",
            "category": "preference",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "User prefers to receive code reviews via GitHub PRs rather than email",
            "category": "preference",
            "episode_type": EpisodeType.OBSERVATION,
        },
        
        # Facts (5)
        {
            "content": "The team uses AWS for cloud infrastructure with EC2 and RDS",
            "category": "fact",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "The project deadline is set for end of Q2 2026",
            "category": "fact",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "The main database is PostgreSQL 15 running on db.t3.medium",
            "category": "fact",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "CI/CD pipeline runs on GitHub Actions with self-hosted runners",
            "category": "fact",
            "episode_type": EpisodeType.OBSERVATION,
        },
        {
            "content": "The application uses Redis for caching and session storage",
            "category": "fact",
            "episode_type": EpisodeType.OBSERVATION,
        },
        
        # TODOs/Tasks (4) - Use explicit markers to pass write gate
        {
            "content": "Remember to refactor the authentication middleware to use JWT tokens",
            "category": "task",
            "episode_type": EpisodeType.ACTION,
        },
        {
            "content": "Don't forget to update API documentation with new endpoints",
            "category": "task",
            "episode_type": EpisodeType.ACTION,
        },
        {
            "content": "Important: Schedule performance testing for next week",
            "category": "task",
            "episode_type": EpisodeType.ACTION,
        },
        {
            "content": "Important: Review security audit report and address critical findings",
            "category": "task",
            "episode_type": EpisodeType.ACTION,
        },
        
        # Relationships/People (4)
        {
            "content": "Sarah is the tech lead for the backend team",
            "category": "relationship",
            "episode_type": EpisodeType.INTERACTION,
        },
        {
            "content": "Mike handles DevOps and infrastructure management",
            "category": "relationship",
            "episode_type": EpisodeType.INTERACTION,
        },
        {
            "content": "User reports to Jennifer who is the Engineering Manager",
            "category": "relationship",
            "episode_type": EpisodeType.INTERACTION,
        },
        {
            "content": "Alex is the frontend specialist working on React components",
            "category": "relationship",
            "episode_type": EpisodeType.INTERACTION,
        },
        
        # Decisions (3)
        {
            "content": "Decision: We will use React Query for server state management",
            "category": "decision",
            "episode_type": EpisodeType.DECISION,
        },
        {
            "content": "Decision: Migrate from REST to GraphQL for API v2",
            "category": "decision",
            "episode_type": EpisodeType.DECISION,
        },
        {
            "content": "Decision: Use Terraform for infrastructure as code",
            "category": "decision",
            "episode_type": EpisodeType.DECISION,
        },
        
        # Reflections/Learnings (2)
        {
            "content": "Reflection: The monorepo approach has improved code sharing but slowed CI",
            "category": "reflection",
            "episode_type": EpisodeType.REFLECTION,
        },
        {
            "content": "Learning: Async database queries significantly improved API response times",
            "category": "reflection",
            "episode_type": EpisodeType.REFLECTION,
        },
    ]
    
    print(f"Adding {len(test_memories)} diverse test memories...")
    
    stored_count = 0
    for i, memory_data in enumerate(test_memories, 1):
        try:
            memory = await pr.record_episode(
                content=memory_data["content"],
                episode_type=memory_data["episode_type"],
                metadata={
                    "category": memory_data["category"],
                    "test_memory": True,
                    "batch": "shadow_validation_v2"
                }
            )
            
            if memory:
                # Also append to MEMORY.md for shadow mode
                await append_to_memory_md(memory_data)
                print(f"  [{i}/{len(test_memories)}] ✓ {memory_data['content'][:50]}...")
                stored_count += 1
            else:
                print(f"  [{i}/{len(test_memories)}] ⚠ Rejected by write gate: {memory_data['content'][:50]}...")
                
        except Exception as e:
            print(f"  [{i}/{len(test_memories)}] ✗ Error: {e}")
    
    await pr.close()
    print(f"\n✅ Successfully stored {stored_count}/{len(test_memories)} memories")
    return stored_count


async def append_to_memory_md(memory_data: dict):
    """Append memory to MEMORY.md for shadow validation."""
    import os
    from datetime import datetime
    
    memory_md_path = Path("/root/.openclaw/workspace/MEMORY.md")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    category = memory_data.get("category", "general")
    
    entry = f"""
## [{timestamp}] {category.upper()}: observation

{memory_data['content']}

---
"""
    
    try:
        with open(memory_md_path, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as e:
        print(f"Warning: Could not append to MEMORY.md: {e}")


if __name__ == "__main__":
    count = asyncio.run(add_test_memories())
    sys.exit(0 if count > 0 else 1)
