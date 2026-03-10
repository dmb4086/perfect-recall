# Implementation Guide

## Quick Start

### Installation

```bash
pip install perfect-recall
```

### Basic Usage

```python
import asyncio
from perfect_recall import PerfectRecall, EpisodeType

async def main():
    # Initialize
    pr = await PerfectRecall.create(
        vector_store_url="postgresql://localhost/memories",
        graph_db_url="neo4j://localhost:7687"
    )
    
    # Start session
    session = await pr.start_session(user_id="user_123")
    
    # Record interactions
    await pr.record_episode(
        content="User: I need help with Python debugging",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id
    )
    
    # Store a fact
    await pr.store_fact(
        subject="user.programming_language",
        predicate="prefers",
        object="Python"
    )
    
    # Later, recall relevant info
    memories = await pr.recall("What does the user need help with?")
    for memory in memories:
        print(f"- {memory.content}")
    
    # End session
    await pr.end_session(session.id)

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with LLM

```python
from openai import AsyncOpenAI

class MemoryEnhancedAgent:
    def __init__(self):
        self.pr = PerfectRecall()
        self.llm = AsyncOpenAI()
    
    async def chat(self, user_id: str, message: str) -> str:
        # Start/resume session
        session = await self.pr.start_session(user_id)
        
        # Retrieve relevant memories
        memories = await self.pr.recall(
            query=message,
            context={"session_id": session.id}
        )
        
        # Build prompt with memories
        memory_context = self.format_memories(memories)
        prompt = f"""{memory_context}

User: {message}
Assistant:"""
        
        # Get response
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        reply = response.choices[0].message.content
        
        # Record interaction
        await self.pr.record_episode(
            content=f"User: {message}\nAssistant: {reply}",
            episode_type=EpisodeType.MESSAGE,
            session_id=session.id
        )
        
        return reply
    
    def format_memories(self, memories):
        if not memories:
            return ""
        
        parts = ["Relevant context from previous conversations:"]
        for m in memories[:5]:  # Top 5
            parts.append(f"- {m.content}")
        
        return "\n".join(parts)
```

## Configuration

```yaml
# config.yaml
perfect_recall:
  vector_store:
    type: pgvector
    url: ${DATABASE_URL}
    embedding_dimension: 1536
  
  graph_db:
    type: neo4j
    url: ${NEO4J_URL}
    username: ${NEO4J_USER}
    password: ${NEO4J_PASS}
  
  embedding:
    provider: openai
    model: text-embedding-3-small
  
  retrieval:
    default_limit: 10
    max_context_tokens: 2000
    salience_weights:
      semantic: 0.35
      recency: 0.20
      importance: 0.20
      frequency: 0.15
      emotional: 0.10
```

## Advanced Features

### Custom Memory Types

```python
from perfect_recall import MemoryNode, MemoryType

class CodePatternMemory(MemoryNode):
    """Custom memory type for code patterns."""
    
    language: str
    pattern_type: str
    usage_count: int
    success_rate: float
```

### Batch Operations

```python
# Efficient batch insertion
episodes = [
    {"content": "Event 1", "type": EpisodeType.OBSERVATION},
    {"content": "Event 2", "type": EpisodeType.ACTION},
    # ... 1000 more
]

await pr.record_episodes_batch(episodes)
```

### Custom Retrieval Strategies

```python
from perfect_recall.retrieval import RetrievalStrategy

class CustomStrategy(RetrievalStrategy):
    async def retrieve(self, query, context):
        # Custom retrieval logic
        pass

pr = PerfectRecall(retrieval_strategy=CustomStrategy())
```
