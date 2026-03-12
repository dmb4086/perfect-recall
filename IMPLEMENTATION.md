# Perfect Recall - Implementation Guide

## Quick Start

```bash
# 1. Start the database
make db-up

# 2. Install dependencies
make install

# 3. Run the example
make example
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PERFECT RECALL                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Session   │  │   Memory    │  │     Retrieval       │ │
│  │   Manager   │  │   Writer    │  │     Pipeline        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                     │            │
│         └────────────────┴─────────────────────┘            │
│                          │                                  │
│                   ┌─────────────┐                           │
│                   │   Database  │                           │
│                   │  (Postgres  │                           │
│                   │  + pgvector)│                           │
│                   └─────────────┘                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Four-Tier Memory Model

### 1. Working Memory
- **Purpose**: Active context (conscious awareness)
- **Scope**: Session-scoped, ephemeral
- **Use case**: Current task, active goals, scratchpad

```python
await pr.add_to_working_memory(
    session_id=session.id,
    content="Current task: Debugging Python error",
    slot_type="task",
    priority=0.9,
)
```

### 2. Episodic Memory
- **Purpose**: Event sequences (autobiographical memory)
- **Scope**: Persistent
- **Use case**: Conversation history, actions, observations

```python
await pr.record_episode(
    content="User: I need help with Python debugging",
    episode_type=EpisodeType.MESSAGE,
    session_id=session.id,
)
```

### 3. Semantic Memory
- **Purpose**: Facts & knowledge (general knowledge)
- **Scope**: Persistent
- **Use case**: User preferences, facts, extracted information

```python
await pr.store_fact(
    subject="user",
    predicate="programming_language_preference",
    object="Python",
    confidence=0.9,
)
```

### 4. Procedural Memory
- **Purpose**: Skills & patterns (muscle memory)
- **Scope**: Persistent
- **Use case**: Successful patterns, skills, procedures

```python
await pr.store_procedural(
    pattern_name="debug_python_keyerror",
    description="Guide for debugging Python KeyErrors",
    trigger_patterns=["KeyError", "dictionary error"],
    applicable_contexts=["python", "debugging"],
)
```

## Core Components

### MemoryWriter
Captures session events and writes to appropriate memory tiers:
- **Write Gate**: Filters low-value content
- **Tier Routing**: Routes to appropriate memory tier
- **Fact Extraction**: Extracts semantic facts from episodes

### SessionManager
Manages session lifecycle:
- Start/End sessions
- Working memory management
- Session resumption from snapshots

### RetrievalPipeline
Retrieves relevant memories:
- Vector similarity search
- Salience scoring
- Multi-tier search

## Database Schema

Key tables:
- `sessions`: Session management
- `memory_nodes`: Unified memory storage
- `working_memory`: Working memory slots
- `episodes`: Episode groupings
- `memory_relationships`: Graph relationships

See `src/perfect_recall/db/schema.sql` for full schema.

## Configuration

Environment variables:
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/perfect_recall
```

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_memory_writer.py -v
```

## Integration with LLMs

Basic integration pattern:

```python
async def chat_with_memory(user_id: str, message: str):
    # Start/resume session
    session = await pr.start_session(user_id)
    
    # Retrieve relevant memories
    memories = await pr.recall_for_session(
        session_id=str(session.id),
        query=message,
    )
    
    # Format for context
    context = pr.format_memories_for_context(memories)
    
    # Build prompt with context
    prompt = f"{context}\n\nUser: {message}\nAssistant:"
    
    # Get LLM response
    response = await llm.complete(prompt)
    
    # Record interaction
    await pr.record_episode(
        content=f"User: {message}\nAssistant: {response}",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id,
    )
    
    return response
```

## API Reference

See the docstrings in:
- `src/perfect_recall/core/perfect_recall.py` - Main API
- `src/perfect_recall/core/memory_writer.py` - Memory writing
- `src/perfect_recall/core/session_manager.py` - Session management
- `src/perfect_recall/core/retrieval.py` - Retrieval
