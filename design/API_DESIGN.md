# Perfect Recall API Design

## Overview

This document specifies the API design for the Perfect Recall memory system.

## Core Principles

1. **Simplicity**: Common operations should be simple
2. **Flexibility**: Advanced use cases should be possible
3. **Async-First**: All operations are async for performance
4. **Type-Safe**: Full type hints and validation

## API Structure

### Memory Operations

```python
# Record a new experience
memory = await pr.record_episode(
    content="User mentioned they prefer dark mode",
    episode_type=EpisodeType.OBSERVATION
)

# Store a fact
fact = await pr.store_fact(
    subject="user.preference.theme",
    predicate="equals",
    object="dark"
)

# Recall relevant memories
memories = await pr.recall("What are the user's preferences?")

# Query with temporal constraints
past_state = await pr.recall_semantic(
    subject="user.project.status",
    at_time=datetime(2025, 1, 15)
)
```

### Session Management

```python
# Start a new session (fresh)
session = await pr.start_session(user_id="alice")

# Resume previous session
session = await pr.start_session(
    user_id="alice",
    resume_from="session_123"
)

# End session (triggers consolidation)
summary = await pr.end_session(session.id)
```

### Maintenance Operations

```python
# Correct a memory
corrected = await pr.correct(
    memory_id="mem_456",
    corrected_content="User prefers light mode, not dark",
    reason="User corrected me"
)

# Consolidate manually
report = await pr.consolidate()

# Search with filters
results = await pr.search(
    query="project status",
    memory_types=[MemoryType.SEMANTIC],
    time_range=(start, end),
    min_confidence=0.8
)
```

## WebSocket Real-Time API

For real-time applications:

```javascript
// Connect to memory stream
const ws = new WebSocket('ws://api.perfect-recall.io/v1/stream');

// Subscribe to session
ws.send(JSON.stringify({
    action: 'subscribe',
    session_id: 'sess_123'
}));

// Receive memory updates
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    // Handle new memory, consolidation events, etc.
};
```

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/episodes` | Record episode |
| GET | `/v1/memories` | Search memories |
| GET | `/v1/memories/{id}` | Get specific memory |
| POST | `/v1/sessions` | Start session |
| POST | `/v1/sessions/{id}/end` | End session |
| GET | `/v1/sessions/{id}` | Get session state |
| POST | `/v1/consolidate` | Trigger consolidation |
| POST | `/v1/memories/{id}/correct` | Correct memory |
| DELETE | `/v1/memories/{id}` | Soft-delete memory |

## Event Schema

```typescript
interface MemoryEvent {
    type: 'memory.created' | 'memory.updated' | 'memory.accessed';
    timestamp: string;
    session_id: string;
    memory: {
        id: string;
        type: 'episodic' | 'semantic' | 'procedural';
        content: string;
    };
}
```
