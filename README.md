# Perfect Recall 🧠

> A Four-Tier Memory System for AI Agents

Perfect Recall transforms stateless AI agents into truly continuous, memory-capable systems using a cognitive-science-inspired architecture.

## The Four-Tier Memory Model

| Tier | Purpose | Human Analog | Persistence |
|------|---------|--------------|-------------|
| **Working Memory** | Active context | Conscious awareness | Session-scoped |
| **Episodic Memory** | Event sequences | Autobiographical memory | Persistent |
| **Semantic Memory** | Facts & knowledge | General knowledge | Persistent |
| **Procedural Memory** | Skills & patterns | Muscle memory | Persistent |

## Quick Start

```bash
# Start PostgreSQL with pgvector
make db-up

# Install dependencies
pip install -r requirements.txt

# Run the example
python examples/basic_usage.py
```

## Usage

```python
import asyncio
from perfect_recall import PerfectRecall, EpisodeType

async def main():
    # Initialize
    pr = await PerfectRecall.create()
    
    # Start session
    session = await pr.start_session(user_id="user_123")
    
    # Record interaction
    await pr.record_episode(
        content="User: I need help with Python debugging",
        episode_type=EpisodeType.MESSAGE,
        session_id=session.id
    )
    
    # Store a fact
    await pr.store_fact(
        subject="user",
        predicate="programming_language_preference",
        object="Python"
    )
    
    # Later, recall relevant info
    memories = await pr.recall("What does the user prefer?")
    for memory in memories:
        print(f"- {memory.memory.content}")
    
    # End session
    await pr.end_session(session.id)

asyncio.run(main())
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PERFECT RECALL                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Session   │  │   Memory    │  │     Retrieval       │ │
│  │   Manager   │  │   Writer    │  │     Pipeline        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PostgreSQL + pgvector                   │   │
│  │  • Sessions  • Memory Nodes  • Working Memory       │   │
│  │  • Episodes  • Relationships • Access Logs          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- ✅ **Four-Tier Memory**: Working, Episodic, Semantic, Procedural
- ✅ **Write Gate**: Intelligent filtering prevents memory spam
- ✅ **Salience Scoring**: Multi-factor relevance ranking
- ✅ **Vector Search**: pgvector-powered similarity retrieval
- ✅ **Session Management**: Cross-session continuity
- ✅ **Fact Extraction**: Automatic semantic fact extraction

## Project Structure

```
perfect-recall/
├── docker-compose.yml          # Postgres + pgvector setup
├── Makefile                    # Development commands
├── requirements.txt            # Python dependencies
├── src/perfect_recall/
│   ├── __init__.py
│   ├── core/
│   │   ├── perfect_recall.py   # Main API
│   │   ├── memory_writer.py    # Memory capture
│   │   ├── session_manager.py  # Session lifecycle
│   │   └── retrieval.py        # Retrieval pipeline
│   ├── models/
│   │   ├── memory.py           # Memory models
│   │   ├── session.py          # Session models
│   │   └── retrieval.py        # Retrieval models
│   └── db/
│       ├── schema.sql          # Database schema
│       ├── connection.py       # DB connection
│       ├── sqlalchemy_models.py # ORM models
│       └── repositories.py     # Data access
├── examples/
│   ├── basic_usage.py          # Basic example
│   └── agent_integration.py    # Agent integration
└── tests/
    └── test_memory_writer.py   # Unit tests
```

## Database Schema

The schema implements the four-tier memory model:

- **sessions**: Session management and snapshots
- **memory_nodes**: Unified storage for all memory tiers
- **working_memory**: Active context references
- **episodes**: Event sequence groupings
- **memory_relationships**: Graph relationships
- **memory_access_log**: Access tracking for salience

See `src/perfect_recall/db/schema.sql` for full schema.

## Configuration

Environment variables:

```bash
DATABASE_URL=postgresql+asyncpg://perfect_recall:perfect_recall_secret@localhost:5432/perfect_recall
```

## Development

```bash
# Start database
make db-up

# Run tests
make test

# Run example
make example

# Reset database
make db-reset
```

## Documentation

- [Implementation Guide](IMPLEMENTATION.md) - Detailed implementation docs
- [Architecture](docs/V1_ARCHITECTURE.md) - Full architecture specification
- [API Design](design/API_DESIGN.md) - API reference

## License

MIT License

---

**Perfect Recall**: Because agents deserve to remember.
