# Memory System Research Notes

## Executive Summary

This document compiles research findings on existing AI memory systems, their architectures, and failure modes. It serves as the foundation for the Perfect Recall architecture.

---

## 1. Memory Taxonomy (CoALA Framework)

Based on Princeton's CoALA framework and cognitive science research:

| Memory Type | Human Analog | AI Implementation | Examples |
|-------------|--------------|-------------------|----------|
| **Working** | Conscious awareness | Context window | Current conversation |
| **Episodic** | Autobiographical | Event logs, timestamps | "Yesterday we discussed..." |
| **Semantic** | General knowledge | Vector DB, facts | "User prefers Python" |
| **Procedural** | Muscle memory | Learned skills | "How to debug TypeScript" |

### Key Insight
Human memory is not a database—it's a **reconstructive, associative system**. AI memory systems must balance:
- Precision (retrievability)
- Associativity (semantic connections)
- Efficiency (storage/compute costs)

---

## 2. Existing Systems Analysis

### 2.1 LangChain Memory

**Architecture**: Prompt-based, in-context memory management

**Types**:
- `ConversationBufferMemory`: Stores raw history
- `ConversationBufferWindowMemory`: Sliding window (last k messages)
- `ConversationSummaryMemory`: LLM-summarized history
- `VectorStoreRetrieverMemory`: Semantic retrieval from vector store

**Limitations**:
1. **Session-bound**: No persistence across sessions
2. **Linear structure**: No rich relationships
3. **Manual selection**: Developer must choose memory type upfront
4. **No temporal awareness**: Can't distinguish "learned yesterday" vs "learned last year"

**Code Pattern**:
```python
# Typical LangChain usage (insufficient)
memory = ConversationBufferWindowMemory(k=10)
# Only remembers last 10 messages
# No persistence, no semantic organization
```

### 2.2 MemGPT / Letta

**Paper**: "MemGPT: Towards LLMs as Operating Systems" (Packer et al., 2023)

**Core Innovation**: OS-inspired virtual memory management

**Architecture**:
```
┌─────────────────────────────────────┐
│           MAIN CONTEXT              │  ← LLM context window
│  (System + Working + FIFO Queue)    │
├─────────────────────────────────────┤
│          EXTERNAL CONTEXT           │  ← Persistent storage
│   (Recall Storage + Archival)       │
└─────────────────────────────────────┘
```

**Mechanism**:
- LLM generates function calls to manage memory
- `page_in()`: Load from external to main context
- `page_out()`: Evict from main to external context
- Self-directed memory management

**Strengths**:
- Unbounded context illusion
- LLM controls its own memory
- Persistent across sessions

**Limitations**:
- Complex implementation
- Requires capable function-calling models
- Limited semantic structure
- No temporal reasoning

**Benchmark**: Deep Memory Retrieval (DMR) - 93.4% accuracy

### 2.3 Mem0

**Approach**: Self-improving memory with automatic extraction

**Features**:
- Automatic fact extraction from conversations
- User-specific memory profiles
- Conflict resolution (handle contradictory facts)
- Multi-level memory (user, session, global)

**Architecture**:
```
Input → Extract Facts → Deduplicate → Store → 
                                       ↓
                              Conflict Resolution
                                       ↓
                                   Update Memory
```

**Strengths**:
- Zero-config memory extraction
- Automatic conflict handling
- Multi-tenant (per-user memory)

**Limitations**:
- Limited episodic structure
- No knowledge graph
- Basic temporal handling

### 2.4 Zep / Graphiti

**Paper**: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (Rasmussen et al., 2025)

**Core Innovation**: Bi-temporal knowledge graph for dynamic data

**Architecture**:
```
Episodes (raw events)
    ↓
Entity Extraction
    ↓
Temporal Knowledge Graph (nodes + edges with validity intervals)
    ↓
Community Detection
    ↓
Hybrid Retrieval (semantic + keyword + graph)
```

**Bi-Temporal Model**:
- **Valid time (T)**: When fact was true in world
- **Transaction time (T')**: When agent learned about it

**Data Structure**:
```python
class TemporalEdge:
    source: Entity
    target: Entity
    relationship: str
    t_valid: datetime      # When became true
    t_invalid: datetime    # When ceased being true
    t_created: datetime    # When learned
```

**Strengths**:
- Excellent temporal reasoning
- Handles changing facts ("X was true until...")
- Graph relationships enable multi-hop reasoning
- Incremental updates (no full recompute)
- Sub-300ms query latency (P95)

**Benchmark Results**:
- DMR: 94.8% (vs MemGPT 93.4%)
- LongMemEval: 71.2% (vs baseline 60.2%)
- Latency: 90% reduction vs full-context

**Limitations**:
- External service dependency
- No working memory integration
- Limited procedural memory

### 2.5 Microsoft GraphRAG

**Approach**: Community-based knowledge graph summarization

**Process**:
1. Extract entities and relationships
2. Build knowledge graph
3. Detect communities (thematic clusters)
4. Pre-compute community summaries
5. Query: Retrieve relevant communities, generate response

**Strengths**:
- Rich global context
- Good for static documents

**Limitations**:
- Batch processing (not real-time)
- High latency (seconds to tens of seconds)
- Full recompute on updates
- Not suitable for dynamic memory

---

## 3. Failure Mode Analysis

### 3.1 The Storage-Retrieval Gap

Most systems focus on **storing** memories but neglect **retrieval quality**:

| Aspect | Current State | Needed |
|--------|--------------|--------|
| What to retrieve? | Simple similarity | Context-aware relevance |
| When to retrieve? | Every query | Triggered by need |
| How to rank? | Cosine similarity | Multi-factor salience |
| How to format? | Raw dump | Compressed, structured |

### 3.2 Context Injection Problems

Simply adding memories to prompts has issues:

**Position Bias**: LLMs favor information at start/end of context
```
Prompt: [Memories] + [Current Query]
        ↑ LLM pays more attention here
        
Prompt: [Current Query] + [Memories]
                          ↑ And here
```

**Attention Dilution**: Too many memories = noise
```
With 5 memories:  85% effective use
With 10 memories: 70% effective use
With 20 memories: 45% effective use
```

**Format Sensitivity**: Raw memory dumps hurt performance

### 3.3 Temporal Blindness

Current systems lack true temporal awareness:

| Query Type | Current Handling | Needed |
|------------|-----------------|--------|
| "What did I say yesterday?" | Keyword match | Temporal index |
| "What was I working on before Project X?" | No support | Temporal graph traversal |
| "When did I change my mind about Y?" | No support | Fact versioning |

### 3.4 Episodic-Semantic Disconnect

Human memory richly connects:
- What happened (episodic)
- What is true (semantic)
- How we responded (procedural)

Current systems silo these with no cross-referencing.

---

## 4. Key Insights for Perfect Recall

### 4.1 Memory is Not a Database

Don't treat memory as key-value storage. Memory should be:
- **Associative**: Retrieved by semantic similarity
- **Temporal**: Ordered in time
- **Causal**: Linked by cause-effect
- **Salient**: Prioritized by importance

### 4.2 The Consolidation Pipeline is Critical

Raw experiences → Processed memories

Key steps:
1. **Extraction**: Identify facts, entities, decisions
2. **Resolution**: Link to existing knowledge
3. **Embedding**: Generate semantic vectors
4. **Graph Update**: Add to knowledge graph
5. **Summarization**: Create compressed representations

### 4.3 Retrieval is Multi-Modal

Effective retrieval combines:
- Semantic search (what's similar)
- Graph traversal (what's connected)
- Temporal query (what's relevant now)
- Salience ranking (what matters most)

### 4.4 Context Injection Needs Intelligence

Don't dump memories into prompts. Instead:
1. Select most relevant (salience filter)
2. Compress if verbose (summarization)
3. Format for LLM consumption (structuring)
4. Position strategically (prompt engineering)

### 4.5 Temporal is Essential

For true continuity, agents need:
- When facts were learned
- When facts ceased being true
- Temporal relationships (before/after)
- Point-in-time queries

---

## 5. Research Gaps

1. **Meta-Memory**: Agents knowing what they know (and don't know)
2. **Active Forgetting**: Intelligent memory pruning
3. **Emotional Weighting**: Affect-influenced memory salience
4. **Dream Consolidation**: Offline memory optimization
5. **Cross-Agent Transfer**: Sharing learned patterns

---

## References

1. Packer, C., et al. (2023). MemGPT: Towards LLMs as Operating Systems.
2. Rasmussen, P., et al. (2025). Zep: A Temporal Knowledge Graph Architecture.
3. Sumers, T., et al. (2023). Cognitive Architectures for Language Agents.
4. Weng, L. (2023). LLM Powered Autonomous Agents.
5. Shinn, N., et al. (2023). Reflexion: Self-Reflective Agents.
