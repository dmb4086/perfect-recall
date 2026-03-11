# Perfect Recall v2: A Memory System for AI Agents

> **"Perfect Recall is our north star, not a guarantee."**

**Version:** 2.0 (Revised)  
**Date:** 2025-03-12  
**Status:** Architecture Specification

---

## ⚠️ Honest Preamble

**"Perfect Recall" is an aspirational name.** We kept it because it captures the goal: memory so good it feels perfect. But let's be clear about what this system actually does and doesn't promise.

### What "Perfect" Means Here

Perfect Recall is achieved when an agent retrieves stored information with **high probability** and **appropriate confidence**, not with certainty. The system acknowledges:

- **Lossy extraction**: LLM-based fact extraction loses nuance
- **Approximate retrieval**: Vector similarity is approximate, not exact
- **Compression artifacts**: Summaries lose information
- **Temporal ambiguity**: "When" is often fuzzy
- **Confidence uncertainty**: The system can be wrong about being right

### What We Actually Promise

| Claim | Reality |
|-------|---------|
| "Same contextual richness as direct experience" | Richer than no memory, lossy compared to full context |
| "Perfect" retrieval | Target 80-90% recall on benchmarks, not 100% |
| "No information loss" | Compression and summarization lose information |
| "Instant recall" | Target <300ms p95 latency, not instantaneous |

### What We Don't Know Yet

- Exact recall rates on real-world tasks (needs measurement)
- Optimal salience scoring weights (needs tuning)
- Cost-benefit break-even point (needs deployment data)
- User satisfaction vs. accuracy tradeoffs (needs user studies)

**Read [HARD_PARTS.md](./HARD_PARTS.md) for the unsolved problems we haven't solved yet.**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The Problem: The Continuity Crisis](#2-the-problem-the-continuity-crisis)
3. [Theory of Agent Memory](#3-theory-of-agent-memory)
4. [Current State Analysis](#4-current-state-analysis)
5. [Proposed Architecture](#5-proposed-architecture)
6. [Honest Limitations](#6-honest-limitations)
7. [Implementation Roadmap v2](#7-implementation-roadmap-v2)
8. [Verification Framework](#8-verification-framework)
9. [Appendices](#9-appendices)

---

## 1. Executive Summary

### 1.1 The Core Problem

AI agents today suffer from **statelessness-induced amnesia**. Each session begins with limited context—the agent reads log files to simulate continuity but possesses no genuine memory system. This document proposes **Perfect Recall**, a memory architecture for continuous, scalable agent memory.

### 1.2 Key Innovation: The Memory Continuum

Perfect Recall organizes memory into four tiers inspired by cognitive science:

| Tier | Human Equivalent | Purpose | Latency Target |
|------|-----------------|---------|----------------|
| **Working Memory** | Conscious awareness | Active context | <10ms |
| **Episodic Memory** | Autobiographical recall | Event sequences | <100ms |
| **Semantic Memory** | General knowledge | Facts & concepts | <100ms |
| **Procedural Memory** | Muscle memory | Skills & patterns | <20ms |

### 1.3 Technical Definition

**Perfect Recall** is a system design that maximizes the probability of retrieving relevant information with appropriate confidence, while minimizing cost and latency. It does not guarantee perfect retrieval.

Key capabilities:
- **Temporal awareness**: Track when information was learned and its evolution
- **Confidence calibration**: Know when information is uncertain
- **Cross-session continuity**: Access relevant history across sessions
- **Efficient retrieval**: Find relevant information without linear search

### 1.4 Success Metrics (Evidence-Based)

| Metric | Target | Measurement Method | Status |
|--------|--------|-------------------|--------|
| DMR Recall@5 | >80% | Standard DMR benchmark | To be measured |
| Cross-session continuity | >70% | Custom multi-session test | To be measured |
| Memory latency (p95) | <300ms | End-to-end retrieval | To be measured |
| Token reduction vs full-context | >40% | Per-query token count | To be measured |
| Abstention accuracy | >75% | Custom benchmark | To be measured |

**Note:** These are targets, not achieved values. See [BENCHMARKS.md](./BENCHMARKS.md) for measurement methodology.

---

## 2. The Problem: The Continuity Crisis

### 2.1 The Stateless Illusion

Modern AI agents operate on context window stuffing—packing recent history into the prompt. This creates an illusion of memory that breaks down in several ways:

### 2.2 Failure Modes

#### 2.2.1 The Cold Start Problem

Every new session begins with limited context. The agent must:
- Re-learn user preferences
- Re-discover project state
- Re-establish conversational context

**Impact:** Significant session time spent re-establishing context from previous sessions.

#### 2.2.2 Context Window Degradation

Effective use of long context is challenging:
- Attention mechanisms dilute focus over long sequences
- Earlier context gets "washed out" by later tokens
- LLMs struggle to retrieve specific information from the middle of long contexts

**Reference:** Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172.

#### 2.2.3 The Relevance Paradox

Context windows treat all information equally. There's no:
- **Salience filtering**: What matters most?
- **Temporal weighting**: Recent vs. distant past
- **Causal prioritization**: Events that changed state

#### 2.2.4 Memory Loss on Session Termination

When a session ends:
- Context window is cleared
- Intermediate reasoning is lost
- Task state must be reconstructed

#### 2.2.5 The Summarization Tradeoff

Recursive summarization compresses history:
- Each summarization loses information
- Cumulative loss increases with depth
- Original details become inaccessible

### 2.3 The Business Case

Memory-enabled agents may improve:
- User satisfaction (reduced repetition)
- Task completion rate (better context)
- Session efficiency (less re-establishment)
- Cost (shorter effective context)

**Note:** These are hypothesized benefits to be validated through benchmarks, not guaranteed outcomes.

---

## 3. Theory of Agent Memory

### 3.1 The Memory Continuum Model

Perfect Recall uses a four-tier architecture:

```
├────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY CONTINUUM                            │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │   WORKING    ────────▶   EPISODIC   ────────▶   SEMANTIC   │      │
│  │   MEMORY     │    │   MEMORY     │    │   MEMORY     │      │
│  │   (Hot)      │    │   (Warm)     │    │   (Cool)     │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│         │                   │                   │              │
│         ├────────────────────────────────────────────────────────────────────────────────────────────────────────┤              │
│                             │                                  │
│                    ┌──────────────────┐                           │
│                    │  PROCEDURAL  │                           │
│                    │   MEMORY     │                           │
│                    │  (Compiled)  │                           │
│                    └──────────────────┘                           │
│                                                                 │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Tier 1: Working Memory

**Purpose**: Active context for immediate reasoning  
**Human Analog**: Conscious awareness  
**Characteristics**:
- Capacity: Limited to context window (4K-8K tokens)
- Lifetime: Session-bound, volatile
- Access: Immediate, zero-latency

```python
class WorkingMemory:
    """Hot memory tier for active context."""
    
    def __init__(self, max_tokens: int = 8192):
        self.messages: List[Message] = []
        self.active_context: Dict[str, Any] = {}
        self.max_tokens = max_tokens
    
    def add_message(self, message: Message) -> None:
        """Add message with automatic overflow management."""
        # Evict oldest non-critical messages when full
        while self.current_tokens + count_tokens(message) > self.max_tokens:
            self._evict_oldest()
        self.messages.append(message)
```

### 3.3 Tier 2: Episodic Memory

**Purpose**: Event sequences with temporal structure  
**Human Analog**: Autobiographical memory  
**Characteristics**:
- Capacity: Effectively unlimited (with storage)
- Lifetime: Persistent across sessions
- Access: Temporal query, similarity search

```python
@dataclass
class EpisodicRecord:
    """A single episode in the agent's experience."""
    id: str
    timestamp: datetime
    session_id: str
    episode_type: EpisodeType  # message | action | decision | observation
    content: str
    embedding: List[float]
    importance_score: float
    access_count: int
    last_accessed: Optional[datetime]
```

### 3.4 Tier 3: Semantic Memory

**Purpose**: Facts, concepts, and general knowledge  
**Human Analog**: Encyclopedia knowledge  
**Characteristics**:
- Capacity: Unlimited
- Lifetime: Persistent, evolves with corrections
- Access: Semantic similarity, structured query

```python
@dataclass
class SemanticFact:
    """A unit of knowledge in semantic memory."""
    subject: str
    predicate: str
    object: str
    valid_from: datetime      # When became true
    valid_until: Optional[datetime]  # When ceased being true
    confidence: float
    source_episode: str
    version: int
    superseded_by: Optional[str]
```

**Bi-temporal model**: Tracks both valid time (when true in world) and transaction time (when learned).

### 3.5 Tier 4: Procedural Memory

**Purpose**: Skills, patterns, and "how-to" knowledge  
**Human Analog**: Muscle memory  
**Key Insight**: This is **learned from experience**, not pre-defined templates.

See [PROCEDURAL_MEMORY.md](./PROCEDURAL_MEMORY.md) for the full specification.

```python
@dataclass
class LearnedSkill:
    """A skill distilled from successful trajectories."""
    trigger_embedding: Vector          # When to use
    context_signature: ContextPattern  # Required conditions
    approach: StrategyGraph            # Decision tree
    success_rate: float                # Track record
    attempt_count: int                 # Experience level
```

### 3.6 Memory Consolidation

New experiences flow through a consolidation pipeline:

```
Ingest Episode → Extract Facts → Resolve Entities → Embed → Store
                     ↓
              Update Graph ← Summarize Session
```

**Note**: Each step is lossy. Extraction loses nuance. Embedding loses precision. Summarization loses detail.

---

## 4. Current State Analysis

### 4.1 Existing Memory Systems

| System | Approach | Strengths | Limitations |
|--------|----------|-----------|-------------|
| **LangChain Memory** | Prompt-based | Simple integration | Session-bound, no persistence |
| **MemGPT/Letta** | OS-inspired virtual memory | Self-directed paging | Complex implementation |
| **Mem0** | Automatic fact extraction | Easy semantic memory | Limited episodic structure |
| **Zep/Graphiti** | Temporal knowledge graph | Excellent temporal handling | External service dependency |

### 4.2 Why Current Systems Fall Short

1. **Storage vs. Retrieval Gap**: Focus on storage, neglect retrieval quality
2. **Context Injection Problem**: Adding memories to prompts is non-trivial
3. **Temporal Blindness**: Lack true temporal awareness
4. **Episodic/Semantic Split**: No rich connections between tiers

---

## 5. Proposed Architecture

### 5.1 System Overview

```
├────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         PERFECT RECALL ARCHITECTURE               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                      CONSOLIDATION LAYER                            │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │  Ingestion │─────▶│ Extraction │─────▶│ Resolution │─────▶│  Embedding │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                    ▼                                 │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                        STORAGE LAYER (Postgres + pgvector)          │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │   Memory   │  │  Episode   │  │ Relationship│  │   Session   │   │   │
│  │  │   Nodes    │  │   Table    │  │    Table    │  │   Table     │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                    ▼                                 │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                       RETRIEVAL LAYER                               │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │   │
│  │  │  Vector    │  │  Keyword   │  │  Salience  │  │  Abstention│   │   │
│  │  │  Search    │  │  Search    │  │  Scoring   │  │  Control   │   │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                     │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Note:** V1 uses Postgres + pgvector only. See [V1_ARCHITECTURE.md](./V1_ARCHITECTURE.md) for details.

### 5.2 Key Algorithms

#### Salience Scoring

```python
def calculate_salience(memory, query, context) -> float:
    """
    Combines multiple factors for relevance ranking.
    Weights are tunable hyperparameters.
    """
    return (
        0.35 * semantic_similarity(memory, query) +
        0.20 * recency_score(memory) +
        0.20 * memory.importance_score +
        0.15 * access_frequency_score(memory) +
        0.10 * contextual_relevance(memory, context)
    )
```

See [V1_ARCHITECTURE.md](./V1_ARCHITECTURE.md) for complete algorithm specifications including write gate logic.

### 5.3 API Overview

```python
class PerfectRecall:
    """Main interface to the memory system."""
    
    async def record_episode(self, content: str, session_id: str) -> MemoryNode:
        """Record a new episode (with write gate filtering)."""
        
    async def recall(self, query: str, limit: int = 10) -> List[RetrievedMemory]:
        """Retrieve relevant memories with salience scoring."""
        
    async def store_fact(self, subject: str, predicate: str, object: str) -> MemoryNode:
        """Store a semantic fact (with conflict detection)."""
        
    async def start_session(self, user_id: str) -> Session:
        """Start/resume session with working memory hydration."""
```

---

## 6. Honest Limitations

### 6.1 What Perfect Recall Cannot Do

| Claim | Reality |
|-------|---------|
| "Perfect" memory | Target 80-90% accuracy, not 100% |
| No information loss | Every processing step loses something |
| Infinite context | Limited by storage and retrieval capacity |
| Instant recall | Retrieval takes 50-300ms |
| Perfect temporal precision | Timestamps are approximate |
| No false memories | LLM extraction hallucinates occasionally |

### 6.2 Known Failure Modes

1. **Embedding Similarity Failures**
   - Semantically different but embedding-similar content gets conflated
   - Sarcasm and nuance often lost

2. **Temporal Confusion**
   - "I used to like X, now I like Y" → may retrieve both
   - Fuzzy boundaries between time periods

3. **False Memory Generation**
   - LLM extraction creates facts that weren't stated
   - Overgeneralization: "You hate all X" from "You hated this one X"

4. **Retrieval Misses**
   - Relevant information has low embedding similarity
   - Important context ranked below less important but more similar content

5. **Confidence Miscalibration**
   - Overconfident about uncertain information
   - Underconfident about well-established facts

6. **Context Injection Problems**
   - Retrieved memories distract from the actual task
   - Too many memories overwhelm the context window

### 6.3 When NOT to Use Perfect Recall

- **Short conversations** (< 10 turns): Overhead exceeds benefit
- **Single-turn Q&A**: No history to remember
- **Security-critical applications**: Need audit trails, not probabilistic recall
- **Regulatory compliance**: May need exact replay, not reconstruction
- **Real-time systems**: Latency requirements may not allow retrieval

### 6.4 Uncertainties We Haven't Resolved

- **Optimal salience weights**: The 0.35/0.20/0.20/0.15/0.10 split is an educated guess
- **Write gate threshold**: 0.6 is arbitrary; needs per-user tuning
- **Abstention calibration**: Balance between helpfulness and accuracy unclear
- **Compression ratios**: How much can we compress before losing utility?
- **Cost-benefit break-even**: At what conversation length does memory pay off?

---

## 7. Implementation Roadmap v2

### 7.1 Phase 1: Minimal Viable Memory (Weeks 1-3)

**Goal:** Working storage and retrieval with Postgres only

**Deliverables**:
- [ ] Postgres + pgvector setup
- [ ] Basic memory node CRUD
- [ ] Vector similarity search
- [ ] Simple write gate (threshold-based)
- [ ] Session management

**Success Criteria**:
- Can store and retrieve memories
- Retrieval latency < 500ms
- Basic cross-session continuity works

### 7.2 Phase 2: Smart Retrieval (Weeks 4-6)

**Goal:** Intelligent retrieval and context injection

**Deliverables**:
- [ ] Salience scoring
- [ ] Temporal queries
- [ ] Conflict detection (basic)
- [ ] Abstention mechanism
- [ ] Context formatting

**Success Criteria**:
- DMR Recall@5 > 60% (measured, not target)
- Abstention accuracy > 65% (measured)
- Cost reduction > 20% vs full-context (measured)

### 7.3 Phase 3: Validation (Weeks 7-9)

**Goal:** Prove value through benchmarks

**Deliverables**:
- [ ] DMR benchmark implementation
- [ ] LongMemEval integration
- [ ] LoCoMo multi-session tests
- [ ] Cost/latency measurement
- [ ] Comparison against baselines

**Success Criteria**:
- DMR Recall@5 > 80% (measured)
- Cost reduction > 40% vs full-context (measured)
- Cross-session continuity > 70% (measured)

### 7.4 Phase 4: Refinement (Weeks 10-12)

**Goal:** Production readiness

**Deliverables**:
- [ ] Performance optimization
- [ ] Error handling
- [ ] Observability
- [ ] Documentation
- [ ] Example implementations

**Success Criteria**:
- Latency p95 < 300ms
- Zero data loss
- Clear upgrade path

### 7.5 Phase 5: Advanced Features (Future)

Add ONLY after Phase 4 succeeds:
- Graph database (if multi-hop queries proven necessary)
- Procedural memory (see [PROCEDURAL_MEMORY.md](./PROCEDURAL_MEMORY.md))
- Advanced conflict resolution
- Cross-agent memory sharing

### 7.6 Decision Gates

At each phase end, ask:
1. Do benchmarks show improvement over baselines?
2. Is the complexity justified by the gains?
3. Are users asking for the next features?

If no to any question, stop and reassess.

---

## 8. Verification Framework

### 8.1 Benchmarks

| Benchmark | What It Tests | Target |
|-----------|---------------|--------|
| **DMR** | Deep retrieval from long history | Recall@5 > 80% |
| **LongMemEval** | Temporal reasoning | Accuracy > 70% |
| **LoCoMo** | Cross-session continuity | Continuity > 70% |
| **Custom** | Write decisions, abstention | Accuracy > 75% |

See [BENCHMARKS.md](./BENCHMARKS.md) for full specification.

### 8.2 Baselines

Compare against:
1. **Full-context LLM**: No memory system
2. **MemGPT-style**: Agent-managed memory
3. **Mem0-style**: Fact-only memory
4. **Zep-style**: Graph-based temporal memory

### 8.3 Evaluation Dimensions

```
                    Qualitative
                         ▲
                         │
    Human Evaluation ───────┼────── End-to-End Scenarios
                         │
         Integration ─────┼──── Benchmarks
                         │
              Unit Tests
                         ▼
                    Quantitative
```

---

## 9. Appendices

### Appendix A: Related Documentation

| Document | Purpose |
|----------|---------|
| [HARD_PARTS.md](./HARD_PARTS.md) | Unsolved problems and hard mechanisms |
| [V1_ARCHITECTURE.md](./V1_ARCHITECTURE.md) | Simplified Postgres-based implementation |
| [PROCEDURAL_MEMORY.md](./PROCEDURAL_MEMORY.md) | Learning-based procedural memory |
| [BENCHMARKS.md](./BENCHMARKS.md) | Measurement methodology and baselines |
| [IMPLEMENTATION_GUIDE.md](./IMPLEMENTATION_GUIDE.md) | Getting started guide |

### Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Memory Continuum** | The four-tier memory architecture |
| **Working Memory** | Active, session-bound context |
| **Episodic Memory** | Time-ordered event sequences |
| **Semantic Memory** | Facts and general knowledge |
| **Procedural Memory** | Learned skills and patterns |
| **Bi-temporal Model** | Tracking valid time and transaction time |
| **Salience** | Current relevance of a memory |
| **Write Gate** | Filter for what gets stored |
| **Abstention** | Declining to answer when uncertain |

### Appendix C: References

1. Packer, C., et al. (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.
2. Rasmussen, P., et al. (2025). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory."
3. Zhang, T., et al. (2024). "LongMemEval: Long-Context Evaluation."
4. Sumers, T., et al. (2023). "Cognitive Architectures for Language Agents." arXiv:2309.02427.
5. Liu, N. F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." arXiv:2307.03172.

---

## Summary

**Perfect Recall** is an aspirational memory system for AI agents. It aims for high-probability, appropriately-confident retrieval of relevant information across sessions.

**Key Principles:**
1. Honest about limitations (this document + [HARD_PARTS.md](./HARD_PARTS.md))
2. Start simple ([V1_ARCHITECTURE.md](./V1_ARCHITECTURE.md))
3. Prove value before adding complexity
4. Measure against real baselines ([BENCHMARKS.md](./BENCHMARKS.md))

**What We've Specified:**
- ✅ Memory continuum architecture
- ✅ Write gate algorithm (see [HARD_PARTS.md](./HARD_PARTS.md))
- ✅ Conflict detection approach (see [HARD_PARTS.md](./HARD_PARTS.md))
- ✅ Abstention mechanism (see [HARD_PARTS.md](./HARD_PARTS.md))
- ✅ Salience scoring formula
- ✅ V1 Postgres architecture
- ✅ Benchmark strategy

**What We Haven't Solved:**
- ❓ Optimal hyperparameters (weights, thresholds)
- ❓ Actual performance on real tasks
- ❓ Cost-benefit validation
- ❓ User experience validation

**Status:** Architecture specified, implementation in progress.
