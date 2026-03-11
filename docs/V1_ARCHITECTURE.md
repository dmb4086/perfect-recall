# V1 Minimal Architecture: Postgres + pgvector + JSONB

> **"Start simple. Add complexity only when proven necessary."**

This document specifies a minimal, production-ready architecture for Perfect Recall v1. It deliberately avoids the complexity of multi-database setups until they are proven necessary.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Architecture Overview](#2-architecture-overview)
3. [Database Schema](#3-database-schema)
4. [Core Algorithms](#4-core-algorithms)
5. [API Specification](#5-api-specification)
6. [Implementation Phases](#6-implementation-phases)
7. [Migration Path](#7-migration-path)

---

## 1. Design Principles

### 1.1 Why Postgres-First?

| Aspect | Original Design | V1 Minimal |
|--------|-----------------|------------|
| Databases | Vector + Graph + Document + Cache | Single Postgres |
| Complexity | High (4 systems to maintain) | Low (1 system) |
| Operations | Complex backups, monitoring | Standard Postgres ops |
| Query complexity | Multi-hop, cross-database | Single SQL/JSONB |
| Cost | 4x infrastructure | 1x infrastructure |
| When to add layers | Immediately | When benchmarks prove need |

### 1.2 When to Add Complexity

Add the graph layer **only when**:
- Multi-hop queries (A -> B -> C) are proven necessary
- Benchmarks show vector-only retrieval is insufficient
- User studies demonstrate need for complex relationship traversal

Add a dedicated cache layer **only when**:
- p95 latency exceeds 200ms consistently
- Query analysis shows repeated identical lookups
- Postgres connection limits become a bottleneck

### 1.3 Core Principles

1. **Single source of truth**: Postgres
2. **JSONB for flexibility**: Schema evolution without migrations
3. **pgvector for search**: Good enough for v1
4. **Indexes for speed**: GIN, IVFFlat, B-tree
5. **Prove before adding**: No premature optimization

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    V1 MINIMAL ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Agent      │──────▶   Memory     │──────▶   Postgres   │  │
│  │   Core       │◀─────│   Service    │◀─────│   + pgvector │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                               │                                 │
│                               ▼                                 │
│                        ┌──────────────┐                        │
│                        │  Embedding   │                        │
│                        │   Service    │                        │
│                        └──────────────┘                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Database | PostgreSQL 15+ | Storage, relationships, metadata |
| Vector Extension | pgvector 0.5+ | Embedding storage & similarity search |
| Cache | In-memory (LRU) | Hot memory during session |
| Embedding | OpenAI/Cohere/Ollama | Text → vector |

### 2.2 What We're NOT Building (Yet)

- ❌ Neo4j graph database
- ❌ MongoDB document store  
- ❌ Redis cache cluster
- ❌ Complex multi-stage retrieval pipelines
- ❌ Dedicated inference service

---

## 3. Database Schema

### 3.1 Core Tables

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ========================================
-- MEMORY NODES (unified storage)
-- ========================================
CREATE TABLE memory_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Type classification
    memory_type VARCHAR(20) NOT NULL CHECK (memory_type IN ('episodic', 'semantic', 'procedural')),
    
    -- Content
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI dimension; adjust for your embedding model
    
    -- Temporal metadata (bi-temporal model)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,  -- NULL = still valid
    
    -- Salience metadata
    importance_score FLOAT NOT NULL DEFAULT 0.5 CHECK (importance_score BETWEEN 0 AND 1),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    emotional_valence FLOAT CHECK (emotional_valence BETWEEN -1 AND 1),
    
    -- Provenance
    confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    source_type VARCHAR(20) NOT NULL DEFAULT 'direct' 
        CHECK (source_type IN ('direct', 'inferred', 'imported', 'corrected', 'compound')),
    source_episode_id UUID,
    
    -- Versioning
    version INTEGER NOT NULL DEFAULT 1,
    supersedes_id UUID REFERENCES memory_nodes(id),
    superseded_by_id UUID REFERENCES memory_nodes(id),
    
    -- Flexible metadata
    metadata JSONB NOT NULL DEFAULT '{}',
    
    -- Search vector for full-text search
    search_vector TSVECTOR
);

-- Indexes for memory_nodes
CREATE INDEX idx_memory_type ON memory_nodes(memory_type);
CREATE INDEX idx_memory_created ON memory_nodes(created_at);
CREATE INDEX idx_memory_valid ON memory_nodes(valid_from, valid_until);
CREATE INDEX idx_memory_importance ON memory_nodes(importance_score DESC);
CREATE INDEX idx_memory_last_accessed ON memory_nodes(last_accessed DESC);
CREATE INDEX idx_memory_metadata ON memory_nodes USING GIN (metadata);
CREATE INDEX idx_memory_search ON memory_nodes USING GIN (search_vector);

-- Vector similarity index (IVFFlat for balance of speed/accuracy)
CREATE INDEX idx_memory_embedding ON memory_nodes 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ========================================
-- EPISODES (groupings of memories)
-- ========================================
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(100) NOT NULL,
    
    -- Temporal bounds
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Summary
    summary TEXT,
    summary_embedding VECTOR(1536),
    
    -- Metadata
    participant_ids TEXT[] DEFAULT '{}',
    topic_tags TEXT[] DEFAULT '{}',
    
    -- Hierarchical linking
    parent_episode_id UUID REFERENCES episodes(id),
    
    -- Metrics
    memory_count INTEGER DEFAULT 0,
    
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_episodes_session ON episodes(session_id);
CREATE INDEX idx_episodes_time ON episodes(started_at, ended_at);
CREATE INDEX idx_episodes_tags ON episodes USING GIN (topic_tags);

-- ========================================
-- EPISODE-MEMORY LINKING
-- ========================================
CREATE TABLE episode_memories (
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,  -- Order within episode
    
    PRIMARY KEY (episode_id, memory_id)
);

CREATE INDEX idx_ep_mem_memory ON episode_memories(memory_id);

-- ========================================
-- MEMORY RELATIONSHIPS (lightweight graph)
-- ========================================
CREATE TABLE memory_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    strength FLOAT NOT NULL DEFAULT 1.0 CHECK (strength BETWEEN 0 AND 1),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}',
    
    UNIQUE (source_memory_id, target_memory_id, relationship_type)
);

CREATE INDEX idx_rel_source ON memory_relationships(source_memory_id);
CREATE INDEX idx_rel_target ON memory_relationships(target_memory_id);
CREATE INDEX idx_rel_type ON memory_relationships(relationship_type);

-- Common relationship types:
-- 'precedes' - temporal ordering
-- 'causes' - causal link
-- 'relates_to' - general association
-- 'supersedes' - version replacement
-- 'mentions' - entity reference

-- ========================================
-- SESSIONS
-- ========================================
CREATE TABLE sessions (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    agent_id VARCHAR(100),
    
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Working memory snapshot (for resumption)
    context_snapshot JSONB,
    
    -- Metrics
    message_count INTEGER DEFAULT 0,
    token_usage INTEGER DEFAULT 0,
    
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_time ON sessions(started_at, ended_at);

-- ========================================
-- ACCESS LOG (for LRU and analytics)
-- ========================================
CREATE TABLE memory_access_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    access_type VARCHAR(20) NOT NULL,  -- 'recall', 'write', 'update'
    session_id VARCHAR(100),
    query_text TEXT,  -- What query triggered this access
    
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_access_memory ON memory_access_log(memory_id);
CREATE INDEX idx_access_time ON memory_access_log(accessed_at);

-- ========================================
-- CONFLICTS (for false memory tracking)
-- ========================================
CREATE TABLE memory_conflicts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    memory_a_id UUID NOT NULL REFERENCES memory_nodes(id),
    memory_b_id UUID NOT NULL REFERENCES memory_nodes(id),
    
    conflict_type VARCHAR(50) NOT NULL,  -- 'contradiction', 'temporal', 'exclusive'
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    resolution_status VARCHAR(20) DEFAULT 'open' 
        CHECK (resolution_status IN ('open', 'auto_resolved', 'user_resolved', 'ignored')),
    resolution_notes TEXT,
    
    UNIQUE (memory_a_id, memory_b_id)
);

CREATE INDEX idx_conflicts_status ON memory_conflicts(resolution_status);
```

### 3.2 JSONB Metadata Schemas

```typescript
// memory_nodes.metadata interface
interface MemoryMetadata {
  // For episodic memories
  episode_type?: 'message' | 'action' | 'decision' | 'observation' | 'reflection';
  participants?: string[];
  
  // For semantic memories (fact extraction)
  extracted_facts?: Array<{
    subject: string;
    predicate: string;
    object: string;
    confidence: number;
  }>;
  
  // For procedural memories
  skill_data?: {
    trigger_patterns: string[];
    success_count: number;
    failure_count: number;
    avg_execution_time_ms: number;
  };
  
  // For all types
  tags?: string[];
  category?: string;
  verification_status?: 'unverified' | 'confirmed' | 'contradicted';
}

// sessions.context_snapshot interface
interface ContextSnapshot {
  working_memories: UUID[];  // Memory IDs in working memory
  active_topics: string[];
  pending_tasks: Array<{
    task_id: string;
    description: string;
    started_at: string;
  }>;
  user_preferences: Record<string, any>;
}
```

---

## 4. Core Algorithms

### 4.1 Salience Scoring Algorithm

```python
class SalienceScorer:
    """
    Calculates how "salient" (relevant, important) a memory is.
    Used for ranking retrieved memories.
    """
    
    def score(
        self,
        memory: MemoryNode,
        query: str,
        query_embedding: List[float],
        current_context: Dict[str, Any]
    ) -> float:
        """
        Combined salience score from multiple factors.
        """
        factors = {
            'semantic': self.semantic_similarity(memory, query_embedding),
            'recency': self.recency_score(memory),
            'importance': self.importance_score(memory),
            'frequency': self.frequency_score(memory),
            'contextual': self.contextual_score(memory, current_context)
        }
        
        # Weights (tunable per use case)
        weights = {
            'semantic': 0.35,
            'recency': 0.20,
            'importance': 0.20,
            'frequency': 0.15,
            'contextual': 0.10
        }
        
        return sum(factors[k] * weights[k] for k in factors)
    
    def semantic_similarity(self, memory: MemoryNode, query_embedding: List[float]) -> float:
        """Cosine similarity between memory and query embeddings."""
        return cosine_similarity(memory.embedding, query_embedding)
    
    def recency_score(self, memory: MemoryNode) -> float:
        """Exponential decay based on age."""
        if not memory.last_accessed:
            age_hours = (now() - memory.created_at).total_seconds() / 3600
        else:
            age_hours = (now() - memory.last_accessed).total_seconds() / 3600
        
        # Half-life of 1 week (168 hours)
        return math.exp(-age_hours / 168)
    
    def importance_score(self, memory: MemoryNode) -> float:
        """Explicit importance score with small boost for high-access memories."""
        access_boost = min(memory.access_count / 20, 0.1)  # Max 0.1 boost
        return min(memory.importance_score + access_boost, 1.0)
    
    def frequency_score(self, memory: MemoryNode) -> float:
        """How often this memory is accessed (frequently used = more salient)."""
        if memory.access_count == 0:
            return 0.0
        return 1 - math.exp(-memory.access_count / 5)
    
    def contextual_score(self, memory: MemoryNode, context: Dict[str, Any]) -> float:
        """Overlap between memory tags/topics and current context."""
        memory_tags = set(memory.metadata.get('tags', []))
        context_topics = set(context.get('active_topics', []))
        
        if not memory_tags or not context_topics:
            return 0.5  # Neutral
        
        overlap = len(memory_tags & context_topics)
        return overlap / max(len(memory_tags), len(context_topics))
```

### 4.2 Memory Write Gate

```python
class WriteGate:
    """
    Decides whether to write a potential memory.
    Prevents spam and low-value storage.
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self.recent_writes = deque(maxlen=100)  # Last 100 writes
    
    def should_write(
        self,
        content: str,
        content_type: str,
        context: Dict[str, Any],
        db: Database
    ) -> WriteDecision:
        """
        Multi-factor decision on whether to write this memory.
        """
        scores = {
            'novelty': self.score_novelty(content, db),
            'utility': self.score_utility(content, context),
            'explicit': self.score_explicit_marking(content),
            'density': self.score_information_density(content),
        }
        
        # Dynamic threshold based on recent write rate
        recent_rate = len([w for w in self.recent_writes 
                          if w > now() - timedelta(minutes=5)])
        dynamic_threshold = self.threshold + (recent_rate / 100) * 0.2
        
        final_score = weighted_average(scores, {
            'novelty': 0.30,
            'utility': 0.30,
            'explicit': 0.25,
            'density': 0.15
        })
        
        if final_score >= dynamic_threshold:
            self.recent_writes.append(now())
            return WriteDecision(write=True, importance=final_score, factors=scores)
        
        return WriteDecision(write=False, factors=scores)
    
    def score_novelty(self, content: str, db: Database) -> float:
        """How different is this from existing memories?"""
        embedding = embed(content)
        similar = db.query_similar(embedding, k=3, threshold=0.85)
        
        if not similar:
            return 1.0
        
        max_sim = max(s.similarity for s in similar)
        return 1.0 - max_sim
    
    def score_utility(self, content: str, context: Dict[str, Any]) -> float:
        """Likely future utility - heuristics-based."""
        lower = content.lower()
        
        # High utility indicators
        if any(kw in lower for kw in ['preference', 'prefer', 'like', 'dislike']):
            return 0.9
        if any(kw in lower for kw in ['decided', 'decision', 'choose', 'chose']):
            return 0.8
        if any(kw in lower for kw in ['project', 'task', 'goal']):
            return 0.7
        if any(kw in lower for kw in ['important', 'remember', 'don\'t forget']):
            return 1.0
        
        # Check if in active task context
        if context.get('active_task') and self.is_task_relevant(content, context['active_task']):
            return 0.6
        
        return 0.3  # Default low utility
    
    def score_explicit_marking(self, content: str) -> float:
        """Did user explicitly mark as important?"""
        markers = ['remember this', "don't forget", 'important:', 'note that', 'for future reference']
        return 1.0 if any(m in content.lower() for m in markers) else 0.0
    
    def score_information_density(self, content: str) -> float:
        """Reward content with facts over fluff."""
        # Simple heuristic: content with entities and specifics
        has_entities = bool(re.search(r'\b[A-Z][a-z]+\b', content))
        has_specifics = bool(re.search(r'\d{4}|\b[A-Z]{2,}\b', content))
        
        score = 0.5
        if has_entities: score += 0.25
        if has_specifics: score += 0.25
        
        return score
```

### 4.3 Retrieval Pipeline

```python
class RetrievalPipeline:
    """
    Multi-stage retrieval optimized for Postgres/pgvector.
    """
    
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
        limit: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant memories in stages.
        """
        # Stage 1: Generate query embedding
        query_embedding = embed(query)
        
        # Stage 2: Vector similarity search (broad recall)
        candidates = await self.vector_search(
            query_embedding, 
            k=limit * 5,  # Over-fetch for reranking
            memory_types=memory_types
        )
        
        # Stage 3: Temporal filtering (if specified)
        if context.temporal_constraints:
            candidates = self.filter_temporal(candidates, context.temporal_constraints)
        
        # Stage 4: Metadata filtering (if specified)
        if context.filters:
            candidates = self.filter_metadata(candidates, context.filters)
        
        # Stage 5: Salience scoring
        scored = []
        for memory in candidates:
            score = self.salience_scorer.score(memory, query, query_embedding, context)
            scored.append((memory, score))
        
        # Stage 6: Sort and select
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:limit]
        
        # Stage 7: Update access metrics
        await self.update_access_metrics([m for m, _ in selected])
        
        return [RetrievedMemory(memory=m, score=s) for m, s in selected]
    
    async def vector_search(
        self,
        query_embedding: List[float],
        k: int,
        memory_types: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> List[MemoryNode]:
        """
        Postgres vector similarity query.
        """
        sql = """
            SELECT m.*, 1 - (m.embedding <=> %s::vector) as similarity
            FROM memory_nodes m
            WHERE m.embedding IS NOT NULL
              AND 1 - (m.embedding <=> %s::vector) >= %s
              AND (m.valid_until IS NULL OR m.valid_until > NOW())
        """
        params = [query_embedding, query_embedding, threshold]
        
        if memory_types:
            sql += " AND m.memory_type = ANY(%s)"
            params.append(memory_types)
        
        sql += " ORDER BY m.embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, k])
        
        return await self.db.query(sql, params)
```

### 4.4 Conflict Detection

```python
class ConflictDetector:
    """
    Detects potential contradictions in memory.
    """
    
    async def detect_conflicts(
        self,
        new_memory: MemoryNode,
        db: Database
    ) -> List[Conflict]:
        """
        Check new memory against existing for contradictions.
        """
        conflicts = []
        
        # Only check semantic memories (facts)
        if new_memory.memory_type != 'semantic':
            return conflicts
        
        # Find similar memories about same subject
        similar = await db.query("""
            SELECT * FROM memory_nodes
            WHERE memory_type = 'semantic'
              AND id != %s
              AND embedding <=> %s::vector < 0.3  -- High similarity
              AND (valid_until IS NULL OR valid_until > NOW())
        """, [new_memory.id, new_memory.embedding])
        
        for candidate in similar:
            if self.are_contradictory(new_memory, candidate):
                conflicts.append(Conflict(
                    memory_a_id=new_memory.id,
                    memory_b_id=candidate.id,
                    type='contradiction',
                    explanation=self.generate_explanation(new_memory, candidate)
                ))
        
        return conflicts
    
    def are_contradictory(self, a: MemoryNode, b: MemoryNode) -> bool:
        """
        Heuristic: Check if two facts contradict.
        Uses LLM for nuanced cases, simple checks for obvious ones.
        """
        # Simple string-based checks first
        content_a = a.content.lower()
        content_b = b.content.lower()
        
        # Direct negation patterns
        negations = [
            ('prefer', 'don\'t prefer'),
            ('like', 'dislike'),
            ('use', 'don\'t use'),
            ('work at', 'left'),
            ('live in', 'moved from'),
        ]
        
        for pos, neg in negations:
            if pos in content_a and neg in content_b:
                return True
            if neg in content_a and pos in content_b:
                return True
        
        # If simple checks fail, use LLM for judgment
        return self.llm_contradiction_check(a.content, b.content)
```

---

## 5. API Specification

### 5.1 Core Operations

```python
class PerfectRecallV1:
    """
    Main API for Perfect Recall v1.
    """
    
    # ─────────────────────────────────────────────────────────────────
    # WRITE OPERATIONS
    # ─────────────────────────────────────────────────────────────────
    
    async def record_episode(
        self,
        content: str,
        episode_type: str,  # 'message' | 'action' | 'decision' | 'observation' | 'reflection'
        session_id: str,
        metadata: Dict[str, Any] = None
    ) -> MemoryNode:
        """
        Record a new episode to memory.
        Runs through write gate and extraction pipeline.
        """
        # Check write gate
        decision = self.write_gate.should_write(content, episode_type, {'session_id': session_id})
        
        if not decision.write:
            logger.debug(f"Write gate rejected: {decision.factors}")
            return None
        
        # Generate embedding
        embedding = self.embed(content)
        
        # Create memory node
        memory = await self.db.insert("""
            INSERT INTO memory_nodes (memory_type, content, embedding, importance_score, metadata, source_episode_id)
            VALUES ('episodic', %s, %s, %s, %s, %s)
            RETURNING *
        """, [content, embedding, decision.importance, json.dumps(metadata or {}), session_id])
        
        # Link to episode
        await self.link_to_episode(memory.id, session_id)
        
        # Extract facts (async background)
        asyncio.create_task(self.extract_facts(memory))
        
        return memory
    
    async def store_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        valid_from: Optional[datetime] = None,
        episode_id: Optional[str] = None
    ) -> MemoryNode:
        """
        Store a semantic fact with conflict detection.
        """
        content = f"{subject} {predicate} {object}"
        embedding = self.embed(content)
        
        # Check for conflicts
        memory = MemoryNode(
            memory_type='semantic',
            content=content,
            embedding=embedding,
            confidence=confidence,
            valid_from=valid_from or now(),
            metadata={'extracted_facts': [{'subject': subject, 'predicate': predicate, 'object': object, 'confidence': confidence}]}
        )
        
        conflicts = await self.conflict_detector.detect_conflicts(memory, self.db)
        
        if conflicts:
            # Store conflict for resolution
            for conflict in conflicts:
                await self.db.insert("""
                    INSERT INTO memory_conflicts (memory_a_id, memory_b_id, conflict_type)
                    VALUES (%s, %s, %s)
                """, [conflict.memory_a_id, conflict.memory_b_id, conflict.type])
        
        # Store memory (even with conflicts - conflicts are flagged, not blocking)
        stored = await self.db.insert("""
            INSERT INTO memory_nodes (memory_type, content, embedding, confidence, valid_from, metadata)
            VALUES ('semantic', %s, %s, %s, %s, %s)
            RETURNING *
        """, [content, embedding, confidence, memory.valid_from, json.dumps(memory.metadata)])
        
        return stored
    
    # ─────────────────────────────────────────────────────────────────
    # READ OPERATIONS
    # ─────────────────────────────────────────────────────────────────
    
    async def recall(
        self,
        query: str,
        context: Dict[str, Any] = None,
        memory_types: List[str] = None,
        limit: int = 10,
        temporal_at: Optional[datetime] = None
    ) -> List[RetrievedMemory]:
        """
        Retrieve relevant memories.
        """
        return await self.retrieval_pipeline.retrieve(
            query=query,
            context=RetrievalContext(
                temporal_constraints=TemporalConstraints(at_time=temporal_at) if temporal_at else None,
                filters=context
            ),
            limit=limit,
            memory_types=memory_types
        )
    
    # ─────────────────────────────────────────────────────────────────
    # SESSION MANAGEMENT
    # ─────────────────────────────────────────────────────────────────
    
    async def start_session(
        self,
        user_id: str,
        resume_from: Optional[str] = None
    ) -> Session:
        """
        Start a new session, optionally resuming previous.
        """
        session_id = generate_id()
        
        # Create session record
        await self.db.insert("""
            INSERT INTO sessions (id, user_id, started_at)
            VALUES (%s, %s, NOW())
        """, [session_id, user_id])
        
        # If resuming, hydrate working memory
        if resume_from:
            prev_session = await self.db.query_one("""
                SELECT * FROM sessions WHERE id = %s
            """, [resume_from])
            
            if prev_session and prev_session['context_snapshot']:
                # Load working memories from snapshot
                snapshot = json.loads(prev_session['context_snapshot'])
                working_memories = await self.db.query("""
                    SELECT * FROM memory_nodes WHERE id = ANY(%s)
                """, [snapshot.get('working_memories', [])])
                
                return Session(
                    id=session_id,
                    user_id=user_id,
                    working_memories=working_memories,
                    resumed_from=resume_from
                )
        
        return Session(id=session_id, user_id=user_id, working_memories=[])
    
    async def end_session(
        self,
        session_id: str,
        working_memories: List[UUID]
    ) -> None:
        """
        End session, save working memory snapshot.
        """
        snapshot = {
            'working_memories': [str(m) for m in working_memories],
            'ended_at': now().isoformat()
        }
        
        await self.db.execute("""
            UPDATE sessions 
            SET ended_at = NOW(), context_snapshot = %s
            WHERE id = %s
        """, [json.dumps(snapshot), session_id])
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Working storage and basic retrieval

```
Week 1:
- [ ] Set up Postgres + pgvector
- [ ] Create schema (memory_nodes, episodes, sessions)
- [ ] Implement basic CRUD
- [ ] Add embedding pipeline

Week 2:
- [ ] Implement vector similarity search
- [ ] Add salience scoring
- [ ] Build retrieval API
- [ ] Write basic tests
```

**Deliverable:** Can store and retrieve memories via API

### Phase 2: Intelligence (Weeks 3-4)

**Goal:** Smart writing and retrieval

```
Week 3:
- [ ] Implement write gate
- [ ] Add fact extraction
- [ ] Build conflict detection
- [ ] Add temporal queries

Week 4:
- [ ] Session management
- [ ] Working memory hydration
- [ ] Context injection formatting
- [ ] Initial benchmarks
```

**Deliverable:** Cross-session memory with intelligent filtering

### Phase 3: Integration (Weeks 5-6)

**Goal:** Production-ready integration

```
Week 5:
- [ ] Agent integration layer
- [ ] Configuration management
- [ ] Error handling & retries
- [ ] Logging & observability

Week 6:
- [ ] Performance optimization
- [ ] Load testing
- [ ] Documentation
- [ ] Example implementations
```

**Deliverable:** Production-ready v1

---

## 7. Migration Path

### 7.1 When to Add Graph Database

**Trigger:** Benchmarks show multi-hop queries are needed

```
Example query that might need graph:
"What projects has Alice worked on with Bob?"
- Requires: Alice -> projects, Bob -> projects, find intersection
- In SQL: Multiple joins, potentially slow
- In Graph: Single traversal

Migration:
1. Keep Postgres as source of truth
2. Add Neo4j for relationship queries
3. Sync writes to both
4. Route queries based on pattern
```

### 7.2 When to Add Dedicated Cache

**Trigger:** Latency requirements exceed 200ms p95

```
Migration:
1. Add Redis layer
2. Cache hot memories (LRU)
3. Cache session state
4. Implement cache invalidation
```

### 7.3 Zero-Downtime Migration Strategy

```
1. Deploy new schema alongside old
2. Dual-write to both schemas
3. Backfill old data
4. Switch read queries
5. Decommission old schema
```

---

## Summary

| Component | V1 Choice | Future Upgrade Path |
|-----------|-----------|---------------------|
| Database | Postgres + pgvector | Add Neo4j for graph queries |
| Cache | In-memory LRU | Redis for distributed cache |
| Embeddings | OpenAI text-embedding-3-small | Fine-tuned domain models |
| Scale | Single instance | Read replicas, partitioning |

**Key Decision:** Start with one database. Prove you need more before adding complexity.
