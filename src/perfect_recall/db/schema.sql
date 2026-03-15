-- ============================================================================
-- Perfect Recall v1 - Database Schema
-- PostgreSQL + pgvector - Four-Tier Memory Model
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- 1. SESSIONS - Working Memory Container
-- Active context (conscious awareness) - ephemeral, session-scoped
-- ============================================================================
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    agent_id VARCHAR(100),
    
    -- Temporal bounds
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Working memory snapshot (for resumption)
    context_snapshot JSONB DEFAULT '{}',
    
    -- Metrics
    message_count INTEGER DEFAULT 0,
    token_usage INTEGER DEFAULT 0,
    
    -- Metadata
    extra_metadata JSONB NOT NULL DEFAULT '{}',
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_agent ON sessions(agent_id);
CREATE INDEX idx_sessions_time ON sessions(started_at, ended_at);
CREATE INDEX idx_sessions_metadata ON sessions USING GIN (extra_metadata);

-- ============================================================================
-- 2. EPISODES - Event Sequences (Episodic Memory)
-- Groupings of related memories - autobiographical memory
-- ============================================================================
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    
    -- Temporal bounds
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    
    -- Episode classification
    episode_type VARCHAR(50) NOT NULL DEFAULT 'interaction'
        CHECK (episode_type IN ('message', 'action', 'decision', 'observation', 'reflection', 'interaction')),
    
    -- Summary for quick scanning
    summary TEXT,
    summary_embedding VECTOR(1536),
    
    -- Participants and topics
    participant_ids TEXT[] DEFAULT '{}',
    topic_tags TEXT[] DEFAULT '{}',
    
    -- Hierarchical linking (sub-episodes)
    parent_episode_id UUID REFERENCES episodes(id),
    
    -- Metrics
    memory_count INTEGER DEFAULT 0,
    
    -- Flexible metadata
    extra_metadata JSONB NOT NULL DEFAULT '{}',
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_episodes_session ON episodes(session_id);
CREATE INDEX idx_episodes_type ON episodes(episode_type);
CREATE INDEX idx_episodes_time ON episodes(started_at, ended_at);
CREATE INDEX idx_episodes_tags ON episodes USING GIN (topic_tags);
CREATE INDEX idx_episodes_participants ON episodes USING GIN (participant_ids);

-- ============================================================================
-- 3. MEMORY NODES - Unified Storage for Semantic, Episodic, Procedural
-- ============================================================================
CREATE TABLE memory_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Memory tier classification
    memory_tier VARCHAR(20) NOT NULL 
        CHECK (memory_tier IN ('working', 'episodic', 'semantic', 'procedural')),
    
    -- Content
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI dimension; adjust for your embedding model
    
    -- Temporal metadata (bi-temporal model)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,  -- NULL = still valid
    
    -- Salience metadata (for ranking)
    importance_score FLOAT NOT NULL DEFAULT 0.5 CHECK (importance_score BETWEEN 0 AND 1),
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    emotional_valence FLOAT CHECK (emotional_valence BETWEEN -1 AND 1),
    
    -- Provenance
    confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
    source_type VARCHAR(20) NOT NULL DEFAULT 'direct' 
        CHECK (source_type IN ('direct', 'inferred', 'imported', 'corrected', 'compound')),
    source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
    
    -- Versioning (for corrections/updates)
    version INTEGER NOT NULL DEFAULT 1,
    supersedes_id UUID REFERENCES memory_nodes(id),
    superseded_by_id UUID REFERENCES memory_nodes(id),
    
    -- Superpowers-inspired metadata fields for context-aware retrieval
    triggers TEXT[] DEFAULT '{}',         -- When to recall this memory (e.g., 'auth error', 'user asks about X')
    symptoms TEXT[] DEFAULT '{}',         -- Error phrases, failure patterns to match against
    aliases TEXT[] DEFAULT '{}',          -- Synonyms and related terms for flexible matching
    anti_triggers TEXT[] DEFAULT '{}',    -- When NOT to use this memory (negative context indicators)
    
    -- Flexible metadata
    extra_metadata JSONB NOT NULL DEFAULT '{}',
    
    -- Full-text search vector
    search_vector TSVECTOR
);

-- Core indexes
CREATE INDEX idx_memory_tier ON memory_nodes(memory_tier);
CREATE INDEX idx_memory_created ON memory_nodes(created_at);
CREATE INDEX idx_memory_valid ON memory_nodes(valid_from, valid_until);
CREATE INDEX idx_memory_importance ON memory_nodes(importance_score DESC);
CREATE INDEX idx_memory_last_accessed ON memory_nodes(last_accessed DESC);
CREATE INDEX idx_memory_metadata ON memory_nodes USING GIN (extra_metadata);
CREATE INDEX idx_memory_search ON memory_nodes USING GIN (search_vector);
CREATE INDEX idx_memory_source_episode ON memory_nodes(source_episode_id);

-- Superpowers metadata indexes for context-aware retrieval
CREATE INDEX idx_memory_triggers ON memory_nodes USING GIN (triggers);
CREATE INDEX idx_memory_symptoms ON memory_nodes USING GIN (symptoms);
CREATE INDEX idx_memory_aliases ON memory_nodes USING GIN (aliases);
CREATE INDEX idx_memory_anti_triggers ON memory_nodes USING GIN (anti_triggers);

-- Vector similarity index (IVFFlat for balance of speed/accuracy)
CREATE INDEX idx_memory_embedding ON memory_nodes 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search trigger
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_search_vector
    BEFORE INSERT OR UPDATE ON memory_nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_search_vector();

-- ============================================================================
-- 4. WORKING MEMORY - Active Context Reference Table
-- Links current session to actively relevant memories
-- ============================================================================
CREATE TABLE working_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    
    -- Working memory specific metadata
    priority FLOAT NOT NULL DEFAULT 0.5 CHECK (priority BETWEEN 0 AND 1),
    slot_type VARCHAR(20) NOT NULL DEFAULT 'context'
        CHECK (slot_type IN ('context', 'goal', 'task', 'scratchpad')),
    
    -- When added to working memory
    added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,  -- NULL = until session ends
    
    -- Position in working memory stack
    position INTEGER NOT NULL DEFAULT 0,
    
    extra_metadata JSONB NOT NULL DEFAULT '{}',
    
    UNIQUE (session_id, memory_id)
);

CREATE INDEX idx_working_memory_session ON working_memory(session_id);
CREATE INDEX idx_working_memory_memory ON working_memory(memory_id);
CREATE INDEX idx_working_memory_priority ON working_memory(session_id, priority DESC);
CREATE INDEX idx_working_memory_position ON working_memory(session_id, position);

-- ============================================================================
-- 5. EPISODE-MEMORY LINKING - Connect events to their constituent memories
-- ============================================================================
CREATE TABLE episode_memories (
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,  -- Order within episode
    
    PRIMARY KEY (episode_id, memory_id)
);

CREATE INDEX idx_ep_mem_memory ON episode_memories(memory_id);
CREATE INDEX idx_ep_mem_position ON episode_memories(episode_id, position);

-- ============================================================================
-- 6. MEMORY RELATIONSHIPS - Lightweight Graph Structure
-- ============================================================================
CREATE TABLE memory_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    target_memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    relationship_type VARCHAR(50) NOT NULL,
    strength FLOAT NOT NULL DEFAULT 1.0 CHECK (strength BETWEEN 0 AND 1),
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    extra_metadata JSONB NOT NULL DEFAULT '{}',
    
    UNIQUE (source_memory_id, target_memory_id, relationship_type)
);

CREATE INDEX idx_rel_source ON memory_relationships(source_memory_id);
CREATE INDEX idx_rel_target ON memory_relationships(target_memory_id);
CREATE INDEX idx_rel_type ON memory_relationships(relationship_type);

-- Common relationship types documentation:
-- 'precedes' - temporal ordering
-- 'causes' - causal link
-- 'relates_to' - general association
-- 'supersedes' - version replacement
-- 'mentions' - entity reference
-- 'part_of' - hierarchical composition
-- 'similar_to' - semantic similarity

-- ============================================================================
-- 7. ACCESS LOG - For LRU tracking and analytics
-- ============================================================================
CREATE TABLE memory_access_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    
    access_type VARCHAR(20) NOT NULL,  -- 'recall', 'write', 'update', 'view'
    query_text TEXT,  -- What query triggered this access
    
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_access_memory ON memory_access_log(memory_id);
CREATE INDEX idx_access_session ON memory_access_log(session_id);
CREATE INDEX idx_access_time ON memory_access_log(accessed_at);
CREATE INDEX idx_access_type ON memory_access_log(access_type);

-- ============================================================================
-- 8. CONFLICTS - False memory tracking and resolution
-- ============================================================================
CREATE TABLE memory_conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_a_id UUID NOT NULL REFERENCES memory_nodes(id),
    memory_b_id UUID NOT NULL REFERENCES memory_nodes(id),
    
    conflict_type VARCHAR(50) NOT NULL,  -- 'contradiction', 'temporal', 'exclusive'
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    resolution_status VARCHAR(20) DEFAULT 'open' 
        CHECK (resolution_status IN ('open', 'auto_resolved', 'user_resolved', 'ignored')),
    resolution_notes TEXT,
    resolved_at TIMESTAMPTZ,
    
    UNIQUE (memory_a_id, memory_b_id)
);

CREATE INDEX idx_conflicts_status ON memory_conflicts(resolution_status);
CREATE INDEX idx_conflicts_memory_a ON memory_conflicts(memory_a_id);
CREATE INDEX idx_conflicts_memory_b ON memory_conflicts(memory_b_id);

-- ============================================================================
-- 9. PROCEDURAL MEMORY PATTERNS - Skill/Pattern storage extension
-- ============================================================================
CREATE TABLE procedural_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id UUID NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    
    -- Pattern metadata
    pattern_name VARCHAR(200),
    trigger_patterns TEXT[] DEFAULT '{}',  -- Keywords/phrases that trigger this
    
    -- Success tracking
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    
    -- Execution metrics
    avg_execution_time_ms FLOAT,
    last_executed_at TIMESTAMPTZ,
    
    -- Context where pattern applies
    applicable_contexts TEXT[] DEFAULT '{}',
    
    extra_metadata JSONB NOT NULL DEFAULT '{}',
    
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_proc_memory ON procedural_patterns(memory_id);
CREATE INDEX idx_proc_triggers ON procedural_patterns USING GIN (trigger_patterns);
CREATE INDEX idx_proc_success_rate ON procedural_patterns(success_rate DESC);
CREATE INDEX idx_proc_contexts ON procedural_patterns USING GIN (applicable_contexts);

-- ============================================================================
-- Views for Convenience
-- ============================================================================

-- Active memories (not expired)
CREATE VIEW active_memories AS
SELECT * FROM memory_nodes
WHERE valid_until IS NULL OR valid_until > NOW();

-- Memories by tier with salience
CREATE VIEW memories_with_salience AS
SELECT 
    m.*,
    -- Simple salience calculation
    (m.importance_score * 0.4 + 
     LEAST(m.access_count / 20.0, 1.0) * 0.3 +
     CASE 
         WHEN m.last_accessed IS NULL THEN 0.1
         ELSE GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW() - m.last_accessed)) / 604800.0) * 0.3
     END
    ) as calculated_salience
FROM memory_nodes m
WHERE m.valid_until IS NULL OR m.valid_until > NOW();

-- Session with memory stats
CREATE VIEW session_stats AS
SELECT 
    s.*,
    COUNT(DISTINCT e.id) as episode_count,
    COUNT(DISTINCT m.id) as memory_count
FROM sessions s
LEFT JOIN episodes e ON e.session_id = s.id
LEFT JOIN memory_nodes m ON m.source_episode_id = e.id
GROUP BY s.id;
