# Memory System Logging Specification

## Log Levels

### DEBUG — Everything
- Every embedding generation (input text, output hash, latency)
- Every database query (raw SQL, params, execution time)
- Vector similarity scores for all candidates (not just top-k)
- Write gate internal scores (novelty, utility, confidence, each computed separately)

### INFO — Operational
- Every write to memory (content, tier, confidence, triggers)
- Every retrieval query (query text, results count, top result score)
- Abstention decisions (why we didn't answer)
- Session boundaries (start/end, memory counts)

### WARN — Suspicious
- Low-confidence retrievals (top score < 0.3)
- Empty result sets on non-trivial queries
- Write gate rejecting borderline memories (score 0.45-0.55)
- Embedding generation failures (falling back to random)

### ERROR — Failures
- Database connection failures
- Storage failures (write attempted, failed)
- Retrieval failures (query crashed)
- Schema mismatches

## Log Format

```json
{
  "timestamp": "2026-03-14T08:05:00+08:00",
  "level": "INFO",
  "component": "memory_writer|retrieval|abstention|embedding",
  "operation": "store_fact|retrieve|should_abstain|generate",
  "session_id": "uuid",
  "latency_ms": 45,
  "payload": {
    // Operation-specific data
  }
}
```

## Specific Log Events

### Memory Write
```json
{
  "operation": "store_fact",
  "content": "dev wants comprehensive logging",
  "tier": "semantic",
  "confidence": 0.95,
  "write_gate_scores": {
    "novelty": 0.9,
    "utility": 0.95,
    "confidence": 0.9,
    "final": 0.92
  },
  "embedding_model": "mock_hash_v1",
  "embedding_dim": 1536,
  "memory_id": "uuid"
}
```

### Memory Retrieval
```json
{
  "operation": "retrieve",
  "query": "how does dev want logging",
  "query_embedding_hash": "a3f2...",
  "results_count": 5,
  "results": [
    {"memory_id": "uuid", "score": 0.87, "tier": "semantic"},
    {"memory_id": "uuid", "score": 0.23, "tier": "episodic"}
  ],
  "abstention": false,
  "latency_ms": 120
}
```

### Abstention
```json
{
  "operation": "should_abstain",
  "query": "what's my favorite color",
  "results_count": 0,
  "decision": "abstain",
  "reason": "No relevant memories found",
  "suggestion": "The system has no information about this query"
}
```

## Log Analysis Queries

**Retrieval quality:**
- "What % of queries return results?" → should be >70% for substantive queries
- "What's the avg top score?" → should be >0.5
- "How often do we abstain?" → track if it's too aggressive

**Write quality:**
- "What's the distribution of write gate scores?" → catch threshold issues
- "How many writes per session?" → too few = missing things, too many = noise
- "Which tier gets most writes?" → should be mostly semantic for facts

**Failure detection:**
- "Any queries with 0 results that should have matched?" → embedding quality issue
- "Any writes that failed silently?" → database issues
- "Embedding generation latency spikes?" → performance issues

## Audit Trail

Every conversation turn should produce:
1. PRE-RESPONSE: Retrieval log (what we searched, what we found)
2. DECISION: Whether we used memories in response
3. POST-RESPONSE: Any new facts extracted and stored
4. CHECKPOINT: Periodic full memory dump for forensics
