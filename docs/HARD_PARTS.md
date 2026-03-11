# The Hard Parts: Unsolved Mechanics in Memory Systems

> **"The difference between a demo and a product is the hard parts."**

This document details the genuinely difficult, unsolved problems in building a production-grade agent memory system. These are the gaps between architecture diagrams and working code.

---

## Table of Contents

1. [The Write Decision Problem](#1-the-write-decision-problem)
2. [False Memory Prevention](#2-false-memory-prevention)
3. [Temporal Truth Maintenance](#3-temporal-truth-maintenance)
4. [Abstention: Knowing When You Don't Know](#4-abstention-knowing-when-you-dont-know)
5. [Context Injection Formatting](#5-context-injection-formatting)
6. [Open Research Questions](#6-open-research-questions)

---

## 1. The Write Decision Problem

### 1.1 The Core Dilemma

**What gets written to memory? When? And how do we avoid spam?**

Every interaction produces potential memories:
- User messages
- Agent responses  
- Tool outputs
- Internal reasoning traces
- System events

Write everything = storage explosion + retrieval noise  
Write too little = missed context + repeated explanations

### 1.2 The Spam Problem

```python
# Bad: Writing everything
await memory.write(user_message)      # "Thanks"
await memory.write(agent_response)    # "You're welcome"
await memory.write(user_message)      # "How are you?"
await memory.write(agent_response)    # "I'm doing well..."
# Result: 1000s of low-value memories drowning out important ones
```

### 1.3 Proposed: The Write Gate Algorithm

```python
class WriteGate:
    """
    Decides whether to write a potential memory.
    Multi-factor decision, not a single threshold.
    """
    
    def should_write(
        self,
        content: str,
        content_type: ContentType,
        context: ConversationContext
    ) -> WriteDecision:
        
        scores = {
            # Factor 1: Information density
            'novelty': self.score_novelty(content),
            
            # Factor 2: Potential future utility
            'utility': self.score_utility(content, context),
            
            # Factor 3: User explicit marking
            'explicit': self.score_explicit_marking(content),
            
            # Factor 4: Relationship to existing memories
            'connection': self.score_connection(content),
            
            # Factor 5: Emotional/relational significance
            'significance': self.score_significance(content),
        }
        
        # Dynamic threshold based on recent write volume
        threshold = self.dynamic_threshold(context.recent_write_rate)
        
        # Weighted combination
        final_score = weighted_sum(scores, weights={
            'novelty': 0.25,
            'utility': 0.25,
            'explicit': 0.20,
            'connection': 0.15,
            'significance': 0.15
        })
        
        if final_score >= threshold:
            return WriteDecision(
                write=True,
                memory_type=self.select_memory_type(content),
                importance=final_score,
                reason=scores
            )
        else:
            return WriteDecision(write=False, reason=scores)
```

### 1.4 Scoring Functions (Heuristic Implementations)

```python
def score_novelty(self, content: str) -> float:
    """
    How different is this from what we already know?
    Uses embedding similarity to existing memories.
    """
    embedding = embed(content)
    similar = vector_db.query(embedding, k=3)
    
    if not similar:
        return 1.0  # Entirely new
    
    max_similarity = max(s.score for s in similar)
    return 1.0 - max_similarity  # Higher = more novel


def score_utility(self, content: str, context: Context) -> float:
    """
    How likely is this to be useful later?
    This is the hardest score - requires prediction.
    """
    # Heuristic: Facts about user preferences are high utility
    if contains_user_preference_indicators(content):
        return 0.8
    
    # Heuristic: Technical decisions have future utility
    if contains_decision_keywords(content):
        return 0.7
    
    # Heuristic: Task context has utility for duration of task
    if context.active_task and is_task_relevant(content, context.active_task):
        return 0.6
    
    # Default: Low predicted utility for chitchat
    return 0.2


def score_explicit_marking(self, content: str) -> float:
    """
    Did the user explicitly mark this as important?
    """
    importance_markers = [
        "remember this",
        "don't forget",
        "important:",
        "note that",
        "for future reference"
    ]
    
    lower = content.lower()
    for marker in importance_markers:
        if marker in lower:
            return 1.0
    
    return 0.0
```

### 1.5 The "Important But Not Now" Problem

Some information is important but not immediately:
- "I'll be on vacation next month" (write date, retrieve near that date)
- "We're planning to migrate to AWS in Q3" (write now, retrieve in Q3)

**Solution: Temporal Write Scheduling**

```python
@dataclass
class DeferredMemory:
    """Memory that should be "activated" at a future time."""
    content: str
    write_immediately: bool  # Store now or later?
    activate_at: Optional[datetime]  # When to start retrieving this
    decay_after: Optional[datetime]  # When it becomes less relevant
```

### 1.6 Open Questions

- How to learn write preferences per user?
- How to handle conflicting write decisions? (User says "remember this" but content is very similar to existing memory)
- What's the right storage cost / retrieval quality tradeoff?

---

## 2. False Memory Prevention

### 2.1 The Problem

LLM-based extraction creates false memories:
- Hallucinated facts
- Misattributed statements  
- Confabulated connections
- Overgeneralized patterns

### 2.2 Types of False Memories

| Type | Example | Source |
|------|---------|--------|
| **Hallucination** | "User works at Google" (never said) | LLM inference |
| **Misattribution** | Attribute Alice's preference to Bob | Entity resolution error |
| **Confabulation** | Connect unrelated events | Pattern matching error |
| **Overgeneralization** | "User hates all JavaScript" (hates one framework) | Extraction overreach |
| **Temporal confusion** | "User prefers X" (used to, now prefers Y) | Update failure |

### 2.3 Proposed: Confidence & Provenance System

```python
@dataclass
class MemoryProvenance:
    """Complete chain of custody for a memory."""
    
    # Source
    source_type: SourceType  # DIRECT, INFERRED, IMPORTED
    source_episode: UUID     # Where this came from
    source_content: str      # Original raw text
    
    # Extraction
    extraction_method: str   # "llm-extract-v1", "user-stated", "inferred"
    extraction_confidence: float  # 0.0 - 1.0
    extraction_prompt: str   # What prompt extracted this
    
    # Verification
    verification_status: VerificationStatus
    verifications: List[VerificationAttempt]
    
    # Lineage
    parent_memory: Optional[UUID]  # If derived from another memory
    child_memories: List[UUID]     # Memories derived from this


class VerificationStatus(Enum):
    UNVERIFIED = "unverified"      # Just extracted
    USER_CONFIRMED = "confirmed"   # User explicitly confirmed
    CROSS_REFERENCED = "xref"      # Matches other sources
    CONTRADICTED = "contradicted"  # Conflicts with other memories
    RETRACTED = "retracted"        # User said this is wrong
```

### 2.4 Conflict Detection Algorithm

```python
class ConflictDetector:
    """Detects when memories contradict each other."""
    
    def find_conflicts(self, new_memory: Memory) -> List[Conflict]:
        conflicts = []
        
        # Type 1: Direct contradiction
        # "User likes X" vs "User dislikes X"
        direct = self.find_direct_contradictions(new_memory)
        conflicts.extend(direct)
        
        # Type 2: Temporal contradiction  
        # "User prefers X" (2024) vs "User prefers Y" (2025)
        temporal = self.find_temporal_contradictions(new_memory)
        conflicts.extend(temporal)
        
        # Type 3: Mutual exclusivity
        # "User lives in NYC" vs "User lives in LA" (can't be both)
        exclusive = self.find_mutually_exclusive(new_memory)
        conflicts.extend(exclusive)
        
        # Type 4: Confidence contradiction
        # High-confidence fact vs low-confidence inference
        confidence = self.find_confidence_discrepancies(new_memory)
        conflicts.extend(confidence)
        
        return conflicts
    
    def find_direct_contradictions(self, memory: Memory) -> List[Conflict]:
        """
        Find memories with opposite predicates about same subject.
        Uses semantic similarity + opposition detection.
        """
        # Query for similar subject-predicate pairs
        candidates = self.query_similar(memory.subject, memory.predicate)
        
        conflicts = []
        for candidate in candidates:
            if self.are_opposites(memory.object, candidate.object):
                conflicts.append(Conflict(
                    type=ConflictType.DIRECT_CONTRADICTION,
                    memory_a=memory.id,
                    memory_b=candidate.id,
                    explanation=f"'{memory.object}' contradicts '{candidate.object}'"
                ))
        
        return conflicts
```

### 2.5 Resolution Strategies

```python
class ConflictResolver:
    """Strategies for handling detected conflicts."""
    
    def resolve(self, conflict: Conflict) -> Resolution:
        """
        Attempt to resolve a conflict automatically or escalate.
        """
        
        # Strategy 1: Temporal resolution
        # If both have valid_time, check if sequential (not conflicting)
        if self.is_temporal_sequence(conflict):
            return Resolution(
                action=ResolutionAction.ACCEPT_BOTH,
                explanation="Sequential truths - both valid in their time periods"
            )
        
        # Strategy 2: Confidence-based
        # Keep higher-confidence, flag lower-confidence as superseded
        if self.confidence_differs_significantly(conflict):
            winner = self.higher_confidence_memory(conflict)
            return Resolution(
                action=ResolutionAction.SUPERSEDE,
                keep=winner.id,
                deprecate=conflict.other(winner).id,
                explanation="Higher confidence memory preferred"
            )
        
        # Strategy 3: Recency-based
        # More recent memory wins (with user confirmation pending)
        if self.recency_differs_significantly(conflict):
            winner = self.more_recent_memory(conflict)
            return Resolution(
                action=ResolutionAction.SUPERSEDE_PENDING_CONFIRMATION,
                keep=winner.id,
                deprecate=conflict.other(winner).id,
                explanation="More recent memory preferred, pending user confirmation"
            )
        
        # Strategy 4: Escalate to user
        # Can't auto-resolve - ask the user
        return Resolution(
            action=ResolutionAction.ESCALATE_TO_USER,
            conflict=conflict,
            explanation="Cannot auto-resolve - requires human judgment"
        )
```

### 2.6 The "User Corrects Themselves" Problem

```
User: I prefer Python.
[Memory stored: user prefers Python]

User: Actually, I prefer JavaScript now.
[New memory: user prefers JavaScript]
[Conflict detected with old memory]
[Resolution: Mark old memory valid_until=new_memory.valid_from]
```

### 2.7 Open Questions

- How to detect subtle contradictions? ("I love X" vs "X is okay I guess")
- What's the right UI for conflict resolution?
- How to prevent confirmation fatigue?
- How to handle "it depends" contradictions?

---

## 3. Temporal Truth Maintenance

### 3.1 The Problem

Truths change over time:
- "I work at Google" → "I work at Meta"
- "I'm single" → "I'm married"
- "We use AWS" → "We migrated to GCP"

Current systems often store facts as timeless. This is wrong.

### 3.2 Bi-Temporal Model (Implemented)

```python
@dataclass
class TemporalFact:
    """
    Bi-temporal: valid_time (when true in world) + 
                transaction_time (when we learned it)
    """
    subject: str
    predicate: str
    object: str
    
    valid_from: datetime      # When became true in world
    valid_until: datetime     # When ceased being true (NULL = still true)
    
    learned_at: datetime      # When we stored this
    superseded_at: datetime   # When we learned it's no longer true
    
    confidence: float
```

### 3.3 Temporal Query Resolution

```python
class TemporalQueryEngine:
    """Query facts as they were at a specific time."""
    
    def query_at(
        self,
        subject: str,
        predicate: str,
        at_time: datetime
    ) -> Optional[TemporalFact]:
        """
        Return the fact as it was known at `at_time`.
        """
        # Find all versions of this fact
        versions = self.db.query(
            subject=subject,
            predicate=predicate,
            order_by="valid_from"
        )
        
        # Find version that was valid at query time
        for version in versions:
            if version.valid_from <= at_time:
                if version.valid_until is None or version.valid_until > at_time:
                    return version
        
        return None  # No fact was true at that time
```

### 3.4 The "Gradual Change" Problem

Some truths don't change instantly:
- "We're planning to migrate" (planning phase)
- "We're migrating" (in progress)
- "We migrated" (completed)
- "We use GCP now" (new steady state)

**Solution: State Transition Model**

```python
class FactState(Enum):
    PLANNED = "planned"       # Intention, not yet true
    TRANSITIONING = "transitioning"  # In progress
    CURRENT = "current"       # True now
    DEPRECATED = "deprecated" # No longer true

@dataclass
class StatefulFact(TemporalFact):
    state: FactState
    transition_history: List[StateTransition]
    expected_completion: Optional[datetime]  # For PLANNED/TRANSITIONING
```

### 3.5 The "Fuzzy Validity" Problem

Some truths are fuzzy:
- "I prefer Python" (usually true, but depends on context)
- "I live in NYC" (true, but traveling for 3 months)

**Solution: Context-Conditional Validity**

```python
@dataclass
class ConditionalFact:
    """Fact that's only true under certain conditions."""
    
    base_fact: TemporalFact
    conditions: List[Condition]
    default_validity: float  # Probability of being true when conditions unknown

@dataclass  
class Condition:
    type: ConditionType  # TIME, LOCATION, ACTIVITY, etc.
    parameters: Dict[str, Any]
```

### 3.6 Open Questions

- How to handle "I don't know when it changed, but it did"?
- How to propagate temporal updates through derived facts?
- What's the UI for "show me what you knew as of [date]"?

---

## 4. Abstention: Knowing When You Don't Know

### 4.1 The Problem

Agents hallucinate when they lack information. We need explicit abstention:
- "I don't know"
- "I don't remember"
- "I don't have enough information"

### 4.2 The Abstention Threshold

```python
class AbstentionController:
    """
    Decides when to abstain vs. attempt answer.
    """
    
    def should_abstain(
        self,
        query: str,
        retrieved_memories: List[RetrievedMemory],
        confidence_scores: List[float]
    ) -> AbstentionDecision:
        
        # Signal 1: Coverage gap
        # Do retrieved memories cover the query topic?
        coverage = self.assess_coverage(query, retrieved_memories)
        
        # Signal 2: Confidence gap
        # Are retrieved memories confident enough?
        avg_confidence = mean(confidence_scores) if confidence_scores else 0
        
        # Signal 3: Retrieval gap
        # Did we retrieve ANY relevant memories?
        retrieval_rate = len(retrieved_memories) / self.expected_recall_size(query)
        
        # Signal 4: Contradiction detected
        # Are retrieved memories contradictory?
        has_contradiction = self.detect_contradictions(retrieved_memories)
        
        # Combined signal
        abstention_score = weighted_sum({
            'coverage_gap': 1.0 - coverage,
            'confidence_gap': 1.0 - avg_confidence,
            'retrieval_gap': 1.0 - retrieval_rate,
            'contradiction': 1.0 if has_contradiction else 0.0
        })
        
        if abstention_score > self.abstention_threshold:
            return AbstentionDecision(
                abstain=True,
                reason=self.generate_abstention_reason(
                    coverage, avg_confidence, has_contradiction
                ),
                confidence=abstention_score
            )
        
        return AbstentionDecision(abstain=False)
```

### 4.3 Coverage Assessment

```python
def assess_coverage(
    self,
    query: str,
    memories: List[RetrievedMemory]
) -> float:
    """
    Assess whether retrieved memories cover the query topic.
    Uses query decomposition + semantic matching.
    """
    # Decompose query into sub-questions
    sub_questions = self.decompose_query(query)
    
    if not sub_questions:
        return 1.0  # Simple query
    
    # Check coverage for each sub-question
    covered = 0
    for sq in sub_questions:
        # Does any memory address this sub-question?
        if any(self.addresses(memory, sq) for memory in memories):
            covered += 1
    
    return covered / len(sub_questions)
```

### 4.4 Types of Abstention

```python
class AbstentionType(Enum):
    # No relevant memories found
    NO_INFORMATION = "I don't have any information about that."
    
    # Found memories but too low confidence
    LOW_CONFIDENCE = "I have some information, but I'm not confident about it."
    
    # Found contradictory memories
    CONTRADICTORY = "I have conflicting information about that."
    
    # Found memories but coverage is incomplete
    INCOMPLETE = "I have partial information, but not enough to answer fully."
    
    # Information exists but is outdated
    OUTDATED = "I have information, but it may be out of date."
    
    # Information exists but requires user confirmation
    UNVERIFIED = "I have unverified information that I should check with you."
```

### 4.5 The "Calibrated Abstention" Problem

We want abstention to be calibrated:
- If we abstain 20% of the time, we should be right 80% of the time
- Better to abstain than confidently hallucinate

**Calibration via feedback:**

```python
class AbstentionCalibrator:
    """
    Learns optimal abstention threshold from user feedback.
    """
    
    def record_outcome(
        self,
        query: str,
        abstention_decision: AbstentionDecision,
        user_feedback: UserFeedback
    ):
        """
        User provides feedback:
        - "You should have known that" → threshold too high
        - "Good thing you asked" → threshold appropriate
        - "You could have answered" → threshold too low
        """
        if user_feedback.type == "SHOULD_HAVE_KNOWN":
            self.adjust_threshold(-0.05)  # Be less conservative
        elif user_feedback.type == "GOOD_ABSTENTION":
            self.adjust_threshold(0.01)   # Slightly more conservative
        elif user_feedback.type == "UNNECESSARY_ABSTENTION":
            self.adjust_threshold(-0.03)  # Be less conservative
```

### 4.6 Open Questions

- How to balance helpfulness vs. accuracy?
- What's the right abstention threshold per domain?
- How to handle "I don't know but I can guess"?
- Should abstention be explicit or graceful degradation?

---

## 5. Context Injection Formatting

### 5.1 The Problem

Retrieved memories must be formatted for the LLM. Format matters:
- Position in prompt (beginning, middle, end)
- Structure (bullets, sections, inline)
- Signaling (special tokens, XML tags)
- Compression level (full, summary, reference)

### 5.2 Format Patterns

```python
class MemoryFormatter:
    """
    Formats retrieved memories for prompt injection.
    """
    
    def format_memories(
        self,
        memories: List[RetrievedMemory],
        format_type: FormatType,
        max_tokens: int
    ) -> str:
        
        if format_type == FormatType.SECTIONED:
            return self.format_sectioned(memories, max_tokens)
        elif format_type == FormatType.XML:
            return self.format_xml(memories, max_tokens)
        elif format_type == FormatType.INLINE:
            return self.format_inline(memories, max_tokens)
        elif format_type == FormatType.JSON:
            return self.format_json(memories, max_tokens)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def format_sectioned(self, memories, max_tokens):
        """
        <memories>
        ## Facts
        - [CONFIDENCE: 0.95] User prefers Python
        - [CONFIDENCE: 0.87] User works at Google
        
        ## Previous Conversations
        - [2024-01-15] Discussed API design
        - [2024-01-10] Debugged authentication issue
        
        ## Relevant Skills
        - FastAPI authentication patterns
        </memories>
        """
        parts = ["<memories>"]
        
        # Group by type
        by_type = groupby(memories, key=lambda m: m.type)
        
        for mem_type, group in by_type:
            parts.append(f"## {mem_type.value.title()}")
            for memory in group:
                line = f"- [CONFIDENCE: {memory.confidence:.2f}] {memory.content}"
                parts.append(line)
            parts.append("")
        
        parts.append("</memories>")
        
        result = "\n".join(parts)
        return truncate_to_tokens(result, max_tokens)
```

### 5.3 Position Matters

```python
class ContextAssembler:
    """
    Assembles final prompt with memories in optimal positions.
    """
    
    def assemble(
        self,
        system_prompt: str,
        memories: List[RetrievedMemory],
        recent_context: List[Message],
        user_query: str
    ) -> str:
        """
        Structure:
        1. System prompt (instructions)
        2. Critical semantic facts (high confidence)
        3. User query context
        4. Supporting episodic memories
        5. Recent conversation
        6. Current user query
        """
        parts = []
        
        # 1. System prompt
        parts.append(f"<system>\n{system_prompt}\n</system>")
        
        # 2. Critical facts (beginning bias)
        critical_facts = [m for m in memories if m.is_critical]
        if critical_facts:
            parts.append("<critical_facts>")
            parts.append(self.format_facts(critical_facts))
            parts.append("</critical_facts>")
        
        # 3. User query context (if available)
        if self.has_query_context(user_query):
            parts.append(f"<context>\n{self.get_query_context(user_query)}\n</context>")
        
        # 4. Supporting memories (middle)
        supporting = [m for m in memories if not m.is_critical]
        if supporting:
            parts.append("<supporting_memories>")
            parts.append(self.format_memories(supporting))
            parts.append("</supporting_memories>")
        
        # 5. Recent conversation (recency bias)
        if recent_context:
            parts.append("<recent_conversation>")
            parts.append(self.format_messages(recent_context))
            parts.append("</recent_conversation>")
        
        # 6. User query
        parts.append(f"<user_query>\n{user_query}\n</user_query>")
        
        return "\n\n".join(parts)
```

### 5.4 Compression Levels

```python
class MemoryCompressor:
    """
    Compresses memories to fit token budget.
    """
    
    def compress(
        self,
        memory: RetrievedMemory,
        level: CompressionLevel
    ) -> str:
        """
        Compression levels:
        - FULL: Complete content
        - SUMMARY: Condensed version (stored summary or on-the-fly)
        - REFERENCE: Just enough to reference (e.g., "the API issue from January")
        - POINTER: Just ID/type (for retrieval if needed)
        """
        if level == CompressionLevel.FULL:
            return memory.content
        
        elif level == CompressionLevel.SUMMARY:
            if memory.summary:
                return memory.summary
            return self.summarize(memory.content, max_tokens=50)
        
        elif level == CompressionLevel.REFERENCE:
            date_str = memory.created_at.strftime("%Y-%m-%d")
            return f"[{date_str}] {self.generate_reference(memory)}"
        
        elif level == CompressionLevel.POINTER:
            return f"[MEMORY:{memory.id}:{memory.type.value}]"
        
        else:
            raise ValueError(f"Unknown compression level: {level}")
```

### 5.5 Open Questions

- What's the optimal format per model? (GPT-4 vs Claude vs Llama)
- How to measure format effectiveness?
- Does position bias vary by model?
- How much context is too much? (context poisoning)

---

## 6. Open Research Questions

### 6.1 Unsolved Problems

1. **Episodic-to-Semantic Distillation**
   - How to convert "what happened" to "what is true" reliably?
   - When is a pattern a rule vs. coincidence?

2. **Memory Forgetting**
   - What should be forgotten?
   - How to forget gracefully (fade vs. delete)?
   - Can we compress without losing?

3. **Cross-Agent Memory Transfer**
   - How to transfer memories between agents?
   - How to handle different contexts/users?
   - Privacy preservation?

4. **Emergent Memory Structure**
   - Can memory structure self-organize?
   - Dynamic schema evolution?
   - Auto-discovered relationship types?

5. **Emotional Memory**
   - Should emotional valence affect storage/retrieval?
   - How to weight "important because emotional" vs "important because useful"?

6. **Dream Consolidation**
   - Can offline processing improve memory organization?
   - Pattern discovery during idle time?
   - Similarity graph optimization?

### 6.2 Measurement Challenges

- How to measure "quality of memory"?
- What's the right benchmark for temporal reasoning?
- How to evaluate subjective user experience?

### 6.3 Ethical Questions

- Right to be forgotten (GDPR)
- Memory ownership (who owns agent memories?)
- Coercion through selective memory
- Privacy of inferred facts

---

## Conclusion

These are the real challenges. The architecture is the easy part. Making it work reliably, at scale, with real users—that's the hard part.

**Status:**  
🟡 Write decisions - Partial solution  
🟡 False memory prevention - Framework defined, needs validation  
🟢 Temporal truth maintenance - Bi-temporal model implemented  
🟡 Abstention - Algorithm defined, needs calibration  
🟡 Context formatting - Heuristics only, needs systematic study

**Next Steps:**
1. Implement write gate with user-tunable thresholds
2. Build conflict detection into storage layer
3. Add abstention metrics to evaluation framework
4. Run format effectiveness study across models
