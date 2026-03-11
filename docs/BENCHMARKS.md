# Benchmarking Strategy: Measuring What Matters

> **"You can't improve what you don't measure—and you can't trust measurements without baselines."**

This document establishes rigorous benchmarks for evaluating Perfect Recall against real baselines with meaningful metrics.

---

## Table of Contents

1. [Benchmarking Philosophy](#1-benchmarking-philosophy)
2. [Baseline Systems](#2-baseline-systems)
3. [Benchmark Datasets](#3-benchmark-datasets)
4. [Metrics Framework](#4-metrics-framework)
5. [Evaluation Protocol](#5-evaluation-protocol)
6. [Cost Analysis](#6-cost-analysis)
7. [When Memory Wins vs. When It Doesn't](#7-when-memory-wins-vs-when-it-doesnt)

---

## 1. Benchmarking Philosophy

### 1.1 Core Principles

1. **Compare against real alternatives** - Not theoretical, but actual working systems
2. **Measure what users care about** - Accuracy AND cost AND latency
3. **Test on realistic data** - Not synthetic cherry-picked examples
4. **Report confidence intervals** - Not single numbers
5. **Show failure modes** - Where does it break?

### 1.2 What We're NOT Doing

- ❌ Comparing against hypothetical "perfect" systems
- ❌ Using toy datasets that don't represent real usage
- ❌ Reporting only aggregate metrics (hiding variance)
- ❌ Ignoring cost (accuracy at any price is not useful)
- ❌ Testing only success cases
- ❌ Citing invented improvement percentages without evidence

---

## 2. Baseline Systems

### 2.1 Baseline A: Full-Context LLM

```python
class FullContextBaseline:
    """
    No memory system - just stuff everything into context.
    Serves as ceiling for retrieval quality, floor for cost.
    """
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 128000):
        self.model = model
        self.max_tokens = max_tokens
        self.context_buffer = []
    
    async def respond(self, user_message: str, conversation_history: List[Message]) -> Response:
        # Append to growing context
        self.context_buffer.extend(conversation_history)
        self.context_buffer.append(Message(role="user", content=user_message))
        
        # When approaching limit, summarize oldest
        while token_count(self.context_buffer) > self.max_tokens * 0.9:
            self.context_buffer = await self.summarize_oldest(self.context_buffer)
        
        return await llm.complete(self.context_buffer)
```

**Why this baseline:** Shows what you get without any memory system investment.

### 2.2 Baseline B: MemGPT-Style

```python
class MemGPTBaseline:
    """
    OS-inspired virtual memory with self-directed paging.
    Agent manages memory via function calls.
    """
    
    def __init__(self, context_window: int = 8000):
        self.main_context = []  # "RAM"
        self.external_context = MemoryStore()  # "Disk"
        self.context_window = context_window
    
    async def respond(self, user_message: str) -> Response:
        # Agent decides what to page in/out via function calls
        tools = [
            memory_search_tool,
            memory_store_tool,
            memory_page_in_tool,
            memory_page_out_tool
        ]
        
        return await llm.complete_with_tools(
            messages=self.main_context,
            tools=tools
        )
```

**Reference:** Packer et al. (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.

**Why this baseline:** Represents current state-of-the-art agent-managed memory.

### 2.3 Baseline C: Mem0-Style

```python
class Mem0Baseline:
    """
    Fact-based memory with automatic extraction and conflict resolution.
    """
    
    async def respond(self, user_message: str) -> Response:
        # Extract facts from conversation
        facts = await extract_facts(user_message)
        await self.memory.add(facts)
        
        # Retrieve relevant facts
        relevant_facts = await self.memory.search(user_message)
        
        # Add to prompt
        context = format_facts(relevant_facts)
        
        return await llm.complete(context + user_message)
```

**Why this baseline:** Represents semantic-only memory without episodic structure.

### 2.4 Baseline D: Zep-Style

```python
class ZepBaseline:
    """
    Temporal knowledge graph with bi-temporal facts.
    """
    
    def __init__(self):
        self.graph = TemporalKnowledgeGraph()
    
    async def respond(self, user_message: str, timestamp: datetime) -> Response:
        # Query graph at specific time
        facts = await self.graph.query(
            query=user_message,
            at_time=timestamp
        )
        
        context = self.format_graph_results(facts)
        return await llm.complete(context + user_message)
```

**Reference:** Rasmussen, P., et al. (2025). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory."

**Why this baseline:** Represents graph-based temporal memory.

---

## 3. Benchmark Datasets

### 3.1 Deep Memory Retrieval (DMR)

**Source:** MemGPT paper (Packer et al. 2023)

**Task:** Retrieve specific "needle" facts embedded in long conversation history.

```python
class DMRBenchmark:
    """
    Tests ability to find specific facts from deep history.
    """
    
    async def run(self, system: MemorySystem) -> DMRResult:
        # Create conversation with embedded needles
        conversation = self.generate_conversation(
            length=1000,  # messages
            needle_count=10,
            needle_positions=[50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
        )
        
        # Feed to system
        for msg in conversation.messages:
            await system.process(msg)
        
        # Query for each needle
        results = []
        for needle in conversation.needles:
            response = await system.query(needle.query)
            found = needle.target in response
            results.append(found)
        
        return DMRResult(
            recall_rate=mean(results),
            needle_positions=conversation.needle_positions,
            per_position_results=results
        )
```

**Metrics:**
- Recall@k: % of needles found in top-k results
- Position-dependent recall: Does recall degrade with depth?

### 3.2 LongMemEval

**Source:** Zhang et al. (2024)

**Task:** Answer questions requiring understanding of long documents with temporal reasoning.

```python
class LongMemEvalBenchmark:
    """
    Tests temporal reasoning over long contexts.
    """
    
    async def run(self, system: MemorySystem) -> LongMemEvalResult:
        # Load test cases
        test_cases = load_longmemeval_dataset()
        
        results = []
        for test in test_cases:
            # Ingest document
            await system.ingest(test.document)
            
            # Answer questions
            for question in test.questions:
                answer = await system.answer(question.text)
                
                # Judge correctness
                correct = await self.judge(answer, question.ground_truth)
                
                results.append({
                    'question_type': question.type,  # 'temporal', 'factual', 'inferential'
                    'correct': correct,
                    'token_position': question.token_position
                })
        
        return LongMemEvalResult(
            overall_accuracy=mean(r['correct'] for r in results),
            by_type=self.group_by_type(results),
            by_position=self.group_by_position(results)
        )
```

**Metrics:**
- Accuracy by question type
- Accuracy by position in document
- Temporal reasoning accuracy

### 3.3 LoCoMo (Long Context Modeling)

**Source:** Long Context Benchmark Suite

**Task:** Multi-session conversations with continuity requirements.

```python
class LoCoMoBenchmark:
    """
    Tests cross-session memory and continuity.
    """
    
    async def run(self, system: MemorySystem) -> LoCoMoResult:
        scenarios = load_locomo_scenarios()
        
        results = []
        for scenario in scenarios:
            # Simulate multi-session interaction
            for session in scenario.sessions:
                # Start/continue session
                await system.start_session(scenario.user_id)
                
                # Have conversation
                for msg in session.messages:
                    await system.interact(msg)
                
                await system.end_session()
            
            # Test continuity
            for test in scenario.continuity_tests:
                answer = await system.answer(test.question)
                correct = await self.judge(answer, test.expected)
                
                results.append({
                    'sessions_ago': test.sessions_ago,
                    'correct': correct,
                    'information_type': test.info_type  # 'fact', 'preference', 'event'
                })
        
        return LoCoMoResult(
            continuity_score=mean(r['correct'] for r in results),
            by_gap=self.group_by_gap(results),
            by_type=self.group_by_type(results)
        )
```

**Metrics:**
- Continuity score: % of cross-session facts recalled
- Degradation curve: How does recall decay over sessions?
- Information type breakdown: Facts vs. preferences vs. events

### 3.4 Custom: Perfect Recall Benchmark

**What existing benchmarks miss:**
- Write decision quality
- Conflict resolution
- Abstention appropriateness
- Context injection effectiveness

```python
class PerfectRecallBenchmark:
    """
    Custom benchmark for Perfect Recall specific capabilities.
    """
    
    async def run_write_decision_test(self, system: MemorySystem) -> WriteDecisionResult:
        """
        Test what gets written vs. what should be written.
        """
        test_cases = [
            {
                'content': 'User: I like Python.',
                'should_write': True,
                'reason': 'User preference'
            },
            {
                'content': 'User: Thanks!',
                'should_write': False,
                'reason': 'Low information value'
            },
            {
                'content': 'User: Remember this: my API key is xyz123',
                'should_write': True,
                'reason': 'Explicit importance marker'
            },
            # ... more cases
        ]
        
        results = []
        for case in test_cases:
            decision = await system.write_gate_decision(case['content'])
            results.append({
                'expected': case['should_write'],
                'actual': decision.should_write,
                'correct': case['should_write'] == decision.should_write
            })
        
        return WriteDecisionResult(
            accuracy=mean(r['correct'] for r in results),
            precision=precision_score(results),
            recall=recall_score(results)
        )
    
    async def run_conflict_resolution_test(self, system: MemorySystem) -> ConflictResolutionResult:
        """
        Test detection and resolution of contradictory memories.
        """
        # Store initial fact
        await system.store_fact("User works at Google")
        
        # Store contradictory fact
        await system.store_fact("User works at Meta")
        
        # Check if conflict detected
        conflicts = await system.get_conflicts()
        
        # Check resolution
        resolved = await system.resolve_conflicts()
        
        # Query - should return most recent or ask for clarification
        answer = await system.query("Where does user work?")
        
        return ConflictResolutionResult(
            conflict_detected=len(conflicts) > 0,
            resolution_appropriate=self.judge_resolution(answer),
            temporal_accuracy=self.check_temporal_handling(answer)
        )
    
    async def run_abstention_test(self, system: MemorySystem) -> AbstentionResult:
        """
        Test appropriate abstention vs. hallucination.
        """
        test_cases = [
            {
                'query': "What is user's favorite color?",
                'has_been_stored': True,
                'stored_value': 'blue',
                'should_abstain': False
            },
            {
                'query': "What is user's blood type?",
                'has_been_stored': False,
                'should_abstain': True
            },
            # ... more cases
        ]
        
        results = []
        for case in test_cases:
            response = await system.query(case['query'])
            abstained = "don't know" in response.lower() or "not sure" in response.lower()
            
            results.append({
                'should_abstain': case['should_abstain'],
                'did_abstain': abstained,
                'correct': case['should_abstain'] == abstained
            })
        
        return AbstentionResult(
            accuracy=mean(r['correct'] for r in results),
            false_positive_rate=mean(  # Abstained when had info
                r['did_abstain'] and not r['should_abstain'] 
                for r in results
            ),
            false_negative_rate=mean(  # Answered when should abstain
                not r['did_abstain'] and r['should_abstain']
                for r in results
            )
        )
```

---

## 4. Metrics Framework

### 4.1 Accuracy Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Recall@k** | % of relevant memories in top-k results | > 85% |
| **Precision@k** | % of top-k results that are relevant | > 75% |
| **MRR** | Mean Reciprocal Rank of first relevant | > 0.7 |
| **NDCG** | Normalized Discounted Cumulative Gain | > 0.75 |
| **Answer Accuracy** | % of questions answered correctly | > 80% |
| **Continuity Score** | % of cross-session facts retained | > 70% |

**Note:** Targets are aspirational goals based on initial estimates. Actual targets should be validated through baseline measurements.

### 4.2 Cost Metrics

| Metric | Definition | Target vs Baseline |
|--------|------------|-------------------|
| **Tokens per Query** | Avg tokens in context | < 50% of full-context |
| **Cost per Session** | API costs (embedding + LLM) | < 60% of full-context |
| **Storage Cost** | Per-user storage cost | < $0.01/day |
| **Compute Cost** | Embedding + retrieval cost | < 30% of LLM cost |

### 4.3 Latency Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **p50 Retrieval** | Median retrieval time | < 100ms |
| **p95 Retrieval** | 95th percentile | < 300ms |
| **p99 Retrieval** | 99th percentile | < 800ms |
| **End-to-End** | Query to response | < 3s |

### 4.4 System Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Write Efficiency** | % of interactions written | 10-30% |
| **Conflict Detection Rate** | % of conflicts caught | > 70% |
| **Abstention Appropriateness** | % correct abstentions | > 75% |
| **Memory Compression** | Compression ratio | 3:1 to 5:1 |

---

## 5. Evaluation Protocol

### 5.1 Standardized Test Harness

```python
class BenchmarkHarness:
    """
    Standardized evaluation environment.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results_db = ResultsDatabase()
    
    async def run_benchmark(
        self,
        system: MemorySystem,
        benchmark: Benchmark,
        repetitions: int = 3
    ) -> BenchmarkReport:
        """
        Run benchmark with multiple repetitions for confidence intervals.
        """
        results = []
        
        for i in range(repetitions):
            # Fresh system for each run
            fresh_system = await self.reset_system(system)
            
            # Run benchmark
            result = await benchmark.run(fresh_system)
            results.append(result)
        
        # Calculate statistics
        report = BenchmarkReport(
            benchmark_name=benchmark.name,
            system_name=system.name,
            mean=self.calculate_mean(results),
            std=self.calculate_std(results),
            ci95=self.calculate_confidence_interval(results, 0.95),
            raw_results=results
        )
        
        await self.results_db.store(report)
        return report
    
    async def compare_systems(
        self,
        systems: List[MemorySystem],
        benchmarks: List[Benchmark]
    ) -> ComparisonReport:
        """
        Compare multiple systems across multiple benchmarks.
        """
        comparison = {}
        
        for benchmark in benchmarks:
            comparison[benchmark.name] = {}
            
            for system in systems:
                report = await self.run_benchmark(system, benchmark)
                comparison[benchmark.name][system.name] = report
        
        return ComparisonReport(comparison)
```

### 5.2 Statistical Significance

```python
def calculate_significance(
    baseline_results: List[float],
    treatment_results: List[float]
) -> SignificanceResult:
    """
    Calculate statistical significance of difference.
    """
    # Paired t-test for repeated measures
    t_stat, p_value = stats.ttest_rel(baseline_results, treatment_results)
    
    # Effect size (Cohen's d)
    cohens_d = (
        mean(treatment_results) - mean(baseline_results)
    ) / pooled_std(baseline_results, treatment_results)
    
    # Power analysis
    power = calculate_power(
        effect_size=cohens_d,
        n=len(baseline_results),
        alpha=0.05
    )
    
    return SignificanceResult(
        p_value=p_value,
        significant=p_value < 0.05,
        effect_size=cohens_d,
        effect_interpretation=interpret_cohens_d(cohens_d),
        power=power
    )
```

### 5.3 Reporting Format

```json
{
  "benchmark": "DMR",
  "date": "2025-03-11",
  "systems": {
    "PerfectRecall": {
      "recall@5": {
        "mean": 0.82,
        "std": 0.04,
        "ci95": [0.78, 0.86],
        "n": 3
      },
      "cost_per_query": {
        "mean": 0.003,
        "currency": "USD"
      },
      "latency_p95": {
        "mean": 245,
        "unit": "ms"
      }
    },
    "FullContext": {
      "recall@5": {
        "mean": 0.78,
        "std": 0.03,
        "ci95": [0.75, 0.81]
      },
      "cost_per_query": {
        "mean": 0.012,
        "currency": "USD"
      }
    },
    "MemGPT": {
      "recall@5": {
        "mean": 0.75,
        "std": 0.05,
        "ci95": [0.70, 0.80]
      },
      "cost_per_query": {
        "mean": 0.005,
        "currency": "USD"
      }
    }
  },
  "significance_tests": {
    "PerfectRecall_vs_FullContext": {
      "p_value": 0.08,
      "significant": false,
      "effect_size": 0.5,
      "interpretation": "Medium effect, not statistically significant"
    }
  }
}
```

---

## 6. Cost Analysis

### 6.1 Cost Model

```python
@dataclass
class CostModel:
    """
    Breakdown of costs per query.
    """
    
    # Embedding costs
    embedding_tokens: int
    embedding_cost_per_1k: float
    
    # LLM costs
    input_tokens: int
    output_tokens: int
    llm_cost_per_1k_input: float
    llm_cost_per_1k_output: float
    
    # Storage costs
    storage_gb: float
    storage_cost_per_gb_month: float
    
    # Compute costs (if self-hosted)
    compute_hours: float
    compute_cost_per_hour: float
    
    def total_per_query(self) -> float:
        embedding = (self.embedding_tokens / 1000) * self.embedding_cost_per_1k
        llm_input = (self.input_tokens / 1000) * self.llm_cost_per_1k_input
        llm_output = (self.output_tokens / 1000) * self.llm_cost_per_1k_output
        
        # Amortize storage/compute over queries
        storage = (self.storage_gb * self.storage_cost_per_gb_month) / (30 * 10000)  # Assume 10k queries/day
        compute = (self.compute_hours * self.compute_cost_per_hour) / 10000
        
        return embedding + llm_input + llm_output + storage + compute
```

### 6.2 Cost Comparison (Estimated)

| System | Embedding | LLM Input | LLM Output | Storage | Compute | Total |
|--------|-----------|-----------|------------|---------|---------|-------|
| Full-Context | $0 | $0.03 | $0.06 | $0 | $0 | **$0.09** |
| MemGPT | $0.0001 | $0.02 | $0.04 | $0.00001 | $0.0001 | **$0.060** |
| Mem0 | $0.0001 | $0.01 | $0.03 | $0.00001 | $0.0001 | **$0.040** |
| PerfectRecall | $0.0001 | $0.008 | $0.02 | $0.00001 | $0.0001 | **$0.028** |

*Assumptions: GPT-4 pricing (as of 2025), 4K input, 1K output, 100 memories retrieved. Actual costs may vary.*

### 6.3 Break-Even Analysis

```
At what point does memory system cost less than full-context?

Break-even = (Memory System Fixed Cost) / (Full-Context Variable Cost - Memory Variable Cost)

Example (hypothetical):
- Full-context: $0.09/query
- PerfectRecall: $0.028/query + $0.10 setup

Break-even = $0.10 / ($0.09 - $0.028) = 1.6 queries

→ PerfectRecall may be cheaper after 2 queries (theoretical)
```

**Important:** These are theoretical estimates. Actual break-even depends on:
- Specific usage patterns
- Token volumes
- Chosen models and their pricing
- Storage duration

---

## 7. When Memory Wins vs. When It Doesn't

### 7.1 Memory Likely Helps

| Scenario | Why Memory Helps | Expected Impact |
|----------|------------------|-----------------|
| **Long conversations** (> 50 turns) | Context window limits require summarization, which loses information | Significant recall improvement expected |
| **Multi-session tasks** | Full-context systems lose all context between sessions | Major continuity improvement expected |
| **Fact-heavy domains** | Semantic retrieval can outperform linear context scan | Moderate accuracy improvement expected |
| **Repeated queries** | Caching and learned retrieval patterns reduce redundant computation | Cost reduction expected |
| **Temporal reasoning** | Bi-temporal models explicitly track fact validity periods | Better handling of changing facts expected |

**Note:** "Expected" impacts are hypotheses to be validated through benchmarks, not guaranteed improvements.

### 7.2 Memory May Not Help

| Scenario | Why Memory Doesn't Help | Recommendation |
|----------|------------------------|----------------|
| **Short conversations** (< 10 turns) | Full-context is sufficient | Don't use memory - adds overhead |
| **Single-turn Q&A** | No history to remember | Don't use memory |
| **Real-time streaming** | Latency requirements may not allow retrieval | Simplified or no memory |
| **Highly unstructured data** | No patterns to learn or retrieve | Full-context may be better |
| **Security-critical applications** | Need audit trails, not probabilistic recall | Hybrid approach with logging |

### 7.3 Decision Matrix

```
                    Low History   High History
                    ──────────────────────────
Low Complexity    │ Full-Context │  Mem0-style  │
                  │   (cheap)    │  (facts only)│
                  ├──────────────┼──────────────┤
High Complexity   │ MemGPT-style │ PerfectRecall│
                  │ (agent mgmt) │  (full power)│
                  └──────────────┴──────────────┘
```

---

## Summary

**Benchmark Checklist:**

- [ ] Implement DMR benchmark
- [ ] Implement LongMemEval benchmark
- [ ] Implement LoCoMo benchmark
- [ ] Create custom PerfectRecall benchmark
- [ ] Run against all baselines
- [ ] Calculate cost per query for each
- [ ] Measure latency distributions
- [ ] Test statistical significance
- [ ] Document failure modes
- [ ] Publish reproducible results

**Success Criteria (to be validated):**

| Metric | Target | Priority |
|--------|--------|----------|
| DMR Recall@5 | > 80% | Critical |
| Cost reduction | > 40% vs full-context | Critical |
| Continuity score | > 70% | Critical |
| Latency p95 | < 300ms | High |
| Abstention accuracy | > 75% | High |
| Conflict detection | > 70% | Medium |

**References:**

1. Packer, C., et al. (2023). "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560.
2. Rasmussen, P., et al. (2025). "Zep: A Temporal Knowledge Graph Architecture for Agent Memory."
3. Zhang, T., et al. (2024). "LongMemEval: Long-Context Evaluation."
4. Sumers, T., et al. (2023). "Cognitive Architectures for Language Agents." arXiv:2309.02427.
