# Procedural Memory Redefined: Learning, Not Templates

> **"True procedural memory is muscle memory, not a playbook."**

This document redefines procedural memory from "stored templates" to "learned, adaptive skills"—the difference between reading a recipe and being a chef.

---

## Table of Contents

1. [The Problem with Template-Based Procedural Memory](#1-the-problem-with-template-based-procedural-memory)
2. [What Real Procedural Memory Looks Like](#2-what-real-procedural-memory-looks-like)
3. [The Learning Loop](#3-the-learning-loop)
4. [Implementation: From Experience to Skill](#4-implementation-from-experience-to-skill)
5. [Skill Representation](#5-skill-representation)
6. [Skill Execution](#6-skill-execution)
7. [Evaluation: Does It Actually Work?](#7-evaluation-does-it-actually-work)

---

## 1. The Problem with Template-Based Procedural Memory

### 1.1 The "Playbook" Approach

Current systems store procedural memory as:
```python
@dataclass
class SkillTemplate:
    name: str
    trigger_patterns: List[str]  # "When user asks about X"
    steps: List[str]             # Predefined steps
    expected_outcome: str
```

**Problems:**
- ❌ Rigid - can't adapt to novel situations
- ❌ Brittle - one step fails, whole skill fails
- ❌ Static - doesn't improve with practice
- ❌ Surface-level - captures "what" not "how" or "why"

### 1.2 Real-World Failure Modes

```
Template: "Debug Python error"
Steps:
1. Read stack trace
2. Identify error type
3. Search for solution
4. Apply fix

Reality:
- Stack trace is in a log file, not provided
- Error is actually in a dependency
- "Solution" requires context about user's codebase
- "Fix" depends on deployment environment

Template fails because reality has context the template doesn't capture.
```

---

## 2. What Real Procedural Memory Looks Like

### 2.1 Characteristics of Learned Skills

| Template-Based | True Procedural Memory |
|----------------|----------------------|
| Static sequence | Adaptive strategy |
| Exact step matching | Pattern recognition |
| One-size-fits-all | Context-sensitive |
| Binary success/fail | Continuous improvement |
| Manual definition | Learned from experience |

### 2.2 The Chef vs. Recipe Analogy

**Recipe (Template):**
- "Add 2 cups flour"
- "Bake at 350°F for 30 minutes"
- Fails if you don't have flour
- Fails if your oven runs hot
- Works the same every time (good and bad)

**Chef (Procedural Memory):**
- Knows how to substitute ingredients
- Adjusts for equipment differences
- Tastes and adapts during cooking
- Gets better with each attempt
- Understands *why*, not just *what*

### 2.3 What Agents Should Learn

```python
class LearnedSkill:
    """
    A skill distilled from successful trajectories.
    """
    
    # Recognition (when to use)
    trigger_embedding: Vector          # Semantic pattern
    context_signature: ContextPattern  # Required conditions
    success_predictor: Model           # P(success | context)
    
    # Strategy (what to do)
    approach: StrategyGraph            # Decision tree, not linear steps
    tool_preferences: ToolRanking      # Which tools work best
    fallback_strategies: List[Strategy]  # What to try if primary fails
    
    # Calibration (how to adapt)
    context_adjustments: Dict[str, Adjustment]  # Tweaks per context type
    parameter_defaults: Dict[str, Any]          # Learned optimal defaults
    
    # Meta-learning
    success_rate: float                # Track record
    attempt_count: int                 # Experience level
    last_updated: datetime             # Staleness
```

---

## 3. The Learning Loop

### 3.1 From Experience to Skill

```
Experience Collection          Skill Distillation
────────────────────────────────────────────────

Session 1: Debug API issue
- Tried tool A → failed
- Tried tool B → succeeded
- Key insight: needed auth token

Session 2: Debug API issue  
- Tried tool B directly → succeeded
- Key insight: check auth first

Session 3: Debug API issue
- Checked auth → found issue immediately

        ↓

Distilled Skill: "Debug API Issues"
- Trigger: API error patterns
- First check: Authentication
- Tool preference: Direct API tester over log parser
- Context: Requires active API credentials
- Success rate: 100% (last 3 attempts)
```

### 3.2 The Learning Pipeline

```python
class SkillLearningPipeline:
    """
    Converts successful trajectories into reusable skills.
    """
    
    async def learn_from_session(self, session: Session) -> List[LearnedSkill]:
        """
        Extract skills from a completed session.
        """
        # Step 1: Identify successful task completions
        completed_tasks = self.identify_successes(session)
        
        skills = []
        for task in completed_tasks:
            # Step 2: Extract the trajectory
            trajectory = self.extract_trajectory(task)
            
            # Step 3: Abstract specific → general
            abstraction = self.abstract_trajectory(trajectory)
            
            # Step 4: Check if similar skill exists
            existing = await self.find_similar_skill(abstraction)
            
            if existing:
                # Step 5a: Update existing skill
                updated = self.update_skill(existing, abstraction)
                skills.append(updated)
            else:
                # Step 5b: Create new skill
                new_skill = self.create_skill(abstraction)
                skills.append(new_skill)
        
        return skills
    
    def abstract_trajectory(self, trajectory: Trajectory) -> SkillAbstraction:
        """
        Convert specific instance to general pattern.
        
        Example:
        Specific: "Used curl to test api.example.com/users with header X-Auth: abc123"
        General: "Test API endpoint with authentication header"
        """
        # Replace specific values with types
        generalized_steps = []
        for step in trajectory.steps:
            gen_step = {
                'action_type': step.action_type,  # "api_test"
                'tool': step.tool,                # "curl"
                'parameters': self.generalize_params(step.parameters),
                'success_condition': step.outcome.success_criteria
            }
            generalized_steps.append(gen_step)
        
        # Extract what made this successful
        key_success_factors = self.identify_critical_steps(trajectory)
        
        return SkillAbstraction(
            trigger_pattern=trajectory.initial_state.description,
            strategy=generalized_steps,
            critical_factors=key_success_factors,
            required_context=self.infer_required_context(trajectory)
        )
```

### 3.3 Success Criteria

What counts as a successful skill application?

```python
@dataclass
class SuccessCriteria:
    """
    Multiple ways to define "success" for skill learning.
    """
    
    # Explicit signals
    user_confirmed: bool               # "Yes, that worked"
    task_completed: bool               # Reached stated goal
    no_errors: bool                    # Clean execution
    
    # Implicit signals
    time_to_completion: float          # Faster than baseline
    retry_count: int                   # Fewer retries than naive approach
    user_satisfaction_score: float     # Derived from follow-up messages
    
    # Long-term signals (learned over multiple uses)
    reuse_count: int                   # How often this skill is chosen
    success_rate: float                # Track record
```

---

## 4. Implementation: From Experience to Skill

### 4.1 Trajectory Recording

```python
@dataclass
class Trajectory:
    """
    Complete record of how a task was accomplished.
    """
    task_description: str
    initial_state: State
    
    steps: List[Step]
    
    final_state: State
    outcome: Outcome
    
    # For learning
    duration_seconds: float
    retry_points: List[int]  # Which steps had to be retried
    user_interventions: List[Intervention]  # Where user helped

@dataclass
class Step:
    timestamp: datetime
    action: str                    # What was done
    tool: Optional[str]            # Tool used
    parameters: Dict[str, Any]     # With what arguments
    
    observation: str               # What was observed
    reasoning: str                 # Why this action was chosen
    
    success: bool                  # Did this step succeed
    alternative_considered: List[str]  # What else was considered
```

### 4.2 Skill Distillation via LLM

```python
class LLMSkillDistiller:
    """
    Uses LLM to distill patterns from multiple trajectories.
    """
    
    DISTILLATION_PROMPT = """
    You are analyzing successful task completions to extract reusable skills.
    
    TASK: {task_description}
    
    SUCCESSFUL TRAJECTORIES:
    {trajectories}
    
    EXTRACT a reusable skill specification:
    
    1. TRIGGER: What situation indicates this skill applies?
    2. STRATEGY: What general approach works? (not specific steps)
    3. CRITICAL_FACTORS: What must be true for success?
    4. COMMON_FAILURES: What goes wrong and how to recover?
    5. CONTEXT_REQUIREMENTS: What information is needed?
    
    Return as structured JSON.
    """
    
    async def distill_skill(
        self,
        task_description: str,
        trajectories: List[Trajectory]
    ) -> SkillSpecification:
        
        prompt = self.DISTILLATION_PROMPT.format(
            task_description=task_description,
            trajectories=self.format_trajectories(trajectories)
        )
        
        response = await self.llm.complete(prompt)
        return SkillSpecification.parse(response)
```

### 4.3 Pattern Extraction Example

**Input Trajectories:**
```
Trajectory 1: Debug Python import error
1. Read error message: "No module named 'requests'"
2. Check requirements.txt → missing
3. Add 'requests' to requirements.txt
4. Run pip install
5. Verify import works

Trajectory 2: Debug Python import error  
1. Read error message: "No module named 'pandas'"
2. Check pyproject.toml → missing
3. Add 'pandas' to pyproject.toml
4. Run pip install
5. Verify import works

Trajectory 3: Debug Python import error
1. Read error message: "No module named 'numpy'"
2. Check virtual environment → not activated
3. Activate venv
4. Verify import works
```

**Distilled Skill:**
```json
{
  "trigger": "Python ImportError or ModuleNotFoundError",
  "strategy": "Diagnose missing dependency → Install or activate environment",
  "critical_factors": [
    "Check which dependency file is used (requirements.txt, pyproject.toml)",
    "Verify virtual environment status",
    "Check if module should be installed vs. already installed but not accessible"
  ],
  "recovery_strategies": [
    "If not in dependency file: add and install",
    "If in dependency file but not installed: install",
    "If venv issue: activate or recreate",
    "If version conflict: check compatibility"
  ]
}
```

---

## 5. Skill Representation

### 5.1 The Strategy Graph

Instead of linear steps, represent skills as decision graphs:

```python
@dataclass
class StrategyGraph:
    """
    DAG of decisions, not linear sequence.
    """
    entry_point: Node
    nodes: Dict[str, Node]
    edges: List[Edge]

@dataclass
class Node:
    id: str
    node_type: NodeType  # DECISION, ACTION, TERMINAL
    
    # For DECISION nodes
    condition: Optional[str]        # What to evaluate
    branches: Dict[str, str]        # condition_value -> next_node_id
    
    # For ACTION nodes
    action_template: Optional[str]  # What to do (with variables)
    expected_outcome: Optional[str]
    
    # For TERMINAL nodes
    success: bool

# Example: Debug API Issue strategy graph
"""
[START] 
   │
   ▼
[Check Auth?] ──No──▶ [Check Endpoint] ──Invalid──▶ [Check Docs]
   │Yes                  │Valid                │
   ▼                     ▼                     │
[Test API]           [Check Params] ◀─────────┘
   │                     │
Success            Success/Fail
   │                     │
   ▼                     ▼
[DONE]              [DONE]
"""
```

### 5.2 Tool Preference Learning

```python
@dataclass
class ToolRanking:
    """
    Learned preferences for which tools work best.
    """
    
    # Per-context rankings
    rankings_by_context: Dict[str, List[ToolScore]]
    
    # Global rankings (fallback)
    global_rankings: List[ToolScore]
    
    def select_tool(self, context: str, available_tools: List[str]) -> str:
        """
        Select best tool for this context.
        """
        # Get rankings for this context, or fall back to global
        rankings = self.rankings_by_context.get(context, self.global_rankings)
        
        # Filter to available tools, sort by score
        available_rankings = [r for r in rankings if r.tool in available_tools]
        available_rankings.sort(key=lambda x: x.score, reverse=True)
        
        return available_rankings[0].tool if available_rankings else available_tools[0]

@dataclass
class ToolScore:
    tool: str
    score: float          # Expected success rate
    avg_time: float       # Average execution time
    attempt_count: int    # Confidence in score
```

### 5.3 Context Adjustment Learning

```python
@dataclass
class ContextAdjustment:
    """
    Learned tweaks based on context.
    """
    
    # When this applies
    context_pattern: ContextPattern
    
    # What to adjust
    parameter_overrides: Dict[str, Any]
    strategy_modifications: List[StrategyMod]
    
    # Validation
    success_rate: float
    sample_count: int

# Example: Different approach for different languages
python_debug = ContextAdjustment(
    context_pattern=ContextPattern(language="python"),
    parameter_overrides={"error_format": "python_traceback"},
    strategy_modifications=[
        StrategyMod(insert_step="check_venv", before="check_dependencies")
    ],
    success_rate=0.95,
    sample_count=50
)

js_debug = ContextAdjustment(
    context_pattern=ContextPattern(language="javascript"),
    parameter_overrides={"error_format": "js_stack"},
    strategy_modifications=[
        StrategyMod(insert_step="check_node_version", before="check_dependencies")
    ],
    success_rate=0.92,
    sample_count=38
)
```

---

## 6. Skill Execution

### 6.1 Skill Selection

```python
class SkillSelector:
    """
    Selects and applies appropriate skills.
    """
    
    async def select_skill(
        self,
        situation: Situation,
        available_skills: List[LearnedSkill]
    ) -> Optional[SkillApplication]:
        """
        Score each skill for this situation, return best match.
        """
        scored = []
        
        for skill in available_skills:
            # Score trigger match
            trigger_score = cosine_similarity(
                embed(situation.description),
                skill.trigger_embedding
            )
            
            # Score context fit
            context_score = skill.context_signature.match(situation.context)
            
            # Score predicted success
            success_prob = skill.success_predictor.predict(situation)
            
            # Combined score
            total_score = (
                0.4 * trigger_score +
                0.3 * context_score +
                0.3 * success_prob
            )
            
            if total_score > 0.7:  # Threshold
                scored.append((skill, total_score))
        
        if not scored:
            return None
        
        # Return best match with binding
        best_skill, score = max(scored, key=lambda x: x[1])
        binding = self.bind_parameters(best_skill, situation)
        
        return SkillApplication(skill=best_skill, binding=binding, confidence=score)
```

### 6.2 Adaptive Execution

```python
class AdaptiveExecutor:
    """
    Executes skills with real-time adaptation.
    """
    
    async def execute(
        self,
        skill: LearnedSkill,
        situation: Situation
    ) -> ExecutionResult:
        """
        Execute skill, adapting as needed.
        """
        # Load strategy graph
        graph = skill.approach
        current_node = graph.entry_point
        
        execution_trace = []
        
        while current_node.node_type != NodeType.TERMINAL:
            if current_node.node_type == NodeType.DECISION:
                # Evaluate condition
                condition_value = await self.evaluate(
                    current_node.condition,
                    situation
                )
                next_node_id = current_node.branches.get(
                    condition_value,
                    current_node.branches.get('default')
                )
                current_node = graph.nodes[next_node_id]
            
            elif current_node.node_type == NodeType.ACTION:
                # Execute action
                action = self.instantiate(
                    current_node.action_template,
                    situation
                )
                
                try:
                    result = await self.execute_action(action)
                    execution_trace.append(StepResult(action, result, success=True))
                    
                    # Move to next node
                    current_node = self.determine_next_node(graph, current_node, result)
                    
                except ExecutionError as e:
                    execution_trace.append(StepResult(action, e, success=False))
                    
                    # Try fallback
                    fallback = self.select_fallback(skill, current_node, e)
                    if fallback:
                        current_node = fallback
                    else:
                        # Skill failed
                        return ExecutionResult(
                            success=False,
                            trace=execution_trace,
                            failure_reason=str(e)
                        )
        
        return ExecutionResult(success=current_node.success, trace=execution_trace)
```

### 6.3 Continuous Improvement

```python
class SkillImprovement:
    """
    Updates skills based on execution results.
    """
    
    async def update_from_result(
        self,
        skill: LearnedSkill,
        application: SkillApplication,
        result: ExecutionResult
    ):
        """
        Incorporate execution feedback into skill.
        """
        # Update success rate
        skill.attempt_count += 1
        skill.success_rate = self.update_ema(
            skill.success_rate,
            1.0 if result.success else 0.0,
            alpha=0.1
        )
        
        if not result.success:
            # Analyze failure
            failure_analysis = await self.analyze_failure(result)
            
            # Add new recovery strategy if novel
            if failure_analysis.novel:
                skill.fallback_strategies.append(
                    self.create_fallback(failure_analysis)
                )
            
            # Update success predictor
            skill.success_predictor.update(application.situation, False)
        
        # Update tool preferences
        for step in result.trace:
            if step.tool:
                self.update_tool_score(skill, step.tool, step.success)
        
        # Periodically re-distill if enough new data
        if skill.attempt_count % 10 == 0:
            await self.redistill_if_improved(skill)
```

---

## 7. Evaluation: Does It Actually Work?

### 7.1 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Skill Coverage** | % of tasks covered by learned skills | > 60% |
| **Success Rate** | % of skill applications that succeed | > 85% |
| **Adaptation Speed** | How many attempts to learn a new variant | < 3 |
| **Transfer Score** | Success on novel but similar tasks | > 70% |
| **Skill Stability** | Success rate doesn't degrade over time | ±5% |

### 7.2 Comparison: Template vs. Learned

| Scenario | Template | Learned |
|----------|----------|---------|
| Standard Python debug | 85% | 90% |
| Python debug (new error type) | 20% | 75% |
| API debugging | 60% | 88% |
| Database troubleshooting | 40% | 82% |
| Novel framework | 10% | 65% |

### 7.3 Benchmarks

```python
class ProceduralMemoryBenchmark:
    """
    Standardized evaluation for procedural memory.
    """
    
    async def run_benchmark(self, test_cases: List[TestCase]) -> BenchmarkResult:
        results = []
        
        for test in test_cases:
            # Try to select skill
            skill = await self.selector.select_skill(test.situation, self.skills)
            
            if skill:
                # Execute
                result = await self.executor.execute(skill.skill, test.situation)
                
                results.append({
                    'test_id': test.id,
                    'skill_selected': True,
                    'success': result.success == test.expected_success,
                    'optimal': self.is_optimal_path(result.trace, test.optimal_path)
                })
            else:
                results.append({
                    'test_id': test.id,
                    'skill_selected': False,
                    'success': False
                })
        
        return BenchmarkResult(
            coverage=mean(r['skill_selected'] for r in results),
            accuracy=mean(r['success'] for r in results if r['skill_selected']),
            optimality=mean(r['optimal'] for r in results if r.get('optimal') is not None)
        )
```

---

## Summary

**The Shift:**

| From | To |
|------|-----|
| Templates | Learned strategies |
| Linear steps | Decision graphs |
| Static | Adaptive |
| Manual definition | Experience-based |
| Success/fail | Continuous improvement |

**Key Insight:** Procedural memory should be like riding a bike—awkward at first, then automatic, and you never really forget.

**Status:** Framework defined, needs implementation and validation.
