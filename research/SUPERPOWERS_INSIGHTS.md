# Research: Superpowers Framework - Context Engineering Insights

**Source:** https://github.com/obra/superpowers  
**Research Date:** March 12, 2026

## Key Insights for Perfect Recall

### 1. Context Engineering = Skill Discovery

Superpowers uses a **trigger-based skill activation** system that maps perfectly to how Perfect Recall should work:

- **Skills have descriptions** that say "Use when..." (triggering conditions)
- **Agents scan descriptions** to decide which skills to load
- **Context determines activation** - not explicit commands

**Mapping to Perfect Recall:**
| Superpowers | Perfect Recall |
|-------------|----------------|
| Skill description | Memory salience/context tags |
| "Use when..." triggers | Retrieval context matching |
| Agent loads relevant skills | Agent recalls relevant memories |
| YAML frontmatter | Memory metadata + embeddings |

### 2. The CSO (Claude Search Optimization) Principle

> "Future Claude needs to FIND your skill"

Critical discovery: **The description field determines retrieval success.**

**Perfect Recall application:**
- Memories need rich, searchable metadata
- Store "triggering conditions" not just content
- Use keywords, error messages, symptoms
- Structure for semantic similarity matching

### 3. Skill Structure Template

```yaml
---
name: skill-name
description: Use when [specific triggering conditions and symptoms]
---

# Overview
Core principle in 1-2 sentences

## When to Use
- Bullet list with SYMPTOMS
- When NOT to use

## Core Pattern
Before/after or key insight

## Quick Reference
Table for scanning

## Common Mistakes
What goes wrong + fixes
```

**Mapping to Memory Schema:**
- `name` → memory title/summary
- `description` → activation triggers/context
- Content sections → memory content with structure
- Keywords → embedding + metadata tags

### 4. RED-GREEN-REFACTOR for Memory Validation

Superpowers applies TDD to skills:
1. **RED**: Test without skill → watch agent fail
2. **GREEN**: Write skill → watch agent comply
3. **REFACTOR**: Close loopholes

**Perfect Recall application:**
The abstention controller should use similar validation:
- Test what agent recalls without the memory
- Validate that retrieved memory changes behavior correctly
- Track "rationalizations" for ignoring memories

### 5. Rationalization Tables

Superpowers documents how agents rationalize breaking rules:

| Excuse | Reality |
|--------|---------|
| "Too simple to test" | Simple code breaks. Test takes 30 seconds. |
| "I'll test after" | Tests passing immediately prove nothing. |

**Perfect Recall insight:**
The abstention controller should track patterns in when memories are ignored vs. used. This is data for improving retrieval.

### 6. Discovery Workflow

How future Claude finds skills:
1. **Encounters problem** → "tests are flaky"
2. **Searches descriptions** → matches "Use when tests have race conditions..."
3. **Scans overview** → confirms relevance
4. **Reads patterns** → gets quick reference
5. **Loads example** → implements

**Perfect Recall equivalent:**
1. **Agent encounters context** → user asks about auth
2. **Retrieval matches** → "auth" + "errors" in memory tags
3. **Reranking** → confirms relevance with embedding similarity
4. **Context injection** → provides memory summary
5. **Agent uses** → behaves differently

## Implementation Notes

### For Memory Storage
- Add `triggers` field to memory metadata
- Store activation conditions separately from content
- Use keyword extraction for searchability
- Include "when NOT to use" for conflict detection

### For Retrieval
- Two-stage retrieval: (1) broad context match, (2) salience reranking
- Description/summary for quick scanning
- Full content for detailed context injection
- Track access patterns like CSO tracks skill usage

### For Abstention Controller
- Learn from Superpowers' "loophole closing"
- Document rationalization patterns
- Test memories with subagents before trusting them
- Validate that retrieved memories actually change behavior

## The Core Parallel

> **Superpowers**: "Skills are discovered through context, not commanded"  
> **Perfect Recall**: "Memories are recalled through context, not queried"

Both systems rely on the same principle: **the right knowledge surfaces at the right time based on context, not explicit requests.**

This validates the Perfect Recall approach and provides a proven pattern for implementation.
