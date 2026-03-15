# Memory System Configuration

## Storage Tiers

### Auto-store (confidence >= 0.8)
- Preferences ("I prefer X", "I like Y")
- Decisions ("I decided to...", "Let's go with...")
- Explicit markers ("Remember this", "Important:")
- Personal info (name, work, location)
- Goals/priorities

### Flag for review (confidence 0.5-0.8)
- Habits ("usually", "sometimes")
- Opinions ("I think", "I believe")
- Soft preferences

### Skip (confidence < 0.5)
- Questions
- Ephemeral chat
- Ambiguous statements

## Retrieval Behavior

- Always retrieve before responding
- Use top 3 memories in context
- Abstain if max score < 0.3
- Format as: "[Relevant context from memory:]"

## Review Workflow

Flagged memories are stored with `review_status='pending'`.
Review via CLI or periodically check logs.

## Integration Pattern

```python
from perfect_recall.memory_integration import get_memory_integration

mem = await get_memory_integration()

# Before responding
context = await mem.get_context(user_message)
memories_text = mem.format_context_for_prompt(context)
# ... add to system prompt ...

# After responding
result = await mem.process_turn(user_message, assistant_response)
# result['stored'] - what was auto-saved
# result['flagged'] - what needs review
```

## Analytics

Check `logs/memory-YYYY-MM-DD.jsonl` for:
- Retrieval hit rate (should be >70%)
- Avg confidence scores
- Facts stored per turn
- Flagged items pending review
