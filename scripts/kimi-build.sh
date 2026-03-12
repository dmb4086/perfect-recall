#!/bin/bash
# Perfect Recall - Kimi CLI Build Script
# This script is called by the cron job to build Phase 2 features

cd /root/.openclaw/workspace/perfect-recall

# Read current progress
PROGRESS_FILE="/root/.openclaw/workspace/memory/perfect-recall-progress.json"

# Determine next task based on progress file
# Priority order for Phase 2:
# 1. superpowers_inspired_metadata
# 2. write_gate  
# 3. fact_extraction
# 4. conflict_detection
# 5. progressive_disclosure
# 6. adversarial_testing

# For now, start with superpowers metadata
TASK="superpowers_metadata"

echo "Starting build: $TASK"

# Run Kimi CLI with proper working directory
kimi --print --yolo --prompt "
I need you to implement the Superpowers-inspired metadata system for Perfect Recall.

Read the research in /root/.openclaw/workspace/perfect-recall/research/SUPERPOWERS_INSIGHTS.md first.

Then modify the existing code in /root/.openclaw/workspace/perfect-recall/ to add:

1. New fields to the memory schema (schema.sql):
   - triggers (TEXT[]) - When to recall this memory
   - symptoms (TEXT[]) - Error phrases, failure patterns  
   - aliases (TEXT[]) - Synonyms and related terms
   - anti_triggers (TEXT[]) - When NOT to use this memory

2. Update SQLAlchemy models (db/sqlalchemy_models.py) with these fields

3. Update Pydantic models (models/memory.py) with these fields

4. Update MemoryWriter (core/memory_writer.py) to populate these fields
   - Extract triggers from content context
   - Store symptoms when recording errors
   - Add aliases for key terms

5. Update RetrievalPipeline (core/retrieval.py) to use these fields
   - Match query against triggers/symptoms first
   - Check anti_triggers to exclude irrelevant memories
   - Use aliases for synonym matching

Follow existing code patterns. Keep changes minimal and focused.

When done:
1. Run: cd /root/.openclaw/workspace/perfect-recall && git add -A && git commit -m \"feat: Superpowers metadata - triggers, symptoms, aliases, anti_triggers\" && git push
2. Update /root/.openclaw/workspace/memory/perfect-recall-progress.json to mark superpowers_inspired_metadata as done
"

# Check if Kimi succeeded
if [ $? -eq 0 ]; then
    echo "Build completed successfully"
    # Update progress file to mark this task done
    # (Kimi should do this, but backup here)
else
    echo "Build failed - will retry next cycle"
fi
