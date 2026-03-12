-- ============================================================================
-- Migration: Add Superpowers-inspired metadata fields to memory_nodes
-- ============================================================================
-- This migration adds triggers, symptoms, aliases, and anti_triggers fields
-- for context-aware memory retrieval based on the Superpowers framework.
--
-- Run: psql -d your_database -f migrate_superpowers_metadata.sql
-- ============================================================================

-- Check if columns exist before adding (idempotent migration)
DO $$
BEGIN
    -- Add triggers column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'memory_nodes' AND column_name = 'triggers'
    ) THEN
        ALTER TABLE memory_nodes ADD COLUMN triggers TEXT[] DEFAULT '{}';
        RAISE NOTICE 'Added triggers column';
    ELSE
        RAISE NOTICE 'triggers column already exists';
    END IF;

    -- Add symptoms column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'memory_nodes' AND column_name = 'symptoms'
    ) THEN
        ALTER TABLE memory_nodes ADD COLUMN symptoms TEXT[] DEFAULT '{}';
        RAISE NOTICE 'Added symptoms column';
    ELSE
        RAISE NOTICE 'symptoms column already exists';
    END IF;

    -- Add aliases column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'memory_nodes' AND column_name = 'aliases'
    ) THEN
        ALTER TABLE memory_nodes ADD COLUMN aliases TEXT[] DEFAULT '{}';
        RAISE NOTICE 'Added aliases column';
    ELSE
        RAISE NOTICE 'aliases column already exists';
    END IF;

    -- Add anti_triggers column
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'memory_nodes' AND column_name = 'anti_triggers'
    ) THEN
        ALTER TABLE memory_nodes ADD COLUMN anti_triggers TEXT[] DEFAULT '{}';
        RAISE NOTICE 'Added anti_triggers column';
    ELSE
        RAISE NOTICE 'anti_triggers column already exists';
    END IF;
END $$;

-- ============================================================================
-- Create indexes for Superpowers metadata fields
-- ============================================================================

-- Index on triggers for fast trigger-based retrieval
CREATE INDEX IF NOT EXISTS idx_memory_triggers ON memory_nodes USING GIN (triggers);

-- Index on symptoms for error pattern matching
CREATE INDEX IF NOT EXISTS idx_memory_symptoms ON memory_nodes USING GIN (symptoms);

-- Index on aliases for synonym matching
CREATE INDEX IF NOT EXISTS idx_memory_aliases ON memory_nodes USING GIN (aliases);

-- Index on anti_triggers for conflict detection
CREATE INDEX IF NOT EXISTS idx_memory_anti_triggers ON memory_nodes USING GIN (anti_triggers);

-- ============================================================================
-- Verify migration
-- ============================================================================
SELECT 
    column_name,
    data_type,
    column_default
FROM information_schema.columns 
WHERE table_name = 'memory_nodes' 
    AND column_name IN ('triggers', 'symptoms', 'aliases', 'anti_triggers')
ORDER BY column_name;
