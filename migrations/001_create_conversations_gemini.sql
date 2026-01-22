-- Migration: Create conversations_gemini table
-- Purpose: Store Gemini Flash transcriptions with Lailix-specific feedback
-- This runs parallel to the existing conversations table for quality comparison

CREATE TABLE IF NOT EXISTS conversations_gemini (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Temporal
  started_at TIMESTAMPTZ NOT NULL,
  ended_at TIMESTAMPTZ,
  duration_seconds INTEGER,

  -- Source
  source_audio_path TEXT,
  source_mp3_path TEXT,
  source_tool TEXT DEFAULT 'MeetingRecorder',

  -- Transcript
  transcript_text TEXT,
  transcript_language TEXT,

  -- Standard metadata
  title TEXT,
  summary TEXT,
  key_points TEXT[],
  tags TEXT[],
  participants JSONB,
  sentiment VARCHAR(20),
  meeting_type VARCHAR(50),

  -- Lailix-specific feedback
  lailix_communication_score INTEGER CHECK (lailix_communication_score BETWEEN 1 AND 10),
  lailix_communication_feedback TEXT,
  lailix_sales_score INTEGER CHECK (lailix_sales_score BETWEEN 1 AND 10),
  lailix_sales_feedback TEXT,
  lailix_strategic_alignment TEXT,
  lailix_improvement_areas TEXT[],
  lailix_strengths TEXT[],

  -- Processing metadata
  gemini_model TEXT DEFAULT 'gemini-2.5-flash',
  gemini_input_tokens INTEGER,
  gemini_output_tokens INTEGER,
  processing_time_seconds FLOAT,
  processing_status TEXT DEFAULT 'pending',
  processing_error TEXT,
  gemini_raw_response JSONB,

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_conversations_gemini_started_at ON conversations_gemini(started_at DESC);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_conversations_gemini_status ON conversations_gemini(processing_status);

COMMENT ON TABLE conversations_gemini IS 'Gemini Flash transcriptions with Lailix coaching feedback. Parallel to conversations table for quality comparison.';
