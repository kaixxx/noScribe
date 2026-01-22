-- Migration: Create conversation_chunks_gemini table
-- Purpose: Store text chunks with embeddings for Gemini transcriptions
-- Enables semantic search over Gemini-processed meeting transcripts

CREATE TABLE IF NOT EXISTS conversation_chunks_gemini (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES conversations_gemini(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  token_count INTEGER,
  embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookup by conversation
CREATE INDEX IF NOT EXISTS idx_chunks_gemini_conversation_id ON conversation_chunks_gemini(conversation_id);

-- Index for vector similarity search (using HNSW for faster queries)
CREATE INDEX IF NOT EXISTS idx_chunks_gemini_embedding ON conversation_chunks_gemini
  USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE conversation_chunks_gemini IS 'Text chunks with embeddings for semantic search over Gemini transcriptions.';
