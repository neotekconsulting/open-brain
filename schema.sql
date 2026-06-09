-- Open Brain: Initial Postgres Schema
-- Vector provider: nomic-embed-text (768 dimensions)
-- Requires: pgvector extension

CREATE EXTENSION IF NOT EXISTS vector;

-- Raw thoughts captured from any source
CREATE TABLE thoughts (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_text    TEXT NOT NULL,
    thought_type TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source      TEXT,
    metadata    JSONB,
    evidence_basis TEXT,
    visibility_verified_by_human_at TIMESTAMPTZ
);

-- Embeddings tied to a thought for retrieval / similarity search
CREATE TABLE embeddings (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thought_id  UUID NOT NULL REFERENCES thoughts(id) ON DELETE CASCADE,
    vector      VECTOR(768) NOT NULL,
    model       VARCHAR NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index on embeddings for cosine similarity
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON embeddings
    USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
