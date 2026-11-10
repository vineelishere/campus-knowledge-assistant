CREATE EXTENSION IF NOT EXISTS vector;

-- access_level: 0 = everyone logged in (student+), 1 = professor+, 2 = administrator only
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('student', 'professor', 'admin')),
    department    TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    title        TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    source_url   TEXT,
    doc_type     TEXT NOT NULL DEFAULT 'policy',      -- policy | syllabus | web | other
    policy_key   TEXT,                                -- same key across versions, e.g. 'prerequisite-policy'
    department   TEXT,
    year         INT NOT NULL,
    access_level INT NOT NULL DEFAULT 0,
    content_hash TEXT UNIQUE NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_policy_key_idx ON documents (policy_key, year DESC);

-- access_level / year / department are denormalised onto chunks so the permission
-- filter runs inside the same index scan as the vector and full-text search.
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    page         INT,
    text         TEXT NOT NULL,
    department   TEXT,
    year         INT NOT NULL,
    access_level INT NOT NULL DEFAULT 0,
    embedding    vector(384) NOT NULL,
    -- 'simple' config: no stemming, so course codes like CS301 or MRU-2026 match exactly
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_access_idx ON chunks (access_level);

CREATE TABLE IF NOT EXISTS query_log (
    id             BIGSERIAL PRIMARY KEY,
    ts             TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id        INT,
    role           TEXT,
    question       TEXT NOT NULL,
    status         TEXT NOT NULL,   -- answered | abstained | stale_only | unsupported | error
    reason         TEXT,
    top_score      REAL,
    latency_ms     INT,
    input_tokens   INT DEFAULT 0,
    output_tokens  INT DEFAULT 0,
    cost_usd       NUMERIC(10,6) DEFAULT 0,
    document_ids   INT[]
);
CREATE INDEX IF NOT EXISTS query_log_ts_idx ON query_log (ts DESC);
