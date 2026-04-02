-- ============================================================
-- Supabase pgvector v2 — Parent-Child Chunking + Hybrid Search
-- 기존 documents 테이블은 유지하고 새 테이블 추가
-- Supabase SQL Editor에서 실행하세요
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ── 1. 부모 청크 테이블 (큰 단위 — LLM 답변 생성용) ─────────────────
CREATE TABLE IF NOT EXISTS parent_chunks (
    id            BIGSERIAL PRIMARY KEY,
    source_file   TEXT NOT NULL,
    category      TEXT,
    page_start    INTEGER,
    page_end      INTEGER,
    content       TEXT NOT NULL,
    token_count   INTEGER,
    metadata      JSONB DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── 2. 자식 청크 테이블 (작은 단위 — 검색용, 임베딩 포함) ────────────
CREATE TABLE IF NOT EXISTS child_chunks (
    id            BIGSERIAL PRIMARY KEY,
    parent_id     BIGINT REFERENCES parent_chunks(id) ON DELETE CASCADE,
    source_file   TEXT NOT NULL,
    category      TEXT,
    page_number   INTEGER,
    chunk_index   INTEGER,
    content       TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(384),   -- multilingual-e5-small
    metadata      JSONB DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ── 3. 인덱스 ───────────────────────────────────────────────────────
-- 벡터 유사도 검색 (HNSW)
CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
    ON child_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 카테고리 필터링
CREATE INDEX IF NOT EXISTS child_chunks_category_idx ON child_chunks (category);
CREATE INDEX IF NOT EXISTS parent_chunks_category_idx ON parent_chunks (category);

-- 부모-자식 조인 최적화
CREATE INDEX IF NOT EXISTS child_chunks_parent_idx ON child_chunks (parent_id);

-- 전체 텍스트 검색 (BM25 대용 — 한국어+영어)
CREATE INDEX IF NOT EXISTS parent_chunks_fts_idx
    ON parent_chunks USING gin (to_tsvector('simple', content));
CREATE INDEX IF NOT EXISTS child_chunks_fts_idx
    ON child_chunks USING gin (to_tsvector('simple', content));

-- ── 4. 벡터 검색 함수 (자식 → 부모 반환) ────────────────────────────
CREATE OR REPLACE FUNCTION match_children_return_parents(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count     INT   DEFAULT 10,
    filter_category TEXT  DEFAULT NULL
)
RETURNS TABLE (
    parent_id      BIGINT,
    parent_content TEXT,
    parent_metadata JSONB,
    child_id       BIGINT,
    child_content  TEXT,
    source_file    TEXT,
    category       TEXT,
    page_number    INTEGER,
    similarity     FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (p.id)
        p.id AS parent_id,
        p.content AS parent_content,
        p.metadata AS parent_metadata,
        c.id AS child_id,
        c.content AS child_content,
        c.source_file,
        c.category,
        c.page_number,
        (1 - (c.embedding <=> query_embedding))::FLOAT AS similarity
    FROM child_chunks c
    JOIN parent_chunks p ON c.parent_id = p.id
    WHERE
        c.embedding IS NOT NULL
        AND (filter_category IS NULL OR c.category = filter_category)
        AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY p.id, (c.embedding <=> query_embedding)
    LIMIT match_count;
END;
$$;

-- ── 5. 하이브리드 검색 함수 (벡터 + 키워드 FTS 결합) ────────────────
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(384),
    query_text      TEXT,
    match_count     INT   DEFAULT 10,
    vector_weight   FLOAT DEFAULT 0.7,   -- 벡터 검색 가중치
    text_weight     FLOAT DEFAULT 0.3,   -- 키워드 검색 가중치
    filter_category TEXT  DEFAULT NULL
)
RETURNS TABLE (
    parent_id      BIGINT,
    parent_content TEXT,
    parent_metadata JSONB,
    source_file    TEXT,
    category       TEXT,
    combined_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT DISTINCT ON (p.id)
            p.id AS pid,
            p.content AS pcontent,
            p.metadata AS pmeta,
            c.source_file AS src,
            c.category AS cat,
            (1 - (c.embedding <=> query_embedding))::FLOAT AS vscore
        FROM child_chunks c
        JOIN parent_chunks p ON c.parent_id = p.id
        WHERE c.embedding IS NOT NULL
            AND (filter_category IS NULL OR c.category = filter_category)
        ORDER BY p.id, c.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    text_results AS (
        SELECT
            p.id AS pid,
            p.content AS pcontent,
            p.metadata AS pmeta,
            p.source_file AS src,
            p.category AS cat,
            ts_rank(to_tsvector('simple', p.content), plainto_tsquery('simple', query_text))::FLOAT AS tscore
        FROM parent_chunks p
        WHERE
            to_tsvector('simple', p.content) @@ plainto_tsquery('simple', query_text)
            AND (filter_category IS NULL OR p.category = filter_category)
        LIMIT match_count * 3
    ),
    combined AS (
        SELECT
            COALESCE(v.pid, t.pid) AS pid,
            COALESCE(v.pcontent, t.pcontent) AS pcontent,
            COALESCE(v.pmeta, t.pmeta) AS pmeta,
            COALESCE(v.src, t.src) AS src,
            COALESCE(v.cat, t.cat) AS cat,
            (COALESCE(v.vscore, 0) * vector_weight +
             COALESCE(t.tscore, 0) * text_weight)::FLOAT AS cscore
        FROM vector_results v
        FULL OUTER JOIN text_results t ON v.pid = t.pid
    )
    SELECT
        c.pid AS parent_id,
        c.pcontent AS parent_content,
        c.pmeta AS parent_metadata,
        c.src AS source_file,
        c.cat AS category,
        c.cscore AS combined_score
    FROM combined c
    ORDER BY c.cscore DESC
    LIMIT match_count;
END;
$$;

-- ── 6. 기존 match_documents도 유지 (하위 호환) ──────────────────────
-- 이미 존재하면 스킵
