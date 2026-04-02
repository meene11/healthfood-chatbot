-- ============================================================
-- Supabase pgvector 테이블 설정
-- Supabase SQL Editor에서 실행하세요
-- ============================================================

-- 1. pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 문서 청크 저장 테이블
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source_file   TEXT,
    category      TEXT,
    description   TEXT,
    page_number   INTEGER,
    chunk_index   INTEGER,
    content       TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(384),          -- multilingual-e5-small (무료 로컬 모델, 384차원)
    metadata      JSONB DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 3. 벡터 유사도 검색 인덱스 (HNSW — 속도 최적화)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4. 카테고리 필터링용 인덱스
CREATE INDEX IF NOT EXISTS documents_category_idx
    ON documents (category);

-- 5. 코사인 유사도 기반 검색 함수
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.5,
    match_count     INT   DEFAULT 5,
    filter_category TEXT  DEFAULT NULL
)
RETURNS TABLE (
    id          BIGINT,
    source_file TEXT,
    category    TEXT,
    page_number INTEGER,
    content     TEXT,
    metadata    JSONB,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.source_file,
        d.category,
        d.page_number,
        d.content,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE
        d.embedding IS NOT NULL
        AND (filter_category IS NULL OR d.category = filter_category)
        AND 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 6. 전체 텍스트 검색용 인덱스 (한국어 지원)
CREATE INDEX IF NOT EXISTS documents_content_fts_idx
    ON documents
    USING gin (to_tsvector('simple', content));
