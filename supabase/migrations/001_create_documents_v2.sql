-- =====================================================
-- 실험 5: documents_v2 테이블 + RPC 함수 생성
-- 목적: bge-m3 임베딩(1024차원) 저장용 새 테이블
-- 기존 documents(384차원) 테이블은 유지 (롤백 가능)
--
-- 실행 방법:
--   Supabase 대시보드 → SQL Editor → 이 파일 전체 복붙 → Run
-- =====================================================


-- ── 1. pgvector 확장 확인 (이미 설치되어 있으면 무시) ──────────────────
CREATE EXTENSION IF NOT EXISTS vector;


-- ── 2. documents_v2 테이블 생성 ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents_v2 (
    id          BIGSERIAL PRIMARY KEY,
    source_file TEXT,
    category    TEXT,
    description TEXT,
    page_number INT,
    chunk_index INT,
    content     TEXT,
    token_count INT,
    embedding   vector(1024),     -- bge-m3: 1024차원 (기존은 384)
    metadata    JSONB,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);


-- ── 3. 벡터 인덱스 생성 (코사인 유사도 기준) ───────────────────────────
-- ivfflat: 근사 최근접 이웃 검색 (빠르지만 약간 부정확)
-- lists 파라미터: 데이터 수 / 1000 권장 (나중에 데이터 크기 보고 조정)
CREATE INDEX IF NOT EXISTS documents_v2_embedding_idx
ON documents_v2
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- 전문 검색 인덱스 (BM25 키워드 검색용)
CREATE INDEX IF NOT EXISTS documents_v2_content_fts
ON documents_v2
USING gin(to_tsvector('simple', content));


-- ── 4. 벡터 유사도 검색 함수 (단순 버전, 폴백용) ───────────────────────
CREATE OR REPLACE FUNCTION match_documents_v2(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.3,
    match_count     int   DEFAULT 5,
    filter_category text  DEFAULT NULL
)
RETURNS TABLE (
    id          bigint,
    source_file text,
    category    text,
    content     text,
    token_count int,
    metadata    jsonb,
    similarity  float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        d.id,
        d.source_file,
        d.category,
        d.content,
        d.token_count,
        d.metadata,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents_v2 d
    WHERE
        (filter_category IS NULL OR d.category = filter_category)
        AND 1 - (d.embedding <=> query_embedding) > match_threshold
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- ── 5. 하이브리드 검색 함수 (벡터 + 키워드 결합) ──────────────────────
-- 기존 hybrid_search 와 동일한 인터페이스, documents_v2 테이블 사용
CREATE OR REPLACE FUNCTION hybrid_search_v2(
    query_embedding vector(1024),
    query_text      text,
    match_count     int   DEFAULT 5,
    vector_weight   float DEFAULT 0.7,
    text_weight     float DEFAULT 0.3,
    filter_category text  DEFAULT NULL
)
RETURNS TABLE (
    id             bigint,
    source_file    text,
    category       text,
    content        text,
    token_count    int,
    metadata       jsonb,
    vector_score   float,
    text_score     float,
    combined_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        -- 벡터 유사도 검색
        SELECT
            d.id,
            d.source_file,
            d.category,
            d.content,
            d.token_count,
            d.metadata,
            1 - (d.embedding <=> query_embedding) AS vscore
        FROM documents_v2 d
        WHERE filter_category IS NULL OR d.category = filter_category
        ORDER BY d.embedding <=> query_embedding
        LIMIT match_count * 3
    ),
    text_results AS (
        -- 전문 검색 (BM25)
        SELECT
            d.id,
            ts_rank(
                to_tsvector('simple', d.content),
                plainto_tsquery('simple', query_text)
            ) AS tscore
        FROM documents_v2 d
        WHERE
            (filter_category IS NULL OR d.category = filter_category)
            AND to_tsvector('simple', d.content) @@ plainto_tsquery('simple', query_text)
        LIMIT match_count * 3
    )
    SELECT
        vr.id,
        vr.source_file,
        vr.category,
        vr.content,
        vr.token_count,
        vr.metadata,
        vr.vscore                                           AS vector_score,
        COALESCE(tr.tscore, 0)                             AS text_score,
        (vector_weight * vr.vscore
         + text_weight * COALESCE(tr.tscore, 0))           AS combined_score
    FROM vector_results vr
    LEFT JOIN text_results tr ON vr.id = tr.id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;


-- ── 6. 확인 쿼리 (실행 후 결과 확인용) ────────────────────────────────
-- 아래 쿼리로 테이블과 함수가 정상 생성됐는지 확인하세요.

-- 테이블 확인
-- SELECT COUNT(*) FROM documents_v2;

-- 함수 확인
-- SELECT routine_name FROM information_schema.routines
-- WHERE routine_name IN ('match_documents_v2', 'hybrid_search_v2');
