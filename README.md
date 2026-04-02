# 건강식품·다이어트 RAG 챗봇

건강식품 및 다이어트 관련 질문에 답변하는 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다.

## 주요 기능

- **Parent-Child Retrieval** — 자식 청크로 정밀 검색 후 부모 청크의 넓은 컨텍스트로 답변 생성
- **Hybrid Search** — 벡터 유사도(70%) + 키워드 BM25(30%) 결합 검색
- **Query Rewriting + Multi-Query** — API 1회 호출로 3가지 관점의 검색 쿼리 자동 생성
- **Reranking** — 검색 점수 + 한국어 키워드 매칭 보너스로 최종 순위 결정
- **메타데이터 필터링** — 푸드올로지 브랜드 쿼리 자동 감지 및 카테고리 필터 적용
- **Streaming 답변** — 실시간 토큰 스트리밍 출력
- **할루시네이션 방지** — 보유 자료 외 내용 생성 차단

## 시스템 아키텍처

```
질문 입력
  │
  ▼
[Query Rewrite + Multi-Query]  ← GPT-4o-mini (1회 호출)
  │  원본 + 3가지 관점 쿼리 생성
  ▼
[Hybrid Search — 병렬]         ← Supabase pgvector
  │  벡터(HNSW) + 키워드(FTS) 결합
  │  카테고리 메타데이터 필터링
  ▼
[Parent-Child 조인]
  │  child_chunks 검색 → parent_chunks 컨텍스트 반환
  ▼
[Reranking]                    ← 로컬 (무료)
  │  결합 점수 기반 TOP-3 선별
  ▼
[LLM 답변 생성]                ← GPT-4o-mini (스트리밍)
  │  출처 [자료 N] 표시
  ▼
답변 출력
```

## 기술 스택

| 구분 | 기술 |
|------|------|
| 임베딩 모델 | `intfloat/multilingual-e5-small` (로컬, 무료) |
| 리랭킹 모델 | `cross-encoder/ms-marco-MiniLM-L-6-v2` (로컬, 무료) |
| LLM | GPT-4o-mini (질문 1회 약 0.1원) |
| 벡터 DB | Supabase pgvector |
| 검색 방식 | Hybrid (HNSW 벡터 + PostgreSQL FTS) |
| 프레임워크 | LangChain LCEL, sentence-transformers, OpenAI SDK |

## 데이터 카테고리

| 카테고리 | 설명 |
|----------|------|
| `건강식품_논문` | 프로바이오틱스, 오메가3, 커큐민 등 임상 논문 |
| `다이어트_논문` | 간헐적 단식, 케토제닉, 저탄수화물 식단 논문 |
| `다이어트영양소_논문` | 단백질, 식이섬유, 폴리페놀 등 영양소 논문 |
| `푸드올로지` | 푸드올로지 제품 정보, 성분, 기업 정보 |
| `네이버블로그` | 건강식품·다이어트 블로그 아티클 |
| `건강_수집데이터` | 직접 작성한 건강식품 가이드 텍스트 |

## 설치 및 실행

### 사전 요구사항

- Python 3.10+
- Supabase 계정 (pgvector 활성화)
- OpenAI API 키

### 1. 의존성 설치

```bash
pip install -r requirements.txt
pip install openai langchain-core
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성합니다:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
```

### 3. Supabase DB 설정

Supabase SQL Editor에서 실행:

```sql
-- src/setup_supabase_v2.sql 내용 실행
```

### 4. 데이터 파싱 및 업로드

```bash
# 전체 데이터 파싱 + Supabase 업로드
python src/parse_and_upload_v2.py
```

### 5. 챗봇 실행

```bash
python src/chatbot_v2.py
```

## 프로젝트 구조

```
healthfood_chatbot/
├── src/
│   ├── chatbot_v2.py           # 메인 챗봇 (RAG 파이프라인)
│   ├── chatbot.py              # 챗봇 v1 (기본 버전)
│   ├── parse_and_upload_v2.py  # 데이터 파싱 + Supabase 업로드
│   ├── parse_all_data.py       # 전체 데이터 파서
│   ├── parse_pdfs.py           # PDF 파서
│   ├── parse_json_data.py      # JSON 데이터 파서
│   ├── parse_glucose_spike.py  # 혈당 관련 데이터 파서
│   ├── run_pipeline.py         # 파이프라인 실행기
│   ├── upload_to_supabase.py   # Supabase 업로더 v1
│   ├── setup_supabase.sql      # DB 스키마 v1
│   └── setup_supabase_v2.sql   # DB 스키마 v2 (Parent-Child)
├── data/
│   ├── raw/                    # 원본 데이터
│   │   ├── papers/             # 논문 PDF (git 제외)
│   │   ├── foodology/          # 푸드올로지 제품/기업 JSON
│   │   ├── blog/               # 블로그 원본
│   │   └── *.txt               # 건강식품 가이드 텍스트
│   └── generated/              # 파싱 후 생성 파일 (git 제외)
├── crawl_papers.py             # 논문 크롤러
├── crawl_all_papers.py         # 전체 논문 크롤러
├── requirements.txt
└── .env                        # API 키 (git 제외)
```

## Supabase DB 스키마

```
parent_chunks           child_chunks
─────────────           ─────────────
id (PK)         ◄──┐   id (PK)
source_file         │   parent_id (FK) ──┘
category            │   source_file
content             │   category
token_count         │   chunk_index
metadata            │   content
                    │   embedding (VECTOR 384)
                    │   token_count
                    └── metadata
```

**검색 함수:**
- `hybrid_search()` — 벡터 + FTS 결합, 카테고리 필터 지원
- `match_children_return_parents()` — 자식 검색 → 부모 컨텍스트 반환
- `match_documents()` — 기존 단순 벡터 검색 (하위 호환)
