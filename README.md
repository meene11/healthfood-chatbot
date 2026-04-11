---
title: 건강식품 RAG 챗봇
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# 건강식품·다이어트 RAG 챗봇

[![HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/meene1212/healthfood-chatbot)

> **라이브 데모**: [https://huggingface.co/spaces/meene1212/healthfood-chatbot](https://huggingface.co/spaces/meene1212/healthfood-chatbot)

건강식품 및 다이어트 관련 논문·자료 기반의 RAG(Retrieval-Augmented Generation) 챗봇입니다.  
18,682개 청크로 구성된 Supabase pgvector DB에서 하이브리드 검색 + 리랭킹으로 정확한 근거를 찾아 답변합니다.

---

## 주요 기능

- **Hybrid Search** — 벡터 유사도(BGE-M3, 70%) + 키워드 BM25(30%) 결합 검색
- **BGE Reranking** — Cross-Encoder(bge-reranker-v2-m3)로 Top5 → Top3 정밀 선별
- **HyDE 지원** — 가상 문서 임베딩으로 한국어 질문 ↔ 영어 논문 매칭 개선 (실험 10)
- **메타데이터 필터링** — 푸드올로지 브랜드 쿼리 자동 감지 및 카테고리 필터 적용
- **Streaming 답변** — 실시간 토큰 스트리밍 출력
- **할루시네이션 방지** — 보유 자료 외 내용 생성 차단

---

## 시스템 아키텍처

```
사용자 질문 (한국어)
  │
  ▼
[BGE-M3 임베딩]              ← BAAI/bge-m3 (로컬, 1024-dim)
  │  (HyDE 적용 시: LLM 가상 문서 생성 후 임베딩)
  ▼
[Hybrid Search]              ← Supabase pgvector (hybrid_search_v2)
  │  벡터(HNSW 70%) + BM25 키워드(30%)
  │  카테고리 메타데이터 필터링
  │  Top-5 반환
  ▼
[BGE Reranker]               ← BAAI/bge-reranker-v2-m3 (로컬)
  │  Top-5 → Top-3 정밀 선별
  ▼
[LLM 답변 생성]              ← GPT-4o-mini (스트리밍)
  │  참고 자료 [자료 N] 출처 표시
  ▼
답변 출력
```

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 임베딩 모델 | `BAAI/bge-m3` (로컬, 1024-dim, 다국어) |
| 리랭킹 모델 | `BAAI/bge-reranker-v2-m3` (로컬, Cross-Encoder) |
| LLM | GPT-4o-mini |
| 벡터 DB | Supabase pgvector (documents_v2, 18,682청크) |
| 검색 방식 | Hybrid (HNSW 벡터 + PostgreSQL FTS) |
| 웹 UI | Streamlit (app.py) |
| 배포 | [HuggingFace Spaces](https://huggingface.co/spaces/meene1212/healthfood-chatbot) |

---

## 데이터 현황

| 카테고리 | 설명 |
|----------|------|
| `건강식품_논문` | 프로바이오틱스, 오메가3, 커큐민 등 임상 논문 |
| `다이어트_논문` | 간헐적 단식, 케토제닉, 저탄수화물 식단 논문 |
| `푸드올로지` | 푸드올로지 제품 정보, 성분, 기업 정보 |
| `건강기사` | 건강식품·다이어트 관련 아티클 |
| `기타` | 직접 작성 건강식품 가이드 텍스트 |

- **총 청크 수**: 18,682개
- **청크 크기**: 200 토큰
- **벡터 차원**: 1,024-dim (BGE-M3)

---

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (.env)

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key        # Llama 사용 시
GOOGLE_API_KEY=your_google_api_key    # Gemini 사용 시
```

### 3. Streamlit 웹 UI 실행

```bash
streamlit run app.py
```

### 4. 터미널 챗봇 실행

```bash
python src/chatbot_v2.py
```

---

## 프로젝트 구조

```
healthfood_chatbot/
├── app.py                          # Streamlit 웹 UI (메인)
├── src/
│   ├── chatbot_v2.py               # 터미널 챗봇 (RAG 파이프라인)
│   ├── parse_and_upload_v2.py      # 데이터 파싱 + Supabase 업로드
│   ├── evaluate_exp9_internal_qa.py  # 실험 9: 내부 QA 모델 비교
│   ├── evaluate_exp10_hyde.py        # 실험 10: HyDE 효과 검증
│   └── evaluate_teammate_qa.py       # 실험 8: 조원 QA 모델 비교
├── data/
│   ├── raw/                        # 원본 데이터 (논문 PDF 등, git 제외)
│   └── generated/
│       └── qa_dataset.json         # 내부 평가용 QA 42문항
├── results/
│   └── model_comparison/           # 실험 결과 및 차트
│       ├── 실험8_모델비교_종합보고서.md
│       ├── 실험9_내부QA_비교보고서.md
│       └── 실험10_HyDE_비교보고서.md
├── requirements.txt
└── .env                            # API 키 (git 제외)
```

---

## 평가 실험 결과 요약

### 실험 8, 9, 10 — 3개 모델 비교 (GPT-4o-mini / Llama 3.3 70B / Gemini Flash Lite)

| 실험 | 데이터 | 평가 방법 | 주요 발견 |
|------|--------|---------|---------|
| 실험 8 | 조원 QA 62문항 | LLM-as-Judge | Llama 1.629로 1위, GPT 완전정답률 최고 |
| 실험 9 | 내부 QA 42문항 | Hit@K + LLM-as-Judge | Gemini 1.262로 1위, 검색 Hit@5 = 9.5% |
| 실험 10 | 내부 QA 42문항 | HyDE 적용 비교 | GPT 품질 +0.214 향상, Llama Hit@5 2배 향상 |

### 3회 실험 통합 답변 품질 순위

| 모델 | 실험8 | 실험9 | 실험10(HyDE) | 평균 | API 비용 |
|------|:----:|:----:|:----------:|:----:|:------:|
| **Gemini Flash Lite** | 1.468 | 1.262 | **1.381** | **1.370** | 무료 |
| GPT-4o-mini | 1.581 | 1.048 | 1.262 | 1.297 | 유료 |
| Llama 3.3 70B | **1.629** | 0.595 | 0.619 | 0.948 | 무료 |

> **Gemini Flash Lite** 가 3회 실험 평균 품질 1.370으로 최고.  
> **GPT-4o-mini + HyDE** 조합이 속도(7.6초)와 품질(1.262)의 균형이 가장 좋음.  
> **Llama 3.3 70B** 는 속도(5.1초)와 무료 장점이 있으나 내부 DB 기반 질문에 취약.

---

## Supabase DB 스키마

```sql
-- 메인 청크 테이블
documents_v2 (
  id            BIGSERIAL PRIMARY KEY,
  source_file   TEXT,
  category      TEXT,
  content       TEXT,
  embedding     VECTOR(1024),     -- BGE-M3 1024-dim
  token_count   INTEGER,
  metadata      JSONB
)

-- 검색 함수
hybrid_search_v2(
  query_embedding  VECTOR,   -- BGE-M3 임베딩
  query_text       TEXT,     -- BM25 키워드
  match_count      INT,      -- Top-K
  vector_weight    FLOAT,    -- 0.7
  text_weight      FLOAT,    -- 0.3
  filter_category  TEXT      -- 카테고리 필터 (NULL=전체)
)
```

---

## 메타데이터 필터링

질문에 푸드올로지 브랜드 키워드 포함 시 해당 카테고리로 검색 범위 자동 좁힘.  
결과 없으면 전체 카테고리 검색으로 자동 폴백.

| 감지 키워드 | 적용 카테고리 |
|-------------|---------------|
| `푸드올로지`, `콜레올로지`, `맨올로지`, `톡스올로지`, `버닝올로지` | `푸드올로지` |
