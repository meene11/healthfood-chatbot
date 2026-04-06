"""
RAG 검색 품질 평가 스크립트 v2
==============================================
실험 1: Baseline — vector 0.7 / keyword 0.3, TOP_K=5, 키워드 보너스 리랭킹
실험 2: 가중치 조정 — vector 0.5 / keyword 0.5, TOP_K=5, 키워드 보너스 리랭킹
실험 3: TOP_K 확대 — vector 0.7 / keyword 0.3, TOP_K=10, 키워드 보너스 리랭킹
실험 4: BGE 리랭커 — vector 0.7 / keyword 0.3, TOP_K=5, BAAI/bge-reranker-v2-m3

평가 지표:
  Hit Rate @1 / @3 / @5  — 상위 K개 결과 안에 관련 문서가 있는지 (0 or 1)
  MRR                    — 첫 번째 관련 문서의 순위 역수 평균
  TOP1 평균 유사도        — combined_score 기준
  종합 등급              — A(우수)/B(양호)/C(보통)/D(미흡)

실행:
  python src/evaluate_v2.py              # 전체 실험
  python src/evaluate_v2.py --exp 5 6 7  # 특정 실험만
"""

import warnings
warnings.filterwarnings("ignore")

import os
import csv
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTENT = 2000
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── 모델 로드 ─────────────────────────────────────────────────────────
print("임베딩 모델 로딩 중...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
bge_reranker: CrossEncoder | None = None   # 실험 4에서 지연 로드
print("완료\n")

# ── 실험 설정 ─────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "id": 1,
        "name": "Exp1_Baseline",
        "label": "실험 1: Baseline",
        "desc": "기존 설정 (vector 0.7 / keyword 0.3, TOP_K=5, 키워드 보너스 리랭킹)",
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "top_k": 5,
        "reranker": "keyword_bonus",
    },
    {
        "id": 2,
        "name": "Exp2_Weight_5050",
        "label": "실험 2: 가중치 50/50",
        "desc": "하이브리드 가중치 변경 (vector 0.5 / keyword 0.5, TOP_K=5)",
        "vector_weight": 0.5,
        "text_weight": 0.5,
        "top_k": 5,
        "reranker": "keyword_bonus",
    },
    {
        "id": 3,
        "name": "Exp3_TopK10",
        "label": "실험 3: TOP_K=10",
        "desc": "검색 범위 확대 (vector 0.7 / keyword 0.3, TOP_K=10)",
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "top_k": 10,
        "reranker": "keyword_bonus",
    },
    {
        "id": 4,
        "name": "Exp4_BGE_Reranker",
        "label": "실험 4: BGE 리랭커",
        "desc": "다국어 크로스인코더 리랭커 (BAAI/bge-reranker-v2-m3, TOP_K=5)",
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "top_k": 5,
        "reranker": "bge-reranker-v2-m3",
        "rpc": "hybrid_search",
    },
    {
        "id": 5,
        "name": "Exp5_BGE_M3_Embedding",
        "label": "실험 5: BGE-M3 임베딩 + 청킹 200토큰",
        "desc": "임베딩 bge-m3(1024차원) + 청크 200토큰 + BGE 리랭커 (vector 0.7 / keyword 0.3)",
        "vector_weight": 0.7,
        "text_weight": 0.3,
        "top_k": 5,
        "reranker": "bge-reranker-v2-m3",
        "rpc": "hybrid_search_v2",
        "embed_model": "BAAI/bge-m3",
    },
    {
        "id": 6,
        "name": "Exp6_BGE_M3_Weight_6040",
        "label": "실험 6: BGE-M3 + 가중치 60/40",
        "desc": "bge-m3 + 청크 200토큰 + BGE 리랭커 (vector 0.6 / keyword 0.4)",
        "vector_weight": 0.6,
        "text_weight": 0.4,
        "top_k": 5,
        "reranker": "bge-reranker-v2-m3",
        "rpc": "hybrid_search_v2",
        "embed_model": "BAAI/bge-m3",
    },
    {
        "id": 7,
        "name": "Exp7_BGE_M3_Weight_5050",
        "label": "실험 7: BGE-M3 + 가중치 50/50",
        "desc": "bge-m3 + 청크 200토큰 + BGE 리랭커 (vector 0.5 / keyword 0.5)",
        "vector_weight": 0.5,
        "text_weight": 0.5,
        "top_k": 5,
        "reranker": "bge-reranker-v2-m3",
        "rpc": "hybrid_search_v2",
        "embed_model": "BAAI/bge-m3",
    },
]

# ── 테스트 케이스 ─────────────────────────────────────────────────────
# judge=True  → Hit Rate / MRR 계산 대상
# judge=False → 주제 외 질문 (폴백 동작 확인용, 지표 계산 제외)
TEST_CASES = [
    # 건강식품
    {"query": "오메가3의 심혈관 건강 효능은?",
     "keywords": ["오메가3", "omega-3", "omega3", "epa", "dha", "심혈관", "fish oil"], "judge": True},
    {"query": "프로바이오틱스가 장 건강에 미치는 영향",
     "keywords": ["프로바이오틱스", "probiotics", "probiotic", "장내", "유산균", "lactobacillus", "gut"], "judge": True},
    {"query": "커큐민의 항염 효과 임상 연구",
     "keywords": ["커큐민", "curcumin", "항염", "강황", "turmeric", "anti-inflammatory"], "judge": True},
    # 다이어트
    {"query": "간헐적 단식 16:8 방법과 체중 감량 효과",
     "keywords": ["간헐적 단식", "간헐적단식", "intermittent fasting", "fasting", "단식", "공복"], "judge": True},
    {"query": "케토제닉 다이어트 혈당 관리 연구",
     "keywords": ["케토제닉", "ketogenic", "keto", "저탄수화물", "혈당", "ketone"], "judge": True},
    {"query": "저탄수화물 식단 메타분석 결과",
     "keywords": ["저탄수화물", "탄수화물", "low-carb", "low carb", "carbohydrate", "메타분석", "meta-analysis"], "judge": True},
    {"query": "단백질 섭취량과 근육 유지 관계",
     "keywords": ["단백질", "protein", "근육", "muscle", "아미노산", "amino"], "judge": True},
    {"query": "식이섬유와 장내 미생물 관계",
     "keywords": ["식이섬유", "dietary fiber", "fiber", "fibre", "microbiome", "microbiota", "장내"], "judge": True},
    # 혈당
    {"query": "혈당 스파이크 원인과 예방법",
     "keywords": ["혈당", "blood glucose", "blood sugar", "인슐린", "insulin", "스파이크", "glucose"], "judge": True},
    {"query": "GI 지수와 혈당 관리 방법",
     "keywords": ["glycemic", "gi index", "혈당", "혈당지수", "당지수", "인슐린", "glucose"], "judge": True},
    # 푸드올로지 제품
    {"query": "푸드올로지 버닝올로지 성분",
     "keywords": ["버닝올로지", "푸드올로지", "버닝", "burning"], "judge": True},
    {"query": "콜레올로지 제품 효능",
     "keywords": ["콜레올로지", "푸드올로지", "콜레스테롤"], "judge": True},
    {"query": "맨올로지 추천 대상",
     "keywords": ["맨올로지", "푸드올로지"], "judge": True},
    # 실생활
    {"query": "닭가슴살 다이어트 식단 추천",
     "keywords": ["닭가슴살", "닭", "식단", "다이어트", "chicken"], "judge": True},
    {"query": "다이어트 보조제 종류와 효과",
     "keywords": ["보조제", "supplement", "다이어트", "지방", "감량", "weight loss"], "judge": True},
    {"query": "단백질 보충제 먹는 방법",
     "keywords": ["단백질", "보충제", "protein", "supplement", "whey"], "judge": True},
    # 비타민 (DB에 없을 가능성 있음 → 진단 포인트)
    {"query": "비타민C 결핍 증상",
     "keywords": ["비타민c", "vitamin c", "비타민 c", "ascorbic", "괴혈병"], "judge": True},
    # 주제 외 (판정 제외)
    {"query": "날씨가 맑은 날 운동하면 좋은가요",
     "keywords": [], "judge": False},
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 검색 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 실험별로 임베딩 모델을 교체할 수 있도록 캐시
_embed_cache: dict[str, SentenceTransformer] = {}

def get_embed_model(model_name: str) -> SentenceTransformer:
    if model_name not in _embed_cache:
        print(f"  임베딩 모델 로딩: {model_name}")
        _embed_cache[model_name] = SentenceTransformer(model_name)
    return _embed_cache[model_name]


def get_embedding(text: str, model_name: str = "intfloat/multilingual-e5-small") -> list[float]:
    """실험에 따라 모델 선택. e5는 query 프리픽스 필요, bge-m3는 불필요."""
    model = get_embed_model(model_name)
    if "e5" in model_name:
        input_text = f"query: {text[:MAX_CONTENT]}"
    else:
        input_text = text[:MAX_CONTENT]
    return model.encode(input_text, normalize_embeddings=True).tolist()


def detect_category(query: str) -> str | None:
    q = query.lower()
    if any(k in q for k in ["푸드올로지", "콜레올로지", "맨올로지", "톡스올로지", "버닝올로지"]):
        return "푸드올로지"
    return None


def search(query: str, vector_weight: float, text_weight: float, top_k: int,
           rpc: str = "hybrid_search", model_name: str = "intfloat/multilingual-e5-small") -> list[dict]:
    """Supabase hybrid_search 호출 (실험 파라미터 + RPC 이름 + 임베딩 모델 선택)"""
    embedding = get_embedding(query, model_name)
    category_filter = detect_category(query)

    def _run(cat_filter):
        try:
            res = supabase.rpc(rpc, {
                "query_embedding": embedding,
                "query_text": query,
                "match_count": top_k,
                "vector_weight": vector_weight,
                "text_weight": text_weight,
                "filter_category": cat_filter,
            }).execute()
            return res.data or []
        except Exception:
            try:
                params = {
                    "query_embedding": embedding,
                    "match_threshold": 0.3,
                    "match_count": top_k,
                }
                if cat_filter:
                    params["filter_category"] = cat_filter
                res = supabase.rpc("match_children_return_parents", params).execute()
                return res.data or []
            except Exception:
                return []

    results = _run(category_filter)
    if not results and category_filter:
        results = _run(None)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 리랭킹 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rerank_keyword_bonus(query: str, docs: list[dict]) -> list[dict]:
    """기존 방식: 검색 점수 + 한국어 키워드 보너스 (cross-encoder 미사용)"""
    keywords = [w for w in query.split() if len(w) >= 2]
    content_key = "parent_content" if docs and "parent_content" in docs[0] else "content"
    for doc in docs:
        base = doc.get("combined_score") or doc.get("similarity") or 0
        content = doc.get(content_key, "")
        hits = sum(1 for kw in keywords if kw in content)
        doc["rerank_score"] = float(base) + min(hits * 0.05, 0.2)
        doc["_content_key"] = content_key
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)


def rerank_bge(query: str, docs: list[dict]) -> list[dict]:
    """BGE 다국어 크로스인코더 리랭킹 (실험 4)"""
    global bge_reranker
    if bge_reranker is None:
        print("    BGE 리랭커 로딩 중 (최초 1회 다운로드)...")
        bge_reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        print("    BGE 로드 완료")

    content_key = "parent_content" if docs and "parent_content" in docs[0] else "content"
    pairs = [(query, doc.get(content_key, "")[:1000]) for doc in docs]
    scores = bge_reranker.predict(pairs)
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
        doc["_content_key"] = content_key
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)


def apply_rerank(query: str, docs: list[dict], reranker_type: str) -> list[dict]:
    if reranker_type == "bge-reranker-v2-m3":
        return rerank_bge(query, docs)
    return rerank_keyword_bonus(query, docs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 지표 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def is_relevant_doc(doc: dict, keywords: list[str]) -> bool:
    """문서가 키워드를 포함하는지 판별 (대소문자 무시)"""
    content_key = doc.get("_content_key", "parent_content" if "parent_content" in doc else "content")
    content = doc.get(content_key, "").lower()
    return any(kw.lower() in content for kw in keywords)


def compute_hit_at_k(ranked_docs: list[dict], keywords: list[str], k: int) -> int:
    """Hit@K: 상위 K개 문서 중 관련 문서 존재 여부 (1 or 0)"""
    for doc in ranked_docs[:k]:
        if is_relevant_doc(doc, keywords):
            return 1
    return 0


def compute_rr(ranked_docs: list[dict], keywords: list[str]) -> float:
    """Reciprocal Rank: 첫 번째 관련 문서의 순위 역수"""
    for rank, doc in enumerate(ranked_docs, 1):
        if is_relevant_doc(doc, keywords):
            return 1.0 / rank
    return 0.0


def grade(hit5: float, avg_sim: float) -> str:
    """종합 등급 판정"""
    if hit5 >= 0.85 and avg_sim >= 0.6:
        return "A (우수)"
    elif hit5 >= 0.70 and avg_sim >= 0.5:
        return "B (양호)"
    elif hit5 >= 0.50:
        return "C (보통)"
    else:
        return "D (미흡)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 실험 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_experiment(config: dict) -> dict:
    """
    한 가지 설정으로 모든 테스트 케이스를 실행하고 결과 반환.
    반환값:
      - per_query: 질문별 상세 결과 리스트
      - summary: Hit@1/3/5, MRR, 평균유사도, 등급
      - diagnosis: 실패 질문 진단 정보
    """
    vector_weight = config["vector_weight"]
    text_weight   = config["text_weight"]
    top_k         = config["top_k"]
    reranker_type = config["reranker"]
    rpc_name      = config.get("rpc", "hybrid_search")
    model_name    = config.get("embed_model", "intfloat/multilingual-e5-small")

    per_query = []
    judged_hits1, judged_hits3, judged_hits5 = [], [], []
    rr_list = []
    sim_scores = []

    # 진단: 검색 실패/성공 분류
    search_fail_queries  = []   # 검색 결과 자체가 없음
    rank_fail_queries    = []   # 검색은 됐지만 관련 문서 Top5 밖
    rerank_hurt_queries  = []   # 리랭킹 전에는 있었지만 리랭킹 후 순위 하락
    success_queries      = []

    total = len(TEST_CASES)
    for idx, tc in enumerate(TEST_CASES, 1):
        query    = tc["query"]
        keywords = tc["keywords"]
        judge    = tc["judge"]

        print(f"  [{idx:02d}/{total}] {query[:40]}", end=" ... ", flush=True)

        t0 = time.time()
        raw_docs = search(query, vector_weight, text_weight, top_k, rpc_name, model_name)
        search_time = time.time() - t0

        if not raw_docs:
            print("결과 없음")
            per_query.append({
                "query": query, "judge": judge,
                "n_results": 0, "hit@1": 0, "hit@3": 0, "hit@5": 0,
                "rr": 0.0, "top1_sim": 0.0, "top1_category": "없음",
                "search_time": round(search_time, 3),
                "pre_rerank_rank": None, "post_rerank_rank": None,
                "hit_keywords": "",
            })
            if judge:
                judged_hits1.append(0); judged_hits3.append(0); judged_hits5.append(0)
                rr_list.append(0.0)
                search_fail_queries.append(query)
            continue

        # ── 리랭킹 전 순위 기록 (진단용) ──
        pre_rerank_rank = None
        for rank, doc in enumerate(raw_docs, 1):
            if is_relevant_doc(doc, keywords):
                pre_rerank_rank = rank
                break

        # ── 리랭킹 ──
        ranked_docs = apply_rerank(query, raw_docs, reranker_type)

        # ── 리랭킹 후 순위 기록 (진단용) ──
        post_rerank_rank = None
        for rank, doc in enumerate(ranked_docs, 1):
            if is_relevant_doc(doc, keywords):
                post_rerank_rank = rank
                break

        # ── 지표 계산 ──
        hit1 = compute_hit_at_k(ranked_docs, keywords, 1)
        hit3 = compute_hit_at_k(ranked_docs, keywords, 3)
        hit5 = compute_hit_at_k(ranked_docs, keywords, 5)
        rr   = compute_rr(ranked_docs, keywords)

        top1_sim = ranked_docs[0].get("combined_score") or ranked_docs[0].get("similarity") or 0
        top1_cat = ranked_docs[0].get("category", "알 수 없음")

        # 적중 키워드 목록
        content_key = ranked_docs[0].get("_content_key", "content")
        hit_kwds = [kw for kw in keywords
                    if kw.lower() in ranked_docs[0].get(content_key, "").lower()]

        print(f"Hit@1={hit1} Hit@5={hit5} RR={rr:.2f} sim={top1_sim:.3f}")

        per_query.append({
            "query": query, "judge": judge,
            "n_results": len(raw_docs),
            "hit@1": hit1, "hit@3": hit3, "hit@5": hit5,
            "rr": round(rr, 4),
            "top1_sim": round(float(top1_sim), 4),
            "top1_category": top1_cat,
            "search_time": round(search_time, 3),
            "pre_rerank_rank": pre_rerank_rank,
            "post_rerank_rank": post_rerank_rank,
            "hit_keywords": ", ".join(hit_kwds) if hit_kwds else "없음",
        })

        if judge:
            judged_hits1.append(hit1)
            judged_hits3.append(hit3)
            judged_hits5.append(hit5)
            rr_list.append(rr)
            sim_scores.append(float(top1_sim))

            # 진단 분류
            if hit5 == 1:
                success_queries.append(query)
            else:
                if pre_rerank_rank is not None and pre_rerank_rank <= top_k:
                    rerank_hurt_queries.append({
                        "query": query,
                        "pre_rank": pre_rerank_rank,
                        "post_rank": post_rerank_rank,
                    })
                else:
                    rank_fail_queries.append(query)

    # ── 요약 지표 ──
    n = len(judged_hits1)
    hr1  = sum(judged_hits1) / n if n else 0
    hr3  = sum(judged_hits3) / n if n else 0
    hr5  = sum(judged_hits5) / n if n else 0
    mrr  = sum(rr_list) / n if n else 0
    avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
    g = grade(hr5, avg_sim)

    summary = {
        "experiment":   config["label"],
        "config":       config["desc"],
        "judged_total": n,
        "hit@1":        round(hr1, 4),
        "hit@3":        round(hr3, 4),
        "hit@5":        round(hr5, 4),
        "mrr":          round(mrr, 4),
        "avg_top1_sim": round(avg_sim, 4),
        "grade":        g,
    }

    diagnosis = {
        "search_fail":   search_fail_queries,
        "rank_fail":     rank_fail_queries,
        "rerank_hurt":   rerank_hurt_queries,
        "success_count": len(success_queries),
    }

    return {"config": config, "per_query": per_query, "summary": summary, "diagnosis": diagnosis}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 저장 함수 (마크다운 우선)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_per_query_md(result: dict) -> Path:
    """질문별 상세 결과 마크다운 저장"""
    cfg = result["config"]
    name = cfg["name"]
    path = RESULTS_DIR / f"{name}_{RUN_TIMESTAMP}.md"

    lines = [
        f"# {cfg['label']} — 질문별 상세 결과",
        f"> {cfg['desc']}",
        f"> 실행 시각: {RUN_TIMESTAMP}",
        "",
        "| # | 질문 | judge | Hit@1 | Hit@3 | Hit@5 | RR | TOP1유사도 | TOP1카테고리 | 적중키워드 |",
        "|---|------|:-----:|:-----:|:-----:|:-----:|:--:|:---------:|:-----------:|-----------|",
    ]
    for i, r in enumerate(result["per_query"], 1):
        judge_mark = "O" if r["judge"] else "-"
        lines.append(
            f"| {i} | {r['query']} | {judge_mark} "
            f"| {r['hit@1']} | {r['hit@3']} | {r['hit@5']} "
            f"| {r['rr']:.2f} | {r['top1_sim']:.4f} "
            f"| {r['top1_category']} | {r['hit_keywords']} |"
        )

    # 진단 섹션
    d = result["diagnosis"]
    lines += [
        "",
        "## 진단",
        f"- 검색 실패: {len(d['search_fail'])}건",
        f"- 순위 실패 (Top-K 밖): {len(d['rank_fail'])}건",
        f"- 리랭킹 순위 악화: {len(d['rerank_hurt'])}건",
    ]
    if d["rank_fail"]:
        lines.append("\n### 순위 실패 질문")
        for q in d["rank_fail"]:
            lines.append(f"- {q}")
    if d["rerank_hurt"]:
        lines.append("\n### 리랭킹 순위 악화 질문")
        for item in d["rerank_hurt"]:
            lines.append(f"- {item['query']} (검색전 {item['pre_rank']}등 → 리랭킹후 {item['post_rank']}등)")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_summary_md(all_results: list[dict]) -> Path:
    """실험 비교 요약 마크다운 저장"""
    path = RESULTS_DIR / f"eval_summary_{RUN_TIMESTAMP}.md"

    lines = [
        "# RAG 평가 실험 비교 요약",
        f"> 실행 시각: {RUN_TIMESTAMP}  |  판정 질문: {all_results[0]['summary']['judged_total']}개",
        "",
        "| 실험 | Hit@1 | Hit@3 | Hit@5 | MRR | 평균유사도 | 등급 |",
        "|------|------:|------:|------:|----:|----------:|------|",
    ]
    for r in all_results:
        s = r["summary"]
        lines.append(
            f"| {s['experiment']} "
            f"| {s['hit@1']:.3f} | {s['hit@3']:.3f} | {s['hit@5']:.3f} "
            f"| {s['mrr']:.3f} | {s['avg_top1_sim']:.3f} | {s['grade']} |"
        )

    lines += ["", "## 설정 상세", ""]
    for r in all_results:
        s = r["summary"]
        lines.append(f"- **{s['experiment']}**: {s['config']}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_summary_json(all_results: list[dict]) -> Path:
    """결과 전체를 JSON으로 저장 (원본 데이터 보존)"""
    path = RESULTS_DIR / f"eval_full_{RUN_TIMESTAMP}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main(exp_ids: list[int] | None = None):
    target_exps = [e for e in EXPERIMENTS if exp_ids is None or e["id"] in exp_ids]

    print("=" * 65)
    print("  건강식품 RAG 챗봇 검색 품질 평가 v2")
    print(f"  실행 시각: {RUN_TIMESTAMP}")
    print(f"  실행 실험: {[e['id'] for e in target_exps]}")
    print(f"  테스트 케이스: {len(TEST_CASES)}개")
    print(f"  결과 저장 위치: {RESULTS_DIR}")
    print("=" * 65)

    all_results = []

    for exp in target_exps:
        print(f"\n{'='*65}")
        print(f"  {exp['label']}")
        print(f"  {exp['desc']}")
        print(f"{'='*65}")

        result = run_experiment(exp)
        all_results.append(result)

        s = result["summary"]
        d = result["diagnosis"]
        print(f"\n  결과 요약:")
        print(f"    Hit@1={s['hit@1']:.4f}  Hit@3={s['hit@3']:.4f}  Hit@5={s['hit@5']:.4f}")
        print(f"    MRR={s['mrr']:.4f}  평균유사도={s['avg_top1_sim']:.4f}  등급={s['grade']}")
        print(f"  진단: 검색실패={len(d['search_fail'])}건  순위실패={len(d['rank_fail'])}건  리랭크악화={len(d['rerank_hurt'])}건")

        md_path = save_per_query_md(result)
        print(f"  저장: {md_path.name}")

    # 요약 마크다운 + JSON
    summary_md   = save_summary_md(all_results)
    summary_json = save_summary_json(all_results)
    print(f"\n비교 요약 저장: {summary_md.name}")
    print(f"전체 결과 저장: {summary_json.name}")

    # 최종 비교 테이블 출력
    print(f"\n{'='*65}")
    print("  최종 실험 비교")
    print(f"{'='*65}")
    header = f"{'실험':<22} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'유사도':>6} {'등급'}"
    print(f"  {header}")
    print(f"  {'-'*70}")
    for r in all_results:
        s = r["summary"]
        label = s["experiment"].replace("실험 ", "E").replace(": ", " ")[:22]
        print(f"  {label:<22} {s['hit@1']:>6.3f} {s['hit@3']:>6.3f} {s['hit@5']:>6.3f} "
              f"{s['mrr']:>6.3f} {s['avg_top1_sim']:>6.3f} {s['grade']}")
    print(f"{'='*65}")
    print(f"\n결과 파일 위치: {RESULTS_DIR}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", type=int,
                        help="실행할 실험 ID (예: --exp 5 6 7). 미지정 시 전체 실행")
    args = parser.parse_args()
    main(exp_ids=args.exp)
