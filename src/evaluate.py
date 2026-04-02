"""
유사도 평가 스크립트 v2
- 평가 기준: 카테고리 적중 → 답변 관련성 (키워드 적중)
- TOP3 문서 중 핵심 키워드가 포함된 문서가 있으면 "관련 있음"으로 판정
- combined_score / rerank_score / 키워드 적중 수 출력
- 관련성 적중률, 평균 유사도, 이상 결과 탐지 리포트 출력

실행: python src/evaluate.py
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

TOP_K = 5
MAX_CONTENT = 2000
RELEVANCE_MIN_KEYWORDS = 1   # TOP3 중 핵심 키워드 최소 N개 이상 → 관련 있음 판정

print("모델 로딩 중...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("완료\n")

# ── 테스트 질문 셋 ────────────────────────────────────────────────────
# (질문, [핵심 키워드 리스트])
# 핵심 키워드 중 하나라도 TOP3 문서에 있으면 관련성 있음으로 판정
TEST_CASES = [
    # 건강식품 관련
    ("오메가3의 심혈관 건강 효능은?",
     ["오메가3", "omega-3", "omega3", "omega 3", "epa", "dha", "심혈관", "fish oil"]),

    ("프로바이오틱스가 장 건강에 미치는 영향",
     ["프로바이오틱스", "probiotics", "probiotic", "장내", "유산균", "lactobacillus", "gut"]),

    ("커큐민의 항염 효과 임상 연구",
     ["커큐민", "curcumin", "항염", "강황", "turmeric", "anti-inflammatory"]),

    # 다이어트 관련
    ("간헐적 단식 16:8 방법과 체중 감량 효과",
     ["간헐적 단식", "간헐적단식", "intermittent fasting", "fasting", "단식", "공복"]),

    ("케토제닉 다이어트 혈당 관리 연구",
     ["케토제닉", "ketogenic", "keto", "저탄수화물", "혈당", "ketone"]),

    ("저탄수화물 식단 메타분석 결과",
     ["저탄수화물", "탄수화물", "low-carb", "low carb", "carbohydrate", "메타분석", "meta-analysis"]),

    ("단백질 섭취량과 근육 유지 관계",
     ["단백질", "protein", "근육", "muscle", "아미노산", "amino"]),

    ("식이섬유와 장내 미생물 관계",
     ["식이섬유", "dietary fiber", "fiber", "fibre", "microbiome", "microbiota", "장내"]),

    # 혈당 관련
    ("혈당 스파이크 원인과 예방법",
     ["혈당", "blood glucose", "blood sugar", "인슐린", "insulin", "스파이크", "glucose"]),

    ("GI 지수와 혈당 관리 방법",
     ["glycemic", "gi index", "혈당", "혈당지수", "당지수", "인슐린", "glucose"]),

    # 푸드올로지 제품
    ("푸드올로지 버닝올로지 성분",
     ["버닝올로지", "푸드올로지", "버닝", "burning"]),

    ("콜레올로지 제품 효능",
     ["콜레올로지", "푸드올로지", "콜레스테롤"]),

    ("맨올로지 추천 대상",
     ["맨올로지", "푸드올로지"]),

    # 실생활 다이어트
    ("닭가슴살 다이어트 식단 추천",
     ["닭가슴살", "닭", "식단", "다이어트", "chicken"]),

    ("곤약 다이어트 효과 후기",
     ["곤약", "konjac", "저칼로리", "다이어트"]),

    # 보충제
    ("다이어트 보조제 종류와 효과",
     ["보조제", "supplement", "다이어트", "지방", "감량", "weight loss"]),

    ("단백질 보충제 먹는 방법",
     ["단백질", "보충제", "protein", "supplement", "whey"]),

    # 주제 외 (판정 제외 — 폴백 확인용)
    ("날씨가 맑은 날 운동하면 좋은가요",
     []),

    ("비타민C 결핍 증상",
     ["비타민c", "vitamin c", "비타민 c", "ascorbic", "괴혈병"]),
]


# ── 검색 함수 ────────────────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    return embed_model.encode(f"query: {text[:MAX_CONTENT]}", normalize_embeddings=True).tolist()


def detect_category(query: str) -> str | None:
    q = query.lower()
    if any(k in q for k in ["푸드올로지", "콜레올로지", "맨올로지", "톡스올로지", "버닝올로지"]):
        return "푸드올로지"
    return None


def search(query: str) -> list[dict]:
    embedding = get_embedding(query)
    category_filter = detect_category(query)

    def _run(cat_filter):
        try:
            res = supabase.rpc("hybrid_search", {
                "query_embedding": embedding,
                "query_text": query,
                "match_count": TOP_K,
                "vector_weight": 0.7,
                "text_weight": 0.3,
                "filter_category": cat_filter,
            }).execute()
            return res.data or []
        except Exception:
            try:
                params = {"query_embedding": embedding, "match_threshold": 0.3, "match_count": TOP_K}
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


def rerank(query: str, docs: list[dict]) -> list[dict]:
    keywords = [w for w in query.split() if len(w) >= 2]
    content_key = "parent_content" if docs and "parent_content" in docs[0] else "content"
    for doc in docs:
        base_score = doc.get("combined_score") or doc.get("similarity") or 0
        content = doc.get(content_key, "")
        keyword_hits = sum(1 for kw in keywords if kw in content)
        doc["rerank_score"] = float(base_score) + min(keyword_hits * 0.05, 0.2)
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)


def check_relevance(docs: list[dict], keywords: list[str], content_key: str) -> tuple[bool, int, str]:
    """TOP3 문서에서 핵심 키워드 적중 여부 확인
    반환: (관련성 있음, 총 적중 키워드 수, 적중 키워드 목록)
    """
    if not keywords:
        return None, 0, ""  # 키워드 없음 = 판정 불가 (주제 외 질문)

    hit_keywords = set()
    for doc in docs[:3]:
        content = doc.get(content_key, "").lower()
        for kw in keywords:
            if kw.lower() in content:
                hit_keywords.add(kw)

    is_relevant = len(hit_keywords) >= RELEVANCE_MIN_KEYWORDS
    return is_relevant, len(hit_keywords), ", ".join(hit_keywords) if hit_keywords else "없음"


# ── 평가 실행 ────────────────────────────────────────────────────────
def evaluate():
    print("=" * 65)
    print("  건강식품 RAG 챗봇 관련성 평가 v2")
    print("  (평가 기준: TOP3 문서 내 핵심 키워드 적중 여부)")
    print("=" * 65)

    total = len(TEST_CASES)
    relevance_hit = 0       # 관련성 있음 판정 수
    relevance_total = 0     # 판정 가능한 질문 수 (키워드 있는 것)
    no_result_count = 0
    low_score_count = 0
    score_sum = 0.0
    score_count = 0
    warnings_log = []

    for idx, (query, keywords) in enumerate(TEST_CASES, 1):
        print(f"\n[{idx:02d}/{total}] 질문: {query}")
        print(f"      핵심 키워드: {', '.join(keywords) if keywords else '없음 (주제 외)'}")

        t0 = time.time()
        docs = search(query)
        elapsed = time.time() - t0

        if not docs:
            print("      결과: 검색 결과 없음")
            no_result_count += 1
            warnings_log.append(f"[{idx:02d}] '{query}' → 검색 결과 없음")
            continue

        ranked = rerank(query, docs)
        content_key = "parent_content" if "parent_content" in ranked[0] else "content"

        print(f"      검색 결과: {len(ranked)}개  ({elapsed:.2f}s)")
        print(f"      {'순위':<4} {'카테고리':<20} {'combined_score':<16} {'rerank_score':<14} {'내용 미리보기'}")
        print(f"      {'-'*4} {'-'*20} {'-'*16} {'-'*14} {'-'*30}")

        for rank, doc in enumerate(ranked, 1):
            cat = doc.get("category", "알 수 없음")
            combined = doc.get("combined_score") or doc.get("similarity") or 0
            rerank_s = doc.get("rerank_score", 0)
            preview = doc.get(content_key, "")[:40].replace("\n", " ")
            marker = " ◀ TOP1" if rank == 1 else ""
            print(f"      {rank:<4} {cat:<20} {combined:<16.4f} {rerank_s:<14.4f} {preview}...{marker}")

            if rank == 1:
                score_sum += combined
                score_count += 1
                if combined < 0.4:
                    low_score_count += 1
                    warnings_log.append(f"[{idx:02d}] '{query}' → TOP1 유사도 낮음 ({combined:.4f})")

        # 관련성 판정
        is_relevant, hit_count, hit_words = check_relevance(ranked, keywords, content_key)

        if is_relevant is None:
            print(f"      관련성 판정: 해당 없음 (주제 외 질문) — TOP1 카테고리: {ranked[0].get('category','?')}")
        elif is_relevant:
            relevance_hit += 1
            relevance_total += 1
            print(f"      관련성 판정: O  (적중 키워드 {hit_count}개: {hit_words})")
        else:
            relevance_total += 1
            print(f"      관련성 판정: X  (키워드 미적중)")
            warnings_log.append(f"[{idx:02d}] '{query}' → 관련 문서 없음 (키워드 미적중)")

    # ── 최종 리포트 ──────────────────────────────────────────────────
    avg_score = score_sum / score_count if score_count else 0
    relevance_rate = relevance_hit / relevance_total if relevance_total else 0

    print("\n" + "=" * 65)
    print("  최종 평가 리포트")
    print("=" * 65)
    print(f"  총 테스트 질문       : {total}개")
    print(f"  관련성 적중률        : {relevance_hit}/{relevance_total} ({relevance_rate*100:.1f}%)")
    print(f"  TOP1 평균 유사도     : {avg_score:.4f}")
    print(f"  검색 결과 없음       : {no_result_count}개")
    print(f"  낮은 유사도 (< 0.4)  : {low_score_count}개")

    if warnings_log:
        print("\n  [주의 항목]")
        for w in warnings_log:
            print(f"    {w}")
    else:
        print("\n  주의 항목 없음 — 모든 질문 관련 문서 검색 성공")

    print("=" * 65)

    # 등급 판정
    if relevance_rate >= 0.85 and avg_score >= 0.6:
        grade = "A  (우수)"
    elif relevance_rate >= 0.70 and avg_score >= 0.5:
        grade = "B  (양호)"
    elif relevance_rate >= 0.50:
        grade = "C  (보통 — 검색 튜닝 권장)"
    else:
        grade = "D  (미흡 — 데이터/임베딩 점검 필요)"

    print(f"\n  종합 등급: {grade}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
