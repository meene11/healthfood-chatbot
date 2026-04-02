"""
유사도 평가 스크립트
- 카테고리별 테스트 질문으로 검색 품질 측정
- combined_score(검색 유사도) + rerank_score(최종 순위 점수) 출력
- 카테고리 적중률, 평균 유사도, 이상 결과 탐지 리포트 출력

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

print("모델 로딩 중...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("완료\n")

# ── 테스트 질문 셋 ────────────────────────────────────────────────────
# (질문, 기대 카테고리)  ← 카테고리명은 DB 기준
TEST_CASES = [
    # 건강식품_논문
    ("오메가3의 심혈관 건강 효능은?",            "건강식품_논문"),
    ("프로바이오틱스가 장 건강에 미치는 영향",    "건강식품_논문"),
    ("커큐민의 항염 효과 임상 연구",              "건강식품_논문"),

    # 다이어트_논문
    ("간헐적 단식 16:8 방법과 체중 감량 효과",   "다이어트_논문"),
    ("케토제닉 다이어트 혈당 관리 연구",          "다이어트_논문"),
    ("저탄수화물 식단 메타분석 결과",              "다이어트_논문"),
    ("단백질 섭취량과 근육 유지 관계",            "다이어트_논문"),
    ("식이섬유와 장내 미생물 관계",               "다이어트_논문"),

    # 혈당스파이크_논문
    ("혈당 스파이크 원인과 예방법",               "혈당스파이크_논문"),
    ("GI 지수와 혈당 관리 방법",                  "혈당스파이크_논문"),

    # 푸드올로지
    ("푸드올로지 버닝올로지 성분",                "푸드올로지"),
    ("콜레올로지 제품 효능",                       "푸드올로지"),
    ("맨올로지 추천 대상",                          "푸드올로지"),

    # 네이버블로그
    ("닭가슴살 다이어트 식단 추천",               "네이버블로그"),
    ("곤약 다이어트 효과 후기",                    "네이버블로그"),

    # 건강_수집데이터
    ("다이어트 보조제 종류와 효과",               "건강_수집데이터"),
    ("단백질 보충제 먹는 방법",                    "건강_수집데이터"),

    # 주제 외 (폴백 테스트)
    ("날씨가 맑은 날 운동하면 좋은가요",          None),
    ("비타민C 결핍 증상",                          None),
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
        results = _run(None)  # 폴백
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


# ── 평가 실행 ────────────────────────────────────────────────────────
def evaluate():
    print("=" * 65)
    print("  건강식품 RAG 챗봇 유사도 평가")
    print("=" * 65)

    total = len(TEST_CASES)
    category_hit = 0       # TOP-1 카테고리 적중
    no_result_count = 0    # 검색 결과 없음
    low_score_count = 0    # 유사도 낮음 (< 0.4)
    score_sum = 0.0
    score_count = 0
    warnings_log = []

    for idx, (query, expected_cat) in enumerate(TEST_CASES, 1):
        print(f"\n[{idx:02d}/{total}] 질문: {query}")
        if expected_cat:
            print(f"      기대 카테고리: {expected_cat}")
        else:
            print(f"      기대 카테고리: 없음 (주제 외 폴백 테스트)")

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

        top1_cat = ranked[0].get("category", "알 수 없음")

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

        # 카테고리 적중 체크
        if expected_cat and top1_cat == expected_cat:
            category_hit += 1
            print(f"      카테고리 적중: O")
        elif expected_cat:
            print(f"      카테고리 적중: X  (실제: {top1_cat})")
            warnings_log.append(f"[{idx:02d}] '{query}' → 카테고리 불일치 (기대: {expected_cat}, 실제: {top1_cat})")
        else:
            print(f"      폴백 동작: TOP1 카테고리 = {top1_cat}")

    # ── 최종 리포트 ──────────────────────────────────────────────────
    expected_count = sum(1 for _, c in TEST_CASES if c is not None)
    avg_score = score_sum / score_count if score_count else 0

    print("\n" + "=" * 65)
    print("  최종 평가 리포트")
    print("=" * 65)
    print(f"  총 테스트 질문       : {total}개")
    print(f"  카테고리 적중률      : {category_hit}/{expected_count} ({category_hit/expected_count*100:.1f}%)")
    print(f"  TOP1 평균 유사도     : {avg_score:.4f}")
    print(f"  검색 결과 없음       : {no_result_count}개")
    print(f"  낮은 유사도 (< 0.4)  : {low_score_count}개")

    if warnings_log:
        print("\n  [주의 항목]")
        for w in warnings_log:
            print(f"    {w}")
    else:
        print("\n  주의 항목 없음 — 모든 질문 정상 검색")

    print("=" * 65)

    # 등급 판정
    hit_rate = category_hit / expected_count if expected_count else 0
    if hit_rate >= 0.85 and avg_score >= 0.6:
        grade = "A  (우수)"
    elif hit_rate >= 0.70 and avg_score >= 0.5:
        grade = "B  (양호)"
    elif hit_rate >= 0.50:
        grade = "C  (보통 — 검색 튜닝 권장)"
    else:
        grade = "D  (미흡 — 데이터/임베딩 점검 필요)"

    print(f"\n  종합 등급: {grade}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
