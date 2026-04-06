"""
QA 쌍 기반 RAG 평가 스크립트
==============================
(질문, 정답_chunk_id) 쌍 데이터셋으로 Hit Rate / MRR 계산.
키워드 매칭이 아닌 정확한 chunk_id 기반 판정 → 더 신뢰도 높은 평가.

비교 대상:
  QA-E4: documents 테이블 (e5-small 384차원, 실험 4 설정)
  QA-E5: documents_v2 테이블 (bge-m3 1024차원, 실험 5 설정)

입력: data/generated/qa_dataset.json
출력:
  results/QA쌍_평가결과.md
  results/QA쌍_질문별상세_QA-E4.md
  results/QA쌍_질문별상세_QA-E5.md

실행:
  python src/evaluate_qa.py
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR     = Path(__file__).resolve().parent.parent
QA_FILE      = BASE_DIR / "data" / "generated" / "qa_dataset.json"
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TOP_K      = 5
MAX_SEARCH = TOP_K * 3   # 리랭킹 전 검색 수
MAX_CONTENT = 2000

# ── 실험 설정 ─────────────────────────────────────────────────────────────
QA_EXPERIMENTS = [
    {
        "id":           "QA-E4",
        "label":        "QA-E4: e5-small (기존, documents)",
        "embed_model":  "intfloat/multilingual-e5-small",
        "embed_prefix": "query: ",          # e5는 query: 프리픽스 필요
        "rpc":          "hybrid_search",
        "vector_weight": 0.7,
        "text_weight":   0.3,
        "table":         "documents",
    },
    {
        "id":           "QA-E5",
        "label":        "QA-E5: bge-m3 (신규, documents_v2)",
        "embed_model":  "BAAI/bge-m3",
        "embed_prefix": "",                 # bge-m3는 프리픽스 불필요
        "rpc":          "hybrid_search_v2",
        "vector_weight": 0.7,
        "text_weight":   0.3,
        "table":         "documents_v2",
    },
]

# ── 모델 캐시 ─────────────────────────────────────────────────────────────
_embed_cache: dict[str, SentenceTransformer] = {}
_reranker:    CrossEncoder | None = None


def get_embed_model(model_name: str) -> SentenceTransformer:
    if model_name not in _embed_cache:
        print(f"임베딩 모델 로딩: {model_name}")
        _embed_cache[model_name] = SentenceTransformer(model_name)
    return _embed_cache[model_name]


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print("리랭커 로딩: BAAI/bge-reranker-v2-m3")
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _reranker


def embed_query(query: str, model_name: str, prefix: str) -> list[float]:
    model = get_embed_model(model_name)
    text  = prefix + query
    vec   = model.encode(text[:MAX_CONTENT], normalize_embeddings=True)
    return vec.tolist()


def search(query: str, exp: dict) -> list[dict]:
    """하이브리드 검색 후 BGE 리랭킹 수행, Top-K 반환."""
    embedding = embed_query(query, exp["embed_model"], exp["embed_prefix"])

    try:
        resp = supabase.rpc(exp["rpc"], {
            "query_embedding": embedding,
            "query_text":      query,
            "match_count":     MAX_SEARCH,
            "vector_weight":   exp["vector_weight"],
            "text_weight":     exp["text_weight"],
        }).execute()
        docs = resp.data or []
    except Exception as e:
        print(f"    [ERROR] 검색 실패: {e}")
        return []

    if not docs:
        return []

    # BGE 리랭킹
    reranker = get_reranker()
    pairs    = [(query, doc.get("content", "")[:1000]) for doc in docs]
    scores   = reranker.predict(pairs)
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
    ranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:TOP_K]


def evaluate_experiment(qa_pairs: list[dict], exp: dict) -> dict:
    """단일 실험에 대해 전체 QA 쌍 평가."""
    label    = exp["label"]
    exp_id   = exp["id"]
    results  = []
    cat_hits = defaultdict(list)   # 카테고리별 Hit@5

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for i, qa in enumerate(qa_pairs, 1):
        question       = qa["question"]
        answer_id      = qa["answer_chunk_id"]
        category       = qa.get("category", "")
        source_file    = qa.get("source_file", "")

        print(f"  [{i:02d}/{len(qa_pairs)}] {question[:50]}...")

        t0   = time.time()
        docs = search(question, exp)
        elapsed = time.time() - t0

        # chunk id 목록 (검색된 순서)
        result_ids = [doc["id"] for doc in docs]

        # Hit 계산
        hit1 = int(answer_id in result_ids[:1])
        hit3 = int(answer_id in result_ids[:3])
        hit5 = int(answer_id in result_ids[:5])

        # Reciprocal Rank
        try:
            rank = result_ids.index(answer_id) + 1  # 1-based
            rr   = 1.0 / rank
        except ValueError:
            rank = None
            rr   = 0.0

        top1_score = docs[0].get("rerank_score", 0.0) if docs else 0.0
        top1_cat   = docs[0].get("category", "") if docs else ""

        results.append({
            "qa_id":       qa["id"],
            "question":    question,
            "answer_id":   answer_id,
            "category":    category,
            "source_file": source_file,
            "n_results":   len(docs),
            "result_ids":  result_ids,
            "hit1":        hit1,
            "hit3":        hit3,
            "hit5":        hit5,
            "rank":        rank,
            "rr":          rr,
            "top1_score":  top1_score,
            "top1_cat":    top1_cat,
            "elapsed":     elapsed,
        })
        cat_hits[category].append(hit5)

        status = f"Hit@5={'O' if hit5 else 'X'}, rank={rank}"
        print(f"    → {status} ({elapsed:.1f}s)")

    # 집계
    n = len(results)
    hit1_rate = sum(r["hit1"] for r in results) / n
    hit3_rate = sum(r["hit3"] for r in results) / n
    hit5_rate = sum(r["hit5"] for r in results) / n
    mrr       = sum(r["rr"]   for r in results) / n

    return {
        "exp_id":    exp_id,
        "label":     label,
        "n":         n,
        "hit1":      round(hit1_rate, 3),
        "hit3":      round(hit3_rate, 3),
        "hit5":      round(hit5_rate, 3),
        "mrr":       round(mrr, 3),
        "results":   results,
        "cat_hits":  {cat: round(sum(hits)/len(hits), 3) for cat, hits in cat_hits.items()},
    }


def save_detail_md(exp_result: dict, filepath: Path):
    """질문별 상세 결과를 마크다운으로 저장."""
    label   = exp_result["label"]
    results = exp_result["results"]

    lines = [
        f"# QA 쌍 평가 - 질문별 상세 결과",
        f"> 실험: {label}  |  질문 수: {len(results)}개",
        "",
        "| # | 질문 | 카테고리 | Hit@1 | Hit@3 | Hit@5 | Rank | RR | Top1 Score |",
        "|---|------|---------|:-----:|:-----:|:-----:|:----:|:--:|:----------:|",
    ]
    for r in results:
        rank_str = str(r["rank"]) if r["rank"] else "-"
        lines.append(
            f"| {r['qa_id']} "
            f"| {r['question'][:45]} "
            f"| {r['category']} "
            f"| {'O' if r['hit1'] else 'X'} "
            f"| {'O' if r['hit3'] else 'X'} "
            f"| {'O' if r['hit5'] else 'X'} "
            f"| {rank_str} "
            f"| {r['rr']:.3f} "
            f"| {r['top1_score']:.3f} |"
        )

    lines += [
        "",
        "## 실패 케이스 (Hit@5 = X)",
        "",
    ]
    failed = [r for r in results if not r["hit5"]]
    if failed:
        for r in failed:
            lines.append(f"- **Q{r['qa_id']}** `{r['question']}`")
            lines.append(f"  - 카테고리: {r['category']}")
            lines.append(f"  - 정답 chunk_id: {r['answer_id']}")
            lines.append(f"  - 검색 결과 ids: {r['result_ids']}")
            lines.append("")
    else:
        lines.append("*모든 질문 Hit@5 성공*")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장 완료: {filepath.name}")


def save_summary_md(exp_results: list[dict], qa_pairs: list[dict], filepath: Path):
    """전체 실험 비교 요약 마크다운 저장."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n   = len(qa_pairs)

    # 카테고리 목록 수집
    categories = sorted({qa["category"] for qa in qa_pairs})

    lines = [
        "# QA 쌍 기반 RAG 평가 결과",
        f"> 생성 일시: {now}  |  QA 쌍: {n}개  |  Top-K: {TOP_K}",
        "",
        "## 1. 평가 방법",
        "",
        "| 항목 | 내용 |",
        "|------|------|",
        "| 데이터셋 | Supabase documents_v2에서 카테고리별 균등 샘플링 |",
        "| 질문 생성 | GPT-4o-mini — 청크가 직접 답할 수 있는 질문 생성 |",
        "| 정답 기준 | 정답 chunk_id가 Top-K 결과에 포함되면 성공 |",
        "| 비교 실험 | QA-E4(e5-small + documents) vs QA-E5(bge-m3 + documents_v2) |",
        "",
        "## 2. 전체 비교 결과",
        "",
        "| 실험 | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|------|------:|------:|------:|----:|",
    ]
    for er in exp_results:
        lines.append(
            f"| {er['label']} "
            f"| {er['hit1']:.3f} "
            f"| {er['hit3']:.3f} "
            f"| {er['hit5']:.3f} "
            f"| {er['mrr']:.3f} |"
        )

    lines += [
        "",
        "## 3. 카테고리별 Hit@5",
        "",
        "| 카테고리 | " + " | ".join(er["exp_id"] for er in exp_results) + " |",
        "|---------|" + "|".join("------:" for _ in exp_results) + "|",
    ]
    for cat in categories:
        row = f"| {cat} "
        for er in exp_results:
            val = er["cat_hits"].get(cat, None)
            row += f"| {val:.3f} " if val is not None else "| - "
        row += "|"
        lines.append(row)

    # 결론
    if len(exp_results) == 2:
        e4, e5 = exp_results[0], exp_results[1]
        diff_hit5 = e5["hit5"] - e4["hit5"]
        diff_mrr  = e5["mrr"]  - e4["mrr"]
        lines += [
            "",
            "## 4. 분석 및 결론",
            "",
            f"- **Hit@5 변화**: {e4['hit5']:.3f} → {e5['hit5']:.3f} ({diff_hit5:+.3f})",
            f"- **MRR 변화**: {e4['mrr']:.3f} → {e5['mrr']:.3f} ({diff_mrr:+.3f})",
            "",
        ]
        if diff_hit5 > 0.05:
            lines.append("bge-m3 임베딩 + 200토큰 청킹이 e5-small 대비 검색 품질을 크게 향상시킴.")
            lines.append("키워드 매칭 방식의 실험 5 결과(Hit@5=0.941)를 정답 기반으로도 확인.")
        elif abs(diff_hit5) <= 0.05:
            lines.append("두 설정의 Hit@5 차이가 크지 않음 (±5% 이내).")
        else:
            lines.append("e5-small이 일부 영역에서 더 나은 결과를 보임.")

    lines += [
        "",
        "## 5. 멘토 질문 대비 답변",
        "",
        "> \"QA 평가를 어떻게 했나요?\"",
        "",
        "> DB에 저장된 청크를 카테고리별로 균등 샘플링해서 총 50개를 뽑았습니다.",
        "> 각 청크에 대해 GPT-4o-mini로 '이 청크가 답할 수 있는 질문'을 생성해서",
        "> (질문, 정답_청크_ID) 쌍의 데이터셋을 만들었습니다.",
        "> 이 데이터셋으로 RAG 검색을 실행하고, 정답 청크가 Top-K 안에 포함됐는지로",
        "> Hit Rate와 MRR을 계산했습니다.",
        "> 키워드 매칭과 달리 정확한 정답이 있어서 평가 신뢰도가 더 높습니다.",
    ]

    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장 완료: {filepath.name}")


def main():
    print("=" * 60)
    print("QA 쌍 기반 RAG 평가")
    print(f"QA 데이터셋: {QA_FILE}")
    print("=" * 60)

    if not QA_FILE.exists():
        print(f"\n[ERROR] QA 데이터셋 파일 없음: {QA_FILE}")
        print("먼저 python src/generate_qa_dataset.py 를 실행하세요.")
        return

    qa_pairs = json.loads(QA_FILE.read_text(encoding="utf-8"))
    print(f"\nQA 쌍 로드 완료: {len(qa_pairs)}개\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_results = []

    for exp in QA_EXPERIMENTS:
        result = evaluate_experiment(qa_pairs, exp)
        exp_results.append(result)

        # 질문별 상세 저장
        detail_path = RESULTS_DIR / f"QA쌍_질문별상세_{exp['id']}_{ts}.md"
        save_detail_md(result, detail_path)

    # 요약 저장
    summary_path = RESULTS_DIR / "QA쌍_평가결과.md"
    save_summary_md(exp_results, qa_pairs, summary_path)

    # 콘솔 출력
    print("\n" + "=" * 60)
    print("최종 결과 요약")
    print("=" * 60)
    print(f"{'실험':<30} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6}")
    print("-" * 60)
    for er in exp_results:
        print(f"{er['label']:<30} {er['hit1']:>6.3f} {er['hit3']:>6.3f} {er['hit5']:>6.3f} {er['mrr']:>6.3f}")

    print(f"\n결과 저장 완료:")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
