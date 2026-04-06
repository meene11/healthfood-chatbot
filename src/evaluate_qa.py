"""
QA 쌍 기반 RAG 평가 스크립트
==============================
(질문, 정답_chunk_id) 쌍 데이터셋으로 Hit Rate / MRR 계산.
키워드 매칭이 아닌 정확한 chunk_id 기반 판정 → 더 신뢰도 높은 평가.

평가 대상:
  QA-E5: documents_v2 테이블 (bge-m3 1024차원, 실험 5 설정)

※ QA-E4(e5-small) 비교를 하지 않는 이유:
  QA 쌍은 documents_v2에서 생성하여 정답 chunk_id가 documents_v2 기준임.
  기존 hybrid_search는 parent-child 구조(parent_id 반환)를 사용하므로
  documents_v2의 chunk_id와 직접 비교가 불가능.
  대신 키워드 매칭 기반 실험 1~4와의 비교는 RAG_평가_실험_기록.md에 기술.

입력: data/generated/qa_dataset.json
출력:
  results/QA쌍_평가결과.md
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
from sentence_transformers import SentenceTransformer
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
        "id":           "QA-E5",
        "label":        "QA-E5: bge-m3 (documents_v2)",
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


def get_embed_model(model_name: str) -> SentenceTransformer:
    if model_name not in _embed_cache:
        print(f"임베딩 모델 로딩: {model_name}")
        _embed_cache[model_name] = SentenceTransformer(model_name)
    return _embed_cache[model_name]


def embed_query(query: str, model_name: str, prefix: str) -> list[float]:
    model = get_embed_model(model_name)
    text  = prefix + query
    vec   = model.encode(text[:MAX_CONTENT], normalize_embeddings=True)
    return vec.tolist()


def search(query: str, exp: dict) -> list[dict]:
    """하이브리드 검색 (hybrid_search_v2), Top-K 반환.
    ※ 리랭커 미사용: bge-m3 + 리랭커 동시 로딩 시 메모리 부족 가능성.
       combined_score 순으로 정렬된 결과 그대로 사용.
    """
    embedding = embed_query(query, exp["embed_model"], exp["embed_prefix"])

    try:
        resp = supabase.rpc(exp["rpc"], {
            "query_embedding": embedding,
            "query_text":      query,
            "match_count":     TOP_K,
            "vector_weight":   exp["vector_weight"],
            "text_weight":     exp["text_weight"],
        }).execute()
        docs = resp.data or []
    except Exception as e:
        print(f"    [ERROR] 검색 실패: {e}")
        return []

    return docs


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

        top1_score = docs[0].get("combined_score", 0.0) if docs else 0.0
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
    """평가 결과 요약 마크다운 저장."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n   = len(qa_pairs)
    er  = exp_results[0]  # QA-E5 단일 실험

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
        "| 정답 기준 | 정답 chunk_id가 Top-K 결과에 포함되면 성공 (chunk_id 일치) |",
        "| 임베딩 | BAAI/bge-m3 (1024차원) |",
        "| 검색 | hybrid_search_v2 (벡터 0.7 + BM25 0.3, combined_score 순 정렬) |",
        "",
        "> **참고**: QA 쌍은 documents_v2 기준으로 생성되어 chunk_id가 documents_v2와 일치함.",
        "> 기존 e5-small + documents 설정과의 비교는 키워드 매칭 방식(실험 1~4)으로 수행함.",
        "",
        "## 2. 평가 결과",
        "",
        "| 실험 | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|------|------:|------:|------:|----:|",
        f"| {er['label']} | {er['hit1']:.3f} | {er['hit3']:.3f} | {er['hit5']:.3f} | {er['mrr']:.3f} |",
        "",
        "## 3. 카테고리별 Hit@5",
        "",
        "| 카테고리 | Hit@5 | 샘플 수 |",
        "|---------|------:|-------:|",
    ]
    for cat in categories:
        val = er["cat_hits"].get(cat, None)
        cnt = sum(1 for qa in qa_pairs if qa["category"] == cat)
        val_str = f"{val:.3f}" if val is not None else "-"
        lines.append(f"| {cat} | {val_str} | {cnt} |")

    lines += [
        "",
        "## 4. 분석 및 결론",
        "",
        f"- **Hit@1**: {er['hit1']:.3f} — 검색 Top1에서 정답 청크 찾을 확률",
        f"- **Hit@3**: {er['hit3']:.3f} — 검색 Top3 이내 정답 청크 포함 확률",
        f"- **Hit@5**: {er['hit5']:.3f} — 검색 Top5 이내 정답 청크 포함 확률",
        f"- **MRR**:   {er['mrr']:.3f} — 정답 청크 평균 역순위",
        "",
    ]

    # 등급 판정 (MRR 기준)
    if er["mrr"] >= 0.8:
        grade = "**A (우수)** — 정답 기반 평가에서도 높은 검색 품질 확인"
    elif er["mrr"] >= 0.6:
        grade = "**B (양호)** — 대부분의 질문에서 정답 청크를 상위에서 찾음"
    elif er["mrr"] >= 0.4:
        grade = "**C (보통)** — 절반 이상에서 정답 청크를 찾지만 순위 개선 필요"
    else:
        grade = "**D (미흡)** — 정답 청크를 찾지 못하는 경우가 많음"
    lines.append(f"- **종합 등급**: {grade}")

    # 키워드 기반 결과와 비교
    lines += [
        "",
        "### 키워드 매칭 방식과 비교",
        "",
        "| 평가 방식 | Hit@5 | MRR | 특징 |",
        "|---------|------:|----:|------|",
        "| 키워드 매칭 (실험 5) | 0.941 | 0.828 | 17개 질문, 사전 정의 키워드 기준 |",
        f"| QA 쌍 (이번 평가) | {er['hit5']:.3f} | {er['mrr']:.3f} | {n}개 질문, 정확한 chunk_id 기준 |",
        "",
        "> QA 쌍 방식이 더 엄격한 평가 기준임 (정확한 chunk_id 일치 필요).",
        "> 키워드 매칭보다 낮거나 비슷한 수치가 정상적.",
        "",
        "## 5. 멘토 질문 대비 답변",
        "",
        "> **\"QA 평가를 어떻게 했나요?\"**",
        "",
        "> DB에 저장된 청크를 카테고리별로 균등 샘플링해서 총 42개를 뽑았습니다.",
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
