"""
실험 10: HyDE (Hypothetical Document Embeddings) 적용 효과 검증
==============================================================
실험 9와의 차이:
  - 검색 방식: 질문 임베딩 → 가상 답변 임베딩으로 검색 (HyDE)
  - BM25는 원본 질문 그대로 사용
  - DB 변경 없음, 파이프라인 코드만 수정

HyDE 원리:
  질문 → LLM → 가상 답변(논문 스타일) → 임베딩 → 벡터 검색
  → 논문 청크와 표현이 비슷해져 매칭률 향상 기대

평가 지표:
  1. 검색 품질: Hit@1/3/5, MRR (실험 9 대비 개선 여부)
  2. 답변 품질: LLM-as-Judge (0~3점)
  3. 응답 시간 (HyDE 오버헤드 포함)

실행:
  python src/evaluate_exp10_hyde.py

결과: results/model_comparison/exp10_*.json/md
"""

import warnings; warnings.filterwarnings("ignore")
import json, os, re, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from supabase import create_client, Client
from openai import OpenAI

BASE_DIR    = Path(__file__).resolve().parent.parent
QA_FILE     = BASE_DIR / "data" / "generated" / "qa_dataset.json"
RESULTS_DIR = BASE_DIR / "results" / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

TOP_K        = 5
RERANK_TOP_K = 3
MAX_CONTENT  = 2000
MAX_CONTEXT  = 4000

MODELS = [
    {"key": "gpt-4o-mini",  "label": "GPT-4o-mini",          "model_id": "gpt-4o-mini",                   "client_type": "openai"},
    {"key": "groq-llama",   "label": "Llama 3.3 70B (Groq)",  "model_id": "llama-3.3-70b-versatile",       "client_type": "groq"},
    {"key": "gemini-flash", "label": "Gemini Flash Lite",     "model_id": "models/gemini-flash-lite-latest","client_type": "gemini"},
]

SYSTEM_PROMPT = """당신은 건강식품과 다이어트 전문 상담사입니다.
[참고 자료]를 바탕으로 한국어로 간결하고 정확하게 답변하세요.
자료에 관련 내용이 없으면 솔직하게 "보유한 자료에서 찾을 수 없습니다"라고 하세요."""

HYDE_PROMPT = """다음 질문에 대해 건강식품/다이어트 분야 논문이나 전문 자료에서 발췌한 것처럼 2~3문장으로 답변을 작성하세요.
정확하지 않아도 됩니다. 검색 품질 향상을 위한 가상 문서입니다. 영어와 한국어를 혼용해도 됩니다.

질문: {query}

가상 답변:"""

# ── 클라이언트 초기화 ────────────────────────────────────────────────
def make_client(cfg: dict):
    if cfg["client_type"] == "openai":
        return OpenAI(api_key=OPENAI_API_KEY)
    elif cfg["client_type"] == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    elif cfg["client_type"] == "gemini":
        return OpenAI(api_key=GOOGLE_API_KEY,
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def generate_answer(client, cfg: dict, system: str, user: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=cfg["model_id"],
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            max_tokens=800, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[오류] {e}"

# ── 모델 로딩 ────────────────────────────────────────────────────────
print("모델 로딩 중...")
embed_model  = SentenceTransformer("BAAI/bge-m3")
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
judge_client = OpenAI(api_key=OPENAI_API_KEY)
print("로드 완료\n")

def get_embedding(text: str) -> list[float]:
    return embed_model.encode(text[:MAX_CONTENT], normalize_embeddings=True).tolist()

# ── HyDE: 가상 문서 생성 ─────────────────────────────────────────────
def generate_hypothetical_doc(query: str) -> str:
    """질문에 대해 논문 스타일의 가상 답변을 생성 (HyDE 핵심)"""
    try:
        resp = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
            max_tokens=200, temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return query  # 실패 시 원본 쿼리로 폴백

# ── HyDE 적용 검색 ───────────────────────────────────────────────────
def hybrid_search_hyde(query: str) -> tuple[list[dict], str]:
    """
    HyDE 적용:
    - 벡터 검색: 가상 문서(hypothetical doc) 임베딩 사용
    - BM25 키워드 검색: 원본 질문 그대로 사용
    """
    hypo_doc = generate_hypothetical_doc(query)
    emb = get_embedding(hypo_doc)  # 가상 문서로 임베딩
    try:
        resp = supabase.rpc("hybrid_search_v2", {
            "query_embedding": emb,
            "query_text": query,       # BM25는 원본 쿼리
            "match_count": TOP_K,
            "vector_weight": 0.7,
            "text_weight": 0.3,
            "filter_category": None,
        }).execute()
        return resp.data or [], hypo_doc
    except:
        return [], hypo_doc

def rerank(query: str, docs: list[dict]) -> list[dict]:
    if not docs: return []
    pairs  = [(query, doc.get("content", "")[:1000]) for doc in docs]
    scores = rerank_model.predict(pairs)
    for doc, s in zip(docs, scores):
        doc["rerank_score"] = float(s)
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_TOP_K]

def build_context(docs: list[dict]) -> str:
    parts = [f"[자료 {i+1}]\n{doc.get('content', '')[:1500]}" for i, doc in enumerate(docs)]
    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT]

# ── 검색 지표 계산 ───────────────────────────────────────────────────
def calc_retrieval_metrics(answer_chunk_id: int, raw_docs: list[dict]) -> dict:
    ids  = [doc.get("id") for doc in raw_docs]
    hit1 = int(answer_chunk_id in ids[:1])
    hit3 = int(answer_chunk_id in ids[:3])
    hit5 = int(answer_chunk_id in ids[:5])
    try:
        rank = ids.index(answer_chunk_id) + 1
        rr   = 1.0 / rank
    except ValueError:
        rank, rr = None, 0.0
    return {"hit1": hit1, "hit3": hit3, "hit5": hit5, "rr": rr, "rank": rank}

# ── LLM-as-Judge ────────────────────────────────────────────────────
def judge_answer(question: str, source_chunk: str, generated: str) -> dict:
    prompt = f"""다음 질문에 대한 답변을 평가하세요.

[질문]
{question}

[정답 기준 (이 청크에서 질문이 만들어짐)]
{source_chunk[:600]}

[생성된 답변]
{generated[:500]}

점수 기준:
0점: 완전히 틀리거나 "모릅니다" 응답
1점: 핵심 내용 일부만 포함 (중요한 정보 누락)
2점: 대부분 정확하나 약간 불완전
3점: 정확하고 완전한 답변

JSON으로만 응답: {{"score": 숫자, "reason": "한 줄 이유"}}"""
    try:
        resp = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100, temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return {"score": max(0, min(3, int(result.get("score", 0)))),
                "reason": result.get("reason", "")}
    except Exception as e:
        return {"score": -1, "reason": f"judge error: {e}"}

# ── 단일 모델 평가 ───────────────────────────────────────────────────
def evaluate_model(qa_pairs: list[dict], cfg: dict) -> dict:
    label  = cfg["label"]
    client = make_client(cfg)
    print(f"\n{'='*55}")
    print(f"  {label} ({len(qa_pairs)}문항) + HyDE")
    print(f"{'='*55}")

    results  = []
    cat_data = defaultdict(lambda: {"quality": [], "hit5": [], "rr": []})

    for i, qa in enumerate(qa_pairs, 1):
        q     = qa["question"]
        cid   = qa["answer_chunk_id"]
        chunk = qa.get("answer_chunk_content", "")
        cat   = qa["category"]

        print(f"  [{i:02d}/{len(qa_pairs)}] {q[:50]}...")
        t0 = time.time()

        # HyDE 검색
        raw_docs, hypo_doc = hybrid_search_hyde(q)
        ranked  = rerank(q, raw_docs)
        ret_m   = calc_retrieval_metrics(cid, raw_docs)

        # 답변 생성 (원본 쿼리 기준)
        context  = build_context(ranked)
        user_msg = f"질문: {q}\n\n[참고 자료]\n{context}\n\n자료를 바탕으로 답변해주세요."
        answer   = generate_answer(client, cfg, SYSTEM_PROMPT, user_msg)
        elapsed  = time.time() - t0

        # Judge
        jd = judge_answer(q, chunk, answer)

        print(f"    Hit@5={'O' if ret_m['hit5'] else 'X'} | 품질={jd['score']}/3 | {elapsed:.1f}s")

        row = {
            "id": qa["id"], "question": q, "category": cat,
            "answer_chunk_id": cid,
            "hypothetical_doc": hypo_doc,
            **ret_m,
            "quality_score":  jd["score"],
            "judge_reason":   jd["reason"],
            "generated":      answer,
            "total_sec":      round(elapsed, 2),
        }
        results.append(row)
        cat_data[cat]["quality"].append(jd["score"])
        cat_data[cat]["hit5"].append(ret_m["hit5"])
        cat_data[cat]["rr"].append(ret_m["rr"])

    # 집계
    valid = [r for r in results if r["quality_score"] >= 0]
    n     = len(valid)
    qdist = defaultdict(int)
    for r in valid: qdist[r["quality_score"]] += 1

    summary = {
        "model":        cfg["key"],
        "label":        label,
        "experiment":   "exp10_hyde",
        "n_valid":      n,
        "avg_quality":  round(sum(r["quality_score"] for r in valid) / n, 3),
        "quality_dist": dict(qdist),
        "hit1":  round(sum(r["hit1"] for r in results) / len(results), 3),
        "hit3":  round(sum(r["hit3"] for r in results) / len(results), 3),
        "hit5":  round(sum(r["hit5"] for r in results) / len(results), 3),
        "mrr":   round(sum(r["rr"]   for r in results) / len(results), 3),
        "avg_time": round(sum(r["total_sec"] for r in results) / len(results), 2),
        "cat_quality": {c: round(sum(v["quality"])/len(v["quality"]), 3) for c, v in cat_data.items()},
        "cat_hit5":    {c: round(sum(v["hit5"])/len(v["hit5"]), 3)    for c, v in cat_data.items()},
        "results": results,
    }
    return summary

# ── 결과 저장 ────────────────────────────────────────────────────────
def save_results(summary: dict, ts: str):
    key   = summary["model"].replace("-", "_")
    n     = summary["n_valid"]
    qdist = summary["quality_dist"]

    jp = RESULTS_DIR / f"exp10_{key}_{ts}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    mp = RESULTS_DIR / f"exp10_{key}_{ts}.md"
    lines = [
        f"# 실험 10: HyDE 평가 - {summary['label']}",
        f"> 평가 일시: {ts}  |  문항: {n}개  |  검색: HyDE (가상 문서 임베딩)",
        "",
        "## 1. 검색 품질 (Retrieval - HyDE 적용)",
        "",
        "| 지표 | 값 | 실험 9 대비 |",
        "|------|-----|-----------|",
        f"| Hit@1 | {summary['hit1']:.3f} | - |",
        f"| Hit@3 | {summary['hit3']:.3f} | - |",
        f"| Hit@5 | {summary['hit5']:.3f} | - |",
        f"| MRR   | {summary['mrr']:.3f} | - |",
        "",
        "## 2. 답변 품질 (LLM-as-Judge)",
        "",
        "| 지표 | 값 |",
        "|------|-----|",
        f"| 평균 품질 (0~3) | {summary['avg_quality']:.3f} |",
        f"| 완전정답 (3점)  | {qdist.get(3,0)}개 ({qdist.get(3,0)/n*100:.1f}%) |",
        f"| 대부분정확 (2점)| {qdist.get(2,0)}개 ({qdist.get(2,0)/n*100:.1f}%) |",
        f"| 부분정답 (1점)  | {qdist.get(1,0)}개 ({qdist.get(1,0)/n*100:.1f}%) |",
        f"| 오답 (0점)      | {qdist.get(0,0)}개 ({qdist.get(0,0)/n*100:.1f}%) |",
        f"| 평균 응답 시간  | {summary['avg_time']}초 |",
        "",
        "## 3. 카테고리별 성능",
        "",
        "| 카테고리 | 평균 품질 | Hit@5 |",
        "|---------|:--------:|:-----:|",
    ]
    for cat in sorted(summary["cat_quality"]):
        lines.append(f"| {cat} | {summary['cat_quality'][cat]:.3f} | {summary['cat_hit5'].get(cat,0):.3f} |")

    lines += ["", "## 4. 질문별 상세", "",
              "| # | 질문 | 카테고리 | Hit@5 | 품질 | Rank |",
              "|---|------|---------|:-----:|:----:|:----:|"]
    for r in summary["results"]:
        rank_s = str(r["rank"]) if r["rank"] else "-"
        lines.append(f"| {r['id']} | {r['question'][:38]} | {r['category']} "
                     f"| {'O' if r['hit5'] else 'X'} | {r['quality_score']} | {rank_s} |")

    mp.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장: {jp.name}")
    print(f"  저장: {mp.name}")
    return summary

# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  실험 10: HyDE 적용 3개 모델 비교 평가")
    print("=" * 55)
    print("  HyDE: 가상 문서 임베딩으로 검색 품질 개선 실험")
    print("=" * 55)

    qa_pairs = json.loads(QA_FILE.read_text(encoding="utf-8"))
    print(f"QA 데이터: {len(qa_pairs)}개 (documents_v2 내부 생성)\n")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_sums = []

    for cfg in MODELS:
        summary = evaluate_model(qa_pairs, cfg)
        saved   = save_results(summary, ts)
        all_sums.append(saved)

    # 최종 비교 출력
    print("\n" + "=" * 55)
    print("  실험 10 (HyDE) 최종 비교")
    print("=" * 55)
    print(f"{'모델':<22} {'품질':>6} {'완전정답':>8} {'Hit@5':>6} {'MRR':>6} {'시간':>6}")
    print("-" * 55)
    for s in all_sums:
        perf = s["quality_dist"].get(3, 0) / s["n_valid"] * 100
        print(f"{s['label']:<22} {s['avg_quality']:>6.3f} {perf:>7.1f}% "
              f"{s['hit5']:>6.3f} {s['mrr']:>6.3f} {s['avg_time']:>5.1f}s")

    print("\n[실험 9 기준값 (HyDE 미적용)]")
    print(f"{'GPT-4o-mini':<22} {'1.048':>6} {'19.0%':>8} {'0.095':>6} {'0.048':>6} {'4.2s':>6}")
    print(f"{'Llama 3.3 70B (Groq)':<22} {'0.595':>6} {'11.9%':>8} {'0.095':>6} {'0.048':>6} {'2.6s':>6}")
    print(f"{'Gemini Flash Lite':<22} {'1.262':>6} {'19.0%':>8} {'0.095':>6} {'0.048':>6} {'7.8s':>6}")

    # 타임스탬프 저장 (차트 생성용)
    ts_file = RESULTS_DIR / "exp10_timestamp.txt"
    ts_file.write_text(ts, encoding="utf-8")
    print(f"\n타임스탬프 저장: {ts}")

if __name__ == "__main__":
    main()
