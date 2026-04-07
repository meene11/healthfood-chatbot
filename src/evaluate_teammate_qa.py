"""
조원 QA 데이터 기반 챗봇 평가 스크립트 (다중 모델 비교 지원)
=============================================================
[실험 8] 모델 비교: GPT-4o-mini vs Llama 3.3 70B vs Gemini 1.5 Flash

평가 지표:
  1. 검색 커버리지 (Retrieval Coverage, 0~1)
  2. 답변 품질 (Answer Quality, 0~3) — LLM-as-Judge
  3. 응답 시간 (Response Time, 초)

실행:
  python src/evaluate_teammate_qa.py                          # gpt-4o-mini (기본)
  python src/evaluate_teammate_qa.py --model groq-llama       # Llama 3.3 70B (Groq)
  python src/evaluate_teammate_qa.py --model gemini-flash     # Gemini 1.5 Flash
  python src/evaluate_teammate_qa.py --n 30                   # 처음 30개만

모델 비교 결과는 results/model_comparison/ 디렉토리에 저장됨.
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from supabase import create_client, Client

BASE_DIR   = Path(__file__).resolve().parent.parent
QA_FILE    = BASE_DIR / "data" / "generated" / "teammate_qa_adapted.json"
RESULTS_DIR = BASE_DIR / "results" / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── 모델 설정 ────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "provider":   "openai",
        "model_id":   "gpt-4o-mini",
        "label":      "GPT-4o-mini (현재 모델)",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "gpt-4o": {
        "provider":   "openai",
        "model_id":   "gpt-4o",
        "label":      "GPT-4o (고성능 모델)",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "claude-haiku": {
        "provider":   "anthropic",
        "model_id":   "claude-haiku-4-5-20251001",
        "label":      "Claude Haiku 4.5 (Anthropic)",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "groq-llama": {
        "provider":   "groq",
        "model_id":   "llama-3.3-70b-versatile",
        "label":      "Llama 3.3 70B (Groq 무료)",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "gemini-flash": {
        "provider":   "gemini",
        "model_id":   "models/gemini-flash-lite-latest",
        "label":      "Gemini Flash Lite (Google 무료)",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
}

# ── RAG 설정 ─────────────────────────────────────────────────────────
TOP_K        = 10
RERANK_TOP_K = 3
MAX_CONTENT  = 2000
MAX_CONTEXT_CHARS = 4000

SYSTEM_PROMPT = """당신은 건강식품과 다이어트 전문 상담사입니다.
[참고 자료]에 있는 내용을 바탕으로 한국어로 간결하고 정확하게 답변하세요.
자료에 관련 내용이 없으면 솔직하게 "보유한 자료에서 찾을 수 없습니다"라고 하세요.
출처는 [자료 N] 형태로 표시하세요."""


def build_llm_client(model_name: str):
    """모델에 따라 LLM 클라이언트 반환"""
    cfg = MODEL_CONFIGS[model_name]

    if cfg["provider"] == "openai":
        from openai import OpenAI
        return OpenAI(api_key=OPENAI_API_KEY)

    elif cfg["provider"] == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 .env에 없습니다.")
        return anthropic.Anthropic(api_key=api_key)

    elif cfg["provider"] == "groq":
        from openai import OpenAI
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            raise ValueError("GROQ_API_KEY가 .env에 없습니다.")
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

    elif cfg["provider"] == "gemini":
        from openai import OpenAI
        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if not google_key:
            raise ValueError("GOOGLE_API_KEY가 .env에 없습니다.")
        # Google의 OpenAI 호환 엔드포인트 사용 (deprecated google-generativeai 불필요)
        return OpenAI(
            api_key=google_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    raise ValueError(f"알 수 없는 provider: {cfg['provider']}")


def generate_answer(llm_client, model_name: str, system_msg: str, user_msg: str) -> str:
    """모델별 답변 생성 (비스트리밍)"""
    cfg = MODEL_CONFIGS[model_name]

    if cfg["provider"] in ("openai", "groq"):
        resp = llm_client.chat.completions.create(
            model=cfg["model_id"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        return resp.choices[0].message.content.strip()

    elif cfg["provider"] == "anthropic":
        resp = llm_client.messages.create(
            model=cfg["model_id"],
            max_tokens=cfg["max_tokens"],
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()

    elif cfg["provider"] == "gemini":
        # OpenAI 호환 엔드포인트 — openai 클라이언트 그대로 사용
        resp = llm_client.chat.completions.create(
            model=cfg["model_id"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
        )
        return resp.choices[0].message.content.strip()

    return ""


# ── 임베딩 / 검색 ────────────────────────────────────────────────────
print("임베딩 + 리랭킹 모델 로딩 중...")
embed_model  = SentenceTransformer("BAAI/bge-m3")
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("모델 로드 완료")


def get_embedding(text: str) -> list[float]:
    return embed_model.encode(text[:MAX_CONTENT], normalize_embeddings=True).tolist()


def hybrid_search(query: str) -> list[dict]:
    emb = get_embedding(query)
    try:
        resp = supabase.rpc("hybrid_search_v2", {
            "query_embedding": emb,
            "query_text":      query,
            "match_count":     TOP_K,
            "vector_weight":   0.7,
            "text_weight":     0.3,
            "filter_category": None,
        }).execute()
        return resp.data or []
    except Exception as e:
        print(f"    [WARN] 검색 실패: {e}")
        return []


def rerank(query: str, docs: list[dict]) -> list[dict]:
    if not docs:
        return []
    pairs  = [(query, doc.get("content", "")[:1000]) for doc in docs]
    scores = rerank_model.predict(pairs)
    for doc, s in zip(docs, scores):
        doc["rerank_score"] = float(s)
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_TOP_K]


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.get("content", "")
        parts.append(f"[자료 {i}]\n{content[:1500]}")
    ctx = "\n\n---\n\n".join(parts)
    return ctx[:MAX_CONTEXT_CHARS]


# ── 검색 커버리지 계산 ────────────────────────────────────────────────
def extract_keywords(text: str) -> set[str]:
    """source_content에서 2글자 이상 핵심 단어 추출"""
    # 영어 단어 + 한글 단어 추출 (불용어 제외)
    stopwords = {
        "the", "and", "for", "that", "this", "with", "are", "was", "from",
        "have", "has", "been", "not", "doi", "et", "al", "in", "of", "to",
        "a", "an", "is", "it", "or", "as", "at", "by", "on",
    }
    words = re.findall(r"[a-zA-Z]{3,}|[가-힣]{2,}", text.lower())
    return {w for w in words if w not in stopwords and len(w) >= 2}


def retrieval_coverage(source_content: str, retrieved_docs: list[dict]) -> float:
    """검색된 문서가 source_content 키워드를 얼마나 커버하는지 (0~1)"""
    ref_kw = extract_keywords(source_content)
    if not ref_kw:
        return 0.0

    retrieved_text = " ".join(doc.get("content", "") for doc in retrieved_docs).lower()
    matched = sum(1 for kw in ref_kw if kw in retrieved_text)
    return round(matched / len(ref_kw), 3)


# ── LLM-as-Judge ────────────────────────────────────────────────────
def judge_answer(
    llm_client,
    judge_model: str,
    question: str,
    reference: str,
    generated: str,
) -> dict:
    """GPT-4o-mini로 생성된 답변을 0~3점으로 평가"""
    prompt = f"""다음 질문에 대한 답변을 평가해주세요.

[질문]
{question}

[정답 (기준 답변)]
{reference}

[생성된 답변]
{generated}

위 생성된 답변을 아래 기준으로 0~3점 사이로 평가하고, 이유를 한 줄로 설명하세요.

점수 기준:
0점: 관련 없거나 틀린 답변 / "모릅니다" 응답
1점: 핵심 내용을 일부만 포함 (중요한 정보 누락)
2점: 대부분 정확하나 약간 불완전하거나 불필요한 내용 포함
3점: 정확하고 완전한 답변 (정답과 동등하거나 더 좋음)

응답 형식 (JSON만):
{{"score": 2, "reason": "주요 내용은 맞지만 수치 정보가 누락됨"}}"""

    try:
        cfg = MODEL_CONFIGS[judge_model]
        if cfg["provider"] in ("openai", "groq"):
            resp = llm_client.chat.completions.create(
                model=cfg["model_id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
        elif cfg["provider"] == "anthropic":
            resp = llm_client.messages.create(
                model=cfg["model_id"],
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()

        elif cfg["provider"] == "gemini":
            resp = llm_client.chat.completions.create(
                model=cfg["model_id"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            import re as _re
            m = _re.search(r'\{.*?\}', raw, _re.DOTALL)
            raw = m.group(0) if m else raw

        result = json.loads(raw)
        score  = int(result.get("score", 0))
        reason = result.get("reason", "")
        return {"score": max(0, min(3, score)), "reason": reason}

    except Exception as e:
        return {"score": -1, "reason": f"judge error: {e}"}


# ── 단일 QA 평가 ─────────────────────────────────────────────────────
def evaluate_one(
    qa: dict,
    llm_client,
    model_name: str,
    judge_client,
    judge_model: str,
    idx: int,
    total: int,
) -> dict:
    question  = qa["question"]
    reference = qa["reference_answer"]
    source_ct = qa["source_content"]
    category  = qa["category"]

    print(f"  [{idx:02d}/{total}] {question[:55]}...")

    t0 = time.time()

    # 1. 검색 + 리랭킹
    raw_docs     = hybrid_search(question)
    ranked_docs  = rerank(question, raw_docs)
    retrieval_t  = time.time() - t0

    # 2. 검색 커버리지
    coverage = retrieval_coverage(source_ct, ranked_docs)

    # 3. 답변 생성
    context  = build_context(ranked_docs)
    user_msg = (
        f"질문: {question}\n\n"
        f"[참고 자료]\n{context}\n\n"
        "위 자료를 바탕으로 간결하게 답변해주세요."
    )
    t1 = time.time()
    try:
        answer = generate_answer(llm_client, model_name, SYSTEM_PROMPT, user_msg)
    except Exception as e:
        answer = f"[생성 오류] {e}"
    answer_t = time.time() - t1
    total_t  = time.time() - t0

    # 4. LLM-as-Judge
    judge_result = judge_answer(judge_client, judge_model, question, reference, answer)

    print(f"    → 커버리지={coverage:.2f} | 품질={judge_result['score']}/3 | {total_t:.1f}s")

    return {
        "id":            qa["id"],
        "question":      question,
        "category":      category,
        "reference":     reference,
        "generated":     answer,
        "coverage":      coverage,
        "quality_score": judge_result["score"],
        "judge_reason":  judge_result["reason"],
        "n_retrieved":   len(raw_docs),
        "n_reranked":    len(ranked_docs),
        "retrieval_sec": round(retrieval_t, 2),
        "answer_sec":    round(answer_t, 2),
        "total_sec":     round(total_t, 2),
    }


# ── 결과 저장 ────────────────────────────────────────────────────────
def save_results(results: list[dict], model_name: str, ts: str):
    cfg   = MODEL_CONFIGS[model_name]
    label = cfg["label"]
    n     = len(results)

    valid = [r for r in results if r["quality_score"] >= 0]

    avg_coverage = round(sum(r["coverage"] for r in valid) / len(valid), 3) if valid else 0
    avg_quality  = round(sum(r["quality_score"] for r in valid) / len(valid), 3) if valid else 0
    avg_time     = round(sum(r["total_sec"] for r in valid) / len(valid), 2) if valid else 0

    # 품질 분포
    dist = defaultdict(int)
    for r in valid:
        dist[r["quality_score"]] += 1
    pct3 = round(dist[3] / len(valid) * 100, 1) if valid else 0  # 완전 정답 비율

    # 카테고리별 품질
    cat_q = defaultdict(list)
    for r in valid:
        cat_q[r["category"]].append(r["quality_score"])
    cat_avg = {c: round(sum(v)/len(v), 3) for c, v in cat_q.items()}

    # JSON 저장
    json_path = RESULTS_DIR / f"eval_{model_name.replace('-','_')}_{ts}.json"
    json_path.write_text(
        json.dumps({
            "model":       model_name,
            "label":       label,
            "timestamp":   ts,
            "n_total":     n,
            "n_valid":     len(valid),
            "avg_coverage": avg_coverage,
            "avg_quality":  avg_quality,
            "avg_time":     avg_time,
            "quality_dist": dict(dist),
            "cat_avg":      cat_avg,
            "results":      results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 마크다운 저장
    md_path = RESULTS_DIR / f"eval_{model_name.replace('-','_')}_{ts}.md"
    lines = [
        f"# 조원 QA 평가 결과 — {label}",
        f"> 평가 일시: {ts}  |  평가 문항: {n}개  |  유효: {len(valid)}개",
        "",
        "## 1. 종합 지표",
        "",
        "| 지표 | 값 |",
        "|------|-----|",
        f"| 모델 | {label} |",
        f"| 평가 문항 수 | {n}개 |",
        f"| 평균 검색 커버리지 | {avg_coverage:.3f} |",
        f"| 평균 답변 품질 (0~3) | {avg_quality:.3f} |",
        f"| 완전 정답 비율 (3점) | {pct3}% ({dist[3]}/{len(valid)}) |",
        f"| 평균 응답 시간 | {avg_time:.2f}초 |",
        "",
        "## 2. 답변 품질 분포 (LLM-as-Judge 0~3점)",
        "",
        "| 점수 | 의미 | 문항 수 | 비율 |",
        "|:----:|------|:-------:|:----:|",
        f"| 3점 | 완전 정답 | {dist[3]} | {dist[3]/len(valid)*100:.1f}% |",
        f"| 2점 | 대부분 정확 | {dist[2]} | {dist[2]/len(valid)*100:.1f}% |",
        f"| 1점 | 부분 정답 | {dist[1]} | {dist[1]/len(valid)*100:.1f}% |",
        f"| 0점 | 오답/모름 | {dist[0]} | {dist[0]/len(valid)*100:.1f}% |",
        "",
        "## 3. 카테고리별 평균 품질",
        "",
        "| 카테고리 | 평균 품질 | 문항 수 |",
        "|---------|:--------:|:------:|",
    ]
    for cat, val in sorted(cat_avg.items()):
        cnt = len(cat_q[cat])
        lines.append(f"| {cat} | {val:.3f} | {cnt} |")

    lines += [
        "",
        "## 4. 질문별 상세 결과",
        "",
        "| # | 질문 | 카테고리 | 커버리지 | 품질 | 이유 |",
        "|---|------|---------|:-------:|:----:|------|",
    ]
    for r in results:
        q = r["question"][:40]
        score_emoji = ["❌", "⚠️", "🔶", "✅"][max(0, r["quality_score"])] if r["quality_score"] >= 0 else "?"
        lines.append(
            f"| {r['id']} | {q} | {r['category']} "
            f"| {r['coverage']:.2f} | {score_emoji} {r['quality_score']} "
            f"| {r['judge_reason'][:40]} |"
        )

    lines += ["", "## 5. 실패 케이스 (품질 0~1점)", ""]
    failed = [r for r in valid if r["quality_score"] <= 1]
    if failed:
        for r in failed:
            lines += [
                f"### Q{r['id']}: {r['question']}",
                f"- **정답**: {r['reference'][:200]}",
                f"- **생성 답변**: {r['generated'][:200]}",
                f"- **평가 이유**: {r['judge_reason']}",
                f"- **커버리지**: {r['coverage']}",
                "",
            ]
    else:
        lines.append("*0~1점 실패 케이스 없음*")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n  JSON: {json_path.name}")
    print(f"  MD:   {md_path.name}")

    return {
        "model":         model_name,
        "label":         label,
        "avg_coverage":  avg_coverage,
        "avg_quality":   avg_quality,
        "pct_perfect":   pct3,
        "avg_time":      avg_time,
        "quality_dist":  dict(dist),
        "cat_avg":       cat_avg,
        "n_valid":       len(valid),
    }


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="평가할 LLM 모델")
    parser.add_argument("--n", type=int, default=None,
                        help="평가할 질문 수 (기본: 전체)")
    parser.add_argument("--judge", default="gpt-4o-mini",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="LLM-as-Judge 모델 (기본: gpt-4o-mini)")
    args = parser.parse_args()

    model_name = args.model
    judge_name = args.judge
    cfg        = MODEL_CONFIGS[model_name]

    print("=" * 60)
    print(f"  조원 QA 평가 - {cfg['label']}")
    print("=" * 60)

    if not QA_FILE.exists():
        print(f"[ERROR] {QA_FILE} 없음. adapt_teammate_qa.py 먼저 실행하세요.")
        sys.exit(1)

    qa_pairs = json.loads(QA_FILE.read_text(encoding="utf-8"))
    if args.n:
        qa_pairs = qa_pairs[: args.n]

    print(f"평가 문항: {len(qa_pairs)}개 | Judge: {judge_name}")
    print()

    # LLM 클라이언트 초기화
    llm_client   = build_llm_client(model_name)
    # Judge는 항상 gpt-4o-mini (일관성 유지)
    judge_client = build_llm_client(judge_name) if judge_name != model_name else llm_client

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    total   = len(qa_pairs)

    for i, qa in enumerate(qa_pairs, 1):
        r = evaluate_one(qa, llm_client, model_name, judge_client, judge_name, i, total)
        results.append(r)

    summary = save_results(results, model_name, ts)

    print("\n" + "=" * 60)
    print("  최종 요약")
    print("=" * 60)
    print(f"  모델:         {summary['label']}")
    print(f"  검색 커버리지: {summary['avg_coverage']:.3f}")
    print(f"  답변 품질:    {summary['avg_quality']:.3f} / 3.0")
    print(f"  완전 정답:    {summary['pct_perfect']}%")
    print(f"  평균 응답 시간: {summary['avg_time']:.2f}초")


if __name__ == "__main__":
    main()
