"""
실험 11: 쿼리 확장 (Query Expansion) 효과 검증
================================================
실험 9(기준)와의 차이:
  - 질문 1개 → LLM이 3가지 다른 표현으로 확장
  - 각 쿼리로 검색 → 결과 합산(Union, 중복 제거)
  - 합산 결과를 리랭커로 최종 Top3 선별
  - DB 변경 없음, HyDE 미적용

효과 기대:
  - 사용자 표현 다양성 커버
  - 더 많은 관련 청크 수집 → 리랭커가 최선 선별

실행: python src/evaluate_exp11_query_expansion.py
결과: results/model_comparison/exp11_*.json/md
"""

import warnings; warnings.filterwarnings("ignore")
import json, os, time
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

EXPANSION_PROMPT = """다음 질문을 검색에 유리하도록 3가지 다른 표현으로 바꿔주세요.
각각 다른 관점이나 용어를 사용하세요. 한국어와 영어를 혼용해도 됩니다.

질문: {query}

JSON으로만 응답: {{"queries": ["변형1", "변형2", "변형3"]}}"""

# ── 클라이언트 ───────────────────────────────────────────────────────
def make_client(cfg):
    if cfg["client_type"] == "openai":
        return OpenAI(api_key=OPENAI_API_KEY)
    elif cfg["client_type"] == "groq":
        return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    elif cfg["client_type"] == "gemini":
        return OpenAI(api_key=GOOGLE_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

def generate_answer(client, cfg, system, user):
    try:
        resp = client.chat.completions.create(
            model=cfg["model_id"],
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
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

def get_embedding(text):
    return embed_model.encode(text[:MAX_CONTENT], normalize_embeddings=True).tolist()

# ── 쿼리 확장 ────────────────────────────────────────────────────────
def expand_query(query: str) -> list[str]:
    """원본 쿼리를 3가지 변형으로 확장"""
    try:
        resp = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":EXPANSION_PROMPT.format(query=query)}],
            max_tokens=200, temperature=0.5,
            response_format={"type":"json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        variants = result.get("queries", [])
        return [query] + variants[:3]  # 원본 + 변형 3개 = 최대 4개
    except:
        return [query]  # 실패 시 원본만

def search_single(query: str) -> list[dict]:
    emb = get_embedding(query)
    try:
        resp = supabase.rpc("hybrid_search_v2", {
            "query_embedding": emb, "query_text": query,
            "match_count": TOP_K, "vector_weight": 0.7,
            "text_weight": 0.3, "filter_category": None,
        }).execute()
        return resp.data or []
    except:
        return []

def search_with_expansion(query: str) -> tuple[list[dict], list[str]]:
    """확장 쿼리로 검색 후 중복 제거하여 합산 반환"""
    queries   = expand_query(query)
    seen_ids  = set()
    all_docs  = []
    for q in queries:
        docs = search_single(q)
        for doc in docs:
            if doc.get("id") not in seen_ids:
                seen_ids.add(doc.get("id"))
                all_docs.append(doc)
    return all_docs, queries

def rerank(query, docs):
    if not docs: return []
    pairs  = [(query, doc.get("content","")[:1000]) for doc in docs]
    scores = rerank_model.predict(pairs)
    for doc, s in zip(docs, scores):
        doc["rerank_score"] = float(s)
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_TOP_K]

def build_context(docs):
    parts = [f"[자료 {i+1}]\n{doc.get('content','')[:1500]}" for i, doc in enumerate(docs)]
    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT]

def calc_retrieval_metrics(answer_chunk_id, raw_docs):
    ids  = [doc.get("id") for doc in raw_docs]
    hit1 = int(answer_chunk_id in ids[:1])
    hit3 = int(answer_chunk_id in ids[:3])
    hit5 = int(answer_chunk_id in ids[:5])
    try:
        rank = ids.index(answer_chunk_id) + 1
        rr   = 1.0 / rank
    except ValueError:
        rank, rr = None, 0.0
    return {"hit1":hit1,"hit3":hit3,"hit5":hit5,"rr":rr,"rank":rank}

def judge_answer(question, source_chunk, generated):
    prompt = f"""다음 질문에 대한 답변을 평가하세요.
[질문] {question}
[정답 기준] {source_chunk[:600]}
[생성된 답변] {generated[:500]}
점수: 0=완전틀림, 1=일부포함, 2=대부분정확, 3=완전정확
JSON: {{"score": 숫자, "reason": "한 줄"}}"""
    try:
        resp = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=100, temperature=0.0,
            response_format={"type":"json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return {"score": max(0,min(3,int(result.get("score",0)))), "reason": result.get("reason","")}
    except Exception as e:
        return {"score":-1, "reason":f"judge error: {e}"}

# ── 단일 모델 평가 ───────────────────────────────────────────────────
def evaluate_model(qa_pairs, cfg):
    label  = cfg["label"]
    client = make_client(cfg)
    print(f"\n{'='*55}\n  {label} + 쿼리확장\n{'='*55}")

    results  = []
    cat_data = defaultdict(lambda: {"quality":[],"hit5":[],"rr":[]})

    for i, qa in enumerate(qa_pairs, 1):
        q, cid, chunk, cat = qa["question"], qa["answer_chunk_id"], qa.get("answer_chunk_content",""), qa["category"]
        print(f"  [{i:02d}/{len(qa_pairs)}] {q[:50]}...")
        t0 = time.time()

        raw_docs, used_queries = search_with_expansion(q)
        ranked  = rerank(q, raw_docs)
        ret_m   = calc_retrieval_metrics(cid, raw_docs)

        context  = build_context(ranked)
        user_msg = f"질문: {q}\n\n[참고 자료]\n{context}\n\n자료를 바탕으로 답변해주세요."
        answer   = generate_answer(client, cfg, SYSTEM_PROMPT, user_msg)
        elapsed  = time.time() - t0

        jd = judge_answer(q, chunk, answer)
        print(f"    Hit@5={'O' if ret_m['hit5'] else 'X'} | 품질={jd['score']}/3 | {elapsed:.1f}s | 쿼리{len(used_queries)}개")

        row = {"id":qa["id"],"question":q,"category":cat,"answer_chunk_id":cid,
               "n_queries":len(used_queries),"n_total_docs":len(raw_docs),
               **ret_m,"quality_score":jd["score"],"judge_reason":jd["reason"],
               "generated":answer,"total_sec":round(elapsed,2)}
        results.append(row)
        cat_data[cat]["quality"].append(jd["score"])
        cat_data[cat]["hit5"].append(ret_m["hit5"])
        cat_data[cat]["rr"].append(ret_m["rr"])

    valid = [r for r in results if r["quality_score"] >= 0]
    n     = len(valid)
    qdist = defaultdict(int)
    for r in valid: qdist[r["quality_score"]] += 1

    summary = {
        "model": cfg["key"], "label": label, "experiment": "exp11_query_expansion",
        "n_valid": n,
        "avg_quality":  round(sum(r["quality_score"] for r in valid) / n, 3),
        "quality_dist": dict(qdist),
        "hit1": round(sum(r["hit1"] for r in results)/len(results),3),
        "hit3": round(sum(r["hit3"] for r in results)/len(results),3),
        "hit5": round(sum(r["hit5"] for r in results)/len(results),3),
        "mrr":  round(sum(r["rr"]   for r in results)/len(results),3),
        "avg_time": round(sum(r["total_sec"] for r in results)/len(results),2),
        "avg_n_docs": round(sum(r["n_total_docs"] for r in results)/len(results),1),
        "cat_quality": {c:round(sum(v["quality"])/len(v["quality"]),3) for c,v in cat_data.items()},
        "cat_hit5":    {c:round(sum(v["hit5"])/len(v["hit5"]),3) for c,v in cat_data.items()},
        "results": results,
    }
    return summary

def save_results(summary, ts):
    key   = summary["model"].replace("-","_")
    n     = summary["n_valid"]
    qdist = summary["quality_dist"]
    jp    = RESULTS_DIR / f"exp11_{key}_{ts}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    mp = RESULTS_DIR / f"exp11_{key}_{ts}.md"
    lines = [
        f"# 실험 11: 쿼리 확장 평가 - {summary['label']}",
        f"> 평가 일시: {ts}  |  문항: {n}개  |  검색: 쿼리 확장 (원본+변형 최대 4개)",
        "", "## 검색 품질", "",
        "| 지표 | 값 |", "|------|-----|",
        f"| Hit@1 | {summary['hit1']:.3f} |", f"| Hit@3 | {summary['hit3']:.3f} |",
        f"| Hit@5 | {summary['hit5']:.3f} |", f"| MRR   | {summary['mrr']:.3f} |",
        f"| 평균 수집 청크 수 | {summary['avg_n_docs']:.1f}개 |",
        "", "## 답변 품질", "",
        "| 지표 | 값 |", "|------|-----|",
        f"| 평균 품질 | {summary['avg_quality']:.3f} |",
        f"| 완전정답(3점) | {qdist.get(3,0)}개 ({qdist.get(3,0)/n*100:.1f}%) |",
        f"| 오답(0점) | {qdist.get(0,0)}개 ({qdist.get(0,0)/n*100:.1f}%) |",
        f"| 평균 응답 시간 | {summary['avg_time']}초 |",
    ]
    mp.write_text("\n".join(lines), encoding="utf-8")
    print(f"  저장: {jp.name} / {mp.name}")
    return summary

def main():
    print("="*55)
    print("  실험 11: 쿼리 확장 3개 모델 비교 평가")
    print("="*55)
    qa_pairs = json.loads(QA_FILE.read_text(encoding="utf-8"))
    print(f"QA 데이터: {len(qa_pairs)}개\n")

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_sums = []
    for cfg in MODELS:
        summary = evaluate_model(qa_pairs, cfg)
        all_sums.append(save_results(summary, ts))

    print("\n"+"="*55)
    print("  실험 11 (쿼리 확장) 최종 비교")
    print("="*55)
    print(f"{'모델':<22} {'품질':>6} {'Hit@5':>6} {'MRR':>6} {'문서수':>6} {'시간':>6}")
    print("-"*55)
    for s in all_sums:
        print(f"{s['label']:<22} {s['avg_quality']:>6.3f} {s['hit5']:>6.3f} {s['mrr']:>6.3f} {s['avg_n_docs']:>6.1f} {s['avg_time']:>5.1f}s")

    print("\n[실험 9 기준값]")
    print(f"{'GPT-4o-mini':<22} {'1.048':>6} {'0.095':>6} {'0.048':>6} {'5.0':>6} {'4.2s':>6}")
    print(f"{'Llama 3.3 70B (Groq)':<22} {'0.595':>6} {'0.095':>6} {'0.048':>6} {'5.0':>6} {'2.6s':>6}")
    print(f"{'Gemini Flash Lite':<22} {'1.262':>6} {'0.095':>6} {'0.048':>6} {'5.0':>6} {'7.8s':>6}")

    ts_file = RESULTS_DIR / "exp11_timestamp.txt"
    ts_file.write_text(ts, encoding="utf-8")

if __name__ == "__main__":
    main()
