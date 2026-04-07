"""
실험 12: 리랭커 임계값 필터링 효과 검증
=========================================
실험 9(기준)와의 차이:
  - 리랭킹 후 score < threshold인 청크 제거
  - bge-reranker-v2-m3: 0 이상 = 관련 있음, 0 미만 = 관련 없음
  - 최소 1개는 항상 유지 (threshold보다 높은 게 없어도 최상위 1개 유지)
  - 노이즈 청크를 LLM에 전달하지 않아 할루시네이션 감소 기대

임계값: 0.0 (logit 기준, bge-reranker-v2-m3 특성)

실행: python src/evaluate_exp12_reranker_threshold.py
결과: results/model_comparison/exp12_*.json/md
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

TOP_K            = 5
MAX_RERANK       = 3
RERANK_THRESHOLD = 0.0   # 0 이상만 LLM에 전달
MAX_CONTENT      = 2000
MAX_CONTEXT      = 4000

MODELS = [
    {"key": "gpt-4o-mini",  "label": "GPT-4o-mini",          "model_id": "gpt-4o-mini",                   "client_type": "openai"},
    {"key": "groq-llama",   "label": "Llama 3.3 70B (Groq)",  "model_id": "llama-3.3-70b-versatile",       "client_type": "groq"},
    {"key": "gemini-flash", "label": "Gemini Flash Lite",     "model_id": "models/gemini-flash-lite-latest","client_type": "gemini"},
]

SYSTEM_PROMPT = """당신은 건강식품과 다이어트 전문 상담사입니다.
[참고 자료]를 바탕으로 한국어로 간결하고 정확하게 답변하세요.
자료에 관련 내용이 없으면 솔직하게 "보유한 자료에서 찾을 수 없습니다"라고 하세요."""

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

print("모델 로딩 중...")
embed_model  = SentenceTransformer("BAAI/bge-m3")
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
judge_client = OpenAI(api_key=OPENAI_API_KEY)
print("로드 완료\n")

def get_embedding(text):
    return embed_model.encode(text[:MAX_CONTENT], normalize_embeddings=True).tolist()

def hybrid_search(query):
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

def rerank_with_threshold(query, docs) -> tuple[list[dict], dict]:
    """리랭킹 후 threshold 필터링. 통계 반환."""
    if not docs: return [], {"n_before":0,"n_after":0,"filtered":0,"min_score":None,"max_score":None}
    pairs  = [(query, doc.get("content","")[:1000]) for doc in docs]
    scores = rerank_model.predict(pairs)
    for doc, s in zip(docs, scores):
        doc["rerank_score"] = float(s)
    ranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:MAX_RERANK]

    # 임계값 필터
    passed = [d for d in ranked if d["rerank_score"] >= RERANK_THRESHOLD]
    if not passed:
        passed = ranked[:1]  # 최소 1개 보장

    stats = {
        "n_before": len(ranked),
        "n_after":  len(passed),
        "filtered": len(ranked) - len(passed),
        "max_score": round(ranked[0]["rerank_score"], 3) if ranked else None,
        "min_score": round(ranked[-1]["rerank_score"], 3) if ranked else None,
    }
    return passed, stats

def build_context(docs):
    parts = [f"[자료 {i+1}]\n{doc.get('content','')[:1500]}" for i, doc in enumerate(docs)]
    return "\n\n---\n\n".join(parts)[:MAX_CONTEXT]

def calc_retrieval_metrics(answer_chunk_id, raw_docs):
    ids = [doc.get("id") for doc in raw_docs]
    hit1 = int(answer_chunk_id in ids[:1])
    hit3 = int(answer_chunk_id in ids[:3])
    hit5 = int(answer_chunk_id in ids[:5])
    try:
        rank = ids.index(answer_chunk_id) + 1; rr = 1.0/rank
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
        return {"score":max(0,min(3,int(result.get("score",0)))),"reason":result.get("reason","")}
    except Exception as e:
        return {"score":-1,"reason":f"judge error: {e}"}

def evaluate_model(qa_pairs, cfg):
    label  = cfg["label"]
    client = make_client(cfg)
    print(f"\n{'='*55}\n  {label} + 리랭커 임계값({RERANK_THRESHOLD})\n{'='*55}")

    results  = []
    cat_data = defaultdict(lambda: {"quality":[],"hit5":[],"rr":[]})
    total_filtered = 0

    for i, qa in enumerate(qa_pairs, 1):
        q, cid, chunk, cat = qa["question"], qa["answer_chunk_id"], qa.get("answer_chunk_content",""), qa["category"]
        print(f"  [{i:02d}/{len(qa_pairs)}] {q[:50]}...")
        t0 = time.time()

        raw_docs = hybrid_search(q)
        ranked, th_stats = rerank_with_threshold(q, raw_docs)
        ret_m = calc_retrieval_metrics(cid, raw_docs)
        total_filtered += th_stats["filtered"]

        context  = build_context(ranked)
        user_msg = f"질문: {q}\n\n[참고 자료]\n{context}\n\n자료를 바탕으로 답변해주세요."
        answer   = generate_answer(client, cfg, SYSTEM_PROMPT, user_msg)
        elapsed  = time.time() - t0

        jd = judge_answer(q, chunk, answer)
        print(f"    Hit@5={'O' if ret_m['hit5'] else 'X'} | 품질={jd['score']}/3 | {elapsed:.1f}s | 필터:{th_stats['filtered']}개제거 남은{th_stats['n_after']}개")

        row = {"id":qa["id"],"question":q,"category":cat,"answer_chunk_id":cid,
               "n_docs_after_filter":th_stats["n_after"],"n_filtered":th_stats["filtered"],
               "max_rerank_score":th_stats["max_score"],
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
        "model": cfg["key"], "label": label, "experiment": "exp12_reranker_threshold",
        "rerank_threshold": RERANK_THRESHOLD,
        "n_valid": n,
        "avg_quality":  round(sum(r["quality_score"] for r in valid)/n, 3),
        "quality_dist": dict(qdist),
        "hit1": round(sum(r["hit1"] for r in results)/len(results),3),
        "hit3": round(sum(r["hit3"] for r in results)/len(results),3),
        "hit5": round(sum(r["hit5"] for r in results)/len(results),3),
        "mrr":  round(sum(r["rr"]   for r in results)/len(results),3),
        "avg_time": round(sum(r["total_sec"] for r in results)/len(results),2),
        "total_filtered_chunks": total_filtered,
        "avg_filtered_per_q": round(total_filtered/len(results),2),
        "cat_quality": {c:round(sum(v["quality"])/len(v["quality"]),3) for c,v in cat_data.items()},
        "cat_hit5":    {c:round(sum(v["hit5"])/len(v["hit5"]),3) for c,v in cat_data.items()},
        "results": results,
    }
    return summary

def save_results(summary, ts):
    key   = summary["model"].replace("-","_")
    n     = summary["n_valid"]
    qdist = summary["quality_dist"]
    jp    = RESULTS_DIR / f"exp12_{key}_{ts}.json"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    mp = RESULTS_DIR / f"exp12_{key}_{ts}.md"
    lines = [
        f"# 실험 12: 리랭커 임계값 필터링 - {summary['label']}",
        f"> 임계값: {summary['rerank_threshold']} | 문항: {n}개 | 총 필터링: {summary['total_filtered_chunks']}개",
        "", "## 검색 품질", "",
        "| 지표 | 값 |", "|------|-----|",
        f"| Hit@5 | {summary['hit5']:.3f} |",
        f"| MRR   | {summary['mrr']:.3f} |",
        f"| 평균 필터링된 청크 수 | {summary['avg_filtered_per_q']:.2f}개/질문 |",
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
    print(f"  실험 12: 리랭커 임계값 필터링 (threshold={RERANK_THRESHOLD})")
    print("="*55)
    qa_pairs = json.loads(QA_FILE.read_text(encoding="utf-8"))
    print(f"QA 데이터: {len(qa_pairs)}개\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_sums = []
    for cfg in MODELS:
        summary = evaluate_model(qa_pairs, cfg)
        all_sums.append(save_results(summary, ts))

    print("\n"+"="*55)
    print("  실험 12 (리랭커 임계값) 최종 비교")
    print("="*55)
    print(f"{'모델':<22} {'품질':>6} {'Hit@5':>6} {'MRR':>6} {'필터':>6} {'시간':>6}")
    print("-"*55)
    for s in all_sums:
        print(f"{s['label']:<22} {s['avg_quality']:>6.3f} {s['hit5']:>6.3f} {s['mrr']:>6.3f} {s['avg_filtered_per_q']:>5.2f}개 {s['avg_time']:>5.1f}s")

    print("\n[실험 9 기준값]")
    print(f"{'GPT-4o-mini':<22} {'1.048':>6} {'0.095':>6} {'0.048':>6} {'-':>6} {'4.2s':>6}")
    print(f"{'Llama 3.3 70B (Groq)':<22} {'0.595':>6} {'0.095':>6} {'0.048':>6} {'-':>6} {'2.6s':>6}")
    print(f"{'Gemini Flash Lite':<22} {'1.262':>6} {'0.095':>6} {'0.048':>6} {'-':>6} {'7.8s':>6}")

    ts_file = RESULTS_DIR / "exp12_timestamp.txt"
    ts_file.write_text(ts, encoding="utf-8")

if __name__ == "__main__":
    main()
