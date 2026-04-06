"""
건강식품 RAG 챗봇 v2 (100% 무료)
- Parent-Child Retrieval: 자식 검색 → 부모 컨텍스트
- Hybrid Search: 벡터 + 키워드(BM25) 결합
- Query Rewriting: 질문 재작성으로 검색 최적화
- Multi-Query: 3가지 관점으로 검색 확장
- Cross-Encoder Reranking: 관련 없는 결과 제거
- Streaming 답변 + 출처 메타데이터 표시
- LCEL 체인 패턴
- Memory Bank: 세션 간 사용자 정보 기억
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from supabase import create_client, Client
from openai import OpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ── 모델 로드 ────────────────────────────────────────────────────────
print("모델 로딩 중...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")
print("임베딩 + 리랭킹 모델 로드 완료!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
llm_client = OpenAI(api_key=OPENAI_API_KEY)

# gpt-4o-mini: $0.15/1M입력 + $0.60/1M출력 — 가장 저렴
# 질문 1회당 약 $0.001 (0.1원) → $5로 약 5,000회 질문 가능
LLM_MODEL = "gpt-4o-mini"
TOP_K = 10            # 초기 검색 수
RERANK_TOP_K = 3      # 리랭킹 후 최종 사용 수
MAX_CONTEXT_CHARS = 4000  # 컨텍스트 최대 글자 수 (Groq 무료 한도 대응)
RERANK_THRESHOLD = -100  # Cross-encoder 점수 기준 (한국어 문서는 음수 정상)
MAX_CONTENT = 2000

# ── 메모리뱅크 경로 ───────────────────────────────────────────────────
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "memory" / "user_memory.json"
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메모리뱅크: 세션 간 사용자 정보 저장/로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_memory() -> dict:
    """저장된 사용자 메모리 로드"""
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"facts": [], "updated_at": ""}


def save_memory(memory: dict) -> None:
    """메모리를 파일에 저장"""
    memory["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MEMORY_PATH.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_memory(user_input: str, assistant_answer: str, existing_facts: list[str]) -> list[str]:
    """대화에서 기억할 만한 사용자 정보를 LLM으로 추출"""
    existing_text = "\n".join(f"- {f}" for f in existing_facts) or "없음"
    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "당신은 건강 상담 기록 분석가입니다.\n"
                    "대화에서 사용자의 건강 관련 개인 정보만 추출하세요.\n"
                    "추출 대상: 건강 목표, 알레르기/금기 식품, 복용 중인 보충제/약, 질환, 식단 방식, 나이/성별/체중 등\n"
                    "규칙:\n"
                    "- 추출된 사실이 있으면 한 줄에 하나씩 출력 (예: 목표: 체중 감량 10kg)\n"
                    "- 기억할 정보가 없으면 '없음' 한 단어만 출력\n"
                    "- 이미 알고 있는 정보와 중복이면 출력하지 않음\n"
                    "- 일반적인 건강 정보나 질문 내용은 추출하지 않음\n"
                    f"\n[이미 기억하고 있는 정보]\n{existing_text}"
                )},
                {"role": "user", "content": f"사용자 발화: {user_input}\n상담사 답변: {assistant_answer[:300]}"},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()
        if result == "없음" or not result:
            return []
        new_facts = [line.strip().lstrip("- ") for line in result.split("\n") if line.strip() and line.strip() != "없음"]
        return new_facts
    except Exception:
        return []


def update_memory(user_input: str, assistant_answer: str) -> None:
    """대화 후 메모리 업데이트 (새 정보만 추가)"""
    memory = load_memory()
    new_facts = extract_memory(user_input, assistant_answer, memory["facts"])
    if new_facts:
        memory["facts"].extend(new_facts)
        save_memory(memory)


def format_memory_for_prompt(memory: dict) -> str:
    """메모리를 시스템 프롬프트용 텍스트로 변환"""
    if not memory["facts"]:
        return ""
    facts_text = "\n".join(f"- {f}" for f in memory["facts"])
    return f"[사용자 기억 정보 — 이전 대화에서 파악한 내용]\n{facts_text}\n"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1+2단계 통합: Query Rewrite + Multi-Query (API 1회 호출로 통합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_query_cache: dict[str, list[str]] = {}  # 동일 질문 캐시


def generate_search_queries(user_input: str) -> list[str]:
    """1회 API 호출로 검색 최적화 쿼리 + 다관점 쿼리 동시 생성"""
    if user_input in _query_cache:
        return _query_cache[user_input]

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "건강식품·다이어트 검색 전문가입니다. "
                    "사용자 질문을 검색에 최적화된 3개의 검색 쿼리로 바꿔주세요.\n"
                    "규칙:\n"
                    "- 한 줄에 하나씩, 번호 없이\n"
                    "- 한국어와 영어 키워드를 적절히 혼합\n"
                    "- 각각 다른 관점(성분, 효과, 방법 등)\n"
                    "- 설명 없이 검색 문장만"
                )},
                {"role": "user", "content": user_input},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        queries = response.choices[0].message.content.strip().split("\n")
        queries = [q.strip().lstrip("0123456789.-) ") for q in queries if len(q.strip()) > 5]
        result = [user_input] + queries[:3]  # 원본 + 생성된 3개
        result = list(dict.fromkeys(result))  # 중복 제거
        _query_cache[user_input] = result
        return result
    except Exception:
        return [user_input]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3단계: Hybrid Search — 벡터 + 키워드 검색 결합
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_query_embedding(text: str) -> list[float]:
    emb = embed_model.encode(f"query: {text[:MAX_CONTENT]}", normalize_embeddings=True)
    return emb.tolist()


def detect_query_category(query: str) -> str | None:
    """질문에서 카테고리 힌트 감지"""
    q = query.lower()
    if "푸드올로지" in q or "콜레올로지" in q or "맨올로지" in q or "톡스올로지" in q or "버닝올로지" in q:
        return "푸드올로지"
    return None


def _run_hybrid_search(query_embedding: list[float], query: str, category_filter: str | None) -> list[dict]:
    """실제 Supabase 검색 실행 (카테고리 필터 선택 적용)"""
    try:
        result = supabase.rpc("hybrid_search", {
            "query_embedding": query_embedding,
            "query_text": query,
            "match_count": TOP_K,
            "vector_weight": 0.7,
            "text_weight": 0.3,
            "filter_category": category_filter,
        }).execute()
        return result.data or []
    except Exception:
        try:
            params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": TOP_K,
            }
            if category_filter:
                params["filter_category"] = category_filter
            result = supabase.rpc("match_children_return_parents", params).execute()
            return result.data or []
        except Exception:
            result = supabase.rpc("match_documents", {
                "query_embedding": query_embedding,
                "match_threshold": 0.3,
                "match_count": TOP_K,
            }).execute()
            return result.data or []


def hybrid_search(query: str) -> list[dict]:
    """벡터 + 키워드 하이브리드 검색 (Parent-Child)
    카테고리 필터 적용 후 결과 없으면 전체 검색으로 폴백
    """
    query_embedding = get_query_embedding(query)
    category_filter = detect_query_category(query)

    results = _run_hybrid_search(query_embedding, query, category_filter)

    # 카테고리 필터 적용했는데 결과 없으면 전체 검색으로 폴백
    if not results and category_filter:
        results = _run_hybrid_search(query_embedding, query, None)

    return results


def multi_query_search(queries: list[str]) -> list[dict]:
    """여러 쿼리를 병렬 실행 후 결과 병합 (중복 제거)"""
    seen_ids = set()
    all_docs = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(hybrid_search, q): q for q in queries}
        for future in as_completed(futures):
            try:
                docs = future.result()
                for doc in docs:
                    doc_id = doc.get("parent_id") or doc.get("id")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
            except Exception:
                pass

    return all_docs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4단계: Cross-Encoder Reranking — 관련 없는 결과 제거
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rerank_documents(query: str, docs: list[dict]) -> list[dict]:
    """BGE 다국어 크로스인코더 리랭킹 (BAAI/bge-reranker-v2-m3)
    한국어/영어 혼재 문서에서 의미 기반 관련성 점수를 직접 계산.
    실험 4 결과: 기존 키워드 보너스 대비 Hit@1 +11.7%, MRR +6.5% 향상.
    """
    if not docs:
        return []

    content_key = "parent_content" if "parent_content" in docs[0] else "content"

    # (질문, 문서) 쌍으로 크로스인코더에 입력
    pairs = [(query, doc.get(content_key, "")[:1000]) for doc in docs]
    scores = rerank_model.predict(pairs)

    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)

    ranked = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:RERANK_TOP_K]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5단계: 컨텍스트 구성 + 출처 메타데이터
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NO_DATA_RESPONSE = (
    "죄송합니다. 현재 보유한 자료에서 해당 정보를 찾을 수 없습니다.\n"
    "건강식품, 다이어트, 영양소 관련 질문을 해주시면 더 정확한 답변을 드릴 수 있습니다."
)

OFF_TOPIC_KEYWORDS = [
    "날씨", "주식", "코딩", "프로그래밍", "게임", "영화", "음악", "정치",
    "축구", "야구", "연예인", "여행", "부동산", "주가", "비트코인",
]

SYSTEM_PROMPT = """당신은 건강식품과 다이어트 전문 상담사입니다.

[절대 규칙 — 할루시네이션 방지]
1. 반드시 [참고 자료]에 있는 내용만으로 답변하세요.
2. [참고 자료]에 없는 내용은 절대 지어내지 마세요.
3. 모르면 "보유한 자료에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
4. 추측, 일반 상식, 외부 지식으로 보충하지 마세요.
5. 의학적 진단이나 처방은 하지 말고 "전문 의사 상담을 권장합니다"라고 안내하세요.

답변 원칙:
- 한국어로 친절하고 명확하게 답변하세요
- [자료 N] 형태로 출처를 반드시 표시하세요
- 건강식품, 다이어트, 영양소 이외의 주제는 "전문 영역이 아닙니다"라고 답변하세요"""


def format_context_with_sources(docs: list[dict]) -> tuple[str, list[dict]]:
    """컨텍스트 텍스트 + 출처 메타데이터 반환"""
    if not docs:
        return "관련 자료를 찾지 못했습니다.", []

    context_parts = []
    sources = []
    content_key = "parent_content" if "parent_content" in docs[0] else "content"

    for i, doc in enumerate(docs, 1):
        source_file = doc.get("source_file", "알 수 없음")
        category = doc.get("category", "")
        rerank_score = doc.get("rerank_score", 0)
        content = doc.get(content_key, "")

        context_parts.append(f"[자료 {i}] ({category} | {source_file})\n{content[:1500]}")
        sources.append({
            "번호": i,
            "파일": source_file,
            "카테고리": category,
            "관련도": f"{rerank_score:.2f}",
        })

    full_context = "\n\n---\n\n".join(context_parts)
    if len(full_context) > MAX_CONTEXT_CHARS:
        full_context = full_context[:MAX_CONTEXT_CHARS] + "\n...(생략)"
    return full_context, sources


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6단계: LLM Streaming 답변 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def stream_llm_response(system_msg: str, user_msg: str):
    """OpenAI API 스트리밍 답변 생성"""
    stream = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1024,
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LCEL 체인 구성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def rag_chain(user_input: str, history: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """
    LCEL 체인: Query Rewrite → Multi-Query → Hybrid Search
               → Rerank → Context Build → Stream LLM → Sources
    """
    # 0. 주제 필터링
    if any(kw in user_input for kw in OFF_TOPIC_KEYWORDS):
        msg = "해당 분야는 제 전문 영역이 아닙니다."
        print(f"\n답변: {msg}")
        return msg, [], history

    t0 = time.time()

    # 1+2. Query Rewrite + Multi-Query (API 1회 호출)
    print("  [1] 검색 쿼리 생성...", end=" ", flush=True)
    search_queries = generate_search_queries(user_input)
    print(f"→ {len(search_queries)}개 ({time.time()-t0:.1f}s)")

    # 3. Hybrid Search (병렬 실행)
    t1 = time.time()
    print("  [2] Hybrid Search (병렬)...", end=" ", flush=True)
    raw_docs = multi_query_search(search_queries)
    print(f"→ {len(raw_docs)}개 결과 ({time.time()-t1:.1f}s)")

    if not raw_docs:
        print(f"\n답변: {NO_DATA_RESPONSE}")
        return NO_DATA_RESPONSE, [], history

    # 4. Cross-Encoder Reranking
    t2 = time.time()
    print(f"  [3] Reranking {len(raw_docs)}개...", end=" ", flush=True)
    ranked_docs = rerank_documents(user_input, raw_docs)
    print(f"→ {len(ranked_docs)}개 통과 ({time.time()-t2:.1f}s)")
    print(f"  총 검색: {time.time()-t0:.1f}s")

    if not ranked_docs:
        print(f"\n답변: {NO_DATA_RESPONSE}")
        return NO_DATA_RESPONSE, [], history

    # 5. 컨텍스트 구성 + 출처
    context, sources = format_context_with_sources(ranked_docs)

    # 6. 프롬프트 구성 (메모리뱅크 주입)
    memory = load_memory()
    memory_text = format_memory_for_prompt(memory)
    system_msg = f"{memory_text}{SYSTEM_PROMPT}\n\n[참고 자료]\n{context}"

    history_text = ""
    for msg in history[-6:]:
        role = "사용자" if msg["role"] == "user" else "상담사"
        history_text += f"{role}: {msg['content']}\n"

    user_msg = (
        f"{history_text}사용자: {user_input}\n\n"
        "위 [참고 자료]를 바탕으로 한국어로 친절하게 답변해주세요. "
        "[자료 N] 형태로 출처를 표시하세요. "
        "자료에 직접적인 답이 없더라도 관련 내용이 있으면 연결해서 답변하세요. "
        "정말 관련 내용이 전혀 없을 때만 모른다고 하세요."
    )

    # 7. Streaming 답변
    print("\n답변: ", end="", flush=True)
    full_answer = ""
    try:
        for token in stream_llm_response(system_msg, user_msg):
            print(token, end="", flush=True)
            full_answer += token
    except Exception as e:
        full_answer = f"[LLM 오류] {str(e)[:200]}"
        print(full_answer)

    print()  # 줄바꿈

    # 히스토리 업데이트
    updated_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": full_answer},
    ]

    # 메모리뱅크 업데이트 (백그라운드 — 오류 무시)
    try:
        update_memory(user_input, full_answer)
    except Exception:
        pass

    return full_answer, sources, updated_history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 60)
    print("  건강식품·다이어트 RAG 챗봇 v2")
    print("  ─────────────────────────────")
    print("  임베딩: multilingual-e5-small (로컬, 무료)")
    print("  리랭킹: BGE-reranker-v2-m3 Cross-Encoder (로컬, 무료, 한/영 지원)")
    print("  LLM: GPT-4o-mini (OpenAI — 질문 1회 약 0.1원)")
    print("  검색: Hybrid (벡터 + 키워드) + Multi-Query")
    print("  명령어: '메모리보기' | '메모리초기화' | 'quit'")
    print("=" * 60)

    # 메모리뱅크 로드 및 표시
    memory = load_memory()
    if memory["facts"]:
        print("\n[메모리뱅크] 이전 대화에서 기억한 정보:")
        for f in memory["facts"]:
            print(f"  - {f}")
        print()

    history = []

    while True:
        try:
            user_input = input("\n질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n챗봇을 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "종료"):
            print("챗봇을 종료합니다.")
            break

        # 메모리 명령어 처리
        if user_input == "메모리보기":
            mem = load_memory()
            if mem["facts"]:
                print("\n[메모리뱅크] 저장된 정보:")
                for f in mem["facts"]:
                    print(f"  - {f}")
                print(f"  (마지막 업데이트: {mem.get('updated_at', '알 수 없음')})")
            else:
                print("[메모리뱅크] 저장된 정보가 없습니다.")
            continue

        if user_input == "메모리초기화":
            save_memory({"facts": []})
            print("[메모리뱅크] 초기화 완료.")
            continue

        print("🔍 검색 중... (Query Rewrite → Multi-Query → Hybrid Search → Rerank)")
        try:
            answer, sources, history = rag_chain(user_input, history)

            # 출처 메타데이터 표시
            if sources:
                print("\n📚 참고 출처:")
                for s in sources:
                    print(f"  [{s['번호']}] {s['파일']} ({s['카테고리']}) 관련도: {s['관련도']}")
        except Exception as e:
            print(f"\n[오류] {e}")


if __name__ == "__main__":
    main()
