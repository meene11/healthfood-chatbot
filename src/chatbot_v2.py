"""
건강식품 RAG 챗봇 v2 (100% 무료)
- Parent-Child Retrieval: 자식 검색 → 부모 컨텍스트
- Hybrid Search: 벡터 + 키워드(BM25) 결합
- Query Rewriting: 질문 재작성으로 검색 최적화
- Multi-Query: 3가지 관점으로 검색 확장
- Cross-Encoder Reranking: 관련 없는 결과 제거
- Streaming 답변 + 출처 메타데이터 표시
- LCEL 체인 패턴
"""
import os
import sys
import time
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
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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


def hybrid_search(query: str) -> list[dict]:
    """벡터 + 키워드 하이브리드 검색 (Parent-Child)"""
    query_embedding = get_query_embedding(query)
    category_filter = detect_query_category(query)

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
    """하이브리드 리랭킹: 검색 점수(90%) + 한국어 키워드 매칭 보너스"""
    if not docs:
        return []

    content_key = "parent_content" if "parent_content" in docs[0] else "content"

    # 질문에서 핵심 키워드 추출
    query_keywords = [w for w in query.split() if len(w) >= 2]

    for doc in docs:
        search_score = doc.get("combined_score") or doc.get("similarity") or 0
        content = doc.get(content_key, "")

        # 한국어 키워드 매칭 보너스 (0~0.2)
        keyword_hits = sum(1 for kw in query_keywords if kw in content)
        keyword_bonus = min(keyword_hits * 0.05, 0.2)

        doc["rerank_score"] = float(search_score) + keyword_bonus

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

    # 6. 프롬프트 구성
    system_msg = f"{SYSTEM_PROMPT}\n\n[참고 자료]\n{context}"

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

    return full_answer, sources, updated_history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 루프
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    print("=" * 60)
    print("  건강식품·다이어트 RAG 챗봇 v2")
    print("  ─────────────────────────────")
    print("  임베딩: multilingual-e5-small (로컬, 무료)")
    print("  리랭킹: ms-marco-MiniLM Cross-Encoder (로컬, 무료)")
    print("  LLM: GPT-4o-mini (OpenAI — 질문 1회 약 0.1원)")
    print("  검색: Hybrid (벡터 + 키워드) + Multi-Query")
    print("  종료: 'quit' 또는 '종료'")
    print("=" * 60)

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
