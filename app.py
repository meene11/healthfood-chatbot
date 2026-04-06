"""
건강식품·다이어트 RAG 챗봇 — Streamlit Web UI
================================================
배포: HuggingFace Spaces (무료, 16GB RAM)
로컬: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# ── 페이지 설정 (가장 먼저 호출해야 함) ───────────────────────────────
st.set_page_config(
    page_title="건강식품·다이어트 RAG 챗봇",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 환경변수 로드 ────────────────────────────────────────────────────
# 로컬: .env 파일 / HuggingFace Spaces: Secrets 설정
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except Exception:
    pass

def _get_secret(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        try:
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val

SUPABASE_URL   = _get_secret("SUPABASE_URL")
SUPABASE_KEY   = _get_secret("SUPABASE_KEY")
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")

# ── 설정값 ────────────────────────────────────────────────────────────
TOP_K          = 10
RERANK_TOP_K   = 3
MAX_CONTENT    = 2000
MAX_CONTEXT_CHARS = 4000
LLM_MODEL      = "gpt-4o-mini"
MEMORY_PATH    = Path(__file__).parent / "data" / "memory" / "user_memory.json"
MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)

OFF_TOPIC_KEYWORDS = [
    "날씨", "주식", "코딩", "프로그래밍", "게임", "영화", "음악", "정치",
    "축구", "야구", "연예인", "여행", "부동산", "주가", "비트코인",
]

SYSTEM_PROMPT = """당신은 건강식품과 다이어트 전문 상담사입니다.

[답변 규칙]
1. [참고 자료]에 직접적인 답이 있으면 그 내용을 중심으로 답변하세요.
2. 직접적인 답이 없더라도 관련 내용(성분, 효과, 주의사항 등)이 조금이라도 있으면 그것을 기반으로 최선을 다해 답변하세요.
3. 자료에 전혀 관련 내용이 없을 때만 "보유한 자료에서 찾을 수 없습니다"라고 안내하세요.
4. 자료에 없는 내용을 지어내거나 추측하지 마세요.
5. 의학적 진단이나 처방은 하지 말고 "전문 의사 상담을 권장합니다"라고 안내하세요.

답변 원칙:
- 한국어로 친절하고 명확하게 답변하세요
- [자료 N] 형태로 출처를 반드시 표시하세요
- 부분적인 정보라도 "자료에 따르면..." 형태로 성실하게 전달하세요
- 건강식품, 다이어트, 영양소 이외의 주제는 "전문 영역이 아닙니다"라고 답변하세요"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 캐시 (Streamlit 재실행 시 재로딩 방지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
@st.cache_resource(show_spinner="AI 모델 로딩 중... (최초 1회, 약 1-2분 소요)")
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embed   = SentenceTransformer("BAAI/bge-m3")
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return embed, reranker


@st.cache_resource(show_spinner=False)
def get_clients():
    from supabase import create_client
    from openai import OpenAI
    sb  = create_client(SUPABASE_URL, SUPABASE_KEY)
    llm = OpenAI(api_key=OPENAI_API_KEY)
    return sb, llm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메모리뱅크
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_memory() -> dict:
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"facts": [], "updated_at": ""}


def save_memory(memory: dict) -> None:
    memory["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    MEMORY_PATH.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_memory(user_input: str, assistant_answer: str, existing_facts: list) -> list:
    _, llm = get_clients()
    existing_text = "\n".join(f"- {f}" for f in existing_facts) or "없음"
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "당신은 건강 상담 기록 분석가입니다.\n"
                    "대화에서 사용자의 건강 관련 개인 정보만 추출하세요.\n"
                    "추출 대상: 건강 목표, 알레르기/금기 식품, 복용 중인 보충제/약, 질환, 식단 방식, 나이/성별/체중 등\n"
                    "규칙:\n"
                    "- 추출된 사실이 있으면 한 줄에 하나씩 출력\n"
                    "- 기억할 정보가 없으면 '없음' 한 단어만 출력\n"
                    "- 이미 알고 있는 정보와 중복이면 출력하지 않음\n"
                    f"\n[이미 기억하고 있는 정보]\n{existing_text}"
                )},
                {"role": "user", "content": f"사용자: {user_input}\n답변: {assistant_answer[:300]}"},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        result = resp.choices[0].message.content.strip()
        if result == "없음" or not result:
            return []
        return [line.strip().lstrip("- ") for line in result.split("\n") if line.strip() and line.strip() != "없음"]
    except Exception:
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 검색 파이프라인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_query_embedding(text: str) -> list:
    embed, _ = load_models()
    return embed.encode(text[:MAX_CONTENT], normalize_embeddings=True).tolist()


def detect_category(query: str) -> str | None:
    q = query.lower()
    if any(k in q for k in ["푸드올로지", "콜레올로지", "맨올로지", "버닝올로지", "톡스올로지"]):
        return "푸드올로지"
    return None


def hybrid_search(query: str) -> list:
    sb, _ = get_clients()
    emb    = get_query_embedding(query)
    cat    = detect_category(query)

    def _search(category_filter):
        try:
            return sb.rpc("hybrid_search_v2", {
                "query_embedding": emb,
                "query_text":      query,
                "match_count":     TOP_K,
                "vector_weight":   0.7,
                "text_weight":     0.3,
                "filter_category": category_filter,
            }).execute().data or []
        except Exception:
            try:
                params = {"query_embedding": emb, "match_threshold": 0.3, "match_count": TOP_K}
                if category_filter:
                    params["filter_category"] = category_filter
                return sb.rpc("match_documents_v2", params).execute().data or []
            except Exception:
                return []

    results = _search(cat)
    if not results and cat:
        results = _search(None)
    return results


def multi_query_search(queries: list) -> list:
    seen, all_docs = set(), []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(hybrid_search, q): q for q in queries}
        for f in as_completed(futures):
            try:
                for doc in f.result():
                    doc_id = doc.get("id")
                    if doc_id and doc_id not in seen:
                        seen.add(doc_id)
                        all_docs.append(doc)
            except Exception:
                pass
    return all_docs


def rerank(query: str, docs: list) -> list:
    _, reranker = load_models()
    if not docs:
        return []
    pairs  = [(query, doc.get("content", "")[:1000]) for doc in docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(docs, scores):
        doc["rerank_score"] = float(score)
    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:RERANK_TOP_K]


def keyword_fallback_search(query: str, exclude_ids: set) -> list:
    """리랭킹 점수가 낮을 때 핵심 키워드로 DB 직접 검색 (ilike).
    벡터 검색이 의미적 정의를 찾지 못할 때 보완.
    예: '퓨린이 뭐야?' → '퓨린' 키워드로 DB에서 직접 찾음
    """
    import re
    sb, _ = get_clients()
    particles = r'(이|가|은|는|을|를|의|에|에서|로|으로|와|과|이야|야|뭐야|란|이란|\?|은\?|가\?)$'
    nouns = []
    for word in query.split():
        cleaned = re.sub(particles, '', word.strip())
        if len(cleaned) >= 2 and cleaned not in {'뭐야', '무엇', '어떻게', '왜', '언제', '어디'}:
            nouns.append(cleaned)

    results = []
    seen = set(exclude_ids)
    for noun in nouns[:2]:
        try:
            resp = sb.table("documents_v2").select(
                "id, content, source_file, category, token_count"
            ).ilike("content", f"%{noun}%").limit(5).execute()
            for r in (resp.data or []):
                if r["id"] not in seen:
                    seen.add(r["id"])
                    r["combined_score"] = 0.35
                    r["rerank_score"]   = 0.0
                    results.append(r)
        except Exception:
            pass
    return results


def generate_queries(user_input: str) -> list:
    _, llm = get_clients()
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "건강식품·다이어트 검색 전문가입니다. "
                    "사용자 질문을 검색에 최적화된 3개의 검색 쿼리로 바꿔주세요.\n"
                    "규칙: 한 줄에 하나씩, 번호 없이, 한/영 혼합, 다른 관점, 설명 없이 검색 문장만"
                )},
                {"role": "user", "content": user_input},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        queries = resp.choices[0].message.content.strip().split("\n")
        queries = [q.strip().lstrip("0123456789.-) ") for q in queries if len(q.strip()) > 5]
        return list(dict.fromkeys([user_input] + queries[:3]))
    except Exception:
        return [user_input]


def build_context(docs: list) -> tuple[str, list]:
    if not docs:
        return "관련 자료를 찾지 못했습니다.", []
    parts, sources = [], []
    for i, doc in enumerate(docs, 1):
        content  = doc.get("content", "")
        src_file = doc.get("source_file", "알 수 없음")
        category = doc.get("category", "")
        score    = doc.get("rerank_score", 0)
        parts.append(f"[자료 {i}] ({category} | {src_file})\n{content[:1500]}")
        sources.append({"번호": i, "파일": src_file, "카테고리": category, "관련도": f"{score:.2f}"})
    ctx = "\n\n---\n\n".join(parts)
    if len(ctx) > MAX_CONTEXT_CHARS:
        ctx = ctx[:MAX_CONTEXT_CHARS] + "\n...(생략)"
    return ctx, sources


def stream_answer(system_msg: str, user_msg: str):
    _, llm = get_clients()
    stream = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
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
# Streamlit UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 사이드바 ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🥗 건강식품 RAG 챗봇")
    st.caption("BAAI/bge-m3 · GPT-4o-mini · Supabase pgvector")
    st.divider()

    # 내 메모리뱅크
    st.subheader("🧠 내 메모리뱅크")
    st.caption("챗봇이 나에 대해 기억한 건강 정보")

    memory = load_memory()

    col1, col2 = st.columns(2)
    with col1:
        show_memory = st.button("📋 내용 보기", use_container_width=True)
    with col2:
        reset_memory = st.button("🗑 초기화", use_container_width=True, type="secondary")

    if reset_memory:
        save_memory({"facts": []})
        st.success("메모리가 초기화되었습니다.")
        st.rerun()

    if show_memory or "show_memory" in st.session_state:
        memory = load_memory()
        if memory["facts"]:
            st.session_state["show_memory"] = True
            with st.expander("기억된 정보", expanded=True):
                for fact in memory["facts"]:
                    st.write(f"• {fact}")
                if memory.get("updated_at"):
                    st.caption(f"마지막 업데이트: {memory['updated_at']}")
        else:
            st.info("아직 기억된 정보가 없습니다.\n건강 목표, 알레르기 등을 말씀해 주시면 기억합니다!")
            st.session_state.pop("show_memory", None)

    st.divider()

    # 모델 정보
    with st.expander("⚙️ 모델 정보"):
        st.markdown("""
        | 항목 | 모델 |
        |------|------|
        | 임베딩 | bge-m3 (1024d) |
        | 리랭커 | bge-reranker-v2-m3 |
        | LLM | GPT-4o-mini |
        | DB | Supabase documents_v2 |
        | 청크 수 | 18,682개 |
        """)
        st.caption("Hit@5=0.941 · MRR=0.828")

    # 대화 초기화
    st.divider()
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("show_memory", None)
        st.rerun()


# ── 메인 채팅 영역 ───────────────────────────────────────────────────
st.title("🥗 건강식품·다이어트 전문 상담사")
st.caption("건강식품, 다이어트, 영양소에 대해 무엇이든 물어보세요!")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 대화 히스토리 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 참고 출처 ({len(msg['sources'])}개)", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['번호']}]** {s['파일']}  \n카테고리: `{s['카테고리']}` · 관련도: `{s['관련도']}`")

# 첫 방문 안내
if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown("""
안녕하세요! 건강식품·다이어트 전문 상담사입니다. 🌿

다음과 같은 질문에 답변드릴 수 있습니다:
- 💊 건강식품 성분 및 효능 (오메가3, 프로바이오틱스, 커큐민 등)
- 🥗 다이어트 방법 (간헐적 단식, 케토제닉, 저탄수화물 등)
- 📊 혈당 관리 및 영양소 정보
- 🛒 푸드올로지 제품 (버닝올로지, 콜레올로지 등)

건강 목표나 현재 상태를 알려주시면 개인 맞춤 답변을 드립니다!
        """)

# 입력창
if user_input := st.chat_input("건강 관련 질문을 입력하세요..."):

    # 사용자 메시지 출력
    with st.chat_message("user"):
        st.markdown(user_input)

    # 주제 필터링
    if any(kw in user_input for kw in OFF_TOPIC_KEYWORDS):
        response = "해당 분야는 제 전문 영역이 아닙니다. 건강식품, 다이어트, 영양소 관련 질문을 해주세요! 🌿"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state["messages"].extend([
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": response, "sources": []},
        ])
        st.stop()

    # RAG 파이프라인 실행
    with st.chat_message("assistant"):
        sources      = []
        full_answer  = ""

        # 진행 상태 표시
        with st.status("검색 중...", expanded=False) as status:
            # 1. 쿼리 생성
            status.update(label="검색 쿼리 생성 중...")
            queries = generate_queries(user_input)

            # 2. 검색
            status.update(label=f"관련 자료 검색 중... ({len(queries)}개 쿼리)")
            raw_docs = multi_query_search(queries)

            if not raw_docs:
                status.update(label="검색 완료", state="complete")
                no_data = "죄송합니다. 관련 자료를 찾지 못했습니다. 다른 방식으로 질문해 보시겠어요?"
                st.markdown(no_data)
                st.session_state["messages"].extend([
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": no_data, "sources": []},
                ])
                st.stop()

            # 3. 리랭킹
            status.update(label=f"관련도 정밀 분석 중... ({len(raw_docs)}개)")
            ranked_docs = rerank(user_input, raw_docs)

            # 리랭킹 점수가 모두 낮으면 키워드 폴백 검색으로 보완
            LOW_RERANK_THRESHOLD = -3.0
            if ranked_docs and max(d.get("rerank_score", -999) for d in ranked_docs) < LOW_RERANK_THRESHOLD:
                already_found = {d.get("id") for d in raw_docs if d.get("id")}
                fallback_docs = keyword_fallback_search(user_input, already_found)
                if fallback_docs:
                    status.update(label=f"키워드 검색으로 추가 자료 {len(fallback_docs)}개 발견...")
                    ranked_docs = (fallback_docs + ranked_docs)[:RERANK_TOP_K]

            # 4. 컨텍스트 구성
            context, sources = build_context(ranked_docs)
            status.update(label="답변 생성 중...", state="running")

        # 5. 메모리 + 히스토리 구성
        memory      = load_memory()
        memory_text = ""
        if memory["facts"]:
            facts = "\n".join(f"- {f}" for f in memory["facts"])
            memory_text = f"[사용자 기억 정보 — 이전 대화에서 파악한 내용]\n{facts}\n\n"

        history_text = ""
        for msg in st.session_state["messages"][-6:]:
            role = "사용자" if msg["role"] == "user" else "상담사"
            history_text += f"{role}: {msg['content']}\n"

        system_msg = f"{memory_text}{SYSTEM_PROMPT}\n\n[참고 자료]\n{context}"
        user_msg   = (
            f"{history_text}사용자: {user_input}\n\n"
            "위 [참고 자료]를 바탕으로 한국어로 친절하게 답변해주세요. "
            "[자료 N] 형태로 출처를 반드시 표시하세요. "
            "자료에 직접적인 정의나 설명이 없더라도, 관련 성분·효과·주의사항 등 부분적인 정보라도 있으면 그것을 기반으로 성실하게 답변하세요. "
            "예를 들어 '퓨린이 뭐야?'라는 질문에 퓨린의 정의는 없지만 퓨린 함량이나 통풍 관련 내용이 있다면 그 내용을 알려주세요. "
            "자료에 전혀 관련 내용이 없을 때만 모른다고 하세요."
        )

        # 6. 스트리밍 답변 출력
        answer_placeholder = st.empty()
        try:
            for token in stream_answer(system_msg, user_msg):
                full_answer += token
                answer_placeholder.markdown(full_answer + "▌")
            answer_placeholder.markdown(full_answer)
        except Exception as e:
            full_answer = f"답변 생성 중 오류가 발생했습니다: {str(e)[:100]}"
            answer_placeholder.markdown(full_answer)

        # 7. 출처 표시
        if sources:
            with st.expander(f"📚 참고 출처 ({len(sources)}개)", expanded=False):
                for s in sources:
                    st.markdown(f"**[{s['번호']}]** {s['파일']}  \n카테고리: `{s['카테고리']}` · 관련도: `{s['관련도']}`")

    # 히스토리 저장
    st.session_state["messages"].extend([
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": full_answer, "sources": sources},
    ])

    # 메모리 업데이트 (백그라운드)
    try:
        memory = load_memory()
        new_facts = extract_memory(user_input, full_answer, memory["facts"])
        if new_facts:
            memory["facts"].extend(new_facts)
            save_memory(memory)
    except Exception:
        pass
