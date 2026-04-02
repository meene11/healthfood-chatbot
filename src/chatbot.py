"""
건강식품 RAG 챗봇 (100% 무료 모델)
- 임베딩: intfloat/multilingual-e5-small (로컬, 무료)
- 챗봇 LLM: Hugging Face 무료 API (Mistral/Llama 등)
- 벡터 검색: Supabase pgvector
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from groq import Groq

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── 무료 로컬 임베딩 모델 ─────────────────────────────────────────────
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_CONTENT      = 2000
TOP_K            = 5
SIMILARITY_TH    = 0.3
HIGH_CONFIDENCE  = 0.5   # 이 이상이면 확실한 자료
LOW_CONFIDENCE   = 0.35  # 이 미만이면 자료 부족 판정
MAX_HISTORY      = 6

print("임베딩 모델 로딩 중...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("모델 로드 완료!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── 무료 LLM (Groq — Llama 3.1 8B, 무료 API) ─────────────────────────
# https://console.groq.com 에서 무료 API 키 발급
groq_client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """당신은 건강식품과 다이어트 전문 상담사입니다.

역할:
- 건강기능식품의 효능과 복용법 안내
- 다이어트 식단, 영양소, 체중 관리 방법 상담
- 푸드올로지 제품 정보 제공
- 논문 및 연구 기반 근거 있는 정보 제공

[절대 규칙 — 할루시네이션 방지]
1. 반드시 [참고 자료]에 있는 내용만으로 답변하세요.
2. [참고 자료]에 없는 내용은 절대 지어내지 마세요.
3. 모르거나 자료가 부족하면 반드시 "죄송합니다. 현재 보유한 자료에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
4. 추측, 일반 상식, 외부 지식으로 답변을 보충하지 마세요.
5. 의학적 진단이나 처방은 하지 말고 "전문 의사 상담을 권장합니다"라고 안내하세요.

답변 원칙:
- 한국어로 친절하고 명확하게 답변하세요
- 출처가 논문인 경우 "(논문 근거)" 표시
- 답변의 근거가 되는 [참고 자료] 번호를 함께 언급하세요 (예: "[자료 1] 기반")
- 건강식품, 다이어트, 영양소 이외의 주제는 "해당 분야는 제 전문 영역이 아닙니다."라고 답변하세요"""

NO_DATA_RESPONSE = (
    "죄송합니다. 현재 보유한 자료에서 해당 정보를 찾을 수 없습니다.\n"
    "건강식품, 다이어트, 영양소 관련 질문을 해주시면 더 정확한 답변을 드릴 수 있습니다."
)

OFF_TOPIC_KEYWORDS = [
    "날씨", "주식", "코딩", "프로그래밍", "게임", "영화", "음악", "정치",
    "축구", "야구", "연예인", "여행", "부동산", "주가", "비트코인", "암호화폐",
]


def get_query_embedding(text: str) -> list[float]:
    """검색 쿼리 임베딩 (query 프리픽스)"""
    embedding = embed_model.encode(
        f"query: {text[:MAX_CONTENT]}",
        normalize_embeddings=True
    )
    return embedding.tolist()


def retrieve_documents(query: str) -> list[dict]:
    """Supabase에서 유사 문서 검색"""
    query_embedding = get_query_embedding(query)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_threshold": SIMILARITY_TH,
        "match_count": TOP_K,
    }).execute()
    return result.data or []


def format_context(docs: list[dict]) -> str:
    if not docs:
        return "관련 자료를 찾지 못했습니다."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("source_file", "")
        category = doc.get("category", "")
        similarity = doc.get("similarity", 0)
        parts.append(
            f"[자료 {i}] ({category} | {source} | 유사도: {similarity:.2f})\n"
            f"{doc['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_msg: str, user_msg: str) -> str:
    """Groq 무료 API로 답변 생성 (Llama 3.1 8B)"""
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM 오류] {str(e)[:200]}"


def is_off_topic(user_input: str) -> bool:
    """건강/다이어트와 무관한 질문인지 판별"""
    return any(kw in user_input for kw in OFF_TOPIC_KEYWORDS)


def assess_confidence(docs: list[dict]) -> str:
    """검색 결과의 신뢰도 평가"""
    if not docs:
        return "no_data"
    max_sim = max(doc.get("similarity", 0) for doc in docs)
    avg_sim = sum(doc.get("similarity", 0) for doc in docs) / len(docs)
    if max_sim >= HIGH_CONFIDENCE:
        return "high"
    elif max_sim >= LOW_CONFIDENCE:
        return "low"
    else:
        return "no_data"


def chat(history: list[dict], user_input: str) -> tuple[str, list[dict]]:
    # 0. 주제 벗어난 질문 필터링
    if is_off_topic(user_input):
        answer = "해당 분야는 제 전문 영역이 아닙니다. 건강식품, 다이어트, 영양소 관련 질문을 해주세요."
        updated_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": answer},
        ]
        return answer, updated_history

    # 1. 관련 문서 검색
    docs = retrieve_documents(user_input)
    confidence = assess_confidence(docs)

    # 2. 자료가 전혀 없으면 LLM 호출 없이 바로 "모름" 응답
    if confidence == "no_data":
        updated_history = history + [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": NO_DATA_RESPONSE},
        ]
        return NO_DATA_RESPONSE, updated_history

    context = format_context(docs)

    # 3. 자료 신뢰도가 낮으면 LLM에 경고 추가
    confidence_warning = ""
    if confidence == "low":
        confidence_warning = (
            "\n[주의] 검색된 자료의 관련성이 낮습니다. "
            "자료에서 직접적인 답을 찾을 수 없다면 반드시 '보유 자료에서 정확한 답변을 찾기 어렵습니다'라고 말하세요.\n"
        )

    # 4. 메시지 구성 (chat completion 형식)
    system_msg = f"""{SYSTEM_PROMPT}
{confidence_warning}
[참고 자료]
{context}"""

    history_text = ""
    for msg in history[-(MAX_HISTORY * 2):]:
        role = "사용자" if msg["role"] == "user" else "상담사"
        history_text += f"{role}: {msg['content']}\n"

    user_msg = f"""{history_text}사용자: {user_input}

위 [참고 자료]에 있는 내용만으로 한국어로 답변해주세요.
자료에 없는 내용은 절대 지어내지 말고, 모르면 "보유한 자료에서 해당 정보를 찾을 수 없습니다"라고 답변하세요."""

    # 5. LLM 답변
    answer = call_llm(system_msg, user_msg)

    # 6. 히스토리 업데이트
    updated_history = history + [
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated_history


def main():
    print("="*60)
    print("건강식품·다이어트 RAG 챗봇 (100% 무료)")
    print("임베딩: multilingual-e5-small (로컬)")
    print("LLM: Llama-3.1-8B (Groq 무료 API)")
    print("종료: 'quit' 또는 'exit' 입력")
    print("="*60)

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

        print("검색 중...", end=" ", flush=True)
        try:
            answer, history = chat(history, user_input)
            print(f"\n\n답변: {answer}")
        except Exception as e:
            print(f"\n[오류] {e}")


if __name__ == "__main__":
    main()
