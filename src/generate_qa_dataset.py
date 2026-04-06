"""
QA 쌍 데이터셋 생성 스크립트
==============================
Supabase documents_v2 테이블에서 카테고리별 청크를 균등 샘플링하고,
GPT-4o-mini로 각 청크가 답할 수 있는 질문을 생성하여
(질문, 정답_chunk_id) 쌍의 데이터셋을 만듭니다.

출력: data/generated/qa_dataset.json

실행:
  python src/generate_qa_dataset.py
"""
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR    = Path(__file__).resolve().parent.parent
OUTPUT_DIR  = BASE_DIR / "data" / "generated"
OUTPUT_FILE = OUTPUT_DIR / "qa_dataset.json"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
OPENAI_KEY   = os.environ["OPENAI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client    = OpenAI(api_key=OPENAI_KEY)

# 카테고리별 샘플 수 (총 50개)
CATEGORY_SAMPLES = {
    "다이어트_논문":        12,
    "건강식품_논문":        10,
    "다이어트영양소_논문":   8,
    "건강기사":              7,
    "푸드올로지":            5,
    "기타":                  8,
}

PROMPT_TEMPLATE = """아래 텍스트를 읽고, 이 텍스트가 직접 답할 수 있는 질문 1개를 만들어줘.

규칙:
- 텍스트에 명시적으로 답이 있는 질문만 만들 것
- 한국어로 작성
- 질문만 출력 (설명 없이)
- 너무 구체적인 숫자/수치 질문은 피할 것

텍스트:
{chunk_content}

질문:"""


def sample_chunks_by_category() -> list[dict]:
    """카테고리별로 청크를 균등 샘플링합니다."""
    sampled = []
    for category, n in CATEGORY_SAMPLES.items():
        print(f"  [{category}] {n}개 샘플링 중...")
        try:
            # 카테고리별 전체 ID 조회 (content는 제외하여 빠르게)
            resp = (
                supabase.table("documents_v2")
                .select("id, source_file, category, content, token_count")
                .eq("category", category)
                .limit(5000)
                .execute()
            )
            rows = resp.data
            if not rows:
                print(f"    [WARN] {category}: 데이터 없음")
                continue

            # 내용이 너무 짧은 청크 제외 (50토큰 미만)
            valid = [r for r in rows if (r.get("token_count") or 0) >= 50]
            if not valid:
                valid = rows  # 폴백: 전체 사용

            chosen = random.sample(valid, min(n, len(valid)))
            sampled.extend(chosen)
            print(f"    → {len(chosen)}개 선택 (풀 크기: {len(valid)})")
        except Exception as e:
            print(f"    [ERROR] {category}: {e}")

    print(f"\n총 {len(sampled)}개 청크 샘플링 완료\n")
    return sampled


def generate_question(chunk_content: str) -> str | None:
    """GPT-4o-mini로 청크 기반 질문을 생성합니다."""
    prompt = PROMPT_TEMPLATE.format(chunk_content=chunk_content[:1500])
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        question = resp.choices[0].message.content.strip()
        # 질문 부호로 끝나지 않으면 "?" 추가
        if question and not question.endswith("?"):
            question += "?"
        return question
    except Exception as e:
        print(f"    [ERROR] 질문 생성 실패: {e}")
        return None


def main():
    print("=" * 60)
    print("QA 쌍 데이터셋 생성")
    print(f"대상 테이블: documents_v2")
    print(f"총 목표: {sum(CATEGORY_SAMPLES.values())}개 QA 쌍")
    print("=" * 60)

    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 카테고리별 샘플링
    print("\n[Step 1] 카테고리별 청크 샘플링...")
    chunks = sample_chunks_by_category()
    if not chunks:
        print("[ERROR] 샘플링된 청크가 없습니다.")
        return

    # 2. 질문 생성
    print("[Step 2] GPT-4o-mini로 질문 생성 중...")
    qa_pairs = []
    for i, chunk in enumerate(chunks, 1):
        chunk_id      = chunk["id"]
        content       = chunk.get("content", "")
        category      = chunk.get("category", "")
        source_file   = chunk.get("source_file", "")

        print(f"  [{i:02d}/{len(chunks)}] chunk_id={chunk_id} ({category})")

        question = generate_question(content)
        if not question:
            print(f"    → 건너뜀 (질문 생성 실패)")
            continue

        qa_pairs.append({
            "id":                   i,
            "question":             question,
            "answer_chunk_id":      chunk_id,
            "answer_chunk_content": content[:500],  # 미리보기용 (앞 500자)
            "category":             category,
            "source_file":          source_file,
        })
        print(f"    → {question}")

        # API 레이트 리밋 방지
        time.sleep(0.5)

    # 3. 저장
    print(f"\n[Step 3] {OUTPUT_FILE} 저장 중...")
    OUTPUT_FILE.write_text(
        json.dumps(qa_pairs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"완료: {len(qa_pairs)}개 QA 쌍 저장 → {OUTPUT_FILE}")
    print(f"카테고리 분포:")
    from collections import Counter
    for cat, cnt in Counter(p["category"] for p in qa_pairs).most_common():
        print(f"  {cat}: {cnt}개")


if __name__ == "__main__":
    main()
