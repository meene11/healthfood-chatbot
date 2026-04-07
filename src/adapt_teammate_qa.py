"""
조원 QA 데이터 → 내 챗봇 평가 형식 변환 스크립트
=====================================================
qa_pairs.json 원본에서:
  1. 내 챗봇(건강식품 RAG)에 적합하지 않은 질문 필터링
  2. 평가에 필요한 형식으로 변환
  3. data/generated/teammate_qa_adapted.json 저장

필터링 기준 (부적합 질문):
  - 특정 저자/연구자 이름 묻는 질문
  - 특정 연도/저널 묻는 질문
  - "이 문서", "이 연구" 등 문맥 의존 질문
  - 연예인·브랜드 모델 등 완전 오프토픽
  - 단순 약어 해석 (ALT=?, FBS=? 등)
    → RAG 챗봇보다 사전 수준 질문이라 평가 의미 낮음

실행:
  python src/adapt_teammate_qa.py
"""

import json
import sys
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
INPUT_FILE = Path("C:/Users/804/Downloads/qa_pairs.json")
OUTPUT_DIR = BASE_DIR / "data" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "teammate_qa_adapted.json"

# ── 부적합 질문 패턴 ────────────────────────────────────────────────
BAD_PATTERNS = [
    # 저자/연구자 이름 기반
    "저자", "연구자들은", "et al", "외의 연구",
    # 연도/저널/출판 기반
    "저널에", "저널에 게재", "어떤 연도", "년에 발표된", "년에 실시된",
    "년에 출판된", "언제 출판", "문서가 언제",
    # 문맥 의존 (이 문서, 이 연구)
    "이 문서의 저자", "이 문서의 제목", "문서의 제목", "이 문서에서 다루",
    "문서에서 다루는 주요 주제", "문서에 언급된",
    "문서에서 연구된", "문서가 언제",
    # 연구 그룹/대상 (문맥 없이 물어봄)
    "Aldrich et al", "Lindqvist et al", "Melanson et al",
    "Abete et al", "Neacsu et al", "Gabel et al",
    # 완전 오프토픽 (연예인, 브랜드 모델)
    "윤성빈", "신봉선", "브랜드 메인 모델", "라인에서 모델",
    # 단순 약어 해석 (사전 수준)
    "FBS는 어떤 측정",
    # 너무 모호한 질문
    "P 그룹에서 관찰된", "연구에서 비교된 대상",
]

# 추가로 부적합한 id (수동 검토)
BAD_IDS = {
    1,   # 2018년에 실시된 연구 (너무 모호)
    8,   # 2021년에 발표된 연구 (너무 모호)
    15,  # BCAAs 6주 지구력 훈련 그룹 (문맥 의존)
    16,  # 어떤 식사 전후 에너지 섭취 (문맥 의존)
    22,  # ME + BCAA 그룹 결과 (문맥 의존)
    28,  # Aldrich et al 연구 결과
    32,  # 비교된 대상 그룹 (모호)
    51,  # P 그룹에서 관찰된 변화 (모호)
    53,  # 이 문서의 주요 주제 (모호)
    54,  # 운동 그룹과 단백질 그룹 차이 (문맥 의존)
    55,  # 연구된 주요 호르몬 (모호)
    59,  # 문서가 언제 출판되었나요?
    71,  # 2019년에 발표된 논문 (모호)
    74,  # Abete et al 연구
    80,  # Melanson et al 연구
    81,  # 문서의 주요 연구 방법 (모호)
    82,  # 어떤 저널에 게재 (저널 이름)
    84,  # 어떤 저널에 실렸나 (저널 이름)
    89,  # Lindqvist et al 환자군
    91,  # 언급된 연구자들은 누구 (저자)
    92,  # 3개월 동안의 연구 측정 (문맥 의존)
    94,  # Gabel et al 연구 방법
    95,  # 2022년 저자
    96,  # 문서의 주요 주제 (모호)
    99,  # Neacsu et al 다이어트
}


def is_bad(item: dict) -> tuple[bool, str]:
    """질문이 부적합한지 판단. (is_bad, reason) 반환"""
    qa_id = item["id"]
    question = item["question"]

    if qa_id in BAD_IDS:
        return True, "manual_review"

    for pattern in BAD_PATTERNS:
        if pattern in question:
            return True, f"pattern: {pattern}"

    return False, ""


def adapt_item(item: dict) -> dict:
    """단일 QA 항목을 평가 형식으로 변환"""
    meta = item.get("source_metadata", {})
    return {
        "id":               item["id"],
        "question":         item["question"],
        "reference_answer": item["reference_answer"],
        "source_content":   item["source_content"],   # 검색 커버리지 평가용
        "source_file":      meta.get("source_file", ""),
        "category":         meta.get("category", "general"),
        "language":         meta.get("language", "en"),
        "paper_title":      meta.get("title", ""),
    }


def main():
    print(f"입력 파일: {INPUT_FILE}")
    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    print(f"원본 QA 쌍: {len(raw)}개")

    adapted = []
    filtered_out = []

    for item in raw:
        bad, reason = is_bad(item)
        if bad:
            filtered_out.append({"id": item["id"], "question": item["question"], "reason": reason})
        else:
            adapted.append(adapt_item(item))

    # 카테고리별 분포
    from collections import Counter
    cat_dist = Counter(a["category"] for a in adapted)

    print(f"\n필터링 후 평가 대상: {len(adapted)}개 / 제외: {len(filtered_out)}개")
    print(f"카테고리 분포: {dict(cat_dist)}")
    print("\n제외된 질문 목록:")
    for f in filtered_out:
        print(f"  [{f['id']:3d}] ({f['reason'][:30]}) {f['question'][:60]}")

    OUTPUT_FILE.write_text(
        json.dumps(adapted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n저장 완료: {OUTPUT_FILE}")
    print(f"평가 대상 질문 수: {len(adapted)}개")

    return len(adapted)


if __name__ == "__main__":
    main()
