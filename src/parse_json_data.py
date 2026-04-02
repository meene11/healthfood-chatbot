"""
JSON 데이터 → 청크 JSON 변환기
- data/raw/ 및 data/generated/ 하위 JSON 파일을 파싱
- 각 항목을 의미 단위 청크로 변환
- 출력: data/generated/chunks/json_chunks.json
"""
import json
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "generated" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "json_chunks.json"

# 파싱할 JSON 파일 목록 (경로, 카테고리, 설명)
JSON_SOURCES = [
    ("data/raw/health_foods.json",                        "건강식품",      "건강식품 기본 정보"),
    ("data/raw/categories.json",                          "건강식품",      "건강식품 카테고리 및 키워드"),
    ("data/raw/faq.json",                                 "FAQ",           "건강식품 자주 묻는 질문"),
    ("data/raw/crawled_nih_omega3.json",                  "건강식품",      "NIH 오메가-3 정보"),
    ("data/raw/crawled_nih_vitamin_d.json",               "건강식품",      "NIH 비타민D 정보"),
    ("data/raw/crawled_pmc_probiotics_research.json",     "건강식품",      "PMC 프로바이오틱스 연구"),
    ("data/raw/foodology/company_info.json",              "푸드올로지",    "푸드올로지 회사 정보"),
    ("data/raw/foodology/products.json",                  "푸드올로지",    "푸드올로지 제품 정보"),
    ("data/raw/blog/health_diet_articles_raw.json",       "다이어트",      "다이어트·건강식품 기사"),
    ("data/generated/unified_knowledge_base.json",        "건강식품",      "통합 지식 베이스"),
    ("data/generated/chatbot_qa_pairs.json",              "FAQ",           "챗봇 Q&A 쌍"),
    ("data/generated/ingredient_index.json",              "건강식품",      "성분 인덱스"),
    ("data/generated/foodology/foodology_knowledge_base.json", "푸드올로지", "푸드올로지 지식 베이스"),
    ("data/generated/blog/health_diet_knowledge_base.json",    "다이어트",  "다이어트·건강식품 지식 베이스"),
    ("data/generated/papers_summary/papers_index.json",        "논문",      "논문 인덱스 (Frontiers)"),
    ("data/generated/papers_summary/new_papers_index.json",    "논문",      "논문 인덱스 (PMC)"),
    ("data/generated/papers_summary/diet_nutrition_index.json","논문",      "다이어트 영양소 논문 인덱스"),
]


def flatten_value(val, prefix: str = "") -> list[str]:
    """JSON 값을 평탄화하여 텍스트 리스트로 변환"""
    texts = []
    if isinstance(val, str) and len(val.strip()) > 10:
        texts.append(f"{prefix}: {val.strip()}" if prefix else val.strip())
    elif isinstance(val, list):
        for item in val:
            texts.extend(flatten_value(item, prefix))
    elif isinstance(val, dict):
        for k, v in val.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            texts.extend(flatten_value(v, new_prefix))
    return texts


def json_item_to_text(item: dict) -> str:
    """JSON 항목 하나를 사람이 읽기 좋은 텍스트로 변환"""
    lines = []

    # 공통 필드 우선 처리
    priority_keys = ["title", "name", "question", "q", "answer", "a", "content",
                     "topic_ko", "key_findings", "chatbot_keywords",
                     "description", "function", "benefits", "precautions"]

    seen_keys = set()
    for key in priority_keys:
        if key in item:
            val = item[key]
            seen_keys.add(key)
            if isinstance(val, str):
                lines.append(f"{key}: {val}")
            elif isinstance(val, list):
                lines.append(f"{key}: {', '.join(str(v) for v in val)}")
            elif isinstance(val, dict):
                lines.append(f"{key}: {json.dumps(val, ensure_ascii=False)}")

    # 나머지 필드
    for key, val in item.items():
        if key in seen_keys or key in ("id", "url", "doi", "local_path", "size_kb"):
            continue
        texts = flatten_value(val, key)
        lines.extend(texts[:5])  # 필드당 최대 5줄

    return "\n".join(lines)


def parse_json_file(rel_path: str, category: str, description: str) -> list[dict]:
    """단일 JSON 파일 → 청크 리스트"""
    file_path = BASE_DIR / rel_path
    if not file_path.exists():
        print(f"  [SKIP] 파일 없음: {rel_path}")
        return []

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [ERROR] {rel_path}: {e}")
        return []

    chunks = []

    # 리스트 최상위
    if isinstance(data, list):
        items = data
    # dict 안의 리스트 찾기
    elif isinstance(data, dict):
        # papers_index 구조: {"papers": {"health_food": [...], "diet": [...]}}
        list_found = None
        for val in data.values():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                list_found = val
                break
            elif isinstance(val, dict):
                # 한 단계 더 내려가기
                for vv in val.values():
                    if isinstance(vv, list) and len(vv) > 0 and isinstance(vv[0], dict):
                        if list_found is None:
                            list_found = []
                        list_found.extend(vv)

        if list_found:
            items = list_found
        else:
            # dict 자체를 하나의 항목으로
            items = [data]
    else:
        return []

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        text = json_item_to_text(item)
        if len(text.strip()) < 20:
            continue

        chunks.append({
            "id":          None,  # 나중에 전역 부여
            "source_file": rel_path,
            "category":    category,
            "description": description,
            "page_number": None,
            "chunk_index": i,
            "content":     text.strip(),
            "token_count": len(text.split()),
            "metadata": {
                "file_path":   rel_path,
                "category":    category,
                "description": description,
                "item_index":  i,
            }
        })

    print(f"  {file_path.name}: {len(chunks)}개 청크")
    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    print(f"JSON 파일 파싱 시작 ({len(JSON_SOURCES)}개)\n{'='*60}")
    for rel_path, category, description in JSON_SOURCES:
        chunks = parse_json_file(rel_path, category, description)
        all_chunks.extend(chunks)

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = i

    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"완료: 총 {len(all_chunks)}개 청크")
    print(f"저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
