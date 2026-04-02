"""
PDF → 청크 JSON 변환기
- data/raw/papers/ 하위 모든 PDF를 파싱
- 페이지별 텍스트 추출 후 의미 단위(청크)로 분할
- source_file, page_number, chunk_index, content, metadata 포함
- 출력: data/generated/chunks/all_chunks.json
"""
import json
import re
from pathlib import Path

import pdfplumber
import tiktoken
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
PAPERS_DIR     = BASE_DIR / "data" / "raw" / "papers"
OUTPUT_DIR     = BASE_DIR / "data" / "generated" / "chunks"
OUTPUT_FILE    = OUTPUT_DIR / "all_chunks.json"

CHUNK_TOKENS   = 400   # 청크당 최대 토큰 수
OVERLAP_TOKENS = 60    # 청크 간 오버랩 토큰 수
MIN_CHARS      = 80    # 이 글자 수 미만 페이지는 건너뜀 (헤더/푸터 등)

TOKENIZER = tiktoken.get_encoding("cl100k_base")  # OpenAI 임베딩 기준


# ── 카테고리 매핑 ─────────────────────────────────────────────────────
CATEGORY_MAP = {
    "health_food":    "건강식품",
    "diet":           "다이어트",
    "diet_nutrition": "다이어트_영양소",
}


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def clean_text(text: str) -> str:
    """PDF 추출 텍스트 전처리"""
    # 연속 공백·줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # 하이픈 줄바꿈 병합 (단어 분리 복원)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def split_into_chunks(text: str, source_file: str, page_num: int, category: str) -> list[dict]:
    """
    텍스트를 CHUNK_TOKENS 단위로 분할, OVERLAP_TOKENS 오버랩 적용.
    반환: chunk dict 리스트
    """
    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = TOKENIZER.decode(chunk_tokens).strip()

        if len(chunk_text) >= MIN_CHARS:
            chunks.append({
                "source_file":  source_file,
                "category":     category,
                "page_number":  page_num,
                "chunk_index":  chunk_idx,
                "content":      chunk_text,
                "token_count":  len(chunk_tokens),
                "metadata": {
                    "file_path": source_file,
                    "category":  category,
                    "page":      page_num,
                    "chunk":     chunk_idx,
                }
            })
            chunk_idx += 1

        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS  # 오버랩

    return chunks


def parse_pdf(pdf_path: Path, category: str) -> list[dict]:
    """단일 PDF → 청크 리스트"""
    all_chunks = []
    source_file = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                if not raw_text:
                    continue

                text = clean_text(raw_text)
                if len(text) < MIN_CHARS:
                    continue

                chunks = split_into_chunks(text, source_file, page_num, category)
                all_chunks.extend(chunks)

    except Exception as e:
        print(f"  [ERROR] {pdf_path.name}: {e}")

    return all_chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    pdf_files = []

    # 모든 PDF 수집
    for subdir in PAPERS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        category_en = subdir.name  # health_food / diet / diet_nutrition
        for pdf in sorted(subdir.glob("*.pdf")):
            pdf_files.append((pdf, category_en))

    print(f"총 {len(pdf_files)}개 PDF 파싱 시작\n{'='*60}")

    for pdf_path, category_en in tqdm(pdf_files, desc="PDF 파싱"):
        category_ko = CATEGORY_MAP.get(category_en, category_en)
        chunks = parse_pdf(pdf_path, category_ko)
        all_chunks.extend(chunks)
        tqdm.write(f"  {pdf_path.name}: {len(chunks)}개 청크")

    # 전체 청크 번호 부여
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = i

    # JSON 저장
    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"완료: 총 {len(all_chunks)}개 청크")
    print(f"저장 위치: {OUTPUT_FILE}")

    # 카테고리별 통계
    from collections import Counter
    stats = Counter(c["category"] for c in all_chunks)
    for cat, cnt in stats.items():
        print(f"  {cat}: {cnt}개 청크")


if __name__ == "__main__":
    main()
