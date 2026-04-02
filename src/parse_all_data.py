"""
전체 raw 데이터 → 청크 JSON 통합 변환기
- data/raw/ 하위 모든 PDF, TXT, JSON 파싱
- PDF: 페이지별 텍스트 추출 후 토큰 단위 청크 분할
- TXT: 텍스트를 토큰 단위 청크 분할
- JSON: 항목별 텍스트 변환 후 청크 분할
- 출력: data/generated/chunks/all_chunks.json

제외 대상: claude-code-main (관련 없는 소스코드), .zip 파일
"""
import json
import re
from pathlib import Path

import pdfplumber
import tiktoken
from tqdm import tqdm

# ── 설정 ─────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
GEN_DIR     = BASE_DIR / "data" / "generated"
OUTPUT_DIR  = BASE_DIR / "data" / "generated" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "all_chunks.json"

CHUNK_TOKENS   = 400
OVERLAP_TOKENS = 60
MIN_CHARS      = 50

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# claude-code 소스코드 제외
EXCLUDE_DIRS = {"claude-code-main (1)"}

# ── 카테고리 자동 매핑 ────────────────────────────────────────────────
CATEGORY_MAP = {
    "papers/health_food":   "건강식품_논문",
    "papers/diet":          "다이어트_논문",
    "papers/diet_nutrition": "다이어트영양소_논문",
    "health_food2":         "건강식품_논문",
    "diet2":                "다이어트_논문",
    "ai_dinner_diet":       "다이어트_AI생성",
    "collected":            "건강_수집데이터",
    "foodology":            "푸드올로지",
    "foodology_thevc_기업투자정보": "푸드올로지",
    "naver_blog":           "네이버블로그",
    "naver_blog_2":         "네이버블로그",
    "blog":                 "건강기사",
}


def detect_category(file_path: Path) -> str:
    """파일 경로에서 카테고리 자동 감지"""
    rel = str(file_path.relative_to(RAW_DIR)).replace("\\", "/")
    for prefix, category in CATEGORY_MAP.items():
        if rel.startswith(prefix):
            return category
    # 루트 JSON 파일
    name = file_path.stem.lower()
    if "food" in name or "health" in name or "nih" in name or "pmc" in name:
        return "건강식품"
    if "diet" in name:
        return "다이어트"
    if "faq" in name:
        return "FAQ"
    if "categor" in name or "ingredient" in name:
        return "건강식품"
    return "기타"


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def split_into_chunks(text: str, source_file: str, page_num: int | None,
                      category: str) -> list[dict]:
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
        start = end - OVERLAP_TOKENS

    return chunks


# ── PDF 파싱 ──────────────────────────────────────────────────────────
def parse_pdf(pdf_path: Path, category: str) -> list[dict]:
    source_file = str(pdf_path.relative_to(BASE_DIR)).replace("\\", "/")
    all_chunks = []
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
        print(f"  [ERROR] PDF {pdf_path.name}: {e}")
    return all_chunks


# ── TXT 파싱 ──────────────────────────────────────────────────────────
def parse_txt(txt_path: Path, category: str) -> list[dict]:
    source_file = str(txt_path.relative_to(BASE_DIR)).replace("\\", "/")
    try:
        for enc in ["utf-8", "cp949", "euc-kr", "latin-1"]:
            try:
                text = txt_path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, ValueError):
                continue
        else:
            print(f"  [ERROR] TXT 인코딩 실패: {txt_path.name}")
            return []

        text = clean_text(text)
        if len(text) < MIN_CHARS:
            return []
        return split_into_chunks(text, source_file, None, category)
    except Exception as e:
        print(f"  [ERROR] TXT {txt_path.name}: {e}")
        return []


# ── JSON 파싱 ─────────────────────────────────────────────────────────
def json_to_text(obj, depth: int = 0) -> str:
    """JSON 객체를 평탄화된 텍스트로 변환"""
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("id", "url", "doi", "local_path", "size_kb", "source_urls"):
                continue
            if isinstance(v, str) and len(v) > 5:
                lines.append(f"{k}: {v}")
            elif isinstance(v, list):
                items = []
                for item in v:
                    if isinstance(item, str):
                        items.append(item)
                    elif isinstance(item, dict):
                        items.append(json_to_text(item, depth + 1))
                if items:
                    lines.append(f"{k}: " + " | ".join(items[:20]))
            elif isinstance(v, dict) and depth < 2:
                sub = json_to_text(v, depth + 1)
                if sub:
                    lines.append(f"{k}: {sub}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                lines.append(json_to_text(item, depth))
            elif isinstance(item, str):
                lines.append(item)
    return "\n".join(lines)


def parse_json_file(json_path: Path, category: str) -> list[dict]:
    source_file = str(json_path.relative_to(BASE_DIR)).replace("\\", "/")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [ERROR] JSON {json_path.name}: {e}")
        return []

    # 항목 리스트 추출
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # 중첩 리스트 찾기
        for val in data.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                items.extend(val)
            elif isinstance(val, dict):
                for vv in val.values():
                    if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                        items.extend(vv)
        if not items:
            items = [data]  # dict 자체를 하나의 항목으로

    all_chunks = []
    for item in items:
        text = json_to_text(item) if isinstance(item, dict) else str(item)
        text = clean_text(text)
        if len(text) < MIN_CHARS:
            continue
        chunks = split_into_chunks(text, source_file, None, category)
        all_chunks.extend(chunks)

    return all_chunks


# ── generated 디렉토리 JSON도 파싱 ─────────────────────────────────────
def collect_generated_jsons() -> list[Path]:
    """generated/ 하위 JSON (chunks 제외)"""
    jsons = []
    for f in GEN_DIR.rglob("*.json"):
        if "chunks" in str(f):
            continue
        jsons.append(f)
    return jsons


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    # 1) raw/ 하위 파일 수집 (claude-code 제외)
    raw_files = {"pdf": [], "txt": [], "json": []}
    for f in RAW_DIR.rglob("*"):
        if not f.is_file():
            continue
        # 제외 디렉토리 확인
        if any(ex in str(f) for ex in EXCLUDE_DIRS):
            continue
        if f.suffix == ".zip":
            continue

        if f.suffix.lower() == ".pdf":
            raw_files["pdf"].append(f)
        elif f.suffix.lower() == ".txt":
            raw_files["txt"].append(f)
        elif f.suffix.lower() == ".json":
            raw_files["json"].append(f)

    # generated/ JSON도 추가
    gen_jsons = collect_generated_jsons()

    total = sum(len(v) for v in raw_files.values()) + len(gen_jsons)
    print(f"{'='*60}")
    print(f"전체 데이터 파싱 시작")
    print(f"  PDF: {len(raw_files['pdf'])}개")
    print(f"  TXT: {len(raw_files['txt'])}개")
    print(f"  JSON (raw): {len(raw_files['json'])}개")
    print(f"  JSON (generated): {len(gen_jsons)}개")
    print(f"  총 {total}개 파일")
    print(f"{'='*60}\n")

    # PDF 파싱
    print("[1/4] PDF 파싱...")
    for f in tqdm(sorted(raw_files["pdf"]), desc="PDF"):
        cat = detect_category(f)
        chunks = parse_pdf(f, cat)
        all_chunks.extend(chunks)
        if chunks:
            tqdm.write(f"  {f.name}: {len(chunks)}청크")

    # TXT 파싱
    print(f"\n[2/4] TXT 파싱...")
    for f in tqdm(sorted(raw_files["txt"]), desc="TXT"):
        cat = detect_category(f)
        chunks = parse_txt(f, cat)
        all_chunks.extend(chunks)

    # raw JSON 파싱
    print(f"\n[3/4] raw JSON 파싱...")
    for f in tqdm(sorted(raw_files["json"]), desc="JSON(raw)"):
        cat = detect_category(f)
        chunks = parse_json_file(f, cat)
        all_chunks.extend(chunks)
        if chunks:
            tqdm.write(f"  {f.name}: {len(chunks)}청크")

    # generated JSON 파싱
    print(f"\n[4/4] generated JSON 파싱...")
    for f in tqdm(sorted(gen_jsons), desc="JSON(gen)"):
        # generated JSON 카테고리는 파일 내용으로 판단
        name = f.stem.lower()
        if "foodology" in name:
            cat = "푸드올로지"
        elif "diet" in name or "blog" in name:
            cat = "다이어트"
        elif "paper" in name or "nutrition" in name:
            cat = "논문인덱스"
        else:
            cat = "건강식품"
        chunks = parse_json_file(f, cat)
        all_chunks.extend(chunks)
        if chunks:
            tqdm.write(f"  {f.name}: {len(chunks)}청크")

    # 전역 ID 부여
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = i

    # JSON 저장
    OUTPUT_FILE.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"완료: 총 {len(all_chunks)}개 청크 생성")
    print(f"저장: {OUTPUT_FILE}")

    # 통계
    from collections import Counter
    stats = Counter(c["category"] for c in all_chunks)
    print(f"\n카테고리별 청크 수:")
    for cat, cnt in stats.most_common():
        print(f"  {cat}: {cnt}개")

    # 파일 유형별 통계
    type_stats = Counter()
    for c in all_chunks:
        src = c["source_file"]
        if src.endswith(".pdf"):
            type_stats["PDF"] += 1
        elif src.endswith(".txt"):
            type_stats["TXT"] += 1
        elif src.endswith(".json"):
            type_stats["JSON"] += 1
    print(f"\n파일 유형별 청크 수:")
    for t, cnt in type_stats.most_common():
        print(f"  {t}: {cnt}개")


if __name__ == "__main__":
    main()
