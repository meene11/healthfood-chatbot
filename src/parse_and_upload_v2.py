"""
Parent-Child Chunking + 임베딩 + Supabase 업로드 (v2)
- Parent: 1600토큰 대형 청크 → LLM 답변 생성용
- Child: 200토큰 소형 청크 → 벡터 검색용 (임베딩 포함)
- 부모-자식 관계를 parent_id로 연결
"""
import json
import re
import os
from pathlib import Path

import pdfplumber
import tiktoken
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ── 청크 설정 ─────────────────────────────────────────────────────────
PARENT_TOKENS  = 1600   # 부모: 큰 단위 (LLM 컨텍스트용)
PARENT_OVERLAP = 200
CHILD_TOKENS   = 200    # 자식: 작은 단위 (검색 정밀도용)
CHILD_OVERLAP  = 40
MIN_CHARS      = 50

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32
UPLOAD_BATCH = 20

EXCLUDE_DIRS = {"claude-code-main (1)"}

CATEGORY_MAP = {
    "papers/health_food": "건강식품_논문",
    "papers/diet": "다이어트_논문",
    "papers/diet_nutrition": "다이어트영양소_논문",
    "papers/glucose_spike": "혈당스파이크_논문",
    "health_food2": "건강식품_논문",
    "diet2": "다이어트_논문",
    "ai_dinner_diet": "다이어트_AI생성",
    "collected": "건강_수집데이터",
    "foodology": "푸드올로지",
    "foodology_thevc_기업투자정보": "푸드올로지",
    "naver_blog": "네이버블로그",
    "naver_blog_2": "네이버블로그",
    "blog": "건강기사",
}


def detect_category(file_path: Path) -> str:
    rel = str(file_path.relative_to(RAW_DIR)).replace("\\", "/")
    for prefix, category in CATEGORY_MAP.items():
        if rel.startswith(prefix):
            return category
    name = file_path.stem.lower()
    if "food" in name or "health" in name:
        return "건강식품"
    if "diet" in name:
        return "다이어트"
    return "기타"


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def tokenize(text: str) -> list[int]:
    return TOKENIZER.encode(text)


def split_tokens(tokens: list[int], chunk_size: int, overlap: int) -> list[list[int]]:
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


# ── 파일 파싱 ─────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """PDF → [{page: int, text: str}]"""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and len(clean_text(text)) >= MIN_CHARS:
                    pages.append({"page": i, "text": clean_text(text)})
    except Exception as e:
        print(f"  [ERROR] {pdf_path.name}: {e}")
    return pages


def extract_text_from_txt(txt_path: Path) -> list[dict]:
    for enc in ["utf-8", "cp949", "euc-kr", "latin-1"]:
        try:
            text = txt_path.read_text(encoding=enc)
            text = clean_text(text)
            if len(text) >= MIN_CHARS:
                return [{"page": None, "text": text}]
            return []
        except (UnicodeDecodeError, ValueError):
            continue
    return []


def extract_text_from_json(json_path: Path) -> list[dict]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    texts = []
    items = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list) and val and isinstance(val[0], dict):
                items.extend(val)
            elif isinstance(val, dict):
                for vv in val.values():
                    if isinstance(vv, list) and vv and isinstance(vv[0], dict):
                        items.extend(vv)
        if not items:
            items = [data]

    for item in items:
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        text = clean_text(text)
        if len(text) >= MIN_CHARS:
            texts.append({"page": None, "text": text})
    return texts


# ── Parent-Child 청크 생성 ────────────────────────────────────────────
def create_parent_child_chunks(pages: list[dict], source_file: str, category: str):
    """
    페이지들을 합쳐서 Parent(1600토큰) → Child(200토큰) 구조 생성
    """
    # 전체 텍스트와 페이지 매핑 생성
    full_text = "\n\n".join(p["text"] for p in pages)
    full_tokens = tokenize(full_text)

    if len(full_tokens) < MIN_CHARS:
        return [], []

    # Parent 청크 생성
    parent_token_chunks = split_tokens(full_tokens, PARENT_TOKENS, PARENT_OVERLAP)
    parents = []
    children = []

    for pidx, ptokens in enumerate(parent_token_chunks):
        parent_text = TOKENIZER.decode(ptokens).strip()
        if len(parent_text) < MIN_CHARS:
            continue

        # 페이지 범위 추정
        page_start = pages[0]["page"] if pages else None
        page_end = pages[-1]["page"] if pages else None

        parent = {
            "source_file": source_file,
            "category": category,
            "page_start": page_start,
            "page_end": page_end,
            "content": parent_text,
            "token_count": len(ptokens),
            "metadata": {
                "file_path": source_file,
                "category": category,
                "parent_index": pidx,
            },
        }
        parents.append(parent)

        # Child 청크 생성 (이 부모 내에서)
        child_token_chunks = split_tokens(ptokens, CHILD_TOKENS, CHILD_OVERLAP)
        for cidx, ctokens in enumerate(child_token_chunks):
            child_text = TOKENIZER.decode(ctokens).strip()
            if len(child_text) < MIN_CHARS:
                continue
            children.append({
                "_parent_index": len(parents) - 1,
                "source_file": source_file,
                "category": category,
                "page_number": page_start,
                "chunk_index": cidx,
                "content": child_text,
                "token_count": len(ctokens),
                "metadata": {
                    "file_path": source_file,
                    "category": category,
                    "parent_index": pidx,
                    "child_index": cidx,
                },
            })

    return parents, children


# ── 메인 ──────────────────────────────────────────────────────────────
def main():
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    # 1. 전체 파일 수집
    print("=" * 60)
    print("[1/4] 파일 수집")
    print("=" * 60)

    files = {"pdf": [], "txt": [], "json": []}
    for f in RAW_DIR.rglob("*"):
        if not f.is_file() or any(ex in str(f) for ex in EXCLUDE_DIRS):
            continue
        if f.suffix == ".zip":
            continue
        if f.suffix.lower() == ".pdf":
            files["pdf"].append(f)
        elif f.suffix.lower() == ".txt":
            files["txt"].append(f)
        elif f.suffix.lower() == ".json":
            files["json"].append(f)

    # generated JSON도 포함
    gen_dir = BASE_DIR / "data" / "generated"
    for f in gen_dir.rglob("*.json"):
        if "chunks" not in str(f):
            files["json"].append(f)

    total = sum(len(v) for v in files.values())
    print(f"  PDF: {len(files['pdf'])}  TXT: {len(files['txt'])}  JSON: {len(files['json'])}  총: {total}")

    # 2. Parent-Child 청크 생성
    print(f"\n{'='*60}")
    print("[2/4] Parent-Child 청크 생성")
    print(f"  Parent: {PARENT_TOKENS}토큰 | Child: {CHILD_TOKENS}토큰")
    print("=" * 60)

    all_parents = []
    all_children = []

    # PDF
    for f in tqdm(sorted(files["pdf"]), desc="PDF"):
        cat = detect_category(f)
        pages = extract_text_from_pdf(f)
        if not pages:
            continue
        src = str(f.relative_to(BASE_DIR)).replace("\\", "/")
        parents, children = create_parent_child_chunks(pages, src, cat)
        all_parents.extend(parents)
        all_children.extend(children)

    # TXT
    for f in tqdm(sorted(files["txt"]), desc="TXT"):
        cat = detect_category(f)
        pages = extract_text_from_txt(f)
        if not pages:
            continue
        src = str(f.relative_to(BASE_DIR)).replace("\\", "/")
        parents, children = create_parent_child_chunks(pages, src, cat)
        all_parents.extend(parents)
        all_children.extend(children)

    # JSON
    for f in tqdm(sorted(files["json"]), desc="JSON"):
        if f.is_relative_to(RAW_DIR):
            cat = detect_category(f)
        else:
            name = f.stem.lower()
            cat = "푸드올로지" if "foodology" in name else "다이어트" if "diet" in name or "blog" in name else "건강식품"
        pages = extract_text_from_json(f)
        if not pages:
            continue
        src = str(f.relative_to(BASE_DIR)).replace("\\", "/")
        parents, children = create_parent_child_chunks(pages, src, cat)
        all_parents.extend(parents)
        all_children.extend(children)

    print(f"\n  Parent 청크: {len(all_parents)}개")
    print(f"  Child 청크:  {len(all_children)}개")
    print(f"  비율: 1 Parent = 평균 {len(all_children)/max(len(all_parents),1):.1f} Children")

    # 3. 임베딩 (Child만)
    print(f"\n{'='*60}")
    print("[3/4] Child 임베딩 생성 (로컬, 무료)")
    print("=" * 60)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    for i in tqdm(range(0, len(all_children), BATCH_SIZE), desc="임베딩"):
        batch = all_children[i:i + BATCH_SIZE]
        texts = [f"passage: {c['content'][:2000]}" for c in batch]
        embs = embed_model.encode(texts, normalize_embeddings=True)
        for child, emb in zip(batch, embs):
            child["embedding"] = emb.tolist()

    # 4. Supabase 업로드
    print(f"\n{'='*60}")
    print("[4/4] Supabase 업로드")
    print("=" * 60)

    # Parent 업로드
    parent_ids = []
    print(f"  Parent {len(all_parents)}개 업로드...")
    for i in tqdm(range(0, len(all_parents), UPLOAD_BATCH), desc="Parent 업로드"):
        batch = all_parents[i:i + UPLOAD_BATCH]
        rows = [{
            "source_file": p["source_file"],
            "category": p["category"],
            "page_start": p["page_start"],
            "page_end": p["page_end"],
            "content": p["content"][:15000],
            "token_count": p["token_count"],
            "metadata": p["metadata"],
        } for p in batch]
        try:
            result = supabase.table("parent_chunks").insert(rows).execute()
            for row in result.data:
                parent_ids.append(row["id"])
        except Exception as e:
            print(f"\n  [ERROR] Parent: {e}")

    print(f"  Parent 업로드 완료: {len(parent_ids)}개")

    # Parent ID 매핑 → Child에 연결
    for child in all_children:
        pidx = child.pop("_parent_index")
        if pidx < len(parent_ids):
            child["parent_id"] = parent_ids[pidx]
        else:
            child["parent_id"] = None

    # Child 업로드
    success = 0
    print(f"  Child {len(all_children)}개 업로드...")
    for i in tqdm(range(0, len(all_children), UPLOAD_BATCH), desc="Child 업로드"):
        batch = all_children[i:i + UPLOAD_BATCH]
        rows = [{
            "parent_id": c.get("parent_id"),
            "source_file": c["source_file"],
            "category": c["category"],
            "page_number": c.get("page_number"),
            "chunk_index": c.get("chunk_index"),
            "content": c["content"][:10000],
            "token_count": c.get("token_count"),
            "embedding": c.get("embedding"),
            "metadata": c.get("metadata", {}),
        } for c in batch]
        try:
            supabase.table("child_chunks").insert(rows).execute()
            success += len(batch)
        except Exception as e:
            print(f"\n  [ERROR] Child: {e}")

    print(f"\n{'='*60}")
    print(f"완료!")
    print(f"  Parent: {len(parent_ids)}개 업로드")
    print(f"  Child:  {success}/{len(all_children)}개 업로드")
    print("=" * 60)


if __name__ == "__main__":
    main()
