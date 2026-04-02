"""혈당 스파이크 논문 PDF 파싱 → 청크 JSON → Supabase 업로드"""
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
PDF_DIR = BASE_DIR / "data" / "raw" / "papers" / "glucose_spike"
OUTPUT = BASE_DIR / "data" / "generated" / "chunks" / "glucose_spike_chunks.json"

TOKENIZER = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 60
MIN_CHARS = 50

EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_DIMENSION = 384


def clean_text(text):
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def split_chunks(text, source, page, category):
    tokens = TOKENIZER.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_TOKENS, len(tokens))
        ct = tokens[start:end]
        ct_text = TOKENIZER.decode(ct).strip()
        if len(ct_text) >= MIN_CHARS:
            chunks.append({
                "source_file": source,
                "category": category,
                "page_number": page,
                "chunk_index": idx,
                "content": ct_text,
                "token_count": len(ct),
                "metadata": {"file_path": source, "category": category, "page": page, "chunk": idx}
            })
            idx += 1
        if end == len(tokens):
            break
        start = end - OVERLAP_TOKENS
    return chunks


def main():
    # 1. PDF 파싱
    print("=" * 60)
    print("[1/3] 혈당 스파이크 논문 PDF 파싱")
    print("=" * 60)

    all_chunks = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        rel_path = str(pdf.relative_to(BASE_DIR)).replace("\\", "/")
        try:
            with pdfplumber.open(pdf) as p:
                for pn, page in enumerate(p.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    text = clean_text(text)
                    if len(text) < MIN_CHARS:
                        continue
                    chunks = split_chunks(text, rel_path, pn, "혈당스파이크_논문")
                    all_chunks.extend(chunks)
            count = sum(1 for c in all_chunks if c["source_file"] == rel_path)
            print(f"  {pdf.name}: {count}개 청크")
        except Exception as e:
            print(f"  [ERROR] {pdf.name}: {e}")

    for i, c in enumerate(all_chunks):
        c["id"] = i

    OUTPUT.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n파싱 완료: {len(all_chunks)}개 청크 → {OUTPUT}")

    # 2. 임베딩 생성
    print(f"\n{'='*60}")
    print("[2/3] 임베딩 생성 (로컬, 무료)")
    print("=" * 60)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    BATCH_SIZE = 32

    all_rows = []
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="임베딩"):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [f"passage: {c['content'][:2000]}" for c in batch]
        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        for chunk, emb in zip(batch, embeddings):
            all_rows.append({
                "source_file": chunk["source_file"],
                "category": chunk["category"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"][:10000],
                "token_count": chunk["token_count"],
                "embedding": emb.tolist(),
                "metadata": chunk["metadata"],
            })

    print(f"임베딩 완료: {len(all_rows)}개")

    # 3. Supabase 업로드
    print(f"\n{'='*60}")
    print("[3/3] Supabase 업로드")
    print("=" * 60)

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    UPLOAD_BATCH = 20
    success = 0

    for i in tqdm(range(0, len(all_rows), UPLOAD_BATCH), desc="업로드"):
        batch = all_rows[i:i + UPLOAD_BATCH]
        try:
            supabase.table("documents").insert(batch).execute()
            success += len(batch)
        except Exception as e:
            print(f"\n  [ERROR] {e}")

    print(f"\n완료: {success}/{len(all_rows)}개 업로드 성공")


if __name__ == "__main__":
    main()
