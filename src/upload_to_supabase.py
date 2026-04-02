"""
청크 JSON → Supabase pgvector 업로드
- data/generated/chunks/all_chunks.json 읽기
- 무료 로컬 임베딩 모델: intfloat/multilingual-e5-small (384차원)
  한국어+영어 다국어 지원, API 비용 없음
- Supabase documents 테이블에 배치 업로드
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR   = Path(__file__).resolve().parent.parent
CHUNKS_DIR = BASE_DIR / "data" / "generated" / "chunks"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

# ── 무료 로컬 임베딩 모델 ─────────────────────────────────────────────
# intfloat/multilingual-e5-small: 384차원, 한국어+영어 다국어 지원
# 첫 실행 시 ~470MB 다운로드, 이후 캐시 사용
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBED_DIMENSION  = 384
BATCH_SIZE       = 32     # 로컬 모델이라 배치 크기 조절
UPLOAD_BATCH     = 20
MAX_CONTENT      = 2000   # 로컬 모델 최대 512토큰 → 한국어 ~2000자

print("임베딩 모델 로딩 중... (첫 실행 시 ~470MB 다운로드)")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"모델 로드 완료: {EMBED_MODEL_NAME} ({EMBED_DIMENSION}차원)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """로컬 모델로 임베딩 생성 (무료, API 호출 없음)"""
    # E5 모델은 "query: " 또는 "passage: " 프리픽스 필요
    prefixed = [f"passage: {t[:MAX_CONTENT]}" for t in texts]
    embeddings = embed_model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


def get_query_embedding(text: str) -> list[float]:
    """검색 쿼리용 임베딩 (query 프리픽스)"""
    embedding = embed_model.encode(
        f"query: {text[:MAX_CONTENT]}",
        normalize_embeddings=True
    )
    return embedding.tolist()


def load_chunks() -> list[dict]:
    """통합 청크 파일 로드"""
    chunk_files = [
        CHUNKS_DIR / "all_chunks.json",
    ]
    all_chunks = []
    for f in chunk_files:
        if f.exists():
            data = json.loads(f.read_text(encoding="utf-8"))
            print(f"  로드: {f.name} ({len(data)}개 청크)")
            all_chunks.extend(data)
        else:
            print(f"  [SKIP] 파일 없음: {f.name}")
    return all_chunks


def chunk_to_row(chunk: dict, embedding: list[float]) -> dict:
    """청크 dict → Supabase insert용 dict"""
    return {
        "source_file": chunk.get("source_file"),
        "category":    chunk.get("category"),
        "description": chunk.get("description"),
        "page_number": chunk.get("page_number"),
        "chunk_index": chunk.get("chunk_index"),
        "content":     chunk.get("content", "")[:10000],
        "token_count": chunk.get("token_count"),
        "embedding":   embedding,
        "metadata":    chunk.get("metadata", {}),
    }


def upload_to_supabase(rows: list[dict]):
    """Supabase에 배치 업로드"""
    supabase.table("documents").insert(rows).execute()


def main():
    print("="*60)
    print("Supabase 벡터 DB 업로드 시작 (무료 로컬 임베딩)")
    print(f"모델: {EMBED_MODEL_NAME} ({EMBED_DIMENSION}차원)")
    print("="*60)

    chunks = load_chunks()
    if not chunks:
        print("업로드할 청크가 없습니다. 먼저 parse_all_data.py를 실행하세요.")
        return

    print(f"\n총 {len(chunks)}개 청크 처리 예정\n")

    all_rows = []

    # 임베딩 생성 (로컬 배치 처리 — API 비용 없음)
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="임베딩 생성 (로컬)"):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        try:
            embeddings = get_embeddings(texts)
        except Exception as e:
            print(f"\n  [ERROR] 임베딩 실패 (배치 {i}~{i+len(batch)}): {e}")
            continue

        for chunk, emb in zip(batch, embeddings):
            all_rows.append(chunk_to_row(chunk, emb))

    print(f"\n임베딩 완료: {len(all_rows)}개 행 준비")

    # Supabase 업로드 (배치)
    success = 0
    for i in tqdm(range(0, len(all_rows), UPLOAD_BATCH), desc="Supabase 업로드"):
        batch = all_rows[i: i + UPLOAD_BATCH]
        try:
            upload_to_supabase(batch)
            success += len(batch)
        except Exception as e:
            print(f"\n  [ERROR] 업로드 실패 (행 {i}~{i+len(batch)}): {e}")

    print(f"\n{'='*60}")
    print(f"완료: {success}/{len(all_rows)}개 행 업로드 성공")
    print(f"Supabase 테이블 'documents'에서 확인하세요.")


if __name__ == "__main__":
    main()
