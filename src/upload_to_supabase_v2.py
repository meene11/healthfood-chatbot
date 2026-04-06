"""
청크 JSON → Supabase documents_v2 업로드 (실험 5)
==============================================
임베딩 모델: BAAI/bge-m3 (1024차원, 한/영 혼재 학술 텍스트 최적화)
대상 테이블: documents_v2  ← 기존 documents(384차원) 와 별도 유지
청크 파일:   data/generated/chunks/all_chunks_v2.json

실험 5 변경점:
  - 모델: multilingual-e5-small(384d) → bge-m3(1024d)
  - 프리픽스: "passage: ..." → 없음 (bge-m3는 프리픽스 불필요)
  - 테이블: documents → documents_v2

실행 전 확인:
  1. Supabase에서 supabase/migrations/001_create_documents_v2.sql 실행 완료
  2. data/generated/chunks/all_chunks_v2.json 존재 확인

실행:
  python src/upload_to_supabase_v2.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_DIR    = Path(__file__).resolve().parent.parent
CHUNKS_FILE = BASE_DIR / "data" / "generated" / "chunks" / "all_chunks_v2.json"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

EMBED_MODEL_NAME = "BAAI/bge-m3"
EMBED_DIMENSION  = 1024
TABLE_NAME       = "documents_v2"
BATCH_SIZE       = 16    # bge-m3는 모델이 크므로 배치 작게
UPLOAD_BATCH     = 20
MAX_CONTENT      = 2000

print(f"임베딩 모델 로딩 중... (첫 실행 시 ~570MB 다운로드)")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print(f"모델 로드 완료: {EMBED_MODEL_NAME} ({EMBED_DIMENSION}차원)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """bge-m3 임베딩 생성 (프리픽스 불필요)"""
    clipped = [t[:MAX_CONTENT] for t in texts]
    embeddings = embed_model.encode(clipped, normalize_embeddings=True)
    return embeddings.tolist()


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        print(f"[ERROR] 청크 파일 없음: {CHUNKS_FILE}")
        print("먼저 python src/parse_all_data.py 를 실행하세요.")
        return []
    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"청크 로드 완료: {len(data)}개 ({CHUNKS_FILE.name})")
    return data


def chunk_to_row(chunk: dict, embedding: list[float]) -> dict:
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


def main():
    print("=" * 60)
    print("Supabase 벡터 DB 업로드 (실험 5 - bge-m3)")
    print(f"모델: {EMBED_MODEL_NAME} ({EMBED_DIMENSION}차원)")
    print(f"테이블: {TABLE_NAME}")
    print("=" * 60)

    chunks = load_chunks()
    if not chunks:
        return

    print(f"\n총 {len(chunks)}개 청크 처리 예정\n")

    all_rows = []
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="임베딩 생성"):
        batch = chunks[i: i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        try:
            embeddings = get_embeddings(texts)
        except Exception as e:
            print(f"\n[ERROR] 임베딩 실패 배치 {i}: {e}")
            continue
        for chunk, emb in zip(batch, embeddings):
            all_rows.append(chunk_to_row(chunk, emb))

    print(f"\n임베딩 완료: {len(all_rows)}개")

    success = 0
    for i in tqdm(range(0, len(all_rows), UPLOAD_BATCH), desc="Supabase 업로드"):
        batch = all_rows[i: i + UPLOAD_BATCH]
        try:
            supabase.table(TABLE_NAME).insert(batch).execute()
            success += len(batch)
        except Exception as e:
            print(f"\n[ERROR] 업로드 실패 행 {i}: {e}")

    print(f"\n{'='*60}")
    print(f"완료: {success}/{len(all_rows)}개 업로드 성공 → {TABLE_NAME}")


if __name__ == "__main__":
    main()
