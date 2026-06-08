from pathlib import Path
import pandas as pd
import chromadb

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DIR = BASE_DIR / "chroma_valve_db"
CHUNKS_CSV = BASE_DIR / "vector_db" / "chunks.csv"

COLLECTION_NAME = "valve_rag_knowledge"

print("=" * 80)
print("ShipLeak-Copilot Vector DB 검증")
print("=" * 80)

# 1. ChromaDB 연결
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

# 2. chunks.csv 확인
chunks = pd.read_csv(CHUNKS_CSV)
csv_chunk_count = len(chunks)

# 3. ChromaDB 저장 개수 확인
db_count = collection.count()

print(f"CSV Chunk Count          : {csv_chunk_count}")
print(f"ChromaDB Collection Count: {db_count}")

if csv_chunk_count == db_count:
    print("검증 결과 1: Chunk 수와 Vector DB 저장 개수가 일치합니다.")
else:
    print("검증 결과 1: Chunk 수와 Vector DB 저장 개수가 일치하지 않습니다.")

# 4. Embedding Dimension 확인
data = collection.get(
    limit=1,
    include=["embeddings", "documents", "metadatas"]
)

embeddings = data.get("embeddings")

if embeddings is not None and len(embeddings) > 0:
    embedding_dim = len(embeddings[0])
    print(f"Embedding Dimension      : {embedding_dim}")

    if embedding_dim == 1536:
        print("검증 결과 2: text-embedding-3-small 기본 차원인 1536차원과 일치합니다.")
    else:
        print("검증 결과 2: 임베딩 차원이 1536이 아닙니다. 모델 또는 DB를 확인하세요.")
else:
    print("Embedding Dimension      : 확인 불가")

print("=" * 80)
print("검증 완료")
print("=" * 80)