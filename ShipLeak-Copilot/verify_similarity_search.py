from pathlib import Path
from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

CHROMA_DIR = BASE_DIR / "chroma_valve_db"
COLLECTION_NAME = "valve_rag_knowledge"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_collection(name=COLLECTION_NAME)

emb = OpenAIEmbeddings(model="text-embedding-3-small")

query = "Seat Leakage 증상은?"
query_vec = emb.embed_query(query)

results = collection.query(
    query_embeddings=[query_vec],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

print("=" * 80)
print("유사도 검색 테스트")
print("=" * 80)
print("질문:", query)

docs = results["documents"][0]
metas = results["metadatas"][0]
distances = results["distances"][0]

for i in range(len(docs)):
    distance = distances[i]
    normalized_similarity = 1 / (1 + distance)

    print("\n" + "=" * 80)
    print(f"Rank: {i + 1}")
    print(f"Source: {metas[i].get('source', 'unknown')}")
    print(f"ChromaDB Distance: {distance:.4f}")
    print(f"Normalized Similarity 참고값: {normalized_similarity:.4f}")
    print("※ ChromaDB Distance는 작을수록 질문과 더 유사합니다.")
    print("-" * 80)
    print(docs[i][:500])