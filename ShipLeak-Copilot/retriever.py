import pickle
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

DB_DIR = "vector_db"

chunks = pd.read_csv(f"{DB_DIR}/chunks.csv")
X = sparse.load_npz(f"{DB_DIR}/tfidf_matrix.npz")
with open(f"{DB_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def search(query, top_k=5):
    q = vectorizer.transform([query])
    q = normalize(q)
    scores = (X @ q.T).toarray().ravel()
    idxs = scores.argsort()[::-1][:top_k]
    results = []
    for rank, idx in enumerate(idxs, 1):
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "source": chunks.iloc[idx]["source"],
            "chunk_id": chunks.iloc[idx]["chunk_id"],
            "text": chunks.iloc[idx]["text"]
        })
    return results

if __name__ == "__main__":
    while True:
        query = input("\n질문을 입력하세요(q 종료): ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            break
        for r in search(query, top_k=5):
            print("\n" + "="*80)
            print(f"[{r['rank']}] score={r['score']:.4f} | {r['source']} | {r['chunk_id']}")
            print(r["text"][:900])