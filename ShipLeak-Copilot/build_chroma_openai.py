"""
선택 사항: OpenAI Embedding + ChromaDB로 Vector DB를 구축하는 코드입니다.
필요 설치:
pip install chromadb langchain-openai langchain-community pandas

.env에 OPENAI_API_KEY를 설정한 뒤 실행하세요.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

print("OPENAI_API_KEY 확인:", os.getenv("OPENAI_API_KEY")[:10])
import pandas as pd
import chromadb
from langchain_openai import OpenAIEmbeddings

chunks = pd.read_csv("vector_db/chunks.csv")
client = chromadb.PersistentClient(path="chroma_valve_db")
collection = client.get_or_create_collection(name="valve_rag_knowledge")

emb = OpenAIEmbeddings(model="text-embedding-3-small")

for i, row in chunks.iterrows():
    vec = emb.embed_query(row["text"])
    collection.add(
        ids=[row["chunk_id"]],
        embeddings=[vec],
        documents=[row["text"]],
        metadatas=[{
            "source": row["source"],
            "chunk_index": int(row["chunk_index"])
        }]
    )

print("Chroma Vector DB 구축 완료:", collection.count())