import os
from pathlib import Path

import streamlit as st
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

st.set_page_config(
    page_title="ShipLeak-Copilot",
    page_icon="🚢",
    layout="wide"
)

st.sidebar.markdown("---")

st.title("🚢 ShipLeak-Copilot RAG 챗봇")
st.info("""
선박 배관/밸브 누설 진단 정비 지식 검색 시스템

기술:
• ChromaDB
• OpenAI Embedding
• GPT-4o-mini
• RAG
""")

client = chromadb.PersistentClient(path="chroma_valve_db")
collection = client.get_or_create_collection(name="valve_rag_knowledge")

emb = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

st.sidebar.title("질문 예시")
st.sidebar.markdown("---")

st.sidebar.markdown("""
### 적용 기술

✅ OpenAI Embedding

✅ ChromaDB

✅ Similarity Search

✅ GPT-4o-mini

✅ Retrieval-Augmented Generation
""")

sample = st.sidebar.selectbox(
    "예시 질문",
    [
        "Seat Leakage 증상은?",
        "Cavitation 원인은?",
        "RMS 증가 시 조치사항은?",
        "밸브 점검 절차는?"
    ]
)

question = st.text_input(
    "질문을 입력하세요",
    sample
)

if st.button("검색 및 답변 생성"):
    query_vec = emb.embed_query(question)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n".join(
        [f"[출처: {metas[i]['source']}]\n{docs[i]}" for i in range(len(docs))]
    )

    prompt = f"""
당신은 선박 배관 및 밸브 누설 진단 전문가입니다.
아래 검색된 정비 문서를 근거로 사용자의 질문에 답변하세요.

규칙:
1. 검색 문서에 근거해서 답변하세요.
2. 추측하지 마세요.
3. 정비 조치가 필요한 경우 단계별로 설명하세요.
4. 마지막에 참고 출처를 표시하세요.

[검색 문서]
{context}

[질문]
{question}

[답변]
"""

    response = llm.invoke(prompt)

    st.subheader("🤖 AI 진단 결과")
    st.success(response.content)

    st.subheader("검색된 근거 문서")
    for i, doc in enumerate(docs):
        score = 1-distances[i]
        
        with st.expander(f"[{i+1}] {metas[i]['source']} (유사도: {score:.3f})"):
            st.write(doc[:500])
