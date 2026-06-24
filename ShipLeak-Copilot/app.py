# app.py
# ShipLeak Copilot RAG Streamlit App
# 실행: streamlit run app.py

import os
import sys
import tempfile
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Streamlit Cloud에서 ChromaDB sqlite 버전 문제 방지용
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma


# ==============================
# 1. 기본 설정값
# ==============================
APP_TITLE = "ShipLeak Copilot RAG"
DEFAULT_DATA_PATH = "./00_ShipLeak_Copilot_Integrated_Knowledge_Base.docx"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
DISTANCE_METRIC = "cosine"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.25


# ==============================
# 2. Topic Keywords 정의
# ==============================
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "Gas Leakage / Pipe Leakage": [
        "Gas Leakage", "Pipe Leakage", "Hydrogen Gas Leak", "가스 누설", "배관 누설",
        "밸브 누설", "Gas Detector", "압력 저하", "누설음", "환기", "비상정지",
        "새요", "샘", "누설", "leak"
    ],
    "Seat Leakage": [
        "Seat Leakage", "시트 누설", "Internal Leakage", "밸브 내부 누설", "시트 손상"
    ],
    "Packing Leakage": [
        "Packing Leakage", "패킹 누설", "Stem Leakage", "스템 누설", "Gland Packing", "패킹 마모"
    ],
    "Cavitation": [
        "Cavitation", "캐비테이션", "공동현상", "고주파 소음", "기포 붕괴"
    ],
    "Valve Vibration / Loose Fastener": [
        "Valve Vibration", "밸브 진동", "Pipe Vibration", "배관 진동", "Loose Fastener",
        "볼트 풀림", "체결 불량", "Mounting Issue", "저주파 진동"
    ],
    "RMS": [
        "RMS", "진동 RMS", "vibration level", "진동 증가", "상태 감시"
    ],
    "FFT Analysis": [
        "FFT", "FFT Analysis", "주파수 분석", "Frequency Analysis", "스펙트럼 분석"
    ],
    "Foreign Object": [
        "Foreign Object", "이물질", "밸브 내부 이물질", "막힘", "유량 저하"
    ],
}


# ==============================
# 3. Prompt
# ==============================
IMPROVED_PROMPT = PromptTemplate.from_template(
    """
당신은 선박 배관 및 밸브 누설 진단 전문가입니다.

아래 [Context]에 있는 내용만 근거로 사용하여 답변하세요.
문서에 없는 내용은 추측하지 말고 "제공된 문서만으로는 확인하기 어렵습니다."라고 답변하세요.

답변 형식:
1. 진단 요약
2. 가능한 원인
3. 점검 방법
4. 권장 조치
5. 근거 문서 요약

[Context]
{context}

[Question]
{question}

[Answer]
"""
)


# ==============================
# 4. 문서/검색 보강 함수
# ==============================
def classify_topics(text: str) -> List[str]:
    matched_topics = []
    lower_text = text.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower_text:
                matched_topics.append(topic)
                break

    return matched_topics


def make_base_docs(raw_docs: List[Document]) -> List[Document]:
    base_docs = []

    for idx, doc in enumerate(raw_docs, start=1):
        topics = classify_topics(doc.page_content)
        base_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"BASE-{idx}",
                    "chunk_type": "base_original_chunk",
                    "topics": " | ".join(topics),
                },
            )
        )

    return base_docs


def make_search_alias(text: str) -> str:
    aliases = []
    lower_text = text.lower()

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["Gas Leakage / Pipe Leakage"]):
        aliases.append(
            """
[Gas Leakage / Pipe Leakage 검색 보강]
대표 질문:
- 밸브에서 가스가 새요?
- 배관에서 누설이 발생하면 어떤 조치를 해야 하나요?
검색 키워드:
가스 누설, 배관 누설, 밸브 누설, Gas Leakage, Pipe Leakage, Hydrogen Gas Leak,
Gas Detector, 압력 저하, 누설음, 환기, 차단, 비상정지
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["Seat Leakage"]):
        aliases.append(
            """
[Seat Leakage 검색 보강]
대표 질문:
- Seat Leakage가 발생하면 어떤 증상이 나타나나요?
- 밸브 시트 누설 증상은 무엇인가요?
검색 키워드:
Seat Leakage, 시트 누설, Internal Leakage, 밸브 내부 누설, 압력 저하, 누설음, RMS 증가
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["Packing Leakage"]):
        aliases.append(
            """
[Packing Leakage 검색 보강]
대표 질문:
- Packing 누설이 발생하면 무엇을 점검해야 하나요?
- 스템 부위에서 누설이 발생합니다.
검색 키워드:
Packing Leakage, 패킹 누설, Stem Leakage, 스템 누설, Gland Packing, 체결 상태, 씰링
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["Cavitation"]):
        aliases.append(
            """
[Cavitation 검색 보강]
대표 질문:
- Cavitation이 발생하면 밸브에 어떤 문제가 생기나요?
- 캐비테이션 소음이 발생합니다.
검색 키워드:
Cavitation, 캐비테이션, 공동현상, 고주파 소음, 진동, 밸브 손상, 압력 변화
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["Valve Vibration / Loose Fastener"]):
        aliases.append(
            """
[Valve Vibration / Loose Fastener 검색 보강]
대표 질문:
- 밸브 진동이 심할 때 가능한 원인은 무엇인가요?
- Loose Fastener가 있으면 어떤 현상이 발생하나요?
검색 키워드:
Valve Vibration, 밸브 진동, Pipe Vibration, 배관 진동, Loose Fastener,
볼트 풀림, 체결 불량, Mounting Issue, 저주파 진동
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["RMS"]):
        aliases.append(
            """
[RMS 검색 보강]
대표 질문:
- RMS 값이 높게 측정되면 어떤 조치를 해야 하나요?
- 진동 RMS가 높으면 어떤 문제가 있나요?
검색 키워드:
RMS, 진동 RMS, vibration level, 진동 증가, 이상 진단, 상태 감시
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["FFT Analysis"]):
        aliases.append(
            """
[FFT / Frequency Analysis 검색 보강]
대표 질문:
- FFT 분석은 밸브 이상 진단에 왜 필요한가요?
- 주파수 분석으로 무엇을 알 수 있나요?
검색 키워드:
FFT Analysis, 주파수 분석, Frequency Analysis, 진동 진단, 이상 주파수, 고주파, 저주파
"""
        )

    if any(keyword.lower() in lower_text for keyword in TOPIC_KEYWORDS["Foreign Object"]):
        aliases.append(
            """
[Foreign Object 검색 보강]
대표 질문:
- Foreign Object가 밸브 내부에 있으면 어떤 문제가 발생하나요?
- 밸브 내부 이물질 문제는 무엇인가요?
검색 키워드:
Foreign Object, 이물질, 밸브 내부 이물질, 막힘, 유량 저하, 이상 소음, 압력 변화
"""
        )

    return "\n".join(aliases)


def make_enriched_docs(base_docs: List[Document]) -> List[Document]:
    enriched_docs = []

    for idx, doc in enumerate(base_docs, start=1):
        alias_text = make_search_alias(doc.page_content)

        if alias_text.strip():
            enriched_content = f"""
[검색 보강 Alias]
{alias_text}

[원본 문서 Chunk]
{doc.page_content}
"""
        else:
            enriched_content = f"""
[원본 문서 Chunk]
{doc.page_content}
"""

        enriched_docs.append(
            Document(
                page_content=enriched_content,
                metadata={
                    **doc.metadata,
                    "chunk_id": f"ALIAS-{idx}",
                    "chunk_type": "alias_enriched_chunk",
                    "alias_added": bool(alias_text.strip()),
                },
            )
        )

    return enriched_docs


def get_faq_knowledge_items() -> List[Dict[str, Any]]:
    return [
        {
            "topic": "Gas Leakage / Pipe Leakage",
            "questions": [
                "밸브에서 가스가 새요?",
                "배관에서 누설이 발생하면 어떤 조치를 해야 하나요?",
                "가스 누설이 발생하면 어떻게 해야 하나요?",
                "Gas Leakage 발생 시 증상과 조치는 무엇인가요?",
                "Pipe Leakage 발생 시 점검 항목은 무엇인가요?",
            ],
            "keywords": TOPIC_KEYWORDS["Gas Leakage / Pipe Leakage"],
            "answer": """
Gas Leakage 또는 Pipe Leakage가 발생하면 가스 감지기 알람, 압력 저하,
누설음, 배관 주변 가스 분출 또는 가스 감지 등의 증상이 나타날 수 있다.
수소 가스는 무색·무취이므로 작업자가 직접 감지하기 어렵기 때문에 Gas Detector를 통해 확인해야 한다.
권장 조치는 안전 확보, 누설 위치 표시, 밸브 격리, 환기, 비상정지, 정비 후 재점검 순서로 수행한다.
""",
        },
        {
            "topic": "Seat Leakage",
            "questions": [
                "Seat Leakage가 발생하면 어떤 증상이 나타나나요?",
                "밸브 시트 누설 증상은 무엇인가요?",
                "밸브 내부 누설 원인은 무엇인가요?",
            ],
            "keywords": TOPIC_KEYWORDS["Seat Leakage"],
            "answer": """
Seat Leakage는 밸브 시트 부위에서 내부 누설이 발생하는 현상이다.
주요 증상은 압력 저하, 누설음, RMS 증가, 밸브 차단 성능 저하 등이다.
점검 시 밸브 시트 손상, 이물질 유입, 마모, 밀봉면 손상 여부를 확인해야 한다.
""",
        },
        {
            "topic": "Packing Leakage",
            "questions": [
                "Packing 누설이 발생했을 때 점검해야 할 항목은 무엇인가요?",
                "스템 부위에서 누설이 발생합니다.",
                "Packing Leakage 조치사항은 무엇인가요?",
            ],
            "keywords": TOPIC_KEYWORDS["Packing Leakage"],
            "answer": """
Packing Leakage는 주로 밸브 스템 또는 Gland Packing 부위에서 발생한다.
점검 항목은 스템 부위 누설 여부, 패킹 마모, 체결 상태, Gland 조임 상태,
씰링 손상 여부이다. 필요 시 패킹 교체 또는 재조임 후 누설 여부를 재확인한다.
""",
        },
        {
            "topic": "Cavitation",
            "questions": [
                "Cavitation이 발생하면 밸브에 어떤 문제가 생기나요?",
                "캐비테이션 소음이 발생합니다.",
                "공동현상 발생 시 조치는 무엇인가요?",
            ],
            "keywords": TOPIC_KEYWORDS["Cavitation"],
            "answer": """
Cavitation은 유체 압력 변화로 기포가 발생하고 붕괴하면서 밸브 내부에 충격을 주는 현상이다.
주요 증상은 고주파 소음, 진동 증가, 밸브 내부 손상, 성능 저하이다.
압력 조건, 유량 조건, 밸브 개도, 손상 여부를 점검해야 한다.
""",
        },
        {
            "topic": "Valve Vibration / Loose Fastener",
            "questions": [
                "밸브 진동이 심할 때 가능한 원인은 무엇인가요?",
                "밸브가 떨려요.",
                "배관 진동이 심합니다.",
                "Loose Fastener가 있으면 밸브 운전 중 어떤 현상이 나타나나요?",
            ],
            "keywords": TOPIC_KEYWORDS["Valve Vibration / Loose Fastener"],
            "answer": """
밸브 또는 배관 진동이 심한 경우 Loose Fastener, Mounting Issue, 체결 불량,
배관 지지 불량, 유동 불안정 등이 원인일 수 있다.
Loose Fastener가 있으면 저주파 진동, 소음 증가, 구조 진동, 체결부 손상 가능성이 있다.
체결 상태, 지지 구조, Mounting 상태를 점검해야 한다.
""",
        },
        {
            "topic": "RMS",
            "questions": [
                "RMS 값이 높게 측정되면 어떤 조치를 해야 하나요?",
                "진동 RMS가 높으면 어떤 문제가 있나요?",
            ],
            "keywords": TOPIC_KEYWORDS["RMS"],
            "answer": """
RMS 값이 높다는 것은 전체 진동 에너지가 증가했다는 의미이다.
이는 체결 불량, 베어링 이상, 밸브 진동, 배관 지지 문제, 유동 이상 등과 관련될 수 있다.
조치로는 진동 원인 분석, 체결 상태 확인, FFT 분석, 운전 조건 확인, 필요 시 정비가 필요하다.
""",
        },
        {
            "topic": "FFT Analysis",
            "questions": [
                "FFT 분석은 밸브 이상 진단에 왜 필요한가요?",
                "주파수 분석으로 무엇을 알 수 있나요?",
            ],
            "keywords": TOPIC_KEYWORDS["FFT Analysis"],
            "answer": """
FFT 분석은 시간 영역의 진동 신호를 주파수 영역으로 변환하여 이상 원인을 구분하는 데 사용된다.
저주파 성분은 체결 불량이나 구조 진동과 관련될 수 있고,
고주파 성분은 누설음, 베어링 이상, Cavitation 등과 관련될 수 있다.
따라서 밸브 이상 진단에서 원인 분류와 정비 판단에 필요하다.
""",
        },
        {
            "topic": "Foreign Object",
            "questions": [
                "Foreign Object가 밸브 내부에 있으면 어떤 문제가 발생하나요?",
                "밸브 내부 이물질 문제는 무엇인가요?",
                "이물질 때문에 유량이 줄어들 수 있나요?",
            ],
            "keywords": TOPIC_KEYWORDS["Foreign Object"],
            "answer": """
Foreign Object가 밸브 내부에 있으면 유로 막힘, 유량 저하, 압력 변화,
이상 소음, 밸브 시트 손상, 누설 등이 발생할 수 있다.
점검 시 밸브 내부 이물질 유입 여부, 시트 손상, 유량 변화, 압력 변화를 확인해야 한다.
""",
        },
    ]


def make_faq_docs(items: List[Dict[str, Any]]) -> List[Document]:
    faq_docs = []

    for idx, item in enumerate(items, start=1):
        content = f"""
[FAQ 검색 전용 Chunk]

주제:
{item["topic"]}

대표 질문:
{chr(10).join("- " + q for q in item["questions"])}

검색 키워드:
{", ".join(item["keywords"])}

답변 근거:
{item["answer"]}
"""

        faq_docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "manual_faq_knowledge",
                    "faq_id": f"FAQ-{idx}",
                    "topic": item["topic"],
                    "topics": item["topic"],
                    "chunk_type": "faq_high_precision_chunk",
                },
            )
        )

    return faq_docs


def get_expanded_queries(query: str) -> List[str]:
    queries = [query]

    if any(word in query for word in ["가스", "누설", "새", "샘", "leak", "Leakage"]):
        queries.append("Gas Leakage Pipe Leakage Hydrogen Gas Leak 가스 누설 배관 누설 Gas Detector 압력 저하 환기 차단")

    if any(word in query for word in ["소리", "소음", "Noise", "이상음"]):
        queries.append("Abnormal Sound Noise Acoustic Emission 누설음 이상 소음 고주파 소음")

    if any(word in query for word in ["떨", "진동", "Vibration"]):
        queries.append("Valve Vibration Pipe Vibration Loose Fastener Mounting Issue 밸브 진동 배관 진동 체결 불량")

    if any(word in query for word in ["Seat", "시트"]):
        queries.append("Seat Leakage Internal Leakage 시트 누설 밸브 내부 누설 시트 손상")

    if any(word in query for word in ["Packing", "패킹", "스템", "Stem"]):
        queries.append("Packing Leakage Stem Leakage Gland Packing 패킹 누설 스템 누설 패킹 마모")

    if any(word in query for word in ["Cavitation", "캐비테이션", "공동"]):
        queries.append("Cavitation 캐비테이션 공동현상 고주파 소음 진동 밸브 손상")

    if "RMS" in query:
        queries.append("RMS vibration level 진동 RMS 진동 증가 이상 진단 상태 감시")

    if any(word in query for word in ["FFT", "주파수"]):
        queries.append("FFT Analysis Frequency Analysis 주파수 분석 진동 진단 스펙트럼 분석")

    if any(word in query for word in ["Foreign", "Object", "이물질"]):
        queries.append("Foreign Object 이물질 밸브 내부 이물질 막힘 유량 저하 압력 변화")

    unique_queries = []
    for q in queries:
        if q not in unique_queries:
            unique_queries.append(q)

    return unique_queries


def cosine_score_from_distance(distance: float) -> float:
    try:
        score = 1.0 - float(distance)
    except Exception:
        score = 0.0

    return round(max(0.0, min(1.0, score)), 4)


def make_doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}

    for key_name in ["chunk_id", "faq_id", "anchor_id"]:
        if metadata.get(key_name):
            return str(metadata.get(key_name))

    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def multi_query_search(
    database: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
    use_query_expansion: bool = True,
) -> List[Dict[str, Any]]:
    search_queries = get_expanded_queries(query) if use_query_expansion else [query]
    candidates: Dict[str, Dict[str, Any]] = {}

    for search_query in search_queries:
        docs_with_distances = database.similarity_search_with_score(search_query, k=k)

        for rank, (doc, raw_distance) in enumerate(docs_with_distances, start=1):
            safe_score = cosine_score_from_distance(raw_distance)
            doc_key = make_doc_key(doc)

            if doc_key not in candidates or safe_score > candidates[doc_key]["score"]:
                candidates[doc_key] = {
                    "doc": doc,
                    "score": safe_score,
                    "raw_distance": raw_distance,
                    "search_query_used": search_query,
                    "rank_in_each_query": rank,
                }

    ranked_results = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    return ranked_results[:k]


# ==============================
# 5. Vector DB 생성
# ==============================
def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:12]


def load_docx_from_bytes(file_bytes: bytes, original_name: str) -> List[Document]:
    suffix = Path(original_name).suffix or ".docx"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        loader = Docx2txtLoader(tmp_path)
        raw_docs = loader.load_and_split(text_splitter=text_splitter)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return raw_docs


@st.cache_resource(show_spinner=False)
def build_high_precision_db(
    file_bytes: bytes,
    file_name: str,
    embedding_model: str,
    api_key: str,
) -> Tuple[Chroma, Dict[str, Any]]:
    os.environ["OPENAI_API_KEY"] = api_key

    raw_docs = load_docx_from_bytes(file_bytes, file_name)
    base_docs = make_base_docs(raw_docs)
    enriched_docs = make_enriched_docs(base_docs)
    faq_docs = make_faq_docs(get_faq_knowledge_items())

    high_precision_docs = enriched_docs + faq_docs

    doc_hash = get_file_hash(file_bytes)
    persist_directory = Path(tempfile.gettempdir()) / "shipleak_chroma" / doc_hash
    marker_path = persist_directory / "build.ok"
    collection_name = f"shipleak_high_precision_{doc_hash}"

    embedding = OpenAIEmbeddings(model=embedding_model)

    if marker_path.exists():
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=str(persist_directory),
        )
    else:
        if persist_directory.exists():
            shutil.rmtree(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)

        db = Chroma.from_documents(
            documents=high_precision_docs,
            embedding=embedding,
            collection_name=collection_name,
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": DISTANCE_METRIC},
        )
        marker_path.write_text("ok", encoding="utf-8")

    stats = {
        "raw_chunk_count": len(raw_docs),
        "base_chunk_count": len(base_docs),
        "alias_chunk_count": len(enriched_docs),
        "faq_chunk_count": len(faq_docs),
        "total_db_docs": len(high_precision_docs),
        "doc_hash": doc_hash,
        "persist_directory": str(persist_directory),
    }

    return db, stats


# ==============================
# 6. 답변 생성
# ==============================
def build_context(ranked_results: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[검색순위 {idx}]\n"
        f"Score: {item['score']}\n"
        f"Chunk Type: {item['doc'].metadata.get('chunk_type', '')}\n"
        f"Topic: {item['doc'].metadata.get('topics', item['doc'].metadata.get('topic', ''))}\n"
        f"Search Query Used: {item['search_query_used']}\n"
        f"Content:\n{item['doc'].page_content}"
        for idx, item in enumerate(ranked_results, start=1)
    )


def answer_with_manual_rag(
    database: Chroma,
    question: str,
    api_key: str,
    llm_model: str,
    k: int,
    min_score: float,
    use_query_expansion: bool = True,
) -> Dict[str, Any]:
    os.environ["OPENAI_API_KEY"] = api_key

    ranked_results = multi_query_search(
        database=database,
        query=question,
        k=k,
        use_query_expansion=use_query_expansion,
    )

    top_score = ranked_results[0]["score"] if ranked_results else 0.0
    context = build_context(ranked_results)

    if not ranked_results or top_score < min_score:
        return {
            "question": question,
            "answer": (
                "제공된 문서에서 관련도가 충분한 근거를 찾지 못했습니다.\n\n"
                f"- 현재 Top Score: {top_score:.4f}\n"
                f"- 최소 기준 점수: {min_score:.4f}\n\n"
                "질문을 선박 배관/밸브 누설, 진동, RMS, FFT, Cavitation, Packing, Seat Leakage 등과 "
                "관련된 표현으로 다시 입력해 주세요."
            ),
            "retrieved_results": ranked_results,
            "context": context,
        }

    llm = ChatOpenAI(model=llm_model, temperature=0)
    rag_chain = IMPROVED_PROMPT | llm | StrOutputParser()
    answer = rag_chain.invoke({"context": context, "question": question})

    return {
        "question": question,
        "answer": answer,
        "retrieved_results": ranked_results,
        "context": context,
    }


# 지표 출력을 위해 콤팩트하게 변환하는 요약 매핑 함수
def results_to_summary_dataframe(ranked_results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for idx, item in enumerate(ranked_results, start=1):
        doc = item["doc"]
        metadata = doc.metadata or {}
        
        # 출처 및 유형 가독성 정제
        chunk_type = metadata.get("chunk_type", "unknown_chunk")
        if chunk_type == "faq_high_precision_chunk":
            source_type = "정밀 FAQ 지식셋"
        elif chunk_type == "alias_enriched_chunk":
            source_type = "검색보강 지식베이스"
        else:
            source_type = "일반 오리지널 문서"
            
        topic = metadata.get("topics", metadata.get("topic", "일반 진단"))
        if not topic:
            topic = "기타 분류"

        rows.append({
            "검색순위": f"{idx}위",
            "유사도 지표(Score)": f"{item['score']:.4f}",
            "참조 출처 분류 (Topic)": topic,
            "문서 청크 유형": source_type
        })
    return pd.DataFrame(rows)


# ==============================
# 7. Streamlit UI
# ==============================
def read_default_docx() -> Tuple[bytes, str]:
    with open(DEFAULT_DATA_PATH, "rb") as f:
        return f.read(), Path(DEFAULT_DATA_PATH).name


def get_openai_api_key() -> str:
    load_dotenv()

    env_key = os.getenv("OPENAI_API_KEY", "")

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    api_key = secret_key or env_key

    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Streamlit Cloud에서는 Settings > Secrets에 OPENAI_API_KEY를 등록하세요.",
        )

    return api_key


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🚢",
        layout="wide",
    )

    st.title("🚢 ShipLeak Copilot RAG")
    st.caption("선박 배관·밸브 누설 진단용 RAG 챗봇")

    with st.sidebar:
        st.header("설정")

        api_key = get_openai_api_key()

        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-large", "text-embedding-3-small"],
            index=0,
        )

        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o"],
            index=0,
        )

        top_k = st.slider("Top-K 검색 문서 수", min_value=1, max_value=5, value=DEFAULT_TOP_K)
        min_score = st.slider(
            "최소 유사도 기준",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_MIN_SCORE,
            step=0.05,
        )

        use_query_expansion = st.toggle("Multi-Query 검색 사용", value=True)

        st.divider()
        st.subheader("지식문서 선택")

        uploaded_file = st.file_uploader(
            "DOCX 지식문서 업로드",
            type=["docx"],
            help="업로드하지 않으면 앱 폴더의 기본 DOCX 파일을 사용합니다.",
        )

    if not api_key:
        st.warning("OpenAI API Key가 필요합니다. 사이드바에서 입력하거나 Streamlit Secrets에 등록하세요.")
        st.stop()

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
    elif Path(DEFAULT_DATA_PATH).exists():
        file_bytes, file_name = read_default_docx()
    else:
        st.error(
            "지식문서가 없습니다. 사이드바에서 DOCX 파일을 업로드하거나 "
            f"앱 폴더에 `{DEFAULT_DATA_PATH}` 파일을 넣어 주세요."
        )
        st.stop()

    with st.spinner("문서를 읽고 High Precision Vector DB를 준비하는 중입니다..."):
        try:
            db, stats = build_high_precision_db(
                file_bytes=file_bytes,
                file_name=file_name,
                embedding_model=embedding_model,
                api_key=api_key,
            )
        except Exception as e:
            st.error("Vector DB 생성 중 오류가 발생했습니다.")
            st.exception(e)
            st.stop()

    sample_questions = [
        "밸브에서 가스가 새요?",
        "Seat Leakage가 발생하면 어떤 증상이 나타나나요?",
        "Packing 누설이 발생했을 때 점검해야 할 항목은 무엇인가요?",
        "Cavitation이 발생하면 밸브에 어떤 문제가 생기나요?",
        "RMS 값이 높게 측정되면 어떤 조치를 해야 하나요?",
        "FFT 분석은 밸브 이상 진단에 왜 필요한가요?",
        "Foreign Object가 밸브 내부에 있으면 어떤 문제가 발생하나요?",
    ]

    # 화면 너비를 절반으로 축소 (5:5 비율 분할로 좌측 컬럼만 사용)
    left_col, _ = st.columns([5, 5])

    with left_col:
        selected_sample = st.selectbox(
            "샘플 질문 선택",
            ["직접 입력"] + sample_questions,
            index=0,
        )

        default_question = "" if selected_sample == "직접 입력" else selected_sample

        question = st.text_area(
            "질문",
            value=default_question,
            height=100,
            placeholder="예: 밸브에서 가스가 새요?",
        )

        ask_button = st.button("답변 생성", type="primary", use_container_width=True)

        if ask_button:
            if not question.strip():
                st.warning("질문을 입력해 주세요.")
                st.stop()

            # 스피너 컴포넌트를 이용한 로딩 상태 구현
            with st.spinner("🔄 답변을 생성하는 중입니다..."):
                try:
                    result = answer_with_manual_rag(
                        database=db,
                        question=question.strip(),
                        api_key=api_key,
                        llm_model=llm_model,
                        k=top_k,
                        min_score=min_score,
                        use_query_expansion=use_query_expansion,
                    )
                except Exception as e:
                    st.error("답변 생성 중 오류가 발생했습니다.")
                    st.exception(e)
                    st.stop()

            st.markdown("---")
            st.subheader("답변")
            st.markdown(result["answer"])

            # 답변 아래에 근거 지표 요약 표 추가 제공
            st.markdown("---")
            st.subheader("📊 답변 근거 출처 및 검색 품질 요약")
            
            if "retrieved_results" in result and result["retrieved_results"]:
                summary_df = results_to_summary_dataframe(result["retrieved_results"])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            else:
                st.info("참조한 근거 문서 정보를 불러오지 못했습니다.")


if __name__ == "__main__":
    main()