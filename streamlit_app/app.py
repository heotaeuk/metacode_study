# app.py
# ShipLeak Copilot RAG Streamlit UI - Multi-Turn V6 Quality Gate
#
# 실행 예:
#   cd C:\Users\wlsrn\New_ShipLeak
#   python -m streamlit run streamlit_app/app.py

import os
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from config import (
    APP_TITLE,
    ROOT_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_TOP_K,
    DEFAULT_MIN_SCORE,
)

from rag_core import (
    build_high_precision_db,
    answer_with_manual_rag,
)


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚢",
    layout="wide",
)


# ==============================
# 1. 환경변수 / API Key
# ==============================
def load_env_files() -> None:
    try:
        load_dotenv(ROOT_DIR / ".env")
    except Exception:
        pass

    try:
        load_dotenv(Path(__file__).parent / ".env")
    except Exception:
        pass


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            api_key = ""

    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )

    return api_key


load_env_files()
OPENAI_API_KEY = get_openai_api_key()


# ==============================
# 2. 기본 PDF 문서
# ==============================
DEFAULT_PDF_FILE_NAME = "00_ShipLeak_Copilot_Integrated_Knowledge_Base.pdf"


def read_default_file() -> tuple[bytes, str]:
    default_path = Path(DEFAULT_DATA_PATH)
    app_dir = Path(__file__).parent

    candidate_paths = [
        default_path.with_suffix(".pdf"),
        default_path.parent / DEFAULT_PDF_FILE_NAME,
        app_dir / DEFAULT_PDF_FILE_NAME,
        app_dir / "data" / DEFAULT_PDF_FILE_NAME,
        ROOT_DIR / DEFAULT_PDF_FILE_NAME,
        ROOT_DIR / "data" / DEFAULT_PDF_FILE_NAME,
        ROOT_DIR / "streamlit_app" / DEFAULT_PDF_FILE_NAME,
        ROOT_DIR / "streamlit_app" / "data" / DEFAULT_PDF_FILE_NAME,
    ]

    for pdf_path in candidate_paths:
        pdf_path = Path(pdf_path)
        if pdf_path.exists() and pdf_path.is_file():
            with open(pdf_path, "rb") as f:
                return f.read(), pdf_path.name

    for root in [app_dir, ROOT_DIR]:
        root = Path(root)
        if root.exists():
            matches = list(root.rglob(DEFAULT_PDF_FILE_NAME))
            if matches:
                pdf_path = matches[0]
                with open(pdf_path, "rb") as f:
                    return f.read(), pdf_path.name

    raise FileNotFoundError(
        f"기본 PDF 문서를 찾을 수 없습니다. 필요한 파일명: {DEFAULT_PDF_FILE_NAME}"
    )


# ==============================
# 3. 질문 처리
# ==============================
def clean_question_prefix(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"^\s*질문\s*\d+\s*[:：.)-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*(q|question)\s*\d+\s*[:：.)-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"^\s*\d+\s*[:：.)-]\s*",
        "",
        text,
    )

    return text.strip()


def split_questions(raw_text: str) -> List[str]:
    """
    여러 질문 또는 복합 요구를 분리합니다.

    지원 예:
    1) 줄바꿈 질문
       질문1
       질문2

    2) 한 줄에 여러 질문
       음향데이터 재측정은 어떻게 수행하나요? 재조립 후 체결 상태를 어떻게 확인하나요?

    3) 물음표 뒤에 '그리고/또한/추가로'로 이어지는 복합 요구
       밸브의 seat leakage 발생시 초동 대처방안은? 그리고 특정 공구 및 작업내용을 상세히 알려줘.

    주의:
    - Multi-Query가 아닙니다.
    - 사용자가 실제로 여러 요구를 입력한 경우 각각 독립 RAG 검색/답변을 수행하기 위한 질문 분리입니다.
    """
    questions: List[str] = []

    connector_pattern = r"\s*(그리고|또한|추가로|추가적으로|아울러|및)\s+"

    for line in raw_text.splitlines():
        line = clean_question_prefix(line)
        if not line:
            continue

        # 물음표가 있으면 첫 질문과 뒤 요구를 분리합니다.
        # 예: "A는? 그리고 B 알려줘" -> ["A는?", "B 알려줘"]
        if "?" in line or "？" in line:
            parts = re.findall(r"[^?？]+[?？]", line)
            tail = re.sub(r".*[?？]", "", line).strip()

            for part in parts:
                cleaned = clean_question_prefix(part.strip())
                if cleaned:
                    questions.append(cleaned)

            if tail:
                tail = re.sub(connector_pattern, "", tail, count=1).strip()
                cleaned_tail = clean_question_prefix(tail)
                if cleaned_tail:
                    questions.append(cleaned_tail)

        else:
            # 물음표가 없어도 "그리고/또한/추가로"가 있으면 요구사항을 분리합니다.
            chunks = re.split(connector_pattern, line)
            chunks = [c.strip() for c in chunks if c.strip() and c.strip() not in ["그리고", "또한", "추가로", "추가적으로", "아울러", "및"]]

            if len(chunks) > 1:
                for chunk in chunks:
                    cleaned = clean_question_prefix(chunk)
                    if cleaned:
                        questions.append(cleaned)
            else:
                questions.append(line)

    return questions


def should_update_active_issue(question: str, result: dict) -> bool:
    """
    독립적인 새 주제 또는 현장 이슈 질문이면 현재 대화 주제로 저장합니다.
    후속 질문은 기존 주제를 유지합니다.
    """
    if result.get("is_followup"):
        return False

    intent = result.get("intent", "diagnosis")

    update_intents = [
        "diagnosis",
        "cause",
        "inspection",
        "action",
        "test_condition",
        "replacement_procedure",
        "seat_leakage_initial_response",
        "seat_leakage_tools_work",
        "packing_displacement_initial_response",
        "packing_detailed_action",
        "cavitation_action_tools",
        "acoustic_remeasurement",
        "reassembly_tightening_check",
        "verification_acceptance_criteria",
        "leak_location_marking",
    ]

    if intent in update_intents:
        return True

    return False


def is_correction_message(text: str) -> bool:
    """
    사용자가 AI 답변을 정정하는 메시지인지 판단합니다.

    예:
    - "아니야, 순서는 3번 2번 1번이 맞아"
    - "정정: Packing 빠짐은 Seat가 아니라 Gland/Packing 중심으로 답해야 함"
    - "점검 순서가 잘못됐어. 먼저 격리하고 압력 제거해야 해"
    """
    t = text.lower().strip()

    correction_markers = [
        "정정",
        "수정",
        "틀렸",
        "잘못",
        "아니야",
        "아닙니다",
        "그게 아니라",
        "이게 아니라",
        "순서가",
        "순서는",
        "반영해",
        "기억해",
        "다음부터",
        "앞으로",
        "맞지 않습니다",
        "부적절",
    ]

    return any(marker in t for marker in correction_markers)


def make_correction_record(text: str, active_issue: str) -> dict:
    """
    사용자의 정정 내용을 다음 답변에 반영하기 위해 메모리 형태로 저장합니다.
    """
    return {
        "active_issue": active_issue or "대화 주제 미지정",
        "correction": text.strip(),
    }


# ==============================
# 4. Source 표시
# ==============================
def get_source_type(doc) -> str:
    metadata = doc.metadata or {}

    source_type = str(metadata.get("source_type", "")).lower().strip()
    source_name = str(
        metadata.get("document_name")
        or metadata.get("file_name")
        or metadata.get("source")
        or ""
    ).lower()

    if source_type:
        return source_type

    if source_name.endswith(".pdf"):
        return "pdf"

    if source_name.endswith(".docx"):
        return "docx"

    return ""


def get_pdf_page_number(doc) -> Optional[int]:
    if get_source_type(doc) != "pdf":
        return None

    metadata = doc.metadata or {}

    value = metadata.get("page")
    if value not in ("", None):
        try:
            return int(value) + 1
        except Exception:
            pass

    value = metadata.get("page_number")
    if value not in ("", None):
        try:
            return int(value)
        except Exception:
            pass

    return None


def make_short_evidence_text(text: str, max_len: int = 160) -> str:
    if not text:
        return ""

    remove_markers = [
        "[공통 검색 보조 키워드]",
        "[Seat Leakage 검색 보조 키워드]",
        "[Disc 검색 보조 키워드]",
        "[Gas Leakage 검색 보조 키워드]",
        "[Pressure Drop 검색 보조 키워드]",
        "[Vibration/RMS/FFT 검색 보조 키워드]",
        "[기록 양식 검색 보조 키워드]",
    ]

    for marker in remove_markers:
        if marker in text:
            text = text.split(marker)[0]

    text = text.replace(chr(13), " ")
    text = text.replace(chr(10), " ")
    text = text.replace(chr(9), " ")
    text = " ".join(text.split())

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "..."

    return text


def build_top_source_item(retrieved_results):
    if not retrieved_results:
        return None

    top_item = retrieved_results[0]
    doc = top_item.get("doc")
    metadata = (doc.metadata or {}) if doc else {}

    source_name = (
        metadata.get("document_name")
        or metadata.get("file_name")
        or metadata.get("source")
        or "알 수 없는 문서"
    )

    source_type = get_source_type(doc) if doc else ""
    pdf_page = get_pdf_page_number(doc) if doc else None
    chunk_index = metadata.get("chunk_index", "")

    if source_type == "pdf" and pdf_page is not None:
        title = f"{source_name}, Page {pdf_page}"
    elif source_type == "supplement":
        title = f"{source_name}, {chunk_index}"
    else:
        if chunk_index not in ("", None):
            title = f"{source_name}, Chunk {chunk_index}"
        else:
            title = source_name

    evidence = make_short_evidence_text(doc.page_content if doc else "")

    return {
        "title": title,
        "evidence": evidence,
        "source_type": source_type,
    }


# ==============================
# 5. Vector DB 캐시
# ==============================
@st.cache_resource(show_spinner=False)
def cached_build_high_precision_db(
    file_bytes: bytes,
    file_name: str,
    embedding_model: str,
    api_key: str,
):
    return build_high_precision_db(
        file_bytes=file_bytes,
        file_name=file_name,
        embedding_model=embedding_model,
        api_key=api_key,
    )


# ==============================
# 6. Session State
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = ""

if "active_issue" not in st.session_state:
    st.session_state.active_issue = ""

if "kb_update_candidates" not in st.session_state:
    st.session_state.kb_update_candidates = []

if "correction_memory" not in st.session_state:
    st.session_state.correction_memory = []


# ==============================
# 7. Sidebar
# ==============================
# 발표/데모용 고정 설정
# 교수님 지적사항 반영:
# - 의미가 명확하지 않은 설정 UI는 제거
# - 사용자는 지식문서 업로드와 대화 초기화만 사용
# - 모델/검색 파라미터는 코드 내부에서 고정 관리
embedding_model = "text-embedding-3-large"
llm_model = "gpt-4o"
top_k = 3
min_score = 0.60

st.sidebar.header("지식문서 선택")

uploaded_file = st.sidebar.file_uploader(
    "지식문서 업로드",
    type=["docx", "pdf"],
    help="문서를 업로드하지 않으면 기본 PDF 문서를 사용합니다.",
)



if st.sidebar.button("대화 초기화"):
    st.session_state.messages = []
    st.session_state.active_issue = ""
    st.session_state.correction_memory = []
    st.session_state.kb_update_candidates = []
    st.rerun()


# ==============================
# 8. 사용할 문서 결정
# ==============================
try:
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name
    else:
        file_bytes, file_name = read_default_file()

except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if st.session_state.last_file_name and st.session_state.last_file_name != file_name:
    st.session_state.messages = []
    st.session_state.active_issue = ""

st.session_state.last_file_name = file_name


# ==============================
# 9. Main
# ==============================
st.title("ShipLeak Copilot RAG 🚢")
st.caption(
    "선박 배관/밸브 누설, 진동, RMS, FFT, Cavitation, Packing, "
    "Seat Leakage 관련 지식문서를 기반으로 답변하는 RAG 챗봇입니다."
)

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY가 없습니다. 좌측 사이드바에 API Key를 입력하세요.")
    st.stop()


# ==============================
# 10. Vector DB 생성
# ==============================
try:
    with st.spinner("Vector DB 생성/로드 중입니다..."):
        database, final_docs = cached_build_high_precision_db(
            file_bytes=file_bytes,
            file_name=file_name,
            embedding_model=embedding_model,
            api_key=OPENAI_API_KEY,
        )
except Exception as e:
    st.error("Vector DB 생성/로드 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# ==============================
# 11. 대화 화면
# ==============================
st.header("무엇이든 물어보세요 🔎")


if not st.session_state.messages:
    st.info("질문을 입력하면 답변이 생성됩니다. 답변을 본 뒤 이어서 추가 질문할 수 있습니다.")

for message in st.session_state.messages:
    role = message.get("role", "")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":
            source = message.get("source")
            if source:
                st.markdown("#### Source")
                st.markdown(f"- [1] {source['title']}")
                if source.get("evidence"):
                    st.caption(f"근거: {source['evidence']}")



# ==============================
# 12. 사용자 질문 입력
# ==============================
user_input = st.chat_input("질문을 입력하세요. 예: 밸브에서 leakage가 발생하면 어떻게 점검해야 하나요?")

if user_input:
    # 사용자가 답변을 정정하는 경우에는 RAG 검색을 하지 않고 Correction Memory에 저장합니다.
    if is_correction_message(user_input):
        correction_record = make_correction_record(
            text=user_input,
            active_issue=st.session_state.active_issue,
        )

        st.session_state.correction_memory.append(correction_record)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        correction_ack = (
            "정정 내용을 반영했습니다. 이후 같은 주제의 후속 질문에는 이 정정 내용을 우선 참고하겠습니다.\n\n"
            f"- 현재 대화 주제: {correction_record['active_issue']}\n"
            f"- 정정 내용: {correction_record['correction']}"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": correction_ack,
            }
        )

        st.rerun()

    questions = split_questions(user_input)

    for question in questions:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        previous_messages = st.session_state.messages[:-1]
        active_issue_before = st.session_state.active_issue

        try:
            with st.spinner("답변 생성 중..."):
                result = answer_with_manual_rag(
                    database=database,
                    question=question,
                    k=top_k,
                    api_key=OPENAI_API_KEY,
                    llm_model=llm_model,
                    conversation_messages=previous_messages,
                    active_issue=active_issue_before,
                    correction_memory=st.session_state.correction_memory,
                    min_relevance_score=float(min_score),
                    strong_relevance_score=0.80,
                )

            if should_update_active_issue(question, result):
                st.session_state.active_issue = question

            retrieved_results = result.get("retrieved_results", [])
            filtered_results = [
                item for item in retrieved_results
                if float(item.get("score", 0.0)) >= float(min_score)
            ]

            source = build_top_source_item(filtered_results)
            assistant_content = result.get("answer", "")

            quality = result.get("quality", {})
            kb_update_candidate = result.get("kb_update_candidate")

            if kb_update_candidate:
                st.session_state.kb_update_candidates.append(kb_update_candidate)

            assistant_message = {
                "role": "assistant",
                "content": assistant_content,
                "source": source,
                "search_question": result.get("search_question", question),
                "intent": result.get("intent", ""),
                "is_followup": result.get("is_followup", False),
                "active_issue_used": active_issue_before,
                "quality": quality,
                "kb_update_candidate": kb_update_candidate,
            }

            st.session_state.messages.append(assistant_message)

            with st.chat_message("assistant"):
                st.markdown(assistant_content)

                if source:
                    st.markdown("#### Source")
                    st.markdown(f"- [1] {source['title']}")
                    if source.get("evidence"):
                        st.caption(f"근거: {source['evidence']}")


        except Exception as e:
            error_message = f"답변 생성 중 오류가 발생했습니다: {e}"
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

            with st.chat_message("assistant"):
                st.error(error_message)
