# rag_core.py
# ShipLeak Copilot RAG Core Logic - Multi-Turn V7
#
# 개선 사항:
# 1. Multi-Query 삭제 유지
# 2. Multi-Turn 후속 질문의 의도 분기 강화
# 3. "누설 위치 표시/마킹/분필/기포 위치 기록" 질문에 별도 답변 포맷 적용
# 4. 기본 진단 포맷은 diagnosis 의도일 때만 사용
# 5. 지식문서 보강: Leak Location Marking Procedure 추가

import os
import re
import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
except ImportError:
    Docx2txtLoader = None
    PyPDFLoader = None


# ==============================
# 1. Prompt 정의
# ==============================
RAG_PROMPT = PromptTemplate.from_template(
    """
당신은 선박 배관 및 밸브 누설 진단 전문가입니다.

아래 정보를 참고하여 사용자의 질문에 답변하세요.

[핵심 원칙]
1. 기술적 사실은 반드시 [Context]에 있는 내용만 사용합니다.
2. [Context]에 없는 내용은 추측하지 말고 "제공된 문서만으로는 확인하기 어렵습니다."라고 답변합니다.
3. 후속 질문은 [현재 대화 주제]와 [대화 이력]을 참고하여 해석합니다.
4. 단, [대화 이력]은 질문 의도 해석용이고, 답변 근거는 [Context]를 우선합니다.
5. 질문 의도에 맞는 답변 형식을 사용합니다.
6. "진단 요약/가능한 원인/점검 방법/권장 조치/근거 문서 요약" 기본 포맷은 질문 의도가 diagnosis일 때만 사용합니다.
7. 후속 질문이 구체적인 방법, 작업도구, 작업내용, 초동 대처를 묻는 경우에는 고정 진단 포맷을 쓰지 말고, 현재 질문에 대한 직접 답변만 작성합니다.
8. Packing 빠짐/이탈 질문은 Seat Leakage가 아니므로 Seat/Disc 중심 답변을 금지하고 Packing/Stem/Gland 중심으로 답변합니다.
9. 합격/개선/판정 기준을 묻는 질문은 진단 포맷을 금지하고 반드시 수치 기준과 판정표를 제시합니다.
10. [사용자 정정 메모리]에 현재 주제와 관련된 정정 내용이 있으면, 문서 근거와 충돌하지 않는 범위에서 그 정정 내용을 우선 반영합니다.
11. 사용자가 "이전 답변이 틀렸다", "순서가 잘못됐다", "이렇게 해야 한다"고 정정한 내용은 다음 답변에서 반복 오류가 나지 않도록 반영합니다.
7. 현재 질문이 "누설 위치 표시", "분필", "마킹", "기포 위치 기록", "교체 방법", "기록 양식", "시험 조건", "작업도구", "준비물", "공구" 중 하나라면 기본 진단 포맷을 절대 사용하지 마세요.
8. 이전 답변을 그대로 반복하지 말고, 현재 질문에서 요구한 부분만 구체적으로 답변하세요.
9. 현재 질문이 절차를 묻는 질문이면 원인 분석을 반복하지 말고 절차, 도구, 기록 항목 중심으로 답변하세요.

[현재 대화 주제]
{active_issue}

[대화 이력]
{conversation_history}

[사용자 정정 메모리]
{correction_memory_text}

[검색에 사용한 질문]
{search_question}

[질문 의도]
{intent}

[답변 형식 지시]
{answer_style_instruction}

[Context]
{context}

[Question]
{question}

[Answer]
"""
)


# ==============================
# 2. 기본 설정
# ==============================
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200


# ==============================
# 3. 문서 로드
# ==============================
def _write_bytes_to_temp_file(file_bytes: bytes, file_name: str) -> Path:
    suffix = Path(file_name).suffix.lower()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(file_bytes)
    temp.flush()
    temp.close()
    return Path(temp.name)


def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    clean_metadata = {}

    for key, value in metadata.items():
        if value is None:
            clean_metadata[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value
        else:
            clean_metadata[key] = str(value)

    return clean_metadata


def load_documents_from_bytes(file_bytes: bytes, file_name: str) -> List[Document]:
    suffix = Path(file_name).suffix.lower()
    temp_path = _write_bytes_to_temp_file(file_bytes, file_name)

    try:
        if suffix == ".docx":
            if Docx2txtLoader is None:
                raise ImportError("DOCX 파일을 읽으려면 langchain-community와 docx2txt가 필요합니다.")
            loader = Docx2txtLoader(str(temp_path))
            docs = loader.load()

        elif suffix == ".pdf":
            if PyPDFLoader is None:
                raise ImportError("PDF 파일을 읽으려면 langchain-community와 pypdf가 필요합니다.")
            loader = PyPDFLoader(str(temp_path))
            docs = loader.load()

        else:
            raise ValueError("지원하지 않는 파일 형식입니다. DOCX 또는 PDF만 사용할 수 있습니다.")

    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    for idx, doc in enumerate(docs):
        metadata = doc.metadata or {}
        metadata["document_name"] = file_name
        metadata["source"] = file_name
        metadata["page_index"] = idx

        if suffix == ".pdf":
            metadata["source_type"] = "pdf"
            if metadata.get("page") not in ("", None):
                try:
                    metadata["page_number"] = int(metadata["page"]) + 1
                except Exception:
                    metadata["page_number"] = ""
            else:
                metadata["page_number"] = ""

        elif suffix == ".docx":
            metadata["source_type"] = "docx"
            metadata["page_number"] = ""

        doc.metadata = _normalize_metadata(metadata)

    return docs


# ==============================
# 4. 지식문서 보강 Supplement KB
# ==============================
def build_supplement_documents() -> List[Document]:
    """
    PDF 원문에 상세 절차가 부족할 때 후속 질문에 답하기 위한 보강 지식문서입니다.
    최종 제출 시에는 이 내용을 기존 PDF 끝에 Appendix로 추가하는 것이 가장 좋습니다.
    """
    supplement_items = [

        {
            "title": "음향 데이터 재측정 합격 및 개선 판단 기준",
            "content": """
[음향 데이터 재측정 합격 및 개선 판단 기준]
적용 대상: 합격 기준, 개선 기준, 판정 기준, 정비 후 검증, 음향 데이터 재측정, RMS, FFT, Dominant Frequency, Spectral Centroid, Leak Test, Pressure Holding Test

목적:
정비 후 음향 데이터를 재측정한 뒤 합격 또는 개선 여부를 정량적으로 판단하기 위한 기준이다.
기본문서에는 RMS, Dominant Frequency, Spectral Centroid의 NORMAL/WARNING/ALERT/DANGER 기준과 정비 후 RMS/FFT/Centroid 변화 비교, Leak Test 또는 압력 유지 시험 결과 기록이 포함되어 있다.

측정 조건 일치 기준:
1. 정비 전과 정비 후의 측정 위치가 동일해야 한다.
2. Valve Body, Inlet Pipe, Outlet Pipe 등 동일한 측정 위치를 사용한다.
3. 운전 압력, 유량, 밸브 개도, 유체 조건을 가능한 동일하게 유지한다.
4. Sampling Rate는 16 kHz로 설정한다.
5. 10초 이상 WAV 데이터를 저장한다.
6. RMS, Peak, Crest Factor, Dominant Frequency, Spectral Centroid, MFCC를 추출한다.

정량 판정 기준:
| Feature | NORMAL | WARNING | ALERT | DANGER |
|---|---:|---:|---:|---:|
| RMS | < 0.097 | 0.097~0.118 | 0.118~0.139 | >= 0.139 |
| Dominant Frequency | < 450.0 Hz | 450.0~690.7 Hz | 690.7~931.5 Hz | >= 931.5 Hz |
| Spectral Centroid | < 2821.9 Hz | 2821.9~3174.6 Hz | 3174.6~3527.3 Hz | >= 3527.3 Hz |

합격 판정 기준:
1. 정비 후 RMS가 NORMAL 범위인 0.097 미만이어야 한다.
2. Dominant Frequency가 450.0 Hz 미만이어야 한다.
3. Spectral Centroid가 2821.9 Hz 미만이어야 한다.
4. 고주파 누설음이 소멸되어야 한다.
5. Leak Test 또는 압력 유지 시험에서 누설이 없어야 한다.
6. Bubble Test를 수행한 경우 기포가 발생하지 않아야 한다.
7. Stem/Packing 문제의 경우 Stem 부위 누설 흔적이 제거되어야 한다.

개선 판정 기준:
1. 정비 전보다 RMS가 감소해야 한다.
2. 정비 전보다 FFT Peak 또는 Dominant Frequency 이상 성분이 감소해야 한다.
3. 정비 전보다 Spectral Centroid가 낮아져야 한다.
4. 누설음 또는 고주파 성분이 줄어야 한다.
5. Leak Test 결과가 정비 전보다 개선되어야 한다.
6. 단, 개선되었더라도 WARNING 이상이면 정상 합격이 아니라 모니터링 또는 추가 점검 대상으로 본다.

불합격 또는 재점검 기준:
1. 정비 후 RMS가 0.118 이상이면 ALERT 이상으로 보고 재점검한다.
2. Dominant Frequency가 690.7 Hz 이상이면 ALERT 이상으로 보고 재점검한다.
3. Spectral Centroid가 3174.6 Hz 이상이면 ALERT 이상으로 보고 재점검한다.
4. Leak Test 또는 압력 유지 시험에서 누설이 확인되면 불합격으로 본다.
5. Bubble Test에서 기포가 계속 발생하면 불합격으로 본다.
6. 정비 후에도 고주파 누설음, 압력 저하, Stem 누설 흔적이 남아 있으면 재점검한다.

기록 항목:
- Valve ID
- 측정 위치
- 운전 압력
- 유량
- 밸브 개도
- Sampling Rate
- 측정 시간
- 정비 전 RMS, Dominant Frequency, Spectral Centroid
- 정비 후 RMS, Dominant Frequency, Spectral Centroid
- Leak Test 결과
- Bubble Test 결과
- 판정: 합격 / 개선 / 재점검 / 불합격
""",
        },


        {
            "title": "Packing 빠짐 또는 이탈 초동조치 및 상세 작업절차",
            "content": """
[Packing 빠짐 또는 이탈 초동조치 및 상세 작업절차]
적용 대상: Packing 빠짐, Packing 이탈, Packing 탈락, Gland Packing 빠짐, Stem 부위 누설, 패킹 누설, 패킹 교체, 초동조치, 상세 조치

상황 정의:
Packing이 빠졌거나 이탈된 상황은 단순한 일반 누설이 아니라 Stem 부위 외부 누설과 직접 관련될 수 있는 이상 상태이다.
기본문서의 Packing / Stem Leakage 항목에서는 Stem 부위 누설, Spectral Centroid 증가, 미세 누설음 또는 고주파 성분 증가를 대표 증상으로 제시한다.
가능 원인은 Packing 노화, Gland 체결력 부족, Stem 표면 손상, 반복 개폐에 의한 Seal 성능 저하이다.

초동조치:
1. 현장 접근 안전을 먼저 확인한다.
2. 고온, 고압, 가연성 유체 여부를 확인한다.
3. 누설이 진행 중이면 무리하게 손으로 Packing을 밀어 넣거나 임의 조립하지 않는다.
4. 해당 배관 또는 밸브 구간을 격리한다.
5. 내부 압력과 잔압을 제거한다.
6. 필요 시 LOTO를 적용하여 오조작을 방지한다.
7. Stem 주변 누설 흔적, Packing 이탈 상태, Gland 체결 상태를 육안으로 확인한다.
8. 운전 조건, 압력, 유체, 발생 위치, 발생 시간을 기록한다.
9. 선임 또는 정비 담당자에게 Packing 이탈 상태를 보고한다.
10. 임시 조치가 필요한 경우에도 정식 정비 전까지 운전 재개 여부는 선임 판단을 받는다.

상세 점검 절차:
1. Stem 주변 누설 흔적을 확인한다.
2. Packing이 완전히 빠졌는지, 일부 돌출되었는지, 손상되었는지 구분한다.
3. Gland follower 또는 Gland nut 체결 상태를 확인한다.
4. Gland 체결력이 부족한 경우 기준 토크 또는 절차에 따라 조정한다.
5. Stem 표면에 흠집, 편심, 오염, 마모 흔적이 있는지 확인한다.
6. Packing이 손상되었거나 이탈된 경우 기존 Packing을 제거하고 교체한다.
7. 새 Packing 삽입 시 방향, 절단면 위치, 층간 배치가 적절한지 확인한다.
8. Gland를 균등하게 조여 Packing이 한쪽으로 쏠리지 않게 한다.
9. 과도한 체결로 Stem 동작이 뻑뻑해지지 않는지 확인한다.
10. 재조립 후 Leak Test 또는 운전 재확인으로 누설이 제거되었는지 확인한다.
11. 재측정 후 누설음 변화와 Spectral Centroid Baseline 복귀 여부를 확인한다.

준비 공구 및 장비:
- 안전 보호구: 보안경, 장갑, 안전화, 필요 시 방진/방독 보호구
- LOTO 장비: 작업 구간 격리 및 오조작 방지
- 압력계: 잔압 제거 및 압력 저하 확인
- 조명 또는 내시경: Stem, Gland, Packing 상태 육안 확인
- 스패너/렌치: Gland nut 또는 체결부 조정
- 토크렌치: 기준 토크 체결 확인
- Packing 제거 공구: 기존 Packing 제거
- 교체 Packing: 동일 사양 Packing
- 세척 도구: Stem 표면 및 Packing Box 오염 제거
- Leak Test 장비 또는 비눗물 용액: 재누설 확인
- 음향 센서 또는 마이크: 정비 전후 누설음 비교
- 점검표/사진기록 장비: 위치, 상태, 조치, 재시험 결과 기록

초보 검사원 보고 문구:
- "밸브 Stem 주변 Packing이 빠지거나 이탈된 상태입니다."
- "현재 누설 여부와 압력 상태를 확인 중입니다."
- "배관 격리와 잔압 제거가 필요합니다."
- "Gland 체결 상태와 Stem 표면 손상 여부 확인이 필요합니다."
- "Packing 교체 후 Leak Test 또는 운전 재확인이 필요합니다."

정비 후 검증:
1. Stem 부위 누설 흔적이 제거되었는지 확인한다.
2. Leak Test 또는 운전 재확인 결과를 기록한다.
3. Spectral Centroid가 Baseline으로 복귀하는지 확인한다.
4. 재측정 후 누설음 변화가 감소했는지 확인한다.
5. 정비 이력과 사진을 저장한다.
""",
        },


        {
            "title": "Seat Leakage 초동 대처 및 작업 공구 체크리스트",
            "content": """
[Seat Leakage 초동 대처 및 작업 공구 체크리스트]
적용 대상: Seat Leakage, 시트 누설, 밸브 내부 누설, 고주파 누설음, 압력 저하, 초동 대처, 작업공구, 작업내용

목적:
Seat Leakage가 의심될 때 초보 품질검사원이 선임에게 바로 문의하기 어려운 상황에서도 안전 확보, 기본 점검, 정비 요청, 재검증 항목을 빠르게 확인하도록 한다.
기본문서에는 Seat Leakage의 증상으로 RMS 증가, 고주파 누설음, 압력 저하 가능성, RMS와 Spectral Centroid 동시 증가 시 미세누설 가능성이 제시되어 있다.
가능 원인은 Seat 마모, 이물질 유입, Disc 손상, Seal 접촉면 오염 또는 손상이다.
점검 방법은 배관 격리와 압력 제거, 외부 누설 및 운전조건 기록, Seat 접촉면과 Disc 상태 육안 점검, 이물질/스크래치/마모 확인, Leak Test 또는 압력 유지 시험이다.

초동 대처 절차:
1. 현장 안전을 먼저 확인한다.
2. 누설음, 압력 저하, 고주파 소음, 진동 등 이상 징후를 확인한다.
3. 작업 전 배관을 격리하고 압력을 제거한다.
4. 외부 누설인지 내부 Seat Leakage인지 구분하기 위해 플랜지, Packing, Stem, Seat 관련 부위를 확인한다.
5. 운전 조건, 압력, 유량, 밸브 개도, 발생 시간을 기록한다.
6. 필요 시 비눗물 테스트, Leak Test, 압력 유지 시험으로 누설 여부를 확인한다.
7. Seat 접촉면과 Disc 상태를 육안으로 점검한다.
8. 이물질, 스크래치, 마모 흔적, Disc 손상을 확인한다.
9. 오염이면 Seat를 세척한다.
10. 손상이 있으면 Seat 또는 Disc 교체를 정비 요청한다.
11. 재조립 후 동일 조건에서 음향 데이터를 재측정한다.
12. 정비 전후 RMS 감소, 고주파 누설음 소멸, Leak Test 합격 여부를 확인한다.

작업 공구 및 준비물:
- 안전 보호구: 보안경, 장갑, 안전화, 필요 시 방진/방독 보호구
- LOTO 장비: 작업 전 격리와 오조작 방지
- 압력계 또는 디지털 압력계: 압력 저하 및 압력 제거 상태 확인
- 비눗물 용액 또는 Leak Test 장비: 누설 위치 확인
- 청소 도구: Seat 접촉면 이물질 제거
- 육안 점검용 조명/내시경: Seat, Disc, Seal 접촉면 점검
- 토크렌치 및 일반 수공구: 분해/재조립 및 체결 상태 확인
- 교체 부품: Seat, Disc, Seal, Gasket, Packing 등 필요 부품
- 음향 센서 또는 마이크: 정비 전후 음향 데이터 비교
- 데이터 분석 도구: RMS, FFT, Spectral Centroid 비교
- 점검표/사진기록 장비: 위치, 증상, 조치, 재시험 결과 기록

작업내용 상세:
1. 안전조치 및 배관 격리
   - 작업 구간을 확인하고 LOTO를 적용한다.
   - 압력이 제거되었는지 확인한다.
2. 누설 상태 확인
   - 고주파 누설음, 압력 저하, 기포 발생 여부를 확인한다.
   - 외부 누설과 내부 누설 가능성을 구분한다.
3. 분해 전 기록
   - Valve ID, 위치, 운전 압력, 유량, 밸브 개도, 발생 시간을 기록한다.
4. Seat/Disc 점검
   - Seat 접촉면과 Disc 표면을 육안으로 확인한다.
   - 이물질, 스크래치, 마모, 손상 여부를 기록한다.
5. 세척 또는 부품 교체
   - 오염이면 Seat를 세척한다.
   - 손상이 있으면 Seat 또는 Disc를 교체한다.
6. 재조립 및 체결 확인
   - 체결 상태를 확인하고 필요 시 토크렌치를 사용한다.
7. 재시험 및 검증
   - Leak Test 또는 압력 유지 시험을 수행한다.
   - 동일 조건에서 음향 데이터를 재측정한다.
   - RMS 감소, 고주파 누설음 소멸, Leak Test 합격 기록을 확인한다.
""",
        },


        {
            "title": "Cavitation 조치사항 및 작업도구 체크리스트",
            "content": """
[Cavitation 조치사항 및 작업도구 체크리스트]
적용 대상: Cavitation 발생, 캐비테이션 발생, 자갈 튀는 소리, 유속 과다, 압력강하 큼, FFT Peak 증가, 진동 증가, 작업도구, 준비물, 공구

목적:
Cavitation이 의심되는 경우 단순히 밸브를 분해하기보다 운전압력, 전후단 압력차, 밸브 개도, 유량, FFT Peak, RMS 변화를 확인하고 운전조건 또는 밸브 사양 적합성을 검토한다.
작업도구는 원인 확인 도구, 운전조건 조정 도구, 정비 작업 도구, 재검증 도구로 구분하여 준비한다.

1. 조치사항:
- 운전압력과 전후단 압력차를 확인한다.
- 밸브 개도 조건과 유량을 기록한다.
- FFT Spectrum에서 특정 Peak 증가 여부를 확인한다.
- Dominant Frequency와 RMS 증가 여부를 확인한다.
- 운전압력 또는 밸브 개도 조건을 조정한다.
- 유속 조건을 재검토한다.
- 밸브 사양과 배관 조건의 적합성을 검토한다.
- 필요 시 밸브 선정 변경 또는 운전 제한을 검토한다.
- 조치 후 동일 조건에서 재측정하여 FFT Peak, RMS, 진동 감소 여부를 확인한다.

2. 문제 해결을 위한 작업도구:
- 압력계 또는 디지털 압력계: 전단/후단 압력과 압력강하 확인
- 차압계 또는 ΔP 측정 장비: 밸브 전후단 압력차 확인
- 유량계 또는 유량 확인 자료: 유속 과다 여부 확인
- 밸브 개도 확인 도구: 밸브 개도율, 포지션, Actuator 위치 확인
- 진동계 또는 가속도 센서: 진동 증가 여부 확인
- 음향 센서 또는 마이크: 자갈 튀는 소리, 고주파 소음 측정
- FFT 분석 프로그램 또는 데이터 분석 노트북: FFT Peak, Dominant Frequency 분석
- RMS/Feature 분석 스크립트: RMS, Peak, Crest Factor, Spectral Centroid 비교
- 토크렌치 및 일반 수공구: 체결 상태 확인, 필요 시 볼트 재체결
- LOTO 장비 및 안전표지: 정비 전 격리와 안전 확보
- 작업기록지 또는 점검표: 운전조건, 유량, 압력, 개도율, 조치 결과 기록

3. 조치 후 검증:
- FFT Peak 감소 여부를 확인한다.
- 진동/RMS 감소 여부를 확인한다.
- 동일 조건에서 재측정 결과를 기록한다.
- 운전압력, 유량, 밸브 개도, 조치 내용을 정비 이력으로 저장한다.

4. 주의사항:
- Cavitation은 누설과 달리 운전 조건, 압력강하, 유속, 밸브 선정 문제와 관련될 수 있으므로 누설 보수 도구만 준비하면 부족하다.
- 배관 격리나 밸브 분해가 필요한 경우에는 고압/고온/가연성 여부를 확인하고 LOTO를 적용한다.
""",
        },


        {
            "title": "정비 후 음향 데이터 재측정 및 재조립 체결 상태 확인",
            "content": """
[정비 후 음향 데이터 재측정 및 재조립 체결 상태 확인]
적용 대상: 음향 데이터 재측정, 재측정 방법, 재조립 후 체결 상태 확인, 체결 확인, 재조립 검증, 정비 후 검증

목적:
정비 후에는 누설이 실제로 개선되었는지 확인하기 위해 정비 전과 동일 조건에서 음향 데이터를 다시 측정하고, 재조립된 밸브와 배관의 체결 상태를 확인한다.
기본문서에는 정비 전후 RMS 감소 여부 확인, 고주파 누설음 소멸 여부 확인, Leak Test 합격 기록, 동일 조건 음향 데이터 재측정, 체결 상태 확인이 포함되어 있다.

음향 데이터 재측정 절차:
1. 정비 전 측정했던 위치를 확인한다. 가능한 경우 Valve Body, Inlet Pipe, Outlet Pipe 등 동일 위치를 사용한다.
2. 정비 전과 동일한 운전 조건을 맞춘다. 압력, 유량, 밸브 개도, 유체 조건을 기록한다.
3. 센서를 동일한 방향과 부착 조건으로 설치한다.
4. 동일한 Sampling Rate와 측정 시간으로 WAV 데이터를 저장한다. 기본 기준은 16 kHz, 10초 이상 측정이다.
5. RMS, Peak, Crest Factor, Dominant Frequency, Spectral Centroid, MFCC를 추출한다.
6. 정비 전후 RMS, FFT Peak, Spectral Centroid 변화를 비교한다.
7. 고주파 누설음 소멸 여부와 Leak Test 결과를 함께 확인한다.
8. 측정 결과를 정비 이력과 Vector DB 업데이트 후보에 저장한다.

재조립 후 체결 상태 확인 절차:
1. 밸브 커버, 플랜지, 가스켓, 피팅, Gland, Packing 부위가 올바르게 조립되었는지 육안 확인한다.
2. 볼트와 너트가 누락 없이 체결되었는지 확인한다.
3. 체결 순서가 대각선 또는 균등 체결 방식으로 수행되었는지 확인한다.
4. 필요한 경우 토크렌치를 사용하여 기준 토크에 맞게 체결되었는지 확인한다.
5. 가스켓 돌출, 씹힘, 편심, Seal 접촉 불량 여부를 확인한다.
6. 밸브 개폐 동작이 부드러운지 확인한다.
7. 압력 유지 시험, 비눗물 테스트, Leak Test로 재누설 여부를 확인한다.
8. 체결 상태, 시험 압력, 재시험 결과를 기록한다.

판정 기준:
- 정비 전보다 RMS 또는 FFT Peak가 감소하면 개선 가능성이 있다.
- 고주파 누설음이 소멸되면 Seat 또는 Seal 관련 누설이 개선된 것으로 볼 수 있다.
- Leak Test 또는 압력 유지 시험에서 이상이 없어야 최종 합격으로 판단한다.
- 재조립 후에도 기포, 압력 저하, 이상 소음이 있으면 재점검한다.
""",
        },

        {
            "title": "누설 위치 표시 및 마킹 절차",
            "content": """
[누설 위치 표시 및 마킹 절차]
적용 대상: 누설 위치 표시, 누설 부위 마킹, 분필 표시, 마커 표시, 태그 부착, 기포 위치 기록, bubble location marking

목적:
누설 위치 표시는 정비자가 실제 누설 부위를 놓치지 않고 후속 조치, 재시험, 사진 기록을 연결하기 위한 절차이다.
누설 위치는 비눗물 테스트, Leak Test, 압력 유지 시험, 가스 검지 결과를 바탕으로 표시한다.

사용 가능한 표시 방법:
1. 분필: 고온이 아니고 표면이 건조한 경우 임시 표시용으로 사용할 수 있다.
2. 페인트 마커 또는 산업용 마킹펜: 금속 표면에 비교적 명확하게 남길 수 있어 현장 표시용으로 적합하다.
3. 내유성/내수성 태그: 누설 부위 근처 밸브, 플랜지, 피팅에 묶어 식별한다.
4. 테이프 또는 라벨: 표면 오염이 적고 접착 가능한 경우 임시 위치 표시로 사용한다.
5. 사진 기록: 누설 부위, 기포 발생 위치, 압력계, 밸브 ID가 함께 보이도록 촬영한다.
6. 도면 또는 체크시트 표시: Valve ID, 위치, 누설 방향, 누설 부위 번호를 기록한다.

권장 절차:
1. 시험 전 유체와 압력 조건을 확인한다.
2. 안전한 시험 매체와 비눗물 용액을 준비한다.
3. 의심 부위에 비눗물 용액을 도포한다.
4. 기포가 발생하는 정확한 위치를 확인한다.
5. 기포 발생 위치와 발생 시간을 기록한다.
6. 압력을 낮추거나 안전 상태를 확보한 뒤 누설 위치를 표시한다.
7. 표시 시에는 분필, 마킹펜, 태그, 사진 기록 중 현장 조건에 맞는 방법을 선택한다.
8. 누설 위치를 표시한 뒤 체결부, Seal, Gasket, Seat, Disc, Packing 등 원인 부위를 확인한다.
9. 조치 후 동일 조건에서 재시험하여 기포가 소멸했는지 확인한다.

주의사항:
1. 가연성 가스 또는 위험 유체가 의심되면 먼저 접근 통제, 점화원 제거, 환기, 가스 검지를 수행한다.
2. 누설 중인 상태에서 무리하게 가까이 접근하지 않는다.
3. 고온 표면, 회전체, 전기 설비 주변에서는 일반 테이프나 분필 사용이 제한될 수 있다.
4. 표시만 하고 끝내지 말고 반드시 사진, Valve ID, 위치, 시험 압력, 시간, 조치 내용을 기록한다.
""",
        },
        {
            "title": "Seat 또는 Disc 교체 방법",
            "content": """
[Seat 또는 Disc 교체 방법]
적용 대상: Seat 교체, Disc 교체, 밸브 시트 교체, 디스크 교체, seat replacement, disc replacement

작업 전 준비:
1. 밸브와 배관을 격리한다.
2. 내부 압력과 잔압을 완전히 제거한다.
3. 유체 종류와 위험성을 확인한다.
4. 필요한 경우 LOTO를 적용한다.
5. 분해 전 밸브 위치, 방향, 체결 상태, 누설 위치를 기록한다.

교체 절차:
1. 밸브 커버 또는 관련 부품을 분해한다.
2. Seat와 Disc 접촉면을 노출시킨다.
3. Seat 접촉면의 마모, 찍힘, 균열, 이물질 유입 여부를 확인한다.
4. Disc 표면의 손상, 편마모, 변형 여부를 확인한다.
5. 세척 가능한 오염이면 Seat와 Disc를 세척한다.
6. 손상이 확인되면 Seat 또는 Disc를 동일 사양 부품으로 교체한다.
7. 교체 부품의 방향과 안착 상태를 확인한다.
8. 밸브를 재조립하고 체결 상태를 확인한다.
9. 개폐 동작이 부드러운지 확인한다.
10. 동일 조건에서 누설 시험 또는 압력 시험을 수행한다.

교체 후 확인:
1. 누설 재발 여부를 확인한다.
2. 압력 저하 여부를 확인한다.
3. 이상 소음 또는 진동 감소 여부를 확인한다.
4. 정비 전후 RMS/FFT/음향 데이터를 비교한다.
5. 정비 결과와 교체 부품 정보를 기록한다.
""",
        },
        {
            "title": "시험 전 유체 및 압력 조건 확인",
            "content": """
[시험 전 유체 및 압력 조건 확인]
적용 대상: 시험 전 유체 확인, 압력 조건 확인, pressure test condition, leak test preparation

시험 전 확인 항목:
1. 시험 유체 종류: 공기, 질소, 물, 실제 운전 유체 등
2. 시험 유체 상태: 청정도, 수분, 이물질, 위험성
3. 시험 압력: 운전 압력, 시험 압력, 허용 압력
4. 압력 상승 속도: 급격한 압력 상승 금지
5. 격리 상태: 시험 구간 이외의 밸브 차단
6. 배기/벤트 상태: 잔압 제거 가능 여부
7. 안전 조치: 작업자 위치, 보호구, 접근 제한
8. 누설 검출 방법: 비눗물, Leak Detector, 압력 강하 관찰

확인 절차:
1. 시험 대상 밸브와 배관 구간을 식별한다.
2. 시험 유체와 압력 조건을 작업 지시서 또는 시험 계획서와 비교한다.
3. 시험 구간을 격리한다.
4. 압력계와 누설 검출 장비의 상태를 확인한다.
5. 단계적으로 압력을 상승시킨다.
6. 누설 여부와 압력 유지 여부를 확인한다.
7. 시험 결과를 기록한다.
""",
        },
        {
            "title": "시험 후 결과 기록 양식",
            "content": """
[시험 후 결과 기록 양식]
적용 대상: 시험 결과 기록, 정비 기록, leak test report, valve maintenance record

기록 목적:
시험 후에는 정비 이력 관리와 재발 분석을 위해 결과를 표준 양식으로 기록한다.
기록에는 밸브 식별 정보, 시험 조건, 점검 결과, 조치 내용, 재시험 결과가 포함되어야 한다.

권장 기록 항목:
- Valve ID
- 위치
- 밸브 종류
- 유체 종류
- 운전 압력
- 시험 압력
- 온도
- 유량
- 누설 위치
- 누설 증상
- 점검 방법
- 조치 내용
- 교체 부품
- 정비 전 RMS/FFT/음향 데이터
- 정비 후 RMS/FFT/음향 데이터
- 재시험 결과
- 판정
- 작업자
- 작업일자

기록 양식 예시:
| No. | 기록 항목 | 기록 내용 | 작성 예시 |
|---|---|---|---|
| 1 | Valve ID | 밸브 식별 번호 | V-101 |
| 2 | 위치 | 설치 위치 | 기관실 연료 라인 |
| 3 | 밸브 종류 | Globe/Gate/Ball 등 | Globe Valve |
| 4 | 유체 | 시험 또는 운전 유체 | Air / N2 / Water |
| 5 | 운전 압력 | 정상 운전 압력 | 6 bar |
| 6 | 시험 압력 | 시험 시 적용 압력 | 8 bar |
| 7 | 누설 위치 | Seat/Disc/Packing/Flange | Seat |
| 8 | 점검 방법 | Leak Test/Pressure Test/Visual | Leak Test |
| 9 | 조치 내용 | 세척/교체/재조립 | Seat 세척 및 Disc 교체 |
| 10 | 재시험 결과 | 합격/불합격 | 합격 |
| 11 | 비고 | 추가 관찰 사항 | RMS 감소 확인 |
""",
        },
    ]

    docs = []
    for idx, item in enumerate(supplement_items, start=1):
        docs.append(
            Document(
                page_content=f"{item['title']}\n\n{item['content']}".strip(),
                metadata=_normalize_metadata(
                    {
                        "document_name": "ShipLeak_MultiTurn_Supplement_KB.md",
                        "source": "ShipLeak_MultiTurn_Supplement_KB.md",
                        "source_type": "supplement",
                        "page_number": "",
                        "chunk_index": f"S{idx}",
                        "chunk_type": "supplement_knowledge",
                        "topic": item["title"],
                    }
                ),
            )
        )

    return docs


# ==============================
# 5. 검색 보조 Alias
# ==============================
def make_search_alias(text: str) -> str:
    text_lower = text.lower()

    aliases = [
        """
[공통 검색 보조 키워드]
선박 배관, 밸브, 누설, 가스 누설, 이상 소음, 진동, 압력 저하, 점검, 조치
ship pipe, valve, leakage, gas leak, abnormal sound, vibration, pressure drop, inspection, action
"""
    ]

    if any(keyword in text_lower for keyword in ["합격", "개선", "판정", "판단 기준", "기준", "모호", "정량", "normal", "warning", "alert", "danger"]):
        aliases.append(
            """
[합격/개선 판정 기준 검색 보조 키워드]
합격 기준, 개선 기준, 판정 기준, 정량 기준, RMS 기준, Dominant Frequency 기준,
Spectral Centroid 기준, NORMAL, WARNING, ALERT, DANGER, Leak Test 합격,
압력 유지 시험, Bubble Test 기포 없음, 정비 후 검증, RMS 감소, FFT Peak 감소
"""
        )

    if any(keyword in text_lower for keyword in ["packing", "패킹", "gland", "stem", "빠졌", "빠짐", "이탈", "탈락"]):
        aliases.append(
            """
[Packing 빠짐/이탈 검색 보조 키워드]
Packing 빠짐, Packing 이탈, Packing 탈락, Gland Packing, Stem 부위 누설, 패킹 누설,
Gland 체결 상태, Gland 체결 토크, Stem 표면 손상, Packing 교체, 초동조치,
배관 격리, 압력 제거, LOTO, Leak Test, Spectral Centroid Baseline 복귀, 누설음 변화
"""
        )


    if any(keyword in text_lower for keyword in ["seat leakage", "seat leak", "시트 누설", "초동", "대처", "작업공구", "작업 공구", "작업내용", "공구"]):
        aliases.append(
            """
[Seat Leakage 초동대처/작업공구 검색 보조 키워드]
Seat Leakage, 시트 누설, 초동 대처, 초기 조치, 작업공구, 작업 내용, 안전조치, 배관 격리,
압력 제거, Seat 점검, Disc 점검, Leak Test, 압력 유지 시험, Seat 세척, Seat 교체,
Disc 교체, 토크렌치, 비눗물 용액, 음향 센서, RMS, FFT, Spectral Centroid
"""
        )


    if any(keyword in text_lower for keyword in ["음향", "재측정", "체결", "재조립", "rms", "fft", "sampling"]):
        aliases.append(
            """
[정비 후 재측정/체결 확인 검색 보조 키워드]
음향 데이터 재측정, 동일 조건 재측정, 정비 후 검증, RMS 감소, FFT Peak 감소,
고주파 누설음 소멸, Leak Test 합격 기록, 재조립 후 체결 상태 확인, 토크렌치,
Sampling Rate 16 kHz, 10초 WAV, Valve Body, Inlet Pipe, Outlet Pipe
"""
        )


    if any(keyword in text_lower for keyword in ["작업도구", "도구", "공구", "준비물", "챙겨", "tool", "tools"]):
        aliases.append(
            """
[작업도구 검색 보조 키워드]
작업도구, 준비물, 공구, 압력계, 차압계, 유량계, 진동계, 음향 센서, FFT 분석, RMS 분석,
토크렌치, LOTO, 안전표지, 점검표, maintenance tools, troubleshooting tools
"""
        )

    if any(keyword in text_lower for keyword in ["cavitation", "캐비테이션", "자갈", "유속", "압력강하"]):
        aliases.append(
            """
[Cavitation 작업도구 검색 보조 키워드]
Cavitation, 캐비테이션, 압력강하, 유속 과다, 밸브 개도, FFT Peak, Dominant Frequency, RMS,
압력계, 차압계, 유량계, 밸브 개도 확인, 진동계, 음향 센서, FFT 분석 프로그램
"""
        )


    if any(keyword in text_lower for keyword in ["bubble", "기포", "버블", "비눗물", "누설 위치", "표시", "마킹"]):
        aliases.append(
            """
[누설 위치 표시 검색 보조 키워드]
누설 위치 표시, 누설 부위 마킹, 분필 표시, 마커 표시, 태그 부착, 사진 기록, 기포 위치 기록,
Bubble Test, Soap Bubble Test, leak location marking, chalk mark, paint marker, tag, photo record
"""
        )

    if any(keyword in text_lower for keyword in ["seat leakage", "seat leak", "시트", "seat"]):
        aliases.append(
            """
[Seat Leakage 검색 보조 키워드]
Seat Leakage, valve seat leakage, 밸브 시트 누설, 내부 누설, seat damage, sealing failure
Seat 교체, Disc 교체, 시트 교체, 디스크 교체, 교체 방법, replacement procedure
"""
        )

    if any(keyword in text_lower for keyword in ["disc", "디스크"]):
        aliases.append(
            """
[Disc 검색 보조 키워드]
Disc damage, Disc replacement, 디스크 손상, 디스크 교체, 밸브 디스크, Seat와 Disc 접촉면
"""
        )

    if any(keyword in text_lower for keyword in ["pressure", "압력"]):
        aliases.append(
            """
[Pressure 검색 보조 키워드]
Pressure test, pressure holding test, pressure drop, 시험 압력, 운전 압력, 압력 조건, 압력 유지 시험
"""
        )

    if any(keyword in text_lower for keyword in ["record", "기록", "양식", "report"]):
        aliases.append(
            """
[기록 양식 검색 보조 키워드]
기록 양식, 시험 결과 기록, Valve ID, 위치, 종류, 유체, 압력, 온도, 유량, 재시험 결과
"""
        )

    return "\n".join(aliases)


# ==============================
# 6. Chroma collection 이름 안전화
# ==============================
def make_safe_chroma_collection_name(file_name: str, embedding_model: str, file_bytes: bytes) -> str:
    file_hash = hashlib.md5(file_bytes).hexdigest()[:10]
    stem = Path(file_name).stem.lower()
    model_name = embedding_model.lower()

    name = f"shipleak_{stem}_{model_name}_{file_hash}_v11"
    name = re.sub(r"[^a-z0-9._-]", "_", name)
    name = re.sub(r"[._-]+", "_", name)
    name = name.strip("._-")

    if len(name) > 60:
        name = f"shipleak_{file_hash}_v11"

    if len(name) < 3:
        name = f"db_{file_hash}"

    return name


# ==============================
# 7. Vector DB 생성
# ==============================
def build_high_precision_db(
    file_bytes: bytes,
    file_name: str,
    embedding_model: str,
    api_key: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 없습니다.")

    os.environ["OPENAI_API_KEY"] = api_key

    raw_docs = load_documents_from_bytes(
        file_bytes=file_bytes,
        file_name=file_name,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_docs = splitter.split_documents(raw_docs)

    final_docs = []

    for idx, doc in enumerate(split_docs, start=1):
        metadata = doc.metadata or {}
        metadata["chunk_index"] = idx
        metadata["chunk_type"] = metadata.get("chunk_type", "knowledge_chunk")
        metadata["document_name"] = metadata.get("document_name", file_name)
        metadata["source"] = metadata.get("source", file_name)

        original_content = doc.page_content.strip()
        alias_content = make_search_alias(original_content)
        enriched_content = f"{original_content}\n\n{alias_content}".strip()

        final_docs.append(
            Document(
                page_content=enriched_content,
                metadata=_normalize_metadata(metadata),
            )
        )

    # Multi-Turn 후속 질문용 보강 지식 추가
    final_docs.extend(build_supplement_documents())

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
    )

    collection_name = make_safe_chroma_collection_name(file_name, embedding_model, file_bytes)

    database = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return database, final_docs


# ==============================
# 8. 검색 점수 보정
# ==============================
def _clamp_score(score: Any) -> float:
    try:
        value = float(score)
    except Exception:
        return 0.0

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return value


def _distance_to_score(distance: Any) -> float:
    try:
        distance_value = float(distance)
    except Exception:
        return 0.0

    if distance_value < 0:
        return 0.0

    return 1.0 / (1.0 + distance_value)


def extract_keywords(text: str) -> List[str]:
    text_lower = text.lower()

    keyword_map = {
        "밸브": ["밸브", "valve"],
        "누설": ["누설", "leakage", "leak"],
        "위치": ["위치", "location"],
        "표시": ["표시", "마킹", "marking", "mark"],
        "분필": ["분필", "chalk"],
        "기포": ["기포", "버블", "bubble"],
        "비눗물": ["비눗물", "soap"],
        "seat": ["seat", "시트"],
        "disc": ["disc", "디스크"],
        "교체": ["교체", "replacement", "replace"],
        "시험": ["시험", "test"],
        "압력": ["압력", "pressure"],
        "유체": ["유체", "fluid"],
        "기록": ["기록", "record", "report"],
        "양식": ["양식", "form", "template"],
        "음향": ["음향", "재측정", "rms", "fft", "sampling", "wav"],
        "체결": ["체결", "재조립", "토크", "토크렌치", "gasket", "seal"],
        "초동": ["초동", "대처", "초기", "initial"],
        "작업": ["작업공구", "작업 공구", "작업내용", "작업 내용", "공구", "준비물", "tool"],
        "packing": ["packing", "패킹", "gland", "stem", "빠졌", "빠짐", "이탈", "탈락"],
        "판정": ["합격", "개선", "판정", "판단 기준", "기준", "모호", "정량", "normal", "warning", "alert", "danger"],
        "도구": ["도구", "공구", "작업도구", "준비물", "tool", "tools"],
        "cavitation": ["cavitation", "캐비테이션", "압력강하", "유속"],
    }

    found = []

    for _, variants in keyword_map.items():
        if any(v in text_lower for v in variants):
            found.extend(variants)

    return list(dict.fromkeys(found))


def keyword_rerank_score(doc: Document, query: str, intent: str) -> float:
    content = (doc.page_content or "").lower()
    metadata = doc.metadata or {}
    source_type = str(metadata.get("source_type", "")).lower()

    score = 0.0

    for kw in extract_keywords(query):
        if kw.lower() in content:
            score += 0.04

    if intent == "verification_acceptance_criteria":
        for kw in ["합격", "개선", "판정", "기준", "rms", "dominant frequency", "spectral centroid", "normal", "warning", "alert", "danger", "leak test", "압력 유지", "기포", "정비 후 검증"]:
            if kw in content:
                score += 0.10

    if intent in ["verification_acceptance_criteria",
        "packing_displacement_initial_response", "packing_detailed_action"]:
        for kw in ["packing", "패킹", "gland", "stem", "빠졌", "빠짐", "이탈", "탈락", "gland 체결", "체결 토크", "stem 표면", "packing 교체", "spectral centroid", "leak test"]:
            if kw in content:
                score += 0.09

    if intent in ["packing_displacement_initial_response",
        "packing_detailed_action",
        "seat_leakage_initial_response", "seat_leakage_tools_work"]:
        for kw in ["seat leakage", "시트 누설", "초동", "대처", "작업공구", "작업 공구", "작업내용", "공구", "배관 격리", "압력 제거", "seat", "disc", "leak test", "rms", "fft", "토크렌치"]:
            if kw in content:
                score += 0.08


    if intent in ["packing_displacement_initial_response",
        "packing_detailed_action",
        "seat_leakage_initial_response",
        "seat_leakage_tools_work",
        "acoustic_remeasurement", "reassembly_tightening_check", "direct_followup"]:
        for kw in ["음향", "재측정", "동일 조건", "rms", "fft", "spectral centroid", "sampling", "16 khz", "10초", "wav", "체결", "재조립", "토크렌치", "leak test", "압력 유지"]:
            if kw in content:
                score += 0.08

    if intent == "cavitation_action_tools":
        for kw in ["cavitation", "캐비테이션", "압력강하", "유속", "밸브 개도", "fft", "dominant frequency", "rms", "압력계", "차압계", "유량계", "진동계", "음향 센서", "작업도구", "준비물", "공구"]:
            if kw in content:
                score += 0.08

    if intent == "leak_location_marking":
        for kw in ["누설 위치", "표시", "마킹", "분필", "chalk", "marker", "태그", "사진", "기포", "bubble", "비눗물"]:
            if kw in content:
                score += 0.07

    elif intent == "replacement_procedure":
        for kw in ["교체", "replacement", "seat", "disc", "시트", "디스크", "분해", "재조립"]:
            if kw in content:
                score += 0.05

    elif intent == "record_format":
        for kw in ["기록", "양식", "valve id", "압력", "유체", "재시험", "report"]:
            if kw in content:
                score += 0.05

    elif intent == "test_condition":
        for kw in ["시험 전", "유체", "압력", "pressure", "fluid", "격리", "안전"]:
            if kw in content:
                score += 0.05

    elif intent == "inspection":
        for kw in ["점검", "확인", "검사", "leak test", "pressure test"]:
            if kw in content:
                score += 0.04

    if source_type == "supplement" and intent in [
        "packing_displacement_initial_response",
        "packing_detailed_action",
        "seat_leakage_initial_response",
        "seat_leakage_tools_work",
        "acoustic_remeasurement",
        "reassembly_tightening_check",
        "direct_followup",
        "cavitation_action_tools",
        "leak_location_marking",
        "replacement_procedure",
        "record_format",
        "test_condition",
    ]:
        score += 0.10

    return score


# ==============================
# 9. 질문 의도 분석
# ==============================
def detect_question_intent(question: str, active_issue: str = "") -> str:
    q = question.lower()
    issue = (active_issue or "").lower()
    combined = f"{issue} {q}"

    has_packing_context = any(word in combined for word in [
        "packing", "패킹", "gland", "stem", "빠졌", "빠짐", "이탈", "탈락"
    ])

    has_seat_context = any(word in combined for word in [
        "seat leakage", "seat leak", "시트 누설", "seat", "disc", "시트", "디스크"
    ])

    if any(word in q for word in ["합격", "개선", "판정", "판단 기준", "기준", "모호", "구체적으로 제시"]):
        return "verification_acceptance_criteria"

    # Packing 빠짐/이탈은 Seat Leakage와 절대 혼동하면 안 됩니다.
    if has_packing_context and any(word in q for word in ["초동", "대처", "초기", "처음", "우선", "빠졌", "빠짐", "이탈", "탈락"]):
        return "packing_displacement_initial_response"

    if has_packing_context and any(word in q for word in ["구체", "자세", "상세", "권장 조치", "조치", "공구", "작업내용", "작업 내용", "어떻게"]):
        return "packing_detailed_action"

    if has_seat_context and any(word in q for word in ["초동", "대처", "초기", "처음", "우선"]):
        return "seat_leakage_initial_response"

    # 공구/상세 질문은 active_issue를 보고 분기합니다.
    # 기존처럼 무조건 Seat Leakage로 보내면 Packing 질문에도 Seat/Disc 답변이 나와서 오답이 됩니다.
    if any(word in q for word in ["작업공구", "작업 공구", "특정 공구", "공구", "준비물", "작업내용", "작업 내용", "상세히"]):
        if has_packing_context:
            return "packing_detailed_action"
        if has_seat_context:
            return "seat_leakage_tools_work"
        if any(word in combined for word in ["cavitation", "캐비테이션"]):
            return "cavitation_action_tools"
        return "direct_followup"

    if any(word in q for word in ["음향데이터", "음향 데이터", "재측정", "rms", "fft"]) and any(word in q for word in ["어떻게", "수행", "방법", "확인"]):
        return "acoustic_remeasurement"

    if any(word in q for word in ["재조립", "체결", "토크", "토크렌치"]) and any(word in q for word in ["어떻게", "확인", "방법", "상태"]):
        return "reassembly_tightening_check"

    if any(word in combined for word in ["cavitation", "캐비테이션"]) and any(word in q for word in ["작업도구", "도구", "공구", "준비물", "챙겨"]):
        return "cavitation_action_tools"

    if any(word in q for word in ["작업도구", "도구", "공구", "준비물", "챙겨"]) and any(word in q for word in ["조치", "문제점", "해결", "점검"]):
        return "cavitation_action_tools"

    if any(word in q for word in ["누설 위치", "위치 표시", "표시", "마킹", "분필", "마커", "태그", "기포 위치", "버블 위치"]):
        return "leak_location_marking"

    if any(word in q for word in ["교체", "replacement", "replace"]) and any(word in q for word in ["seat", "disc", "시트", "디스크"]):
        return "replacement_procedure"

    if any(word in q for word in ["기록", "양식", "form", "template", "로그", "이력", "report"]):
        return "record_format"

    if any(word in q for word in ["시험 전", "유체", "압력 조건", "시험 조건", "압력은", "압력 확인"]):
        return "test_condition"

    if any(word in q for word in ["교체 방법", "방법", "절차", "순서"]) and any(word in q for word in ["seat", "disc", "시트", "디스크", "밸브"]):
        return "replacement_procedure"

    if any(word in q for word in ["조치", "대응", "수정", "보완", "해결"]):
        return "action"

    if any(word in q for word in ["점검", "확인", "검사", "test", "검증"]):
        return "inspection"

    if any(word in q for word in ["원인", "왜", "cause"]):
        return "cause"

    return "diagnosis"

def is_followup_question(question: str, active_issue: str = "") -> bool:
    q = question.strip().lower()

    if not active_issue:
        return False

    followup_markers = [
        "그럼",
        "그러면",
        "앞에서",
        "위에서",
        "방금",
        "이 경우",
        "그 경우",
        "추가",
        "구체적으로",
        "자세히",
        "어떤 방식",
        "어떤 양식",
        "어떻게",
        "다른 무엇",
    ]

    if any(marker in q for marker in followup_markers):
        return True

    if len(q) <= 90 and any(word in q for word in [
        "조치", "점검", "확인", "기록", "양식", "조건", "원인", "교체", "방법", "절차",
        "표시", "마킹", "분필", "위치", "도구", "공구", "준비물", "챙겨", "음향", "재측정", "재조립", "체결", "작업내용", "상세히", "초동", "대처"
    ]):
        return True

    return False


def build_search_question(question: str, active_issue: str = "") -> Tuple[str, bool, str]:
    intent = detect_question_intent(question, active_issue)
    followup = is_followup_question(question, active_issue)

    if followup:
        base = f"현재 문제: {active_issue}. 후속 질문: {question}."
    else:
        base = question

    if intent == "verification_acceptance_criteria":
        base += " 합격 기준 개선 기준 판정 기준 정량 기준 RMS NORMAL WARNING ALERT DANGER Dominant Frequency Spectral Centroid Leak Test 압력 유지 시험 Bubble Test 기포 없음 정비 후 검증 RMS 감소 FFT Peak 감소"

    elif intent == "packing_displacement_initial_response":
        base += " Packing 빠짐 Packing 이탈 패킹 탈락 Gland Packing Stem 부위 누설 초동조치 현장 안전 배관 격리 압력 제거 LOTO Stem 주변 누설 Gland 체결 상태 Packing 교체 Leak Test 운전 재확인"

    elif intent == "packing_detailed_action":
        base += " Packing 빠짐 Packing 이탈 패킹 누설 상세 조치 작업공구 Gland 체결 토크 Stem 표면 손상 Packing 교체 토크렌치 LOTO 압력계 Leak Test 비눗물 음향 센서 Spectral Centroid Baseline 복귀 정비 후 검증"

    if intent == "seat_leakage_initial_response":
        base += " Seat Leakage 시트 누설 초동 대처 초기 조치 안전조치 배관 격리 압력 제거 누설음 압력 저하 Seat 점검 Disc 점검 Leak Test 압력 유지 시험 Seat 세척 교체 정비 후 검증"

    elif intent == "seat_leakage_tools_work":
        base += " Seat Leakage 시트 누설 특정 공구 작업공구 작업내용 준비물 안전보호구 LOTO 압력계 비눗물 Leak Test 장비 조명 내시경 토크렌치 수공구 Seat Disc Seal Gasket Packing 음향 센서 RMS FFT 점검표 사진기록"


    if intent == "acoustic_remeasurement":
        base += " 음향 데이터 재측정 동일 조건 재측정 정비 후 검증 RMS 감소 FFT Peak 감소 Spectral Centroid 고주파 누설음 소멸 Sampling Rate 16 kHz 10초 WAV Valve Body Inlet Pipe Outlet Pipe"

    elif intent == "reassembly_tightening_check":
        base += " 재조립 후 체결 상태 확인 볼트 너트 가스켓 Seal Packing Gland 플랜지 토크렌치 기준 토크 균등 체결 Leak Test 압력 유지 시험"

    if intent == "cavitation_action_tools":
        base += " Cavitation 캐비테이션 조치사항 문제 해결 작업도구 준비물 압력계 차압계 유량계 밸브 개도 확인 진동계 음향 센서 FFT 분석 RMS 분석 토크렌치 LOTO 점검표"

    if intent == "leak_location_marking":
        base += " 누설 위치 표시 방법 분필 마커 태그 사진 기록 기포 발생 위치 비눗물 테스트 누설 부위 마킹"

    elif intent == "replacement_procedure":
        base += " Seat Disc 교체 방법 교체 절차 분해 세척 손상 확인 동일 사양 부품 교체 재조립 누설 시험"

    elif intent == "record_format":
        base += " 시험 후 결과 기록 양식 Valve ID 위치 종류 유체 압력 온도 유량 누설 위치 조치 내용 재시험 결과"

    elif intent == "test_condition":
        base += " 시험 전 유체 압력 조건 확인 시험 압력 운전 압력 격리 안전 Leak Test Pressure Test"

    elif intent == "action":
        base += " 권장 조치 조치 순서 세척 교체 재조립 재시험 검증"

    elif intent == "inspection":
        base += " 점검 방법 확인 방법 Leak Test Pressure Test 누설 위치 압력 유지"

    elif intent == "cause":
        base += " 원인 cause failure leakage seat disc packing flange pressure"

    return base, followup, intent


def get_answer_style_instruction(intent: str) -> str:
    if intent == "verification_acceptance_criteria":
        return """
사용자는 합격/개선 판단 기준이 모호하므로 구체적인 판정 기준을 요구하고 있습니다.
진단 요약/가능한 원인/점검 방법/권장 조치의 고정 포맷을 절대 사용하지 마세요.
반드시 정량 기준, 합격/개선/불합격 기준을 표로 제시하세요.

아래 형식으로 답변하세요.

1. 판정 전제
   - 동일 조건 재측정이 전제임을 설명합니다.

2. 정량 판정 기준표
   - RMS, Dominant Frequency, Spectral Centroid의 NORMAL/WARNING/ALERT/DANGER 기준을 표로 작성합니다.

3. 합격 기준
   - RMS < 0.097
   - Dominant Frequency < 450.0 Hz
   - Spectral Centroid < 2821.9 Hz
   - Leak Test 또는 압력 유지 시험 합격
   - Bubble Test 시 기포 없음
   - 고주파 누설음 소멸
   을 포함합니다.

4. 개선 기준
   - 정비 전 대비 RMS 감소
   - FFT Peak 또는 Dominant Frequency 이상 성분 감소
   - Spectral Centroid 감소
   - 누설음 감소
   단, WARNING 이상이면 추가 모니터링 대상으로 설명합니다.

5. 불합격/재점검 기준
   - RMS >= 0.118
   - Dominant Frequency >= 690.7 Hz
   - Spectral Centroid >= 3174.6 Hz
   - Leak Test 실패
   - Bubble Test 기포 발생
   - 압력 저하 지속
   을 포함합니다.

6. 현장 판정 예시
   - 초보 검사원이 바로 판단할 수 있도록 예시를 작성합니다.

7. 기록 항목
   - 판정에 필요한 기록 항목을 작성합니다.

8. 문서 근거
   - 검색된 문서에서 확인된 근거를 요약합니다.
"""

    if intent == "packing_displacement_initial_response":
        return """
사용자는 'Packing이 빠졌다/이탈되었다'는 상황의 초동조치사항을 묻고 있습니다.
Seat Leakage, Seat/Disc 점검, Seat 세척 답변을 하지 마세요.
Packing/Stem/Gland 중심으로 답변하세요.
진단 요약/가능한 원인/점검 방법/권장 조치의 고정 포맷을 사용하지 마세요.

아래 형식으로 답변하세요.

1. 상황 판단
   - Packing 빠짐/이탈이 왜 중요한지 설명합니다.

2. 즉시 초동조치
   - 접근 안전 확인
   - 고온/고압/가연성 확인
   - 배관 격리
   - 압력 및 잔압 제거
   - 필요 시 LOTO
   - 임의 삽입 금지
   순서로 작성합니다.

3. 현장 확인 항목
   - Stem 주변 누설 흔적
   - Packing 이탈 상태
   - Gland 체결 상태
   - Stem 표면 손상
   - 운전 조건 기록
   을 작성합니다.

4. 정비 요청 또는 조치 방향
   - Gland 체결 조정
   - Packing 교체
   - Stem 표면 세척 또는 교정
   - Leak Test 또는 운전 재확인
   을 작성합니다.

5. 기록 및 보고 항목
   - 초보 검사원이 선임에게 보고할 내용을 작성합니다.

6. 문서 근거
   - 검색된 문서에서 확인된 Packing/Stem Leakage 근거를 요약합니다.
"""

    if intent == "packing_detailed_action":
        return """
사용자는 앞선 Packing 빠짐/이탈 질문에 이어 권장조치를 더 구체적으로 묻고 있습니다.
Seat Leakage, Seat/Disc 공구, Seat 세척 중심으로 답변하지 마세요.
Packing/Stem/Gland 작업 중심으로 상세히 답변하세요.
진단 요약/가능한 원인/점검 방법/권장 조치의 고정 포맷을 사용하지 마세요.

아래 형식으로 답변하세요.

1. 구체 조치 순서
   - 안전조치부터 재검증까지 순서대로 작성합니다.

2. 준비 공구 및 장비
   - 반드시 표로 작성합니다.
   - 표 컬럼은 [구분, 준비 공구/장비, 사용 목적]입니다.
   - LOTO, 압력계, 조명/내시경, 스패너/렌치, 토크렌치, Packing 제거 공구, 교체 Packing, Leak Test 장비, 점검표/사진기록 장비를 포함합니다.

3. 작업내용 상세
   - Gland 체결 확인
   - Stem 표면 확인
   - Packing 제거
   - Packing 교체
   - Gland 균등 체결
   - Leak Test 또는 운전 재확인
   순서로 작성합니다.

4. 작업 시 주의사항
   - 압력 잔류, 과도한 체결, 임의 재삽입 금지 등을 포함합니다.

5. 정비 후 검증
   - Stem 부위 누설 흔적 제거, Spectral Centroid Baseline 복귀, Leak Test 또는 운전 재확인을 포함합니다.

6. 문서 근거
   - 검색된 문서에서 확인된 근거를 요약합니다.
"""

    if intent == "seat_leakage_initial_response":
        return """
사용자는 Seat Leakage 발생 시 초동 대처방안을 묻고 있습니다.
진단 요약/가능한 원인/점검 방법/권장 조치의 고정 포맷을 사용하지 마세요.
초보 품질검사원이 현장에서 바로 확인할 수 있는 초기 조치 중심으로 답변하세요.

아래 형식으로 답변하세요.

1. 초동 판단
2. 즉시 안전조치
3. 현장 확인 순서
4. 정비 요청 또는 조치 판단
5. 정비 후 검증
6. 기록 및 보고 항목
7. 문서 근거
"""

    if intent == "seat_leakage_tools_work":
        return """
사용자는 앞선 Seat Leakage 질문에 이어 특정 공구와 작업내용을 상세히 묻고 있습니다.
진단 요약/가능한 원인/점검 방법/권장 조치의 고정 포맷을 사용하지 마세요.
공구와 작업내용을 분리해서 실무형으로 답변하세요.

아래 형식으로 답변하세요.

1. 준비해야 할 공구 및 장비
   - 반드시 표로 작성합니다.
   - 표 컬럼은 [구분, 준비 공구/장비, 사용 목적]으로 구성합니다.

2. 작업내용 상세
   - 안전조치
   - 누설 확인
   - Seat/Disc 점검
   - 세척 또는 교체
   - 재조립
   - 재시험
   순서로 작성합니다.

3. 작업 시 주의사항

4. 기록 항목

5. 문서 근거
"""

    if intent == "acoustic_remeasurement":
        return """
사용자는 후속 질문으로 '음향 데이터 재측정 방법'만 묻고 있습니다.
진단 요약/가능한 원인/권장 조치 포맷을 사용하지 마세요.
현재 질문에 대한 직접 답변만 작성하세요.

아래 형식으로 답변하세요.

1. 수행 목적
2. 재측정 절차
3. 비교해야 할 데이터
4. 합격/개선 판단 기준
5. 기록 항목
6. 문서 근거
"""

    if intent == "reassembly_tightening_check":
        return """
사용자는 후속 질문으로 '재조립 후 체결 상태 확인 방법'만 묻고 있습니다.
진단 요약/가능한 원인/권장 조치 포맷을 사용하지 마세요.
현재 질문에 대한 직접 답변만 작성하세요.

아래 형식으로 답변하세요.

1. 확인 목적
2. 체결 상태 확인 절차
3. 확인 도구
4. 이상 판단 기준
5. 기록 항목
6. 문서 근거
"""

    if intent == "direct_followup":
        return """
사용자는 앞선 답변에 대한 구체적인 후속 질문을 하고 있습니다.
진단 요약/가능한 원인/점검 방법/권장 조치의 고정 포맷을 사용하지 마세요.
현재 질문에 필요한 항목만 간단하고 직접적으로 답변하세요.

답변은 다음 원칙을 따르세요.
- 질문이 2개 이상이면 [질문 1], [질문 2]로 나누어 답변합니다.
- 문서에서 확인되는 내용만 답변합니다.
- 문서에 없는 내용은 해당 항목에서 '제공된 문서만으로는 확인하기 어렵습니다'라고 씁니다.
"""

    if intent == "cavitation_action_tools":
        return """
사용자는 Cavitation 발생 시 조치사항과 문제 해결을 위한 작업도구를 동시에 묻고 있습니다.
진단 요약/가능한 원인/점검 방법/권장 조치 기본 포맷을 사용하지 마세요.
질문에 포함된 두 요구사항인 '조치사항'과 '작업도구'를 반드시 분리해서 답변하세요.

아래 형식으로 답변하세요.

1. 상황 판단
   - Cavitation 의심 근거를 1~2문장으로 요약합니다.

2. 조치사항
   - 운전압력/전후단 압력차 확인
   - 밸브 개도 조건과 유량 확인
   - 운전압력 또는 개도 조건 조정
   - 유속 조건 재검토
   - 밸브 사양과 배관 조건 적합성 검토
   - 필요 시 밸브 선정 변경 또는 운전 제한 검토

3. 문제 해결을 위한 작업도구
   - 반드시 표로 작성합니다.
   - 표 컬럼은 [구분, 준비 도구, 사용 목적]으로 구성합니다.
   - 압력계/차압계/유량계/밸브 개도 확인 도구/진동계/음향 센서/FFT 분석 프로그램/토크렌치/LOTO/점검표를 포함합니다.

4. 작업 순서
   - 현장에서 수행할 순서대로 작성합니다.

5. 조치 후 검증
   - FFT Peak 감소, 진동/RMS 감소, 동일 조건 재측정 결과 기록을 포함합니다.

6. 근거 문서 요약
   - 검색된 문서에서 확인된 근거를 간단히 요약합니다.
"""

    if intent == "leak_location_marking":
        return """
사용자는 누설 위치를 어떻게 표시/마킹해야 하는지 묻고 있습니다.
진단 요약/가능한 원인/권장 조치 포맷을 사용하지 마세요.
분필 사용 가능 여부, 대체 표시 방법, 기록 방법을 직접 답변하세요.

아래 형식으로 답변하세요.

1. 결론
   - 분필을 사용할 수 있는지, 어떤 경우에 적합한지 간단히 답합니다.

2. 누설 위치 표시 방법
   - 분필
   - 페인트 마커/산업용 마킹펜
   - 태그/라벨
   - 사진 기록
   - 도면 또는 점검표 기록
   중 현장 조건에 맞게 설명합니다.

3. 권장 절차
   - 비눗물 테스트 또는 Leak Test로 기포 발생 위치를 확인합니다.
   - 압력을 낮추거나 안전 상태를 확보합니다.
   - 위치를 표시하고 사진/기록을 남깁니다.
   - 조치 후 재시험합니다.

4. 주의사항
   - 가연성 가스, 고온 표면, 젖은 표면, 회전체 주변 등 주의사항을 정리합니다.

5. 기록 항목
   - Valve ID, 위치, 시험 압력, 표시 방법, 사진 번호, 조치 내용, 재시험 결과를 정리합니다.

6. 근거 문서 요약
   - 검색된 문서에서 확인된 근거를 간단히 요약합니다.
"""

    if intent == "replacement_procedure":
        return """
사용자는 Seat 또는 Disc 교체 방법을 묻고 있습니다.
진단 요약/가능한 원인 포맷을 사용하지 마세요.
이미 손상이 있다는 상황을 전제로, 실제 작업 절차 중심으로 답변하세요.

1. 작업 전 안전조치
2. 분해 및 손상 확인
3. Seat 또는 Disc 교체 절차
4. 재조립 및 조치 후 확인
5. 기록 항목
6. 근거 문서 요약
"""

    if intent == "record_format":
        return """
사용자는 시험 후 결과 기록 양식을 묻고 있습니다.
진단/원인/조치 포맷을 사용하지 마세요.

1. 기록 목적
2. 권장 기록 양식
   - Markdown 표로 작성합니다.
   - 표 컬럼은 [No., 기록 항목, 기록 내용, 작성 예시]로 구성합니다.
3. 필수 기록 항목
4. 작성 시 주의사항
5. 근거 문서 요약
"""

    if intent == "test_condition":
        return """
사용자는 시험 전 유체와 압력 조건 확인 방법을 묻고 있습니다.
진단/원인 중심으로 답변하지 마세요.

1. 확인 목적
2. 시험 전 확인 항목
3. 확인 절차
4. 판정 및 기록 항목
5. 근거 문서 요약
"""

    if intent == "action":
        return """
사용자는 권장 조치 또는 해결 방법을 묻고 있습니다.
원인 분석을 반복하지 말고 조치 중심으로 답변하세요.

1. 조치 방향
2. 권장 조치
3. 조치 순서
4. 조치 후 확인
5. 근거 문서 요약
"""

    if intent == "inspection":
        return """
사용자는 점검 또는 확인 방법을 묻고 있습니다.
원인 분석보다 점검 절차 중심으로 답변하세요.

1. 점검 목적
2. 사전 확인 사항
3. 점검 절차
4. 판정 기준
5. 근거 문서 요약
"""

    if intent == "cause":
        return """
사용자는 원인을 묻고 있습니다.

1. 원인 요약
2. 가능한 원인
3. 원인별 확인 방법
4. 우선 확인 순서
5. 근거 문서 요약
"""

    return """
사용자는 증상에 대한 진단을 묻고 있습니다.

1. 진단 요약
2. 가능한 원인
3. 점검 방법
4. 권장 조치
5. 근거 문서 요약
"""


# ==============================
# 10. 검색
# ==============================
def single_query_search(
    database,
    query: str,
    k: int = 5,
    intent: str = "diagnosis",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    fetch_k = max(k * 4, 12)

    try:
        docs_with_scores: List[Tuple[Document, float]] = database.similarity_search_with_relevance_scores(
            query,
            k=fetch_k,
        )

        for doc, score in docs_with_scores:
            base_score = _clamp_score(score)
            boost = keyword_rerank_score(doc, query, intent)
            results.append(
                {
                    "doc": doc,
                    "score": min(base_score + boost, 1.0),
                    "base_score": base_score,
                    "boost_score": boost,
                    "search_query_used": query,
                }
            )

    except Exception:
        docs_with_scores = database.similarity_search_with_score(
            query,
            k=fetch_k,
        )

        for doc, raw_score in docs_with_scores:
            base_score = _distance_to_score(raw_score)
            boost = keyword_rerank_score(doc, query, intent)
            results.append(
                {
                    "doc": doc,
                    "score": min(base_score + boost, 1.0),
                    "base_score": base_score,
                    "boost_score": boost,
                    "search_query_used": query,
                }
            )

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:k]


# ==============================
# 11. 대화 이력 처리
# ==============================
def format_conversation_history(messages: List[Dict[str, str]], max_turns: int = 8) -> str:
    if not messages:
        return "이전 대화 없음"

    recent_messages = messages[-max_turns:]
    lines = []

    for message in recent_messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            lines.append(f"사용자: {content}")
        elif role == "assistant":
            short_content = content[:800]
            lines.append(f"AI: {short_content}")

    return "\n".join(lines) if lines else "이전 대화 없음"


def format_correction_memory(
    correction_memory: Optional[List[Dict[str, str]]] = None,
    active_issue: str = "",
    max_items: int = 5,
) -> str:
    """
    사용자가 이전 답변을 정정한 내용을 프롬프트에 주입합니다.
    같은 주제의 후속 질문에서 반복 오류가 나지 않도록 하기 위한 간단한 메모리입니다.
    """
    correction_memory = correction_memory or []

    if not correction_memory:
        return "사용자 정정 없음"

    active_issue_text = (active_issue or "").lower()
    selected = []

    # 현재 대화 주제와 관련된 정정을 우선 사용합니다.
    for item in reversed(correction_memory):
        issue = str(item.get("active_issue", "")).lower()
        correction = str(item.get("correction", "")).strip()

        if not correction:
            continue

        if active_issue_text and (
            active_issue_text in issue
            or issue in active_issue_text
            or any(token in correction.lower() for token in active_issue_text.split() if len(token) >= 2)
        ):
            selected.append(item)

        if len(selected) >= max_items:
            break

    # 관련 항목이 없으면 최근 정정도 일부 참고합니다.
    if not selected:
        selected = list(reversed(correction_memory))[:max_items]

    lines = []
    for idx, item in enumerate(selected, start=1):
        lines.append(
            f"[정정 {idx}] 주제: {item.get('active_issue', '미지정')} / 내용: {item.get('correction', '')}"
        )

    return "\n".join(lines) if lines else "사용자 정정 없음"


# ==============================
# 12. Context 생성
# ==============================
def build_context_from_results(ranked_results: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for idx, item in enumerate(ranked_results, start=1):
        doc = item["doc"]
        metadata = doc.metadata or {}

        source = (
            metadata.get("document_name")
            or metadata.get("file_name")
            or metadata.get("source")
            or "알 수 없는 문서"
        )

        page_number = metadata.get("page_number", "")
        page_text = f"Page: {page_number}" if page_number else "Page: N/A"

        block = f"""
[검색순위 {idx}]
Score: {item.get("score")}
Base Score: {item.get("base_score", "")}
Boost Score: {item.get("boost_score", "")}
Source: {source}
{page_text}
Chunk Index: {metadata.get("chunk_index", "")}
Chunk Type: {metadata.get("chunk_type", "")}
Topic: {metadata.get("topic", "")}
Search Query Used: {item.get("search_query_used", "")}

Content:
{doc.page_content}
"""
        context_blocks.append(block)

    return "\n\n".join(context_blocks)




# ==============================
# 14. 답변 품질 판단 / KB 업데이트 후보
# ==============================
def required_keywords_for_intent(intent: str) -> Dict[str, List[str]]:
    """
    질문 의도별로 검색 결과에 반드시 포함되면 좋은 핵심 키워드입니다.
    모든 키워드가 들어가야 하는 것은 아니고, 그룹별로 일부가 포함되는지 판단합니다.
    """
    rules = {
        "diagnosis": {
            "symptom_or_fault": ["누설", "leak", "leakage", "진동", "소음", "cavitation", "캐비테이션", "pressure", "압력"],
        },
        "action": {
            "action": ["조치", "세척", "교체", "재조립", "재시험", "검증", "action"],
        },
        "inspection": {
            "inspection": ["점검", "확인", "검사", "test", "leak test", "pressure test"],
        },
        "cavitation_action_tools": {
            "cavitation": ["cavitation", "캐비테이션", "압력강하", "유속", "fft", "rms"],
            "tools": ["작업도구", "도구", "공구", "준비물", "압력계", "차압계", "유량계", "진동계", "음향 센서", "fft 분석", "loto"],
        },
        "leak_location_marking": {
            "marking": ["누설 위치", "표시", "마킹", "분필", "마커", "태그", "사진", "기포", "bubble"],
        },
        "replacement_procedure": {
            "replacement": ["교체", "replacement", "seat", "disc", "시트", "디스크", "분해", "재조립"],
        },
        "record_format": {
            "record": ["기록", "양식", "valve id", "시험 압력", "재시험", "report"],
        },
        "test_condition": {
            "condition": ["시험 전", "유체", "압력", "격리", "안전", "pressure", "fluid"],
        },
        "cause": {
            "cause": ["원인", "cause", "마모", "손상", "이물질", "압력강하", "유속"],
        },
    }
    return rules.get(intent, rules["diagnosis"])


def evaluate_retrieval_quality(
    question: str,
    intent: str,
    ranked_results: List[Dict[str, Any]],
    min_relevance_score: float = 0.60,
    strong_relevance_score: float = 0.80,
) -> Dict[str, Any]:
    """
    RAG 답변 전에 검색 품질을 판단합니다.

    판단 결과:
    - strong: 기본문서 근거가 충분함
    - partial: 일부 근거는 있으나 질문의 일부 요구사항이 부족함
    - insufficient: 근거 부족. 답변 생성하지 않고 KB 업데이트 후보로 처리
    """
    if not ranked_results:
        return {
            "status": "insufficient",
            "top_score": 0.0,
            "missing_keyword_groups": [],
            "reason": "검색 결과가 없습니다.",
            "needs_kb_update": True,
            "needs_code_update": False,
        }

    top_score = float(ranked_results[0].get("score", 0.0))

    combined_text = "\n".join(
        (item.get("doc").page_content if item.get("doc") else "")
        for item in ranked_results[:3]
    ).lower()

    required_groups = required_keywords_for_intent(intent)
    missing_groups = []

    for group_name, keywords in required_groups.items():
        if not any(keyword.lower() in combined_text for keyword in keywords):
            missing_groups.append(group_name)

    if top_score < min_relevance_score:
        return {
            "status": "insufficient",
            "top_score": top_score,
            "missing_keyword_groups": missing_groups,
            "reason": f"최고 유사도 {top_score:.3f}가 최소 기준 {min_relevance_score:.3f} 미만입니다.",
            "needs_kb_update": True,
            "needs_code_update": False,
        }

    if missing_groups:
        return {
            "status": "partial",
            "top_score": top_score,
            "missing_keyword_groups": missing_groups,
            "reason": "일부 근거는 검색되었지만 질문 의도에 필요한 세부 키워드가 부족합니다.",
            "needs_kb_update": True,
            "needs_code_update": False,
        }

    if top_score < strong_relevance_score:
        return {
            "status": "partial",
            "top_score": top_score,
            "missing_keyword_groups": missing_groups,
            "reason": f"유사도 {top_score:.3f}로 참고 답변은 가능하지만 강한 근거는 아닙니다.",
            "needs_kb_update": False,
            "needs_code_update": False,
        }

    return {
        "status": "strong",
        "top_score": top_score,
        "missing_keyword_groups": [],
        "reason": "기본문서 근거가 충분합니다.",
        "needs_kb_update": False,
        "needs_code_update": False,
    }


def make_kb_update_candidate(
    question: str,
    search_question: str,
    intent: str,
    quality: Dict[str, Any],
    active_issue: str = "",
) -> Dict[str, Any]:
    """
    미흡한 질문을 지식문서 업데이트 후보로 저장하기 위한 데이터 구조입니다.
    """
    return {
        "question": question,
        "search_question": search_question,
        "intent": intent,
        "active_issue": active_issue,
        "quality_status": quality.get("status", ""),
        "top_score": quality.get("top_score", 0.0),
        "reason": quality.get("reason", ""),
        "missing_keyword_groups": ", ".join(quality.get("missing_keyword_groups", [])),
        "suggested_kb_action": suggest_kb_update_action(intent),
    }


def suggest_kb_update_action(intent: str) -> str:
    suggestions = {
        "cavitation_action_tools": "KB-14. Cavitation 조치사항 및 작업도구 체크리스트 추가",
        "leak_location_marking": "KB-13. 누설 위치 표시 및 마킹 절차 추가",
        "replacement_procedure": "Seat 또는 Disc 교체 절차 Appendix 추가",
        "record_format": "시험 후 결과 기록 양식 Appendix 추가",
        "test_condition": "시험 전 유체/압력 조건 확인 절차 추가",
        "inspection": "점검 절차와 판정 기준 보강",
        "action": "조치 순서와 조치 후 검증 항목 보강",
        "cause": "원인별 확인 방법 보강",
        "diagnosis": "증상-원인-점검-조치-검증 Knowledge Block 보강",
    }
    return suggestions.get(intent, "해당 질문에 대한 Knowledge Block 추가")


def make_insufficient_answer(
    question: str,
    search_question: str,
    intent: str,
    quality: Dict[str, Any],
) -> str:
    """
    근거 부족 시 LLM을 호출하지 않고 고정 안전 답변을 반환합니다.
    """
    return f"""
제공된 기본 지식문서만으로는 해당 질문에 충분히 답변하기 어렵습니다.

1. 판단 결과
- 질문 의도: {intent}
- 검색에 사용한 질문: {search_question}
- 최고 유사도: {quality.get('top_score', 0.0):.3f}
- 판단 사유: {quality.get('reason', '')}

2. 답변 제한
- RAG 원칙상 기본문서에 없는 내용을 AI가 임의로 만들어 답변하지 않습니다.
- 현재 질문은 지식문서 업데이트 후보로 저장하는 것이 적절합니다.

3. 권장 조치
- {suggest_kb_update_action(intent)}
- 기본문서에 해당 항목을 Appendix 또는 새로운 KB 섹션으로 추가한 뒤 Vector DB를 다시 생성하세요.
""".strip()


def build_evidence_policy_instruction(quality: Dict[str, Any]) -> str:
    """
    검색 품질에 따라 프롬프트에 추가할 근거 사용 정책입니다.
    """
    status = quality.get("status", "")

    if status == "strong":
        return """
[근거 사용 정책]
검색 근거가 충분합니다.
Context에 있는 내용을 바탕으로 질문 의도에 맞게 답변하세요.
"""

    if status == "partial":
        return f"""
[근거 사용 정책]
검색 근거가 일부만 충분합니다.
확인 가능한 내용과 문서에 부족한 내용을 분리해서 답변하세요.
문서에 없는 세부 내용은 추측하지 말고 '기본문서에 명확히 정리되어 있지 않습니다'라고 표시하세요.
판단 사유: {quality.get('reason', '')}
"""

    return """
[근거 사용 정책]
검색 근거가 부족합니다.
기술적 답변을 생성하지 말고 문서 업데이트가 필요하다고 답변해야 합니다.
"""


# ==============================
# 13. RAG 답변
# ==============================
def answer_with_manual_rag(
    database,
    question: str,
    k: int = 5,
    api_key: str = "",
    llm_model: str = "gpt-4o-mini",
    conversation_messages: Optional[List[Dict[str, str]]] = None,
    active_issue: str = "",
    correction_memory: Optional[List[Dict[str, str]]] = None,
    min_relevance_score: float = 0.60,
    strong_relevance_score: float = 0.80,
) -> Dict[str, Any]:
    """
    RAG 답변 함수.

    처리 원칙:
    1. 질문 의도 분석
    2. Single-Query 검색
    3. 검색 품질 평가
    4. 근거 부족이면 LLM 답변 생성 중단
    5. 부분 근거면 '확인 가능/문서 부족'을 분리하여 답변
    6. 충분한 근거면 질문 의도별 포맷으로 답변
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    conversation_messages = conversation_messages or []
    conversation_history = format_conversation_history(conversation_messages)
    correction_memory_text = format_correction_memory(
        correction_memory=correction_memory,
        active_issue=active_issue,
    )

    search_question, is_followup, intent = build_search_question(
        question=question,
        active_issue=active_issue,
    )

    ranked_results = single_query_search(
        database=database,
        query=search_question,
        k=k,
        intent=intent,
    )

    quality = evaluate_retrieval_quality(
        question=question,
        intent=intent,
        ranked_results=ranked_results,
        min_relevance_score=min_relevance_score,
        strong_relevance_score=strong_relevance_score,
    )

    kb_update_candidate = None

    if quality.get("needs_kb_update"):
        kb_update_candidate = make_kb_update_candidate(
            question=question,
            search_question=search_question,
            intent=intent,
            quality=quality,
            active_issue=active_issue,
        )

    if quality.get("status") == "insufficient":
        answer = make_insufficient_answer(
            question=question,
            search_question=search_question,
            intent=intent,
            quality=quality,
        )

        return {
            "question": question,
            "search_question": search_question,
            "is_followup": is_followup,
            "intent": intent,
            "answer": answer,
            "retrieved_results": ranked_results,
            "context": build_context_from_results(ranked_results),
            "quality": quality,
            "kb_update_candidate": kb_update_candidate,
        }

    context = build_context_from_results(ranked_results)

    answer_style_instruction = (
        get_answer_style_instruction(intent)
        + "\n\n"
        + build_evidence_policy_instruction(quality)
    )

    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
    )

    rag_chain = RAG_PROMPT | llm | StrOutputParser()

    answer = rag_chain.invoke(
        {
            "active_issue": active_issue if active_issue else "현재 대화 주제 없음",
            "conversation_history": conversation_history,
            "correction_memory_text": correction_memory_text,
            "search_question": search_question,
            "intent": intent,
            "answer_style_instruction": answer_style_instruction,
            "context": context,
            "question": question,
        }
    )

    return {
        "question": question,
        "search_question": search_question,
        "is_followup": is_followup,
        "intent": intent,
        "answer": answer,
        "retrieved_results": ranked_results,
        "context": context,
        "correction_memory_text": correction_memory_text,
        "quality": quality,
        "kb_update_candidate": kb_update_candidate,
    }
