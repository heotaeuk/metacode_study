# config.py
# ShipLeak Copilot RAG Streamlit 앱 설정값

from pathlib import Path

APP_TITLE = "ShipLeak Copilot RAG"

# 현재 파일(config.py)이 있는 폴더 = streamlit_app
BASE_DIR = Path(__file__).resolve().parent

# 프로젝트 루트 폴더 = New_ShipLeak
ROOT_DIR = BASE_DIR.parent

# 기본 지식문서는 streamlit_app 폴더 안에 보관
DEFAULT_DATA_PATH = BASE_DIR / "00_ShipLeak_Copilot_Integrated_Knowledge_Base.docx"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
DISTANCE_METRIC = "cosine"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.25
