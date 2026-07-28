"""환경변수 기반 설정.

origin/config.py 와 동일하다. app 추가분은 ************* 로 표시.
사내 게이트웨이 키 등 민감정보는 코드가 아니라 .env 에만 둔다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 모듈 import 시점에 .env 를 한 번 로드한다.
# override=False — 이미 설정된 OS 환경변수는 덮어쓰지 않는다.
# *************  [app — Windows 메모장이 붙이는 UTF-8 BOM 을 벗겨 읽는다]
load_dotenv(PROJECT_ROOT / ".env", override=False, encoding="utf-8-sig")
# *************


def _get(name: str, default: str = "") -> str:
    """환경변수 조회. 없거나 비어 있으면 기본값."""
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


# ── LLM 게이트웨이 (OpenAI 호환) ──────────────────────────────────────────
# _llm.py 가 참조하는 세 값.
API_BASE_TEMPLATE = _get("API_BASE_TEMPLATE", "http://hcp.llm.skhynix.com/v1")
MODEL_LIST_ENDPOINT = _get("MODEL_LIST_ENDPOINT", "/models")
api_key = _get("LLM_GATEWAY_API_KEY", "")

# ── FAB 목록 / 별칭 ───────────────────────────────────────────────────────
# fab_extract_tool 은 DB 를 보지 않고 이 표만 대조해서 FAB 을 정규화한다.
VALID_FABS = ["M16", "M14", "M11"]

FAB_ALIASES = {
    # "별칭": "정규 FAB명"
    "엠십육": "M16",
    "16라인": "M16",
}

# ── 로그: {LOG_DIR}/{YYYY-MM}/{YYYY-MM-DD}.jsonl ──────────────────────────
LOG_DIR = _get("LOG_DIR", "./devLogs")

# ── 서버 ──────────────────────────────────────────────────────────────────
API_HOST = _get("API_HOST", "0.0.0.0")
API_PORT = int(_get("API_PORT", "8000"))

# *************  [app 전용 — origin 에 없음]  *************

# Streamlit 데모 UI 가 붙을 백엔드 주소
API_BASE_URL = _get("API_BASE_URL", "http://localhost:8000/llm/api")

# 사내망 밖(로컬 ollama 등)에서 돌릴 때 모델명을 .env 로 갈아끼우기 위한 값.
# 비워두면 사내 기본값이 그대로 쓰인다 — 사내에서는 아무것도 안 바뀐다.
#   DEFAULT_MODEL=glm4:latest
#   AVAILABLE_MODELS=glm4:latest,llama3.2:latest
DEFAULT_MODEL = _get("DEFAULT_MODEL", "")
AVAILABLE_MODELS = [m.strip() for m in _get("AVAILABLE_MODELS", "").split(",")
                    if m.strip()]

# HITL 루프 가드
MAX_COLLECT = int(_get("MAX_COLLECT", "5"))       # 파라미터 재질문 상한
MAX_VALIDATE = int(_get("MAX_VALIDATE", "3"))     # 검증 재시도 상한
MAX_HOPS = int(_get("MAX_HOPS", "3"))             # needs-핸드오프 왕복 상한

# *************  [app 전용 끝]  *************
