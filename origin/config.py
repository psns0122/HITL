"""환경변수 기반 설정.  [원본·규약]

사내 게이트웨이 키 등 민감정보는 코드가 아니라 .env 에만 둔다.
변수 이름은 사내 기존 프로젝트(pptx-vision-rag/config.py)의 규약을 따른다.

원본에는 HITL 관련 설정(MAX_COLLECT / MAX_VALIDATE / MAX_HOPS)이 없다.
그건 app/config.py 에서 추가된 것이다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 모듈 import 시점에 .env 를 한 번 로드한다.
# override=False — 이미 설정된 OS 환경변수는 덮어쓰지 않는다.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get(name: str, default: str = "") -> str:
    """환경변수 조회. 없거나 비어 있으면 기본값."""
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()


# ── LLM 게이트웨이 (OpenAI 호환) ──────────────────────────────────────────
LLM_GATEWAY_BASE_URL = _get("LLM_GATEWAY_BASE_URL", "http://hcp.llm.skhynix.com/v1")

# 사내 게이트웨이는 인증 키를 요구하지 않아 빈 값을 허용한다.
# (OpenAI SDK 가 빈 키를 거부해서 _llm.py 에서 'EMPTY' 로 대체한다)
LLM_GATEWAY_API_KEY = _get("LLM_GATEWAY_API_KEY", "")

LLM_CHAT_MODEL = _get("LLM_CHAT_MODEL", "GaiA-LLM-Latest")
LLM_MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "4000"))
LLM_TIMEOUT = int(_get("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(_get("LLM_MAX_RETRIES", "2"))

# ── FAB 목록 / 별칭  [원본·추정] ──────────────────────────────────────────
# fab_extract_tool 은 DB 를 보지 않고 이 표만 대조해서 FAB 을 정규화한다.
# 사용자가 별칭으로 불러도 인식해야 하므로 암묵지 별칭을 함께 둔다.
# **실제 목록은 사내 config.py 에서 가져와 덮어쓸 것.**
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
