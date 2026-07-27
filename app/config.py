"""환경변수 기반 설정.

사내 게이트웨이 키 등 민감정보는 코드가 아니라 .env 에만 둔다.

네이밍은 사내 기존 프로젝트(pptx-vision-rag)의 컨벤션을 그대로 따른다
(`LLM_GATEWAY_BASE_URL` / `LLM_GATEWAY_API_KEY` / `LLM_CHAT_MODEL`).
→ 기존 .env 값을 그대로 복사해 쓸 수 있다.
구버전 이름(OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME)도 폴백으로 받는다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 이미 설정된 OS 환경변수는 덮어쓰지 않는다 (사내 컨벤션과 동일)
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get(name: str, default: str = "", *aliases: str) -> str:
    """환경변수 조회. name 이 비어 있으면 aliases 를 순서대로 본다."""
    val = os.getenv(name)
    if val is None or val.strip() == "":
        for alt in aliases:
            val = os.getenv(alt)
            if val is not None and val.strip() != "":
                return val
        return default
    return val


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ── LLM 게이트웨이 (OpenAI 호환) ──────────────────────────────────────────
# 1 이면 LLM 없이 규칙 기반 목업으로 동작 (HITL 검증/데모용)
FAKE_LLM = _as_bool(_get("FAKE_LLM", "1"))

GATEWAY_BASE_URL = _get("LLM_GATEWAY_BASE_URL", "http://hcp.llm.skhynix.com/v1",
                        "OPENAI_BASE_URL")
# 사내 게이트웨이는 키를 요구하지 않으므로 빈 값 허용
# (OpenAI SDK 가 빈 키를 거부해서 _llm.py 에서 'EMPTY' 로 대체한다)
GATEWAY_API_KEY = _get("LLM_GATEWAY_API_KEY", "", "OPENAI_API_KEY")
CHAT_MODEL = _get("LLM_CHAT_MODEL", "glm-5.1", "MODEL_NAME")
TEMPERATURE = float(_get("TEMPERATURE", "0"))
MAX_TOKENS = int(_get("LLM_MAX_TOKENS", "4000"))
LLM_TIMEOUT = int(_get("LLM_TIMEOUT", "120"))
LLM_MAX_RETRIES = int(_get("LLM_MAX_RETRIES", "2"))

# 구 이름 호환 별칭 (기존 코드/노트북이 참조할 수 있어 남겨둔다)
OPENAI_BASE_URL = GATEWAY_BASE_URL
OPENAI_API_KEY = GATEWAY_API_KEY
MODEL_NAME = CHAT_MODEL

# ── 로그: logs/{LOG_DIR basename}/{YYYY-MM}/{YYYY-MM-DD}.jsonl ────────────
LOG_DIR = _get("LOG_DIR", "./devLogs")

# ── 서버 ──────────────────────────────────────────────────────────────────
API_HOST = _get("API_HOST", "0.0.0.0")
API_PORT = int(_get("API_PORT", "8000"))
API_BASE_URL = _get("API_BASE_URL", "http://localhost:8000/llm/api")

# ── HITL 루프 가드 ────────────────────────────────────────────────────────
MAX_COLLECT = int(_get("MAX_COLLECT", "5"))       # 파라미터 재질문 상한
MAX_VALIDATE = int(_get("MAX_VALIDATE", "3"))     # 검증 재시도 상한
MAX_HOPS = int(_get("MAX_HOPS", "3"))             # needs-핸드오프 왕복 상한

# ── 중간 에이전트 토큰 스트리밍(thinking 채널) ────────────────────────────
SHOW_THINKING_TOKENS = _as_bool(_get("SHOW_THINKING_TOKENS", "0"))

# ── 디버그 ────────────────────────────────────────────────────────────────
DEBUG = _as_bool(_get("DEBUG", "false"))
