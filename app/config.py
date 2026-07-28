"""환경변수 기반 설정.

사내 게이트웨이 키 등 민감정보는 코드가 아니라 .env 에만 둔다.

네이밍은 사내 기존 프로젝트(pptx-vision-rag)의 컨벤션을 그대로 따른다
(`LLM_GATEWAY_BASE_URL` / `LLM_GATEWAY_API_KEY` / `LLM_CHAT_MODEL`).
→ 기존 .env 값을 그대로 복사해 쓸 수 있다.
구버전 이름(OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME)도 폴백으로 받는다.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env():
    """.env 를 읽는다. 무슨 일이 있어도 앱을 죽이지 않는다.

    Windows 메모장으로 .env 를 저장하면 파일 앞에 UTF-8 BOM(EF BB BF)이 붙는다.
    dotenv 를 기본 utf-8 로 읽으면 첫 줄을 못 파싱해
    "Python-dotenv could not parse statement starting at line 1" 에러가 난다.
    그래서 utf-8-sig 로 읽어 BOM 을 벗겨낸다.

    dotenv 모듈이 없거나(.env 없이 OS 환경변수만 쓰는 경우) .env 문법이 깨져도
    조용히 넘어간다 — 그때는 아래 기본값으로 동작한다.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except Exception as e:
        print(f"[config] python-dotenv 없음 -> OS 환경변수만 사용 ({e})", flush=True)
        return

    try:
        # utf-8-sig: BOM 이 있으면 벗기고, 없으면 그냥 utf-8 로 읽는다.
        # override=False: 이미 설정된 OS 환경변수는 덮어쓰지 않는다(사내 컨벤션).
        load_dotenv(env_path, override=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[config] .env 파싱 실패 -> 기본값 사용 ({e})", flush=True)


_load_env()


def _get(name: str, default: str = "", *aliases: str) -> str:
    """환경변수 조회. name 이 비어 있으면 aliases 를 순서대로 본다.

    값 앞뒤 공백은 벗긴다. (Windows 에서 .env 값 뒤에 공백이 붙는 경우가 있어
    int()/float() 변환이 깨지는 걸 막는다.)
    """
    val = os.getenv(name)
    if val is None or val.strip() == "":
        for alt in aliases:
            val = os.getenv(alt)
            if val is not None and val.strip() != "":
                return val.strip()
        return default
    return val.strip()


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_int(name: str, default: int) -> int:
    """정수 환경변수. 값이 이상하면 기본값으로 떨어진다(앱을 안 죽인다)."""
    raw = _get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        print(f"[config] {name}={raw!r} 정수 변환 실패 -> 기본값 {default} 사용", flush=True)
        return default


def _as_float(name: str, default: float) -> float:
    """실수 환경변수. 값이 이상하면 기본값으로 떨어진다."""
    raw = _get(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        print(f"[config] {name}={raw!r} 실수 변환 실패 -> 기본값 {default} 사용", flush=True)
        return default


# ── LLM 게이트웨이 (OpenAI 호환) ──────────────────────────────────────────
GATEWAY_BASE_URL = _get("LLM_GATEWAY_BASE_URL", "http://hcp.llm.skhynix.com/v1",
                        "OPENAI_BASE_URL")
# 사내 게이트웨이는 키를 요구하지 않으므로 빈 값 허용
# (OpenAI SDK 가 빈 키를 거부해서 _llm.py 에서 'EMPTY' 로 대체한다)
GATEWAY_API_KEY = _get("LLM_GATEWAY_API_KEY", "", "OPENAI_API_KEY")
CHAT_MODEL = _get("LLM_CHAT_MODEL", "GaiA-LLM-Latest", "MODEL_NAME")
TEMPERATURE = _as_float("TEMPERATURE", 0.0)
MAX_TOKENS = _as_int("LLM_MAX_TOKENS", 4000)
LLM_TIMEOUT = _as_int("LLM_TIMEOUT", 120)
LLM_MAX_RETRIES = _as_int("LLM_MAX_RETRIES", 2)

# 구 이름 호환 별칭 (기존 코드/노트북이 참조할 수 있어 남겨둔다)
OPENAI_BASE_URL = GATEWAY_BASE_URL
OPENAI_API_KEY = GATEWAY_API_KEY
MODEL_NAME = CHAT_MODEL

# ── 로그: logs/{LOG_DIR basename}/{YYYY-MM}/{YYYY-MM-DD}.jsonl ────────────
LOG_DIR = _get("LOG_DIR", "./devLogs")

# ── 서버 ──────────────────────────────────────────────────────────────────
API_HOST = _get("API_HOST", "0.0.0.0")
API_PORT = _as_int("API_PORT", 8000)
API_BASE_URL = _get("API_BASE_URL", "http://localhost:8000/llm/api")

# ── HITL 루프 가드 ────────────────────────────────────────────────────────
MAX_COLLECT = _as_int("MAX_COLLECT", 5)       # 파라미터 재질문 상한
MAX_VALIDATE = _as_int("MAX_VALIDATE", 3)     # 검증 재시도 상한
MAX_HOPS = _as_int("MAX_HOPS", 3)             # needs-핸드오프 왕복 상한

# ── 중간 에이전트 토큰 스트리밍(thinking 채널) ────────────────────────────
SHOW_THINKING_TOKENS = _as_bool(_get("SHOW_THINKING_TOKENS", "0"))

# ── 디버그 ────────────────────────────────────────────────────────────────
DEBUG = _as_bool(_get("DEBUG", "false"))
