"""전역 설정. 모든 사내 의존 값은 .env 로 주입한다."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# LLM
FAKE_LLM = _as_bool(os.getenv("FAKE_LLM", "1"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# 로그: logs/{LOG_DIR basename}/{YYYY-MM}/{YYYY-MM-DD}.jsonl
LOG_DIR = os.getenv("LOG_DIR", "./devLogs")

# 서버
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/llm/api")

# HITL 루프 가드
MAX_COLLECT = int(os.getenv("MAX_COLLECT", "5"))
MAX_VALIDATE = int(os.getenv("MAX_VALIDATE", "3"))
MAX_HOPS = int(os.getenv("MAX_HOPS", "3"))

# thinking 토큰 채널(미구현 훅)
SHOW_THINKING_TOKENS = _as_bool(os.getenv("SHOW_THINKING_TOKENS", "0"))
