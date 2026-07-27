"""공용 헬퍼.

LLM 응답 파싱, 메시지 훑기, SSE 트레이스 이벤트 발행처럼
여러 모듈이 함께 쓰는 잡동사니를 모아둔다.
"""
import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app._llm import ECHO_MARKER, get_llm


# ─────────────────────────────────────────────────────────────────────────
# SSE 트레이스
# ─────────────────────────────────────────────────────────────────────────

def emit(config, name: str, data: dict):
    """SSE 트레이스용 커스텀 이벤트 발행.

    astream_events 의 on_custom_event 로 잡힌다.
    이벤트 발행이 실패해도 그래프 실행이 깨지면 안 되므로 조용히 무시한다.
    """
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data, config=config)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# 메시지 훑기
# ─────────────────────────────────────────────────────────────────────────

def message_content_to_text(content) -> str:
    """LLM 응답의 content 를 문자열로 정규화한다.

    content 는 모델/버전에 따라 세 가지 형태로 온다.
      1) 문자열                      -> 그대로
      2) 블록 리스트                 -> text 블록만 이어붙임
      3) 그 외(None, dict 등)        -> str() 로 강제
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # 멀티모달/블록 형태: [{"type": "text", "text": "..."}, ...]
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    return str(content)


def last_user_text(messages: list) -> str:
    """가장 최근 사용자 발화. 없으면 빈 문자열."""
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return message_content_to_text(msg.content)
    return ""


# 사내 코드에서 쓰던 이름 (동일 동작)
last_human_text = last_user_text


def member_answered_this_turn(messages: list, members: list) -> AIMessage | None:
    """이번 user turn 안에서 member 에이전트가 남긴 마지막 AIMessage.

    직전 HumanMessage 를 만나면 멈춘다 = 이전 턴 결과는 보지 않는다.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return None
        if isinstance(msg, AIMessage) and getattr(msg, "name", None) in members:
            return msg
    return None


def agent_ran_this_turn(messages: list, agent_name: str) -> bool:
    """이번 턴에 특정 에이전트가 이미 실행됐는지.

    ExtractAgent 를 턴당 한 번만 태우는 판정에 쓴다.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return False
        if isinstance(msg, AIMessage) and getattr(msg, "name", None) == agent_name:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# LLM 출력 파싱
# ─────────────────────────────────────────────────────────────────────────

# ```json ... ``` 같은 코드펜스로 감싸 오는 모델이 많아서 먼저 벗겨낸다
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """응답 문자열에서 JSON 객체 하나를 꺼낸다. 실패하면 빈 dict.

    모델이 JSON 앞뒤에 설명을 붙이거나 코드펜스로 감싸는 경우가 잦아
    세 단계로 시도한다.
      1) 통째로 파싱
      2) 코드펜스 안쪽만 파싱
      3) 첫 '{' ~ 마지막 '}' 구간만 파싱
    """
    if not text:
        return {}

    raw = text.strip()

    # 1) 통째로
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # 2) 코드펜스 안쪽
    fence = _FENCE_RE.search(raw)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    # 3) 중괄호 구간만
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    return {}


def normalize_route_label(content: str) -> str:
    """라우터 응답을 'general' | 'supervisor' 로 정규화한다.

    JSON 으로 왔으면 route 키를 보고, 아니면 본문에서 단어를 찾는다.
    판단이 안 되면 supervisor 로 보낸다 — 조회를 놓치는 것보다
    불필요하게 조회하는 편이 낫기 때문.
    """
    parsed = extract_json_object(content)
    candidate = str(parsed.get("route", "")) if parsed else ""

    # JSON 이 아니면 본문 전체를 후보로 본다
    if not candidate:
        candidate = content or ""

    lowered = candidate.lower()

    if "general" in lowered:
        return "general"
    if "supervisor" in lowered:
        return "supervisor"

    return "supervisor"


# ─────────────────────────────────────────────────────────────────────────
# 목업 모드 보조
# ─────────────────────────────────────────────────────────────────────────

def fake_llm_echo(role: str, payload: str, config=None, model_name: str = None,
                  extra_context: str = "") -> AIMessage:
    """FAKE_LLM 모드에서 규칙 기반 결정을 '모델 호출'처럼 통과시킨다.

    이렇게 해야 on_chat_model_end / usage_metadata 가 에이전트별로 잡혀서
    토큰 원장 집계가 실제와 같은 경로로 검증된다.
    """
    llm = get_llm(model_name)
    prompt = f"[ROLE:{role}]\n{extra_context}\n{ECHO_MARKER}{payload}"
    return llm.invoke([HumanMessage(content=prompt)], config=config)
