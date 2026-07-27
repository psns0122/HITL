"""공용 헬퍼."""
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app._llm import ECHO_MARKER, get_llm


def emit(config, name: str, data: dict):
    """SSE 트레이스용 커스텀 이벤트 발행 (astream_events 의 on_custom_event 로 수신).

    이벤트 발행 실패가 그래프 실행을 깨면 안 되므로 조용히 무시한다.
    """
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data, config=config)
    except Exception:
        pass


def last_human_text(messages: list) -> str:
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def member_answered_this_turn(messages: list, members: list) -> AIMessage | None:
    """이번 user turn 안에서 member 에이전트가 남긴 마지막 AIMessage."""
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            return None
        if isinstance(m, AIMessage) and getattr(m, "name", None) in members:
            return m
    return None


def fake_llm_echo(role: str, payload: str, config=None, extra_context: str = "") -> AIMessage:
    """FAKE_LLM 모드에서 규칙 기반 결정을 '모델 호출'로 통과시켜
    on_chat_model_end / usage_metadata 가 에이전트별로 잡히게 한다."""
    llm = get_llm()
    prompt = f"[ROLE:{role}]\n{extra_context}\n{ECHO_MARKER}{payload}"
    return llm.invoke([HumanMessage(content=prompt)], config=config)
