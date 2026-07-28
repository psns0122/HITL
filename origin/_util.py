"""공용 헬퍼.  [원본·직접]

  agent_node             : create_react_agent 계열 워커 노드 전부가 여기 위임한다
  slice_new_messages     : 이번 호출로 새로 늘어난 메시지만 잘라낸다
  message_to_dict        : 메시지를 로그/직렬화용 dict 로 편다
  last_user_text         : 마지막 사용자 발화 텍스트
  extract_json_object    : 응답 문자열에서 JSON 객체 하나 꺼내기
  message_content_to_text: LLM content 를 문자열로 정규화
  normalize_route_label  : 라우터 응답을 general / supervisor 로 정규화
"""
import json
import re
from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from origin import _state


async def agent_node(state: _state.AgentState, agent, name: str, *,
                     next_value=None) -> _state.AgentState:
    """react agent 를 실행하고, 이번 호출로 새로 늘어난 메시지만 골라 돌려준다.

    Args:
        state      : 현재 그래프 상태
        agent      : create_react_agent 로 만든 실행기
        name       : 에이전트 이름. 새 AIMessage 의
                     additional_kwargs["agent_name"] 에 박힌다.
        next_value : 주면 patch 에 next 로 실어 보낸다 (키워드 전용)

    실패해도 그래프를 죽이지 않는다 — 에러도 사용자에게 보여줄 메시지로 바꿔
    돌려주고, 다음 노드가 평소대로 이어받는다.
    """
    print(f"[NODE] {name} entered", flush=True)

    before = list(state.get("messages", []) or [])

    try:
        out = await agent.ainvoke({"messages": before})

    except Exception as e:
        print(f"[ERROR] {name} 실행 실패: {type(e).__name__}: {e}", flush=True)
        error_msg = AIMessage(
            content=f"{name} 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            additional_kwargs={"agent_name": name},
        )
        patch: _state.AgentState = {
            "messages": [error_msg],
            "step": state.get("step", 0) + 1,
        }
        if next_value is not None:
            patch["next"] = next_value
        return patch

    # react agent 는 받은 messages 뒤에 자기 작업 내역을 이어 붙여 돌려준다.
    # 그래서 늘어난 뒷부분만 잘라내야 같은 메시지를 두 번 싣지 않는다.
    after = list(out.get("messages", []) or [])
    append = after[len(before):] if len(after) >= len(before) else after

    if not append:
        print(f"[WARN] {name} 이(가) 새 메시지를 만들지 않았습니다", flush=True)
        empty_msg = AIMessage(
            content=f"{name}에서 유효한 응답을 생성하지 못했습니다.",
            additional_kwargs={"agent_name": name},
        )
        append = [empty_msg]

    # 누가 만든 메시지인지 표시해 둔다. Supervisor 가 이 값을 읽는다.
    for msg in append:
        if isinstance(msg, AIMessage):
            msg.additional_kwargs["agent_name"] = name

            content = msg.content or ""
            if "STATUS:" in content:
                header_line = content.strip().split("\n")[0]
                print(f"[NODE] {name} {header_line}", flush=True)

    patch: _state.AgentState = {
        "messages": append,
        "step": state.get("step", 0) + 1,
    }
    if next_value is not None:
        patch["next"] = next_value
    return patch


def slice_new_messages(all_messages: Sequence[BaseMessage],
                       start_idx: int) -> List[BaseMessage]:
    """start_idx 이후에 새로 붙은 메시지만 잘라낸다."""
    if start_idx is None or start_idx < 0:
        start_idx = 0
    return list(all_messages[start_idx:])


def message_to_dict(m: BaseMessage) -> Dict[str, Any]:
    """메시지를 로그/직렬화용 dict 로 편다.

    타입별로 있는 필드만 더 담는다.
      ToolMessage : tool_name, tool_call_id
      AIMessage   : usage_metadata, response_metadata, tool_calls
    """
    d: Dict[str, Any] = {
        "type": type(m).__name__,
        "name": getattr(m, "name", None),
        "content": getattr(m, "content", None),
    }

    if isinstance(m, ToolMessage):
        d["tool_name"] = getattr(m, "name", None)
        d["tool_call_id"] = getattr(m, "tool_call_id", None)

    if isinstance(m, AIMessage):
        d["usage_metadata"] = getattr(m, "usage_metadata", None)
        d["response_metadata"] = getattr(m, "response_metadata", None)
        d["tool_calls"] = getattr(m, "tool_calls", None)

    return d


def last_user_text(state: Any) -> str:
    """마지막 사용자 발화 텍스트.

    state(dict) / messages(list) / 객체 어느 형태로 들어와도 받는다.
    HumanMessage 가 하나도 없으면 마지막 메시지 내용으로 대신한다.
    """
    if isinstance(state, dict):
        msgs = state.get("messages", []) or []
    elif isinstance(state, list):
        msgs = state
    else:
        msgs = getattr(state, "messages", []) or []

    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage):
            return message_content_to_text(msg.content)

    if msgs:
        return message_content_to_text(getattr(msgs[-1], "content", ""))

    return ""


def extract_json_object(text: str) -> Optional[dict]:
    """응답 문자열에서 JSON 객체 하나를 꺼낸다. 못 꺼내면 None.

    통째로 파싱해 보고, 실패하면 첫 '{' ~ 마지막 '}' 구간만 다시 시도한다.
    """
    text = (text or "").strip()

    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def message_content_to_text(content: Any) -> str:
    """LLM 응답의 content 를 문자열로 정규화한다.

    content 는 모델/버전에 따라 문자열, 블록 리스트, 그 외로 온다.
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))

        return "\n".join(parts).strip()

    return str(content).strip()


def normalize_route_label(raw: str) -> Literal["general", "supervisor"]:
    """라우터 응답을 general / supervisor 로 정규화한다.

    1) JSON 으로 왔으면 route 키를 본다
    2) 아니면 본문에서 먼저 나오는 라벨 단어를 찾는다
    3) 둘 다 실패하면 general 로 떨어진다
    """
    text = (raw or "").strip()
    parsed = extract_json_object(text)

    if parsed:
        route_value = str(parsed.get("route", "")).strip().upper()

        if route_value in {"SUPERVISOR", "SUPERVISOR_AGENT"}:
            return "supervisor"

        if route_value in {"GENERAL", "GENERAL_AGENT"}:
            return "general"

    upper_text = text.upper()

    first_label = re.search(
        r"\b(SUPERVISOR_AGENT|SUPERVISOR|GENERAL_AGENT|GENERAL)\b", upper_text)

    if first_label:
        label = first_label.group(1)

        if label in {"SUPERVISOR", "SUPERVISOR_AGENT"}:
            return "supervisor"

        if label in {"GENERAL", "GENERAL_AGENT"}:
            return "general"

    return "general"
