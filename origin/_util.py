"""공용 헬퍼.  [원본·직접]

  agent_node          : create_react_agent 계열 워커 노드 전부가 여기 위임한다
  slice_new_messages  : 이번 호출로 새로 늘어난 메시지만 잘라낸다
  message_to_dict     : 메시지를 로그/직렬화용 dict 로 편다
"""
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

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
