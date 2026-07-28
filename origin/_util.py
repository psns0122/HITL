"""공용 헬퍼.  [원본·첨부]

핵심은 agent_node — 모든 create_react_agent 계열 워커 노드가 이 한 줄에
위임하므로, 그 노드들은 "에이전트를 만들고 넘긴다" 외에 할 일이 없다.

여기 없는 함수는 임의로 만들지 않는다. 이 파일에 무엇이 더 있는지는
사내 실물 확인 대기. (safe_tool 은 _util 이 아니라 _tool.py 에 있다)
"""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage

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


# ─────────────────────────────────────────────────────────────────────────
# 메시지 훑기
# ─────────────────────────────────────────────────────────────────────────

def last_user_text(source) -> str:
    """가장 최근 사용자 발화. 없으면 빈 문자열.

    호출부가 messages 리스트를 주기도 하고 state 를 통째로 주기도 한다
    (Router 는 state, GeneralAgent 는 messages 를 넘긴다). 둘 다 받는다.
    """
    messages = source.get("messages", []) if isinstance(source, dict) else source

    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return message_content_to_text(msg.content)
    return ""


# 같은 동작의 다른 이름 — 사내 코드에 둘 다 등장한다
last_human_text = last_user_text


def message_content_to_text(content) -> str:
    """LLM 응답의 content 를 문자열로 정규화한다.  [원본·추정]

    content 는 모델/버전에 따라 세 가지 형태로 온다.
      1) 문자열       -> 그대로
      2) 블록 리스트  -> text 블록만 이어붙임
      3) 그 외        -> str() 로 강제
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


# ─────────────────────────────────────────────────────────────────────────
# LLM 출력 파싱  [원본·추정]
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
