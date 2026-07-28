"""그래프 노드들.  [원본·직접]

사용자가 직접 제공한 코드. extract_node 와 supervisor_node 가 원문이고,
나머지 워커 노드는 extract_node 와 완전히 같은 틀이라 그대로 복제했다.

워커 노드의 틀 (예외 없음)
    1. try 안에서 에이전트를 만든다
    2. _util.agent_node 에 위임한다
    3. except 에서 에러 메시지 + next="FINISH" 로 안전하게 빠져나온다

노드가 판단하는 건 아무것도 없다. 판단은 전부 에이전트(LLM)가 한다.
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from origin import _agent, _llm, _prompt, _state, _util


# ─────────────────────────────────────────────────────────────────────────
# Supervisor 로스터
# ─────────────────────────────────────────────────────────────────────────

members = ["StatusAgent", "LocationAgent", "LogAgent", "ActionAgent", "ExtractAgent"]

options_for_next = ["FINISH", "FinalAnswerAgent"] + members

# 모델이 'location_agent' / 'LOCATION-AGENT' 처럼 답해도 구제하기 위한 정규화 맵
options_lower_map = {
    opt.lower().replace("_", "").replace("-", ""): opt
    for opt in options_for_next
}


class RouteResponse(TypedDict):
    next: Annotated[Literal[tuple(options_for_next)], "다음에 실행할 노드"]


# 프롬프트 본문에 중괄호가 들어 있으면 ChatPromptTemplate 이 변수로 오인한다.
# 그래서 이스케이프한 뒤에 템플릿에 넣는다.
safe_supervisor_prompt = (
    _prompt.supervisor_agent_prompt().replace("{", "{{").replace("}", "}}")
)

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", safe_supervisor_prompt + """
Given the conversation above, who should act next? Or should we FINISH?
You must select exactly one option from the list below:

{options}

[Strict Output Rule]
You MUST respond in JSON format with a single key 'next'. Do NOT output anything else.
Do NOT explain. Example: {{"next": "LocationAgent"}}"""),
    MessagesPlaceholder("messages"),
]).partial(
    options=str(options_for_next),
    members=", ".join(members),
)


# ─────────────────────────────────────────────────────────────────────────
# Supervisor 노드
# ─────────────────────────────────────────────────────────────────────────

async def supervisor_node(state: _state.AgentState) -> _state.AgentState:
    """다음에 일할 워커를 LLM 에게 고르게 한다.

    모델이 뭘 뱉든 그래프가 죽으면 안 되므로, 응답을 3단계로 방어한다.
      1. dict 인가
      2. next 값이 문자열인가
      3. 그 문자열이 아는 노드 이름인가 (정규화 후 한 번 더)
    어느 하나라도 어긋나면 FinalAnswerAgent 로 떨어뜨린다.
    """
    print("[NODE] Supervisor entered")

    # 빈 content 메시지는 게이트웨이가 400 을 뱉는 경우가 있어 미리 걸러낸다
    raw_messages = state.get("messages", []) or []
    clean_messages = []
    for msg in raw_messages:
        content = getattr(msg, "content", None)
        if content is None or str(content).strip() == "":
            continue
        clean_messages.append(msg)

    supervisor_chain = supervisor_prompt | _llm.get_llm(
        model_name=state.get("model_name"), temperature=0
    ).with_structured_output(RouteResponse, method="json_mode")

    next_node = "FinalAnswerAgent"

    # [원본 그대로] 성공 경로에서는 기존 messages(리스트), 실패 경로에서는
    # 에러 문자열이 들어간다. 타입이 갈리지만 원문의 모양이라 손대지 않았다.
    final_message = state.get("messages", [])

    try:
        response = await supervisor_chain.ainvoke({"messages": clean_messages})

        if isinstance(response, dict):
            raw_next = response.get("next", "")

            if isinstance(raw_next, str):
                clean_next = raw_next.strip().replace("'", "").replace('"', "")

                if clean_next in options_for_next:
                    next_node = clean_next
                else:
                    # 대소문자/구분자만 다른 경우를 구제한다
                    normalized_next = clean_next.lower().replace("_", "").replace("-", "")
                    if normalized_next in options_lower_map:
                        next_node = options_lower_map[normalized_next]
                    else:
                        print(f"[DEBUG] Supervisor returned an INVALID route word: "
                              f"'{raw_next}' Defaulting to FinalAnswerAgent.")
                        next_node = "FinalAnswerAgent"
                        final_message = "Supervisor 라우팅에 문제 발생"
            else:
                print(f"[DEBUG] Supervisor returned a NON-STRING value for 'next': "
                      f"'{raw_next}' Defaulting to FinalAnswerAgent.")
                next_node = "FinalAnswerAgent"
                final_message = "Supervisor 라우팅에 문제 발생"
        else:
            print(f"[DEBUG] Supervisor response is NOT a dictionary: "
                  f"'{response}' Defaulting to FinalAnswerAgent.")
            next_node = "FinalAnswerAgent"
            final_message = "Supervisor 라우팅에 문제 발생"

    except Exception as e:
        print(f"[ERROR] Supervisor LLM invoke failed: {e}")
        next_node = "FinalAnswerAgent"
        final_message = "Supervisor 라우팅에 문제 발생"

    return {
        "messages": final_message,
        "next": next_node,
        "step": state.get("step", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────
# 워커 노드들
#   전부 같은 틀이다. 에이전트를 만들고 _util.agent_node 에 넘길 뿐.
# ─────────────────────────────────────────────────────────────────────────

async def extract_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_extract_agent(model_name=state.get("model_name"))
        return await _util.agent_node(state, agent, "ExtractAgent")
    except Exception as e:
        print(f"[ERROR] Failed to execute ExtractAgent: {e}")
        return {
            "messages": [AIMessage(content="ExtractAgent 초기화/실행 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


async def status_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_status_agent(model_name=state.get("model_name"))
        return await _util.agent_node(state, agent, "StatusAgent")
    except Exception as e:
        print(f"[ERROR] Failed to execute StatusAgent: {e}")
        return {
            "messages": [AIMessage(content="StatusAgent 초기화/실행 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


async def location_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_location_agent(model_name=state.get("model_name"))
        return await _util.agent_node(state, agent, "LocationAgent")
    except Exception as e:
        print(f"[ERROR] Failed to execute LocationAgent: {e}")
        return {
            "messages": [AIMessage(content="LocationAgent 초기화/실행 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


async def log_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_log_agent(model_name=state.get("model_name"))
        return await _util.agent_node(state, agent, "LogAgent")
    except Exception as e:
        print(f"[ERROR] Failed to execute LogAgent: {e}")
        return {
            "messages": [AIMessage(content="LogAgent 초기화/실행 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


async def action_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_action_agent(model_name=state.get("model_name"))
        return await _util.agent_node(state, agent, "ActionAgent")
    except Exception as e:
        print(f"[ERROR] Failed to execute ActionAgent: {e}")
        return {
            "messages": [AIMessage(content="ActionAgent 초기화/실행 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


# ─────────────────────────────────────────────────────────────────────────
# Router / GeneralAgent
# ─────────────────────────────────────────────────────────────────────────

async def router_node(state: _state.AgentState) -> _state.AgentState:
    """일반 질의면 GeneralAgent, 업무 질의면 Supervisor 로 보낸다.

    _agent.router_agent 가 route 를 판단하고, 노드가 그 값을 노드 이름으로
    바꿔 next 에 싣는다.
    """
    messages = state.get("messages", []) or []
    if messages and isinstance(messages[-1], HumanMessage):
        print(f"[USER] {messages[-1].content}")

    print("[NODE] Router entered")

    result = await _agent.router_agent({
        "messages": state["messages"],
        "model_name": state["model_name"],
    })

    route = result.get("route", "supervisor")
    if route == "supervisor":
        next_node = "Supervisor"
    else:
        next_node = "GeneralAgent"

    return {"route": route, "handoff": False, "next": next_node, "step": 1}


async def general_node(state: _state.AgentState) -> _state.AgentState:
    """일반 대화. 업무 질의로 재판정되면 Supervisor 로 handoff 한다."""
    print("[NODE] General entered")

    input_messages = state.get("messages", []) or []
    q = _util.last_user_text(state)

    # 세이프티 핸드오프 재판정
    route2 = await _agent.classify_route_with_llm(q)

    if route2 == "supervisor":
        return {
            "messages": [],
            "handoff": True,
            "route": "supervisor",
            "next": "Supervisor",
            "step": state.get("step", 0) + 1,
        }

    # 일반 대화 확정 -> react agent(툴 포함)로 답변 생성
    general_agent = _agent.create_general_agent(model_name=state.get("model_name"))

    result = general_agent.invoke({"messages": state["messages"]})

    result_messages = result.get("messages", []) or []

    # 이번 호출로 새로 늘어난 메시지만 잘라낸다
    prev_len = len(input_messages)
    new_messages = result_messages[prev_len:]

    if not new_messages and result_messages:
        new_messages = result_messages

    return {
        "messages": new_messages,
        "handoff": False,
        "route": "general",
        "next": "FINISH",
        "step": state.get("step", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 노드
#   사용자는 이 노드의 토큰만 스트리밍으로 본다.
# ─────────────────────────────────────────────────────────────────────────

async def final_node(state: _state.AgentState) -> _state.AgentState:
    print("[NODE] FinalAnswer entered")

    # create_final_agent 는 _ainvoke(state) 함수를 돌려준다 — 만들어서 바로 부른다
    out = await _agent.create_final_agent(model_name=state.get("model_name"))(state)

    msgs = out.get("messages", []) or []

    if msgs == []:
        print("[ERROR] 비어있는 FINAL 응답")
    else:
        msg = msgs[0]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        finish = msg.response_metadata.get("finish_reason")

        # 빈 본문이거나 정상 종료(stop)가 아니면 폴백 문구로 교체한다
        is_empty = not content.strip()
        is_weird_finish = finish not in ("stop", None)

        if is_empty or is_weird_finish:
            print(f"[ERROR] 비정상 FINAL 응답 종료 "
                  f"content={content[:80]!r} finish_reason={finish}")
            msgs = [AIMessage(content="응답 생성에 실패했습니다. 다시 시도해주세요.")]

    return {
        "messages": msgs,
        "next": "END",
        "step": state.get("step", 0) + 1,
    }


async def final_general_node(state: _state.AgentState) -> _state.AgentState:
    """final_node 와 같은 구조. 에이전트만 FinalGeneral 로 다르다."""
    print("[NODE] FinalGeneral entered")

    out = await _agent.create_final_general_agent(model_name=state.get("model_name"))(state)

    msgs = out.get("messages", []) or []

    if msgs == []:
        print("[ERROR] 비어있는 FINAL_GENERAL 응답")
    else:
        msg = msgs[0]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        finish = msg.response_metadata.get("finish_reason")

        is_empty = not content.strip()
        is_weird_finish = finish not in ("stop", None)

        if is_empty or is_weird_finish:
            print(f"[ERROR] 비정상 FINAL_GENERAL 응답 종료 "
                  f"content={content[:80]!r} finish_reason={finish}")
            msgs = [AIMessage(content="응답 생성에 실패했습니다. 다시 시도해주세요.")]

    return {
        "messages": msgs,
        "next": "END",
        "step": state.get("step", 0) + 1,
    }
