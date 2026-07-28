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
    """명령 실행.

    원본은 다른 워커와 구분되는 점이 하나도 없다 — 파라미터가 없어도 되묻지
    않고, 실행 전에 승인을 받지도 않는다. 이 노드가 이번 작업의 교체 대상이다.
    """
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

    이 노드는 판단하지 않는다. _agent.router_agent 가 route / handoff / next
    를 다 채워서 주고, 여기서는 step 만 얹는다.
    next 에는 노드 이름이 아니라 route 값("general"/"supervisor")이 들어간다.
    """
    print("[NODE] Router entered")

    try:
        result = await _agent.router_agent(state)
        return {**result, "step": 1}

    except Exception as e:
        # 판단이 안 되면 supervisor 로 보낸다 — 조회를 놓치는 것보다
        # 불필요하게 조회하는 편이 낫다.
        print(f"[ERROR] Failed to execute Router: {e}")
        return {"route": "supervisor", "handoff": True, "next": "supervisor", "step": 1}


async def general_node(state: _state.AgentState) -> _state.AgentState:
    """일반 대화. 업무 질의로 재판정되면 Supervisor 로 handoff 한다.

    GeneralAgent 는 react agent 가 아니라서 _util.agent_node 를 쓰지 않는다.
    (_agent.build_general_agent 가 만든 함수를 직접 부른다)
    """
    try:
        result = await _agent.general_agent(state)

        next_node = "Supervisor" if result.get("handoff") else "FINISH"
        return {**result, "next": next_node, "step": state.get("step", 0) + 1}

    except Exception as e:
        print(f"[ERROR] Failed to execute GeneralAgent: {e}")
        return {
            "messages": [AIMessage(content="GeneralAgent 초기화/실행 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 노드
#   사용자는 이 노드의 토큰만 스트리밍으로 본다.
# ─────────────────────────────────────────────────────────────────────────

async def final_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_final_agent(model_name=state.get("model_name"))
        result = await _util.agent_node(state, agent, "FinalAnswerAgent")
        result["next"] = "FINISH"
        return result
    except Exception as e:
        print(f"[ERROR] Failed to execute FinalAnswerAgent: {e}")
        return {
            "messages": [AIMessage(content="답변 생성 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }


async def final_general_node(state: _state.AgentState) -> _state.AgentState:
    try:
        agent = _agent.create_final_general_agent(model_name=state.get("model_name"))
        result = await _util.agent_node(state, agent, "FinalGeneralAgent")
        result["next"] = "FINISH"
        return result
    except Exception as e:
        print(f"[ERROR] Failed to execute FinalGeneralAgent: {e}")
        return {
            "messages": [AIMessage(content="답변 생성 중 오류가 발생했습니다.")],
            "next": "FINISH",
            "step": state.get("step", 0) + 1,
        }
