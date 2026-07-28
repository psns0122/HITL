"""에이전트 정의.  [원본·직접]

사용자가 직접 제공한 코드.

에이전트가 세 종류로 갈린다 — 이게 이 파일의 구조다.

  1. create_*_agent  : create_react_agent. 툴과 프롬프트만 다르고 나머지는 동일.
                       Status / Location / Log / Extract / Action
  2. build_*_agent   : react agent 가 아니라 직접 만든 async 함수.
                       state 를 읽고 AgentState 조각을 돌려준다.
                       Router / General
  3. (없음)          : Supervisor 는 여기 없다. _node.py 가 체인으로 직접 돌린다.

로그는 공통 헬퍼 없이 각 함수가 직접 print 한다.
"""
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

from origin import _llm, _prompt, _state, _tool, _util


def disable_tool_caching(tools_list):
    """툴 결과 캐싱을 끈다.

    사내 데이터는 조회 시점마다 값이 달라진다. 캐시가 남아 있으면
    직전 조회 결과를 그대로 돌려줘서 오답이 된다.
    """
    for t in tools_list:
        t.cache = False
    return tools_list


# ─────────────────────────────────────────────────────────────────────────
# Router — 일반 질의인지 업무 질의인지
#   react agent 가 아니다. 분류 한 번 하고 끝이라 툴 루프가 필요 없다.
# ─────────────────────────────────────────────────────────────────────────

async def classify_route_with_llm(
    user_query: str,
    model_name: str = None,
    temperature: float = 0.0,
):
    """질의를 general / supervisor 로 분류한다.

    Args:
        user_query  : 사용자 발화
        model_name  : 프론트에서 고른 모델 (None 이면 기본 모델)
        temperature : 분류는 흔들리면 안 되므로 기본 0.
                      GeneralAgent 의 재확인에서는 0.5 로 올려 부른다.
    """
    system_prompt = _prompt.router_agent_prompt().strip()

    user_prompt = (
        "다음 질문을 general 또는 supervisor 중 하나로 분류하시오.\n"
        "반드시 아래 JSON 형식으로만 답하시오. 다른 말은 덧붙이지 마시오.\n"
        '{"route": "general" 또는 "supervisor"}\n\n'
        f"질문: {user_query}"
    ).strip()

    llm = _llm.get_llm(model_name, temperature)
    resp = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    content = _util.message_content_to_text(getattr(resp, "content", ""))
    route = _util.normalize_route_label(content)
    print(f"[AGENT] router(llm) -> {route}", flush=True)

    return route


def build_router_agent():
    """Router 노드가 호출할 함수를 만든다."""

    async def _ainvoke(state: _state.AgentState) -> Dict[str, Any]:
        query = _util.last_user_text(state)
        model_name = state.get("model_name")

        route = await classify_route_with_llm(query, model_name, temperature=0.0)

        # next 에는 노드 이름이 아니라 route 값이 그대로 들어간다.
        # 분기표(_builder)가 "general"/"supervisor" 를 노드로 매핑한다.
        return {
            "route": route,
            "handoff": route == "supervisor",
            "next": route,
        }

    return _ainvoke


router_agent = build_router_agent()


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent — 일반 대화. 업무 질의로 판명되면 supervisor 로 handoff
#   여기도 react agent 가 아니라 prompt | llm 체인이다.
# ─────────────────────────────────────────────────────────────────────────

def build_general_agent():
    """GeneralAgent 노드가 호출할 함수를 만든다."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", _prompt.general_agent_prompt().strip()),
        MessagesPlaceholder("messages"),
    ])

    async def _ainvoke(state: _state.AgentState) -> Dict[str, Any]:
        messages: List[BaseMessage] = state.get("messages", []) or []
        query = _util.last_user_text(messages)
        model_name = state.get("model_name")

        # 제너럴에서도 한 번 더 세이프티 핸드오프 판단.
        # 온도를 올려 라우터의 첫 판단과 다른 시각으로 보게 한다.
        route2 = await classify_route_with_llm(query, model_name, temperature=0.5)

        if route2 == "supervisor":
            print("[AGENT] general -> supervisor handoff (업무 질의로 재판정)", flush=True)
            return {"handoff": True, "route": "supervisor", "messages": []}

        # 일반 대화로 확정 -> 여기서 답변을 만든다
        llm = _llm.get_llm(model_name, temperature=0.0)
        chain = prompt | llm
        resp = await chain.ainvoke({"messages": messages})

        print("[AGENT] general -> FINISH (일반 대화 확정)", flush=True)
        return {
            "handoff": False,
            "route": "general",
            "messages": [AIMessage(content=_util.message_content_to_text(resp.content),
                                   name="GeneralAgent")],
        }

    return _ainvoke


general_agent = build_general_agent()


# ─────────────────────────────────────────────────────────────────────────
# 워커 에이전트 — 전부 create_react_agent. tools 와 prompt 만 다르다.
# ─────────────────────────────────────────────────────────────────────────

def create_extract_agent(model_name: str = None):
    """FAB / 파라미터 추출."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.fab_extract_tool,
            _tool.params_extract_tool,
        ]),
        prompt=_prompt.extract_agent_prompt(),
    )


def create_status_agent(model_name: str = None):
    """큐 / 서버 / 설비 상태, 패치 계획, 담당자 조회."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.queue_status_tool,
            _tool.server_status_search_tool,
            _tool.sys_admin_tool,
            _tool.patch_plan_search_tool,
            _tool.eqp_search_tool,
        ]),
        prompt=_prompt.status_agent_prompt(),
    )


def create_location_agent(model_name: str = None):
    """캐리어 현재 위치 조회."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.location_search_tool,
        ]),
        prompt=_prompt.location_agent_prompt(),
    )


def create_log_agent(model_name: str = None):
    """반송 이력 / 에러 로그 분석."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.log_search_tool,
        ]),
        prompt=_prompt.log_agent_prompt(),
    )


def create_action_agent(model_name: str = None):
    """명령 실행 (반송요청명령 / 목적지요청).

    원본에서는 다른 워커와 완전히 같은 모양이다 — HITL 도, 파라미터 수집도,
    승인 절차도 없다. 툴을 그냥 부른다.
    이 자리를 턴 기반 HITL 노드로 바꾸는 게 이번 작업이다.
    (app/actions/node.py 와 비교)
    """
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.transport_tool,
            _tool.dest_req_tool,
        ]),
        prompt=_prompt.action_agent_prompt(),
    )


# ─────────────────────────────────────────────────────────────────────────
# Supervisor
#   여기에 없다. _node.py 가 ChatPromptTemplate | llm.with_structured_output
#   체인을 직접 만들어 돌린다. 왜 얘만 예외인지는 확인되지 않았다.
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 에이전트  [원본·추정 — 실물 대기]
#   사용자가 토큰 스트리밍으로 보게 되는 건 이 둘의 출력뿐이다.
#   사내 실물은 이보다 두껍다는 확인을 받았다(메시지 정리/재시도/폴백 등).
#   아래는 자리만 잡아둔 최소 형태다. **실물로 덮어쓸 것.**
# ─────────────────────────────────────────────────────────────────────────

def create_final_agent(model_name: str = None):
    """워커 결과를 받아 최종 답변. 툴 없음."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=[],
        prompt=_prompt.final_agent_prompt(),
    )


def create_final_general_agent(model_name: str = None):
    """일반 대화의 최종 답변. 툴 없음."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=[],
        prompt=_prompt.final_general_agent_prompt(),
    )
