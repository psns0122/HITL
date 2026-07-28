"""에이전트 정의.

에이전트마다 하는 일은 같고 붙는 툴만 다르다.
그래서 create_*_agent 들은 전부 같은 모양이고, tools 인자만 바뀐다.

  Router          : 일반 질의 / 업무 질의 분류
  GeneralAgent    : 일반 대화 (필요하면 supervisor 로 handoff)
  StatusAgent     : 큐/서버/설비/패치 상태
  LocationAgent   : 캐리어 위치
  LogAgent        : 반송 이력·에러 분석
  ExtractAgent    : FAB/파라미터 추출 (모든 워커에 선행)
  ActionAgent     : 명령 실행 (HITL — actions/node.py 가 담당)
  FinalAnswerAgent / FinalGeneralAgent : 최종 응답 생성(스트리밍)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from origin import _llm, _prompt, _state, _tool, _util
from app.actions.registry import ACTION_REGISTRY


# ─────────────────────────────────────────────────────────────────────────
# 툴 캐싱 비활성화
# ─────────────────────────────────────────────────────────────────────────

def disable_tool_caching(tools_list: list) -> list:
    """툴 결과 캐싱을 끈다.

    설비/캐리어 상태는 계속 바뀌므로, 같은 질문이라도 매번 실제 DB 를 봐야 한다.
    캐시가 켜져 있으면 이전 턴의 낡은 값을 그대로 답해버린다.
    """
    disabled_tools = []

    for t in tools_list:
        if hasattr(t, "cache"):
            t.cache = False        # 캐시 비활성화
        disabled_tools.append(t)

    return disabled_tools


# ─────────────────────────────────────────────────────────────────────────
# Router — 일반 질의인지 업무 질의인지
# ─────────────────────────────────────────────────────────────────────────

async def classify_route_with_llm(
    user_query: str,
    model_name: str = None,
    temperature: float = 0.0,
    return_raw: bool = False,
):
    """질의를 general / supervisor 로 분류한다.

    Args:
        user_query  : 사용자 발화
        model_name  : 프론트에서 고른 모델 (None 이면 기본 모델)
        temperature : 분류는 흔들리면 안 되므로 기본 0
        return_raw  : True 면 원문·파싱결과까지 함께 돌려준다 (디버그용)
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

    if return_raw:
        return {"query": user_query, "route": route,
                "raw": content, "parsed": _util.extract_json_object(content)}
    return route


def build_router_agent():
    """Router 노드가 호출할 함수를 만든다."""

    async def _ainvoke(state: _state.AgentState) -> Dict[str, Any]:
        query = _util.last_user_text(state)
        model_name = state.get("model_name")

        route = await classify_route_with_llm(query, model_name, temperature=0.0)

        # next 는 노드 이름이 아니라 route 값 그대로다.
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
        # 온도를 살짝 올려 라우터의 첫 판단과 다른 시각으로 보게 한다.
        route2 = await classify_route_with_llm(query, model_name, temperature=0.5)

        if route2 == "supervisor":
            print("[AGENT] general -> supervisor handoff (업무 질의로 재판정)", flush=True)
            return {"handoff": True, "route": "supervisor", "messages": []}

        # 일반 대화로 확정 -> 여기서 답변을 만든다.
        llm = _llm.get_llm(model_name, temperature=0.0)
        chain = prompt | llm
        resp = chain.invoke({"messages": messages})

        print("[AGENT] general -> FINISH (일반 대화 확정)", flush=True)
        return {
            "handoff": False,
            "route": "general",
            "messages": [resp],
        }

    return _ainvoke


general_agent = build_general_agent()


# ─────────────────────────────────────────────────────────────────────────
# 워커 에이전트들
#   구조는 전부 동일하고 tools 만 다르다.
# ─────────────────────────────────────────────────────────────────────────

def create_general_agent(model_name: str = None):
    """일반 대화 + 사내 문서 RAG."""
    return create_react_agent(
        model=_llm.get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.general_tool,
            _tool.amhs_rag_tool,
        ]),
        prompt=_prompt.general_agent_prompt().strip(),
    )


def create_status_agent(model_name: str = None):
    """큐/서버/설비/패치 상태 조회."""
    return create_react_agent(
        model=_llm.get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.queue_status_tool,
            _tool.server_status_tool,
            _tool.sysadmin_tool,
            _tool.patch_plan_search_tool,
            _tool.eqp_search_tool,
        ]),
        prompt=_prompt.status_agent_prompt().strip(),
    )


def create_location_agent(model_name: str = None):
    """캐리어 위치 조회."""
    return create_react_agent(
        model=_llm.get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.location_search_tool,
        ]),
        prompt=_prompt.location_agent_prompt().strip(),
    )


def create_log_agent(model_name: str = None):
    """반송 이력·에러 로그 분석."""
    return create_react_agent(
        model=_llm.get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.log_search_tool,
        ]),
        prompt=_prompt.log_agent_prompt().strip(),
    )


def create_extract_agent(model_name: str = None):
    """FAB/파라미터 추출.

    이 에이전트는 특이하게도 Supervisor 진입 후 **가장 먼저** 실행되어
    다른 모든 워커에 선행한다. 뒤 단계가 쓸 ID 재료를 만드는 역할이다.
    """
    return create_react_agent(
        model=_llm.get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.fab_extract_tool,
            _tool.params_extract_tool,
        ]),
        prompt=_prompt.extract_agent_prompt().strip(),
    )


# ─────────────────────────────────────────────────────────────────────────
