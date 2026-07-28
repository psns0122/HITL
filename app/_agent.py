"""에이전트 정의.

에이전트마다 하는 일은 같고 붙는 툴만 다르다.
그래서 create_*_agent 들은 전부 같은 모양이고, tools 인자만 바뀐다.

  Router          : 일반 질의 / 업무 질의 분류
  GeneralAgent    : 일반 대화 (필요하면 supervisor 로 handoff)
  StatusAgent     : 큐/서버/설비/패치 상태
  LocationAgent   : 캐리어 위치
  LogAgent        : 반송 이력·에러 분석
  ExtractAgent    : FAB/파라미터 추출 (모든 워커에 선행)
  ActionAgent     : 명령 실행 (HITL — _util.ActionService 가 담당)
  FinalAnswerAgent / FinalGeneralAgent : 최종 응답 생성(스트리밍)

파일 구성: 위쪽은 origin/_agent.py 와 같고, ActionAgent 판단부는 파일 맨 아래
`[app 전용]` 블록에 모여 있다. 이식할 때는 그 블록만 들고 가면 된다.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app import _llm, _prompt, _state, _tool, _util


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


# *************  [app — origin 의 create_action_agent 는 여기 없다]  *************
# origin 은 이 자리에서 transport_tool / dest_req_tool 을 붙인 react agent 를
# 만들었다. HITL 은 react 루프로 표현할 수 없어(승인 전 실행을 막을 수 없다)
# 턴 기반 상태기계 `_util.ActionService` 로 대체했다. 판단부는 이 파일 맨 아래
# [app 전용] 블록에 있다. 사내 반입 시 origin 의 create_action_agent 는 지운다.
# *************


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 에이전트
#   react agent 가 아니다. prompt | llm 체인을 만들고 _ainvoke 를 돌려준다.
#   사용자가 토큰 스트리밍으로 보게 되는 건 이 둘의 출력뿐이다.
# ─────────────────────────────────────────────────────────────────────────

def create_final_agent(model_name: str = None):
    """워커 결과를 받아 최종 답변을 만든다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _prompt.final_agent_prompt().strip()),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | _llm.get_llm(model_name, temperature=0.2)

    # *************  [app — config 인자 추가. origin 은 _ainvoke(state) 다]
    # astream 의 on_chat_model_stream 이벤트를 SSE 로 흘리려면 그래프의 config
    # (콜백 매니저)를 체인까지 내려 줘야 한다. 그 외 로직은 origin 그대로다.
    async def _ainvoke(state: _state.AgentState, config=None,
                       context: str = "") -> Dict[str, Any]:
        # *************
        raw_messages = state.get("messages", []) or []

        # 1. 오염된 메세지 필터링
        messages = []
        for m in raw_messages:
            if isinstance(m, AIMessage):
                c = m.content if isinstance(m.content, str) else str(m.content)
                if not c.strip():
                    continue
                messages.append(AIMessage(content=c))
            else:
                messages.append(m)

        # *************  [app — 마지막에 사람 차례 한 줄. origin 에는 없다]
        # 두 가지를 동시에 해결한다.
        #  1) 게이트웨이가 "assistant 메시지 2개 이상으로 끝나는" 대화를 400 으로
        #     거부한다 (ollama: "Cannot have 2 or more assistant messages at the
        #     end of the list"). 워커가 연달아 답한 턴(Extract -> Location)이
        #     정확히 그 모양이라, 사람 차례로 닫아 줘야 한다.
        #  2) 이번 턴에 워커가 낸 결과(context)를 명시적으로 다시 실어 준다.
        #     final 프롬프트는 "실행했다면 Job ID, 거절되었다면 사유를 반드시
        #     포함" 을 요구하는데, 그 근거가 긴 대화 속에 묻히면 모델이 두루뭉술
        #     하게 바꿔 말한다(실측: 거절 사유가 사라짐).
        tail = "위 처리 결과를 바탕으로 사용자 질문에 답하세요."
        if context:
            tail = (f"[처리 결과]\n{context}\n\n{tail}\n"
                    "처리 결과에 적힌 Job ID·상태·사유는 요약하거나 바꾸지 말고 "
                    "그 값을 그대로 답변에 포함하세요.")
        messages = messages + [HumanMessage(content=tail)]
        # *************

        # 2. LLM 호출 + 빈 응답이면 1회 재시도
        for attempt in range(2):
            # *************  [app — ainvoke 대신 astream. 토큰 스트리밍용]
            resp = await _astream_final(chain, messages, config)
            # *************
            c = resp.content if isinstance(resp.content, str) else str(resp.content)
            if c.strip():
                return {"messages": [resp]}
            print(f"[ERROR] FINAL 응답 재시도 중, retry {attempt + 1}")

        # 여전히 비어있는 응답이라면 폴백
        return {"messages": [AIMessage(content="응답 생성에 실패했습니다. 다시 시도해주세요.")]}

    return _ainvoke


def create_final_general_agent(model_name: str = None):
    """일반 대화의 최종 답변. create_final_agent 와 프롬프트만 다르다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _prompt.final_general_agent_prompt().strip()),
        MessagesPlaceholder("messages"),
    ])
    chain = prompt | _llm.get_llm(model_name, temperature=0.2)

    async def _ainvoke(state: _state.AgentState, config=None,
                       context: str = "") -> Dict[str, Any]:
        raw_messages = state.get("messages", []) or []

        # 1. 오염된 메세지 필터링
        messages = []
        for m in raw_messages:
            if isinstance(m, AIMessage):
                c = m.content if isinstance(m.content, str) else str(m.content)
                if not c.strip():
                    continue
                messages.append(AIMessage(content=c))
            else:
                messages.append(m)

        # *************  [app — 마지막에 사람 차례 한 줄. origin 에는 없다]
        # 두 가지를 동시에 해결한다.
        #  1) 게이트웨이가 "assistant 메시지 2개 이상으로 끝나는" 대화를 400 으로
        #     거부한다 (ollama: "Cannot have 2 or more assistant messages at the
        #     end of the list"). 워커가 연달아 답한 턴(Extract -> Location)이
        #     정확히 그 모양이라, 사람 차례로 닫아 줘야 한다.
        #  2) 이번 턴에 워커가 낸 결과(context)를 명시적으로 다시 실어 준다.
        #     final 프롬프트는 "실행했다면 Job ID, 거절되었다면 사유를 반드시
        #     포함" 을 요구하는데, 그 근거가 긴 대화 속에 묻히면 모델이 두루뭉술
        #     하게 바꿔 말한다(실측: 거절 사유가 사라짐).
        tail = "위 처리 결과를 바탕으로 사용자 질문에 답하세요."
        if context:
            tail = (f"[처리 결과]\n{context}\n\n{tail}\n"
                    "처리 결과에 적힌 Job ID·상태·사유는 요약하거나 바꾸지 말고 "
                    "그 값을 그대로 답변에 포함하세요.")
        messages = messages + [HumanMessage(content=tail)]
        # *************

        # 2. LLM 호출 + 빈 응답이면 1회 재시도
        for attempt in range(2):
            resp = await _astream_final(chain, messages, config)
            c = resp.content if isinstance(resp.content, str) else str(resp.content)
            if c.strip():
                return {"messages": [resp]}
            print(f"[ERROR] FINAL 응답 재시도 중, retry {attempt + 1}")

        # 여전히 비어있는 응답이라면 폴백
        return {"messages": [AIMessage(content="응답 생성에 실패했습니다. 다시 시도해주세요.")]}

    return _ainvoke


# *************  [app 전용 — origin 에 없음]  *************
#
# 아래는 전부 HITL(ActionAgent) 때문에 더해진 것이다. 사내 반입 시 이 블록만
# 통째로 가져가면 된다.
#
# 구조화 출력은 origin 관례를 따른다 — 스키마를 **TypedDict** 로 쓰고
# `llm.with_structured_output(스키마)` 로 호출한다 (origin 선례: _node.RouteResponse).
#
# 딱 한 가지만 origin 과 다르다: method="json_mode" 를 쓰지 않는다.
#   json_mode 는 "유효한 JSON 인지" 만 보장하고 **스키마를 모델에 보내지 않는다.**
#   origin 의 유일한 구조화 출력인 RouteResponse 는 필드가 next 하나뿐이라
#   프롬프트 본문에 규칙을 적는 것으로 충분했다. 반면 아래 IntentOut 은 필드가
#   6개이고 "옮길 대상 캐리어 vs 위치 기준 캐리어" 처럼 헷갈리는 짝이 있다.
#   json_mode 로 두면 그 구분이 모델에 전달되지 않아 두 ID 가 서로 바뀐다
#   (실측: carrier_id 와 reference_carrier_id 가 뒤집힘). 그래서 스키마를 실어
#   보내는 기본 경로를 쓴다. Supervisor 쪽(_node.RouteResponse)은 origin 그대로
#   json_mode 를 유지한다.
#
# 판단이 실패하면 값을 지어내지 않고 안전한 쪽(재질문/미승인)으로 떨어진다.

from typing import Annotated, TypedDict   # noqa: E402  (app 전용 import)


def _judge(schema, system: str, user: str, config=None, model_name: str = None):
    """구조화 출력 한 번. 실패하면 예외를 그대로 올려 호출부가 폴백한다."""
    runner = _llm.get_llm(model_name, temperature=0.0).with_structured_output(schema)
    return runner.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)], config=config)


# ── needs-핸드오프 배분 (Supervisor 소관) ─────────────────────────────────

class DispatchOut(TypedDict):
    agent: Annotated[str, "도와줄 워커 이름. 없으면 NONE"]
    query: Annotated[str, "그 워커에게 보낼 한 문장 질의"]


def needs_dispatch(needs: dict, members: list, config=None,
                   model_name: str = None) -> dict:
    """ActionAgent 의 상담 요청을 받아 도와줄 워커를 고른다.

    입력은 ActionAgent 가 넘긴 원문 세 가지뿐이다.
      question : ActionAgent 가 사용자에게 물은 것
      answer   : 사용자가 실제로 답한 것 (원문)
      fill     : 필요한 값의 이름

    로스터(members + 프롬프트의 워커 설명)를 보고 LLM 이
      - 이 답변을 값으로 바꿔줄 수 있는 워커 하나와
      - 그 워커에게 보낼 질의문
    을 고른다. 확신이 없으면 NONE — 그러면 사용자에게 직접 다시 묻는다.

    워커를 새로 붙일 때 할 일은 로스터 프롬프트에 설명 한 줄을 더하는 것뿐이다.
    ActionAgent 도, 워커 본문도 건드리지 않는다.

    반환: {"agent": 워커명 or None, "query": 질의문 or None}
    """
    answer = str(needs.get("answer") or "")

    try:
        out = _judge(
            DispatchOut,
            _prompt.needs_dispatch_prompt(members).strip(),
            (f"ActionAgent 가 사용자에게 물은 것: {needs.get('question')}\n"
             f"사용자의 답변(원문): {answer}\n"
             f"필요한 값: {needs.get('fill')}\n"
             f"지금까지 확정된 파라미터: {needs.get('params')}"),
            config=config, model_name=model_name,
        )
        agent = out.get("agent") if out.get("agent") in members else None
        query = out.get("query") or (answer if agent else None)
        print(f"[AGENT] needs_dispatch(llm) -> {agent} query='{query}'", flush=True)
        return {"agent": agent, "query": query}

    except Exception as e:
        print(f"[AGENT] needs_dispatch llm 실패({e}) -> NONE (사용자에게 직접 질문)", flush=True)
        return {"agent": None, "query": None}


# ── ActionAgent 의도/파라미터 추출 ────────────────────────────────────────

class IntentOut(TypedDict):
    action: Annotated[Literal["transport", "dest_req", "unknown"], "실행할 명령"]
    carrier_id: Annotated[str, "옮길 대상 캐리어 ID (8자 영숫자). "
                               "'X 를 ~' 의 X. 없으면 빈 문자열"]
    eqp_id: Annotated[str, "목적지 장비 ID (영문3자+숫자3자, 예 STK102). "
                           "캐리어 ID 를 넣지 말 것. 없으면 빈 문자열"]
    reference_kind: Annotated[Literal["", "carrier_location", "log_analysis"],
                              "목적지를 리터럴이 아니라 참조로 말한 경우만 채운다. "
                              "'다른 캐리어가 있는 위치로'=carrier_location, "
                              "'로그 분석해 원인 장비 피해서'=log_analysis"]
    reference_carrier_id: Annotated[str, "carrier_location 일 때 위치의 기준이 되는 "
                                         "캐리어 ID (옮길 대상이 아닌 쪽). 없으면 빈 문자열"]
    cancel: Annotated[bool, "취소 의사면 true"]


def extract_intent(text: str, config=None, model_name: str = None) -> dict:
    """자연어 -> (액션, 파라미터, 참조, 취소) 구조화 추출.

    반환: {"action": str|None, "params": dict, "reference": dict|None, "cancel": bool}

    LLM 이 실패하면 빈 결과를 돌려준다 — 그러면 ActionAgent 가 파라미터를
    사용자에게 물어보는 정상 경로로 흘러간다(추측하지 않는다).
    """
    empty = {"action": None, "params": {}, "reference": None, "cancel": False}

    try:
        spec_desc = "\n".join(
            f"- {name}({meta['label']}): 필수 {meta['required_params']}"
            for name, meta in _prompt.action_catalog().items()
        )

        out = _judge(
            IntentOut,
            _prompt.action_agent_prompt().strip(),
            ("사용자 발화에서 액션과 파라미터를 추출하라.\n"
             f"{spec_desc}\n"
             "eqp_id 가 '다른 캐리어가 있는 위치' 로 표현되면 "
             "reference_kind=carrier_location 이고, 그 기준 캐리어를 "
             "reference_carrier_id 에 넣는다 (carrier_id 에 넣지 않는다).\n"
             "'로그를 분석해 원인 장비로' 처럼 표현되면 "
             "reference_kind=log_analysis 로 표시하라.\n"
             "eqp_id 에는 장비 ID(영문3자+숫자3자)만 넣는다. "
             "carrier_location / log_analysis 같은 참조 표시를 eqp_id 에 쓰지 마라 "
             "— 그건 reference_kind 필드다.\n"
             "목적지를 아예 말하지 않았으면 reference_kind 는 빈 문자열이다. "
             "'다른 캐리어 있는 위치로' 나 '로그 분석해서' 같은 말이 실제로 "
             "발화에 있을 때만 채운다.\n"
             '예: "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘"\n'
             "   -> carrier_id=6PDMQ283, reference_carrier_id=9ZXCV456, "
             "reference_kind=carrier_location, eqp_id=빈값\n"
             '예: "로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘"\n'
             "   -> carrier_id=6PDMQ283, reference_kind=log_analysis, eqp_id=빈값\n"
             f"발화: {text}"),
            config=config, model_name=model_name,
        )

        action = str(out.get("action") or "").strip()
        ref_kind = str(out.get("reference_kind") or "").strip()

        r = {
            "action": None if action in ("", "unknown") else action,
            "params": {
                k: v for k, v in
                {"carrier_id": str(out.get("carrier_id") or "").strip(),
                 "eqp_id": str(out.get("eqp_id") or "").strip()}.items() if v
            },
            "reference": (
                {"kind": ref_kind, "fill": "eqp_id",
                 "carrier_id": str(out.get("reference_carrier_id") or "").strip()}
                if ref_kind in ("carrier_location", "log_analysis") else None
            ),
            "cancel": bool(out.get("cancel")),
        }
        print(f"[AGENT] extract_intent(llm) -> {r}", flush=True)
        return r

    except Exception as e:
        print(f"[AGENT] extract_intent llm 실패({e}) -> 빈 결과 (사용자에게 물어본다)", flush=True)
        return empty


# ── ActionAgent 답변 분류 / 승인 판정 ─────────────────────────────────────

class CollectAnswerOut(TypedDict):
    # 필드를 하나로 합쳐 둔다. kind + action_choice 두 칸으로 두면 작은 모델이
    # action_choice 만 주고 kind 를 통째로 빠뜨린다(실측). 그러면 분류 불명으로
    # 떨어져 같은 질문을 무한 반복한다. 한 칸이면 빠뜨릴 칸이 없다.
    kind: Annotated[Literal["cancel", "consult", "switch", "value", "empty",
                            "action_transport", "action_dest_req"],
                    "답변의 종류"]


def classify_collect_answer(fieldname: str, answer, current_action: str | None,
                            question: str = None, config=None,
                            model_name: str = None) -> dict:
    """파라미터 질문에 대한 사용자 답변을 분류한다.

    반환: {"kind": cancel|consult|switch|action|value|empty, "text"/"value"/"note"...}

    value 로 분류돼도 실제 ID 인식·존재 확인은 판독기 툴(params_extract_tool)이 한다 —
    LLM 은 종류만 판단하고 값은 만들어내지 않는다.
    """
    text = str(answer or "")

    # /chat/stop 등이 보내는 기계 센티널 — 모델에 물을 것도 없다
    if isinstance(answer, dict) and answer.get("aborted"):
        return {"kind": "cancel"}

    # action_transport / action_dest_req 는 '어떤 명령인지' 를 묻는 중일 때만
    # 유효하다. 다른 파라미터를 묻는 중인데도 답변에 명령형 어미가 붙으면
    # ("9ZXCV456 있는 위치로 채워줘") 모델이 그쪽으로 샌다(실측).
    # 이 문장은 user 메시지 **맨 앞**에 둔다 — 뒤에 붙이면 잘 안 먹는다.
    guard = ("" if fieldname == "action" else
             "[중요] 지금은 명령 종류를 묻는 중이 아닙니다. "
             "action_transport 와 action_dest_req 는 후보에서 제외하고 "
             "나머지 중에서만 고르세요.\n")

    try:
        out = _judge(
            CollectAnswerOut,
            _prompt.action_collect_answer_prompt().strip(),
            (guard
             + f"진행 중인 명령: {current_action or '미확정'}\n"
             f"물어본 것: {question or fieldname}\n"
             f"묻는 파라미터: {fieldname}\n"
             f"사용자의 답변(원문): {text}"),
            config=config, model_name=model_name,
        )
        kind = str(out.get("kind") or "").strip()
        print(f"[AGENT] classify_collect_answer(llm) -> {kind}", flush=True)

        if kind == "cancel":
            return {"kind": "cancel"}
        if kind == "consult":
            return {"kind": "consult", "text": text}
        if kind == "switch":
            return {"kind": "switch", "text": text}
        if kind in ("action_transport", "action_dest_req"):
            if fieldname == "action":
                return {"kind": "action", "value": kind[len("action_"):]}
            # action 을 묻던 게 아닌데 action 이라 답함 -> 재질문으로 강등
            return {"kind": "empty", "note": "답변을 이해하지 못했습니다."}
        if kind == "value":
            return {"kind": "value", "text": text}
        return {"kind": "empty", "note": f"답변에서 {fieldname} 값을 찾지 못했습니다."}

    except Exception as e:
        # 추측하지 않는다 — 다시 묻는 게 가장 안전하다
        print(f"[AGENT] classify_collect_answer llm 실패({e}) -> 재질문", flush=True)
        return {"kind": "empty", "note": "답변을 이해하지 못했습니다. 다시 알려주세요."}


class ConfirmOut(TypedDict):
    verdict: Annotated[Literal["approve", "reject", "unclear"], "승인 판정"]


def classify_confirm(answer, action: str = None, params: dict = None,
                     config=None, model_name: str = None) -> str:
    """승인 질문에 대한 답변 판정 -> approve | reject | unclear.

    approve 는 명시적 동의일 때만. 정정 시도("STK103 으로 바꿔줘")는
    unclear 로 돌려서 호출부가 수집 루프로 되돌릴 수 있게 한다.
    """
    # /chat/stop 등이 보내는 기계 센티널 — 모델에 물을 것도 없다
    if isinstance(answer, dict):
        if answer.get("aborted"):
            return "reject"
        if "approved" in answer:
            return "approve" if answer["approved"] else "reject"

    try:
        out = _judge(
            ConfirmOut,
            _prompt.action_confirm_prompt().strip(),
            (f"실행하려는 명령: {action or '?'} (파라미터: {params})\n"
             f"사용자의 답변(원문): {answer}"),
            config=config, model_name=model_name,
        )
        verdict = str(out.get("verdict") or "").strip()
        if verdict not in ("approve", "reject", "unclear"):
            print(f"[AGENT] classify_confirm(llm) 알 수 없는 값 {verdict!r} -> unclear",
                  flush=True)
            return "unclear"
        print(f"[AGENT] classify_confirm(llm) -> {verdict}", flush=True)
        return verdict

    except Exception as e:
        # 실행은 위험하다 — 판정 못 하면 절대 승인하지 않는다
        print(f"[AGENT] classify_confirm llm 실패({e}) -> unclear (미승인)", flush=True)
        return "unclear"


# ── action_node 가 가지는 판단 에이전트 ───────────────────────────────────

class _ActionAgent:
    """action_node 가 가지는 판단 에이전트.

    노드는 흐름(수집 루프/턴 종료)만 잡고, 아래 판단은 전부 여기로 위임한다.
      - extract_intent            : 최초 발화 -> 의도/파라미터/참조
      - classify_collect_answer   : 파라미터 질문의 답 -> 종류 분류
      - classify_confirm          : 승인 질문의 답 -> approve/reject/unclear
    """
    extract_intent = staticmethod(extract_intent)
    classify_collect_answer = staticmethod(classify_collect_answer)
    classify_confirm = staticmethod(classify_confirm)


action_agent = _ActionAgent()


# ── 최종 응답 스트리밍 헬퍼 (create_final_* 가 쓴다) ──────────────────────

async def _astream_final(chain, messages: list, config):
    """체인을 스트리밍으로 돌려 AIMessage 하나로 합친다.

    origin 은 `chain.ainvoke(...)` 를 쓴다. 여기서 astream 을 쓰는 이유는
    on_chat_model_stream 이벤트가 발생해야 라우터가 그 토큰을 SSE 로
    사용자 화면에 흘릴 수 있기 때문이다. 반환 모양(AIMessage)은 origin 과 같다.
    """
    parts = []

    async for chunk in chain.astream({"messages": messages}, config=config):
        text = _util.message_content_to_text(getattr(chunk, "content", ""))
        if text:
            parts.append(text)

    return AIMessage(content="".join(parts))

# *************  [app 전용 끝]  *************
