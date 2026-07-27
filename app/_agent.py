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
import json
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

import app.config as cfg
from app import _prompt, _tool, _util
from app._llm import ECHO_MARKER, get_llm, structured_invoke
from app.actions import resolvers
from app.actions.registry import ACTION_REGISTRY


def _log(msg: str):
    print(f"[AGENT] {msg}", flush=True)


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

# 목업 모드에서 supervisor 로 보낼 키워드
_SUPERVISOR_HINT = (
    "반송", "이송", "옮겨", "이동", "목적지", "위치", "어디", "상태",
    "로그", "이력", "에러", "원인", "캐리어", "장비", "추출", "명령",
)


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

    # 목업 모드: LLM 없이 키워드로 판단하되, 호출 경로는 동일하게 태운다
    if cfg.FAKE_LLM:
        hit = (
            any(k in user_query for k in _SUPERVISOR_HINT)
            or bool(resolvers.extract_ids(user_query)["carrier_ids"])
        )
        route = "supervisor" if hit else "general"

        content = json.dumps({"route": route}, ensure_ascii=False)
        _util.fake_llm_echo("router", content, model_name=model_name)
        _log(f"router(fake) -> {route}")

        if return_raw:
            return {"query": user_query, "route": route,
                    "raw": content, "parsed": {"route": route}}
        return route

    # 실제 게이트웨이 호출
    llm = get_llm(model_name, temperature)
    resp = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    content = _util.message_content_to_text(getattr(resp, "content", ""))
    route = _util.normalize_route_label(content)
    _log(f"router(llm) -> {route}")

    if return_raw:
        return {"query": user_query, "route": route,
                "raw": content, "parsed": _util.extract_json_object(content)}
    return route


def build_router_agent():
    """Router 노드가 호출할 함수를 만든다."""

    async def _ainvoke(state: dict) -> dict:
        messages = state.get("messages", []) or []
        query = _util.last_user_text(messages)
        model_name = state.get("model_name")

        route = await classify_route_with_llm(query, model_name=model_name, temperature=0.0)
        return {"route": route}

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

    async def _ainvoke(state: dict) -> dict:
        messages = state.get("messages", []) or []
        query = _util.last_user_text(messages)
        model_name = state.get("model_name")

        # 라우터가 general 로 보냈어도 한 번 더 확인한다.
        # 온도를 살짝 올려 첫 판단과 다른 시각으로 보게 한다.
        route2 = await classify_route_with_llm(query, model_name=model_name, temperature=0.5)

        if route2 == "supervisor":
            _log("general -> supervisor handoff (업무 질의로 재판정)")
            return {"handoff": True, "route": "supervisor", "messages": []}

        # 일반 대화로 확정. 실제 답변 생성은 FinalGeneralAgent 가 스트리밍으로 한다.
        _log("general -> FINISH (일반 대화 확정)")
        return {"handoff": False, "route": "general", "messages": []}

    return _ainvoke


general_agent = build_general_agent()


# ─────────────────────────────────────────────────────────────────────────
# 워커 에이전트들
#   구조는 전부 동일하고 tools 만 다르다.
# ─────────────────────────────────────────────────────────────────────────

def create_general_agent(model_name: str = None):
    """일반 대화 + 사내 문서 RAG."""
    return create_react_agent(
        model=get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.general_tool,
            _tool.amhs_rag_tool,
        ]),
        prompt=_prompt.general_agent_prompt().strip(),
    )


def create_status_agent(model_name: str = None):
    """큐/서버/설비/패치 상태 조회."""
    return create_react_agent(
        model=get_llm(model_name, temperature=0.2),
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
        model=get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.location_search_tool,
        ]),
        prompt=_prompt.location_agent_prompt().strip(),
    )


def create_log_agent(model_name: str = None):
    """반송 이력·에러 로그 분석."""
    return create_react_agent(
        model=get_llm(model_name, temperature=0.2),
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
        model=get_llm(model_name, temperature=0.2),
        tools=disable_tool_caching([
            _tool.fab_extract_tool,
            _tool.params_extract_tool,
        ]),
        prompt=_prompt.extract_agent_prompt().strip(),
    )


# ─────────────────────────────────────────────────────────────────────────
# Supervisor — 어떤 워커로 보낼지 결정
# ─────────────────────────────────────────────────────────────────────────

class SupervisorOut(BaseModel):
    next: str


def supervisor_agent(text: str, members: list, config=None, model_name: str = None) -> str:
    """워커 선택.

    needs-핸드오프 / ExtractAgent 선행 / 진행 중 액션 같은 결정적 규칙은
    supervisor_node 가 먼저 처리하고, 여기는 그 뒤에만 불린다.
    """
    # 목업 모드: 키워드 기반 선택
    if cfg.FAKE_LLM:
        if resolvers.detect_intent(text) or any(k in text for k in ("명령", "실행", "요청")):
            nxt = "ActionAgent"
        elif "위치" in text or "어디" in text:
            nxt = "LocationAgent"
        elif "상태" in text:
            nxt = "StatusAgent"
        elif any(k in text for k in ("로그", "이력", "에러", "원인")):
            nxt = "LogAgent"
        else:
            nxt = "FinalAnswerAgent"

        _util.fake_llm_echo("supervisor", json.dumps({"next": nxt}, ensure_ascii=False),
                            config=config, model_name=model_name)
        _log(f"supervisor(fake) -> {nxt}")
        return nxt

    # 실제 LLM 호출
    try:
        out = structured_invoke(
            get_llm(model_name, temperature=0.0),
            SupervisorOut,
            [
                SystemMessage(content=_prompt.supervisor_agent_prompt(members).strip()),
                HumanMessage(content=f"질문: {text}"),
            ],
            config=config,
        )
        allowed = members + ["FinalAnswerAgent"]
        nxt = out.next if out.next in allowed else "FinalAnswerAgent"
        _log(f"supervisor(llm) -> {nxt}")
        return nxt

    except Exception as e:
        _log(f"supervisor llm 실패({e}) -> FinalAnswerAgent 폴백")
        return "FinalAnswerAgent"


# ─────────────────────────────────────────────────────────────────────────
# needs-핸드오프 배분 (Supervisor 소관)
# ─────────────────────────────────────────────────────────────────────────

class DispatchOut(BaseModel):
    agent: str = Field("NONE", description="도와줄 워커 이름. 없으면 NONE")
    query: Optional[str] = Field(
        None, description="그 워커에게 보낼 한 문장 질의 (사용자 답변 속 대상 그대로)")


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

    # 목업 모드: 참조 패턴 규칙이 LLM 자리를 대신한다 (데모/테스트용)
    if cfg.FAKE_LLM:
        ref = resolvers.detect_reference(answer)
        agent, query = None, None
        if ref and ref["kind"] == "carrier_location" and "LocationAgent" in members:
            agent = "LocationAgent"
            query = f"{ref.get('carrier_id')} 위치 알려줘"
        elif ref and ref["kind"] == "log_analysis" and "LogAgent" in members:
            agent = "LogAgent"
            query = f"{ref.get('carrier_id') or ''} 반송 로그 분석해줘".strip()

        _util.fake_llm_echo("needs_dispatch",
                            json.dumps({"agent": agent, "query": query},
                                       ensure_ascii=False),
                            config=config, model_name=model_name)
        _log(f"needs_dispatch(fake) -> {agent} query='{query}'")
        return {"agent": agent, "query": query}

    # 실제 LLM 호출
    try:
        out: DispatchOut = structured_invoke(
            get_llm(model_name, temperature=0.0),
            DispatchOut,
            [
                SystemMessage(content=_prompt.needs_dispatch_prompt(members).strip()),
                HumanMessage(content=(
                    f"ActionAgent 가 사용자에게 물은 것: {needs.get('question')}\n"
                    f"사용자의 답변(원문): {answer}\n"
                    f"필요한 값: {needs.get('fill')}\n"
                    f"지금까지 확정된 파라미터: {needs.get('params')}"
                )),
            ],
            config=config,
        )
        agent = out.agent if out.agent in members else None
        query = out.query or (answer if agent else None)
        _log(f"needs_dispatch(llm) -> {agent} query='{query}'")
        return {"agent": agent, "query": query}

    except Exception as e:
        _log(f"needs_dispatch llm 실패({e}) -> NONE (사용자에게 직접 질문)")
        return {"agent": None, "query": None}


# ─────────────────────────────────────────────────────────────────────────
# ActionAgent 의도/파라미터 추출
# ─────────────────────────────────────────────────────────────────────────

class IntentOut(BaseModel):
    action: Literal["transport", "dest_req", "unknown"] = "unknown"
    carrier_id: Optional[str] = Field(None, description="명령 대상 캐리어 ID (8자 영숫자)")
    eqp_id: Optional[str] = Field(None, description="목적지 장비 ID (영문3+숫자3)")
    reference_kind: Optional[Literal["carrier_location", "log_analysis"]] = Field(
        None, description="eqp_id 가 리터럴이 아니라 참조로 표현된 경우 그 종류")
    reference_carrier_id: Optional[str] = Field(
        None, description="carrier_location 참조의 대상 캐리어 ID")
    cancel: bool = False


def extract_intent(text: str, config=None, model_name: str = None) -> resolvers.IntentResult:
    """자연어 -> (액션, 파라미터, 참조, 취소) 구조화 추출."""
    # 목업 모드: 규칙 기반 파서
    if cfg.FAKE_LLM:
        r = resolvers.parse_intent(text)
        _util.fake_llm_echo(
            "action_intent",
            json.dumps({"action": r.action, "params": r.params}, ensure_ascii=False),
            config=config,
            model_name=model_name,
        )
        return r

    # 실제 LLM 호출
    try:
        spec_desc = "\n".join(
            f"- {s.name}({s.label}): 필수 {s.required_params}"
            for s in ACTION_REGISTRY.values()
        )

        out: IntentOut = structured_invoke(
            get_llm(model_name, temperature=0.0),
            IntentOut,
            [
                SystemMessage(content=_prompt.action_agent_prompt().strip()),
                HumanMessage(content=(
                    "사용자 발화에서 액션과 파라미터를 추출하라.\n"
                    f"{spec_desc}\n"
                    "eqp_id 가 '다른 캐리어가 있는 위치' 로 표현되면 "
                    "reference_kind=carrier_location,\n"
                    "'로그를 분석해 원인 장비로' 처럼 표현되면 "
                    "reference_kind=log_analysis 로 표시하라.\n"
                    f"발화: {text}"
                )),
            ],
            config=config,
        )

        r = resolvers.IntentResult(
            action=None if out.action == "unknown" else out.action,
            params={
                k: v for k, v in
                {"carrier_id": out.carrier_id, "eqp_id": out.eqp_id}.items() if v
            },
            reference=(
                {"kind": out.reference_kind, "fill": "eqp_id",
                 "carrier_id": out.reference_carrier_id}
                if out.reference_kind else None
            ),
            cancel=out.cancel,
        )
        _log(f"extract_intent(llm) -> {r}")
        return r

    except Exception as e:
        _log(f"extract_intent llm 실패({e}) -> 규칙 폴백")
        return resolvers.parse_intent(text)


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 에이전트
#   FinalAnswerAgent 와 FinalGeneralAgent 는 프롬프트만 다르고 로직은 같다.
# ─────────────────────────────────────────────────────────────────────────

# 이 문구가 나가면 응답 생성이 두 번 다 실패했다는 뜻
FALLBACK_TEXT = "응답 생성에 실패했습니다. 다시 시도해 주세요."


def _clean_messages(messages: list) -> list:
    """오염된 메시지를 걸러낸다.

    - 내용이 빈 AIMessage : 모델이 헛돌아 만든 껍데기. 컨텍스트만 잡아먹는다.
    - tool_calls 가 달린 AIMessage : 짝이 되는 ToolMessage 없이 남으면
      게이트웨이가 400 을 뱉는다.
    """
    cleaned = []

    for msg in messages or []:
        if isinstance(msg, AIMessage):
            text = _util.message_content_to_text(msg.content)

            if not text.strip():
                continue
            if getattr(msg, "tool_calls", None):
                continue

        cleaned.append(msg)

    return cleaned


def _build_final_chain(system_prompt: str, model_name: str = None):
    """프롬프트 + LLM 체인 하나."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("messages"),
    ])
    return prompt | get_llm(model_name, temperature=0.2)


async def _astream_final(chain, messages: list, config) -> str:
    """체인을 스트리밍으로 돌려 전체 텍스트를 모은다.

    astream 을 써야 on_chat_model_stream 이벤트가 발생하고,
    라우터가 그 토큰을 사용자 화면으로 흘려보낼 수 있다.
    """
    parts = []

    async for chunk in chain.astream({"messages": messages}, config=config):
        text = _util.message_content_to_text(getattr(chunk, "content", ""))
        if text:
            parts.append(text)

    return "".join(parts)


def _make_final_agent(system_prompt: str, role: str, model_name: str = None):
    """최종 응답 에이전트 생성기 (Answer/General 공용)."""

    chain = _build_final_chain(system_prompt, model_name)

    async def _ainvoke(state: dict, config=None, context: str = "") -> str:
        # 1) 오염된 메시지 제거
        messages = _clean_messages(state.get("messages", []) or [])

        # 목업 모드에서는 FakeEcho 가 [ECHO] 뒤 내용을 그대로 뱉으므로,
        # 앞 단계 처리 결과를 그 마커에 실어 붙인다.
        if cfg.FAKE_LLM:
            messages = messages + [HumanMessage(content=f"[ROLE:{role}]\n{ECHO_MARKER}{context}")]

        # 2) 1차 호출
        text = await _astream_final(chain, messages, config)

        # 3) 비었으면 1회 재시도
        if not text.strip():
            _log(f"{role}: 빈 응답 -> 1회 재시도")
            text = await _astream_final(chain, messages, config)

        # 4) 그래도 비었으면 폴백 문구
        if not text.strip():
            _log(f"{role}: 재시도도 실패 -> 폴백 문구")
            return FALLBACK_TEXT

        return text

    return _ainvoke


def create_final_agent(model_name: str = None):
    """워커 결과를 받아 최종 답변을 만드는 에이전트."""
    return _make_final_agent(
        _prompt.final_agent_prompt().strip(),
        role="final",
        model_name=model_name,
    )


def create_final_general_agent(model_name: str = None):
    """일반 대화의 최종 답변을 만드는 에이전트. create_final_agent 와 같은 모양."""
    return _make_final_agent(
        _prompt.final_general_agent_prompt().strip(),
        role="final_general",
        model_name=model_name,
    )
