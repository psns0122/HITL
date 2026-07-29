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

파일 구성: origin/_agent.py 와 거의 같다. app 추가분은 ************* 로 표시하고
파일 맨 아래 `[app 전용]` 블록에 모아 두었다 (지금은 스트리밍 헬퍼 하나뿐).

ActionAgent 판단부는 `_util.py`, needs-핸드오프 배분은 `_node.py` 에 있다 —
origin 관례대로 "스키마는 그것을 쓰는 파일에" 두었다.
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


def create_action_agent(model_name: str = None):
    """명령 실행 판단 (반송요청명령 / 목적지요청 / ...).

    origin 과 같은 자리, 같은 모양의 react agent 다. 바인딩 툴만 다르다:
    origin 은 transport_tool/dest_req_tool (즉시 실행)을 붙였지만, HITL 에서는
    실행 전에 수집·검증·승인을 거쳐야 하므로 그 앞 단계 툴들을 붙인다.

      params_extract_tool  : 발화에서 캐리어/장비 ID 판독 (DB 조회)
      param_check_tool     : 명령별 필수 파라미터 충족 확인
      *_validate_tool      : 파라미터가 다 모이면 유효성 검증

    ★ {action}_confirm_tool / {action}_execute_tool 은 바인딩하지 않는다.
      승인 질문은 턴을 닫을 때 ActionService 가 만들고(confirm), 실행은
      사용자의 명시적 승인 이후 ActionService 만 호출한다(execute).
      에이전트에 붙이면 LLM 이 승인 절차를 건너뛸 길이 생긴다.

    에이전트는 판단(어떤 명령인지, 값이 뭔지)과 툴 호출을 하고, 턴을 닫는
    흐름(질문하고 기다리기 / 승인 후 실행)은 _util.ActionService 가 잡는다.
    """
    return create_react_agent(
        model=_llm.get_llm(model_name, temperature=0.0),
        tools=disable_tool_caching([
            _tool.params_extract_tool,
            _tool.param_check_tool,
            _tool.transport_validate_tool,
            _tool.dest_req_validate_tool,
        ]),
        prompt=_prompt.action_agent_prompt().strip(),
    )


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
# 이 파일에 남은 app 추가분은 최종 응답 스트리밍 헬퍼 하나뿐이다.
#
# ActionAgent 판단부(어떤 명령인지 / 파라미터가 무엇인지 / 답변이 무슨 뜻인지 /
# 승인인지)는 `_util.py` 로 옮겼다. 그 판단을 쓰는 것이 `_util.ActionService`
# 이고, origin 관례가 "스키마는 그것을 쓰는 파일에 둔다" 이기 때문이다
# (origin 선례: supervisor_node 가 쓰는 RouteResponse 가 _node.py 에 있다).
# 덕분에 _util -> _agent 순환 참조를 피하려고 두었던 지연 로더도 사라졌다.
#
# needs-핸드오프 배분(needs_dispatch)은 Supervisor 소관이라 `_node.py` 로
# 옮겼다. 같은 이유다.
#
# 결과적으로 이 파일은 origin/_agent.py 와 거의 같다. 이식할 때 비교하기 쉽다.

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
