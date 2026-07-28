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
    try:
        out = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            SupervisorOut,
            [
                SystemMessage(content=_prompt.supervisor_agent_prompt(members).strip()),
                HumanMessage(content=f"질문: {text}"),
            ],
            config=config,
        )
        allowed = members + ["FinalAnswerAgent"]
        nxt = out.next if out.next in allowed else "FinalAnswerAgent"
        print(f"[AGENT] supervisor(llm) -> {nxt}", flush=True)
        return nxt

    except Exception as e:
        print(f"[AGENT] supervisor llm 실패({e}) -> FinalAnswerAgent 폴백", flush=True)
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

    try:
        out: DispatchOut = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
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
        print(f"[AGENT] needs_dispatch(llm) -> {agent} query='{query}'", flush=True)
        return {"agent": agent, "query": query}

    except Exception as e:
        print(f"[AGENT] needs_dispatch llm 실패({e}) -> NONE (사용자에게 직접 질문)", flush=True)
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


@dataclass
class IntentResult:
    """의도 추출 결과 (ActionAgent 가 스크래치를 만들 때 쓴다)."""
    action: str | None = None            # transport | dest_req | None
    params: dict = field(default_factory=dict)
    reference: dict | None = None        # {kind, carrier_id, fill}
    cancel: bool = False


def extract_intent(text: str, config=None, model_name: str = None) -> IntentResult:
    """자연어 -> (액션, 파라미터, 참조, 취소) 구조화 추출.

    LLM 이 실패하면 빈 결과를 돌려준다 — 그러면 ActionAgent 가 파라미터를
    사용자에게 물어보는 정상 경로로 흘러간다(추측하지 않는다).
    """
    try:
        spec_desc = "\n".join(
            f"- {name}({meta['label']}): 필수 {meta['required_params']}"
            for name, meta in _prompt.action_catalog().items()
        )

        out: IntentOut = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
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

        r = IntentResult(
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
        print(f"[AGENT] extract_intent(llm) -> {r}", flush=True)
        return r

    except Exception as e:
        print(f"[AGENT] extract_intent llm 실패({e}) -> 빈 결과 (사용자에게 물어본다)", flush=True)
        return IntentResult()


# ─────────────────────────────────────────────────────────────────────────
# ActionAgent 판단부 — action_node 가 가지는 에이전트
#
# 사내 규칙: 이런 결정은 전부 LLM 이 한다. 노드는 판단하지 않고 이 에이전트를
# 호출만 한다. 규칙 기반 판단은 이 코드베이스에 존재하지 않는다.
# LLM 이 실패하면 추측하지 않고 안전한 쪽(재질문/미승인)으로 떨어진다.
# ─────────────────────────────────────────────────────────────────────────

class CollectAnswerOut(BaseModel):
    kind: Literal["cancel", "consult", "switch", "action", "value", "empty"]
    # kind == "action" 일 때만 채운다 (반송/목적지 선택)
    action_choice: Optional[Literal["transport", "dest_req"]] = None


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

    try:
        out: CollectAnswerOut = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            CollectAnswerOut,
            [
                SystemMessage(content=_prompt.action_collect_answer_prompt().strip()),
                HumanMessage(content=(
                    f"진행 중인 명령: {current_action or '미확정'}\n"
                    f"물어본 것: {question or fieldname}\n"
                    f"묻는 파라미터: {fieldname}\n"
                    f"사용자의 답변(원문): {text}"
                )),
            ],
            config=config,
        )
        print(f"[AGENT] classify_collect_answer(llm) -> {out.kind}", flush=True)

        if out.kind == "cancel":
            return {"kind": "cancel"}
        if out.kind == "consult":
            return {"kind": "consult", "text": text}
        if out.kind == "switch":
            return {"kind": "switch", "text": text}
        if out.kind == "action":
            if fieldname == "action" and out.action_choice:
                return {"kind": "action", "value": out.action_choice}
            # action 을 묻던 게 아닌데 action 이라 답함 -> 재질문으로 강등
            return {"kind": "empty", "note": "답변을 이해하지 못했습니다."}
        if out.kind == "value":
            return {"kind": "value", "text": text}
        return {"kind": "empty", "note": f"답변에서 {fieldname} 값을 찾지 못했습니다."}

    except Exception as e:
        # 추측하지 않는다 — 다시 묻는 게 가장 안전하다
        print(f"[AGENT] classify_collect_answer llm 실패({e}) -> 재질문", flush=True)
        return {"kind": "empty", "note": "답변을 이해하지 못했습니다. 다시 알려주세요."}


class ConfirmOut(BaseModel):
    verdict: Literal["approve", "reject", "unclear"]


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
        out: ConfirmOut = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            ConfirmOut,
            [
                SystemMessage(content=_prompt.action_confirm_prompt().strip()),
                HumanMessage(content=(
                    f"실행하려는 명령: {action or '?'} (파라미터: {params})\n"
                    f"사용자의 답변(원문): {answer}"
                )),
            ],
            config=config,
        )
        print(f"[AGENT] classify_confirm(llm) -> {out.verdict}", flush=True)
        return out.verdict

    except Exception as e:
        # 실행은 위험하다 — 판정 못 하면 절대 승인하지 않는다
        print(f"[AGENT] classify_confirm llm 실패({e}) -> unclear (미승인)", flush=True)
        return "unclear"


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
    return prompt | _llm.get_llm(model_name, temperature=0.2)


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

        # 앞 단계 처리 결과를 마지막 컨텍스트로 붙여 준다
        if context:
            messages = messages + [
                HumanMessage(content=f"[처리 결과]\n{context}")]

        # 2) 1차 호출
        text = await _astream_final(chain, messages, config)

        # 3) 비었으면 1회 재시도
        if not text.strip():
            print(f"[AGENT] {role}: 빈 응답 -> 1회 재시도", flush=True)
            text = await _astream_final(chain, messages, config)

        # 4) 그래도 비었으면 폴백 문구
        if not text.strip():
            print(f"[AGENT] {role}: 재시도도 실패 -> 폴백 문구", flush=True)
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
