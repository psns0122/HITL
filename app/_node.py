"""그래프 노드들.

Router / Supervisor / Final* 은 사내 코드(shared_code.md) 형태를 유지하고,
Location / Status / Log / Extract 는 목업 스텁이다.
(단, needs-핸드오프 계약과 facts 적재는 실제로 동작한다.)

ActionAgent 는 actions/node.py 의 턴 기반 단일 노드가 담당한다.
"""
import ast
import json
import re
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, create_model

import app.config as cfg
from app import _agent, _llm, _prompt, _state, _util
from app._util import (
    agent_ran_this_turn,
    emit,
    extract_json_object,
    last_user_text,
    member_answered_this_turn,
    message_content_to_text,
)
# *************  [app — 워커 스텁이 목업 DB 와 ID 판독기 툴을 직접 본다.
#                  사내 노드는 에이전트에 위임하므로 이 두 import 가 없다]  *************
from app import _db as mock_db
from app import _tool
# *************

# Supervisor 밑에 붙는 워커들
members = ["StatusAgent", "LocationAgent", "LogAgent", "ActionAgent", "ExtractAgent"]

# 액션 파이프라인 3노드. 입구(ActionAgent)만 members/로스터에 있고,
# Validator/Executor 는 직접 엣지·배관으로만 도달한다 (LLM 직행 차단).
ACTION_NODES = ["ActionAgent", "ActionValidator", "ActionExecutor"]

# ExtractAgent 는 '답변'을 내는 워커가 아니라 재료를 뽑는 선행 단계다.
# 그래서 "이번 턴에 워커가 답을 냈나?" 판정에서는 빼야 한다.
# (안 빼면 Extract 가 돌자마자 턴이 끝나버린다.)
# Validator/Executor 의 발화(승인 질문·실행 완료·거절 안내)도 '워커의 답'이다 —
# 안 넣으면 Extract 가 턴 중간에 재실행되고 Job ID 가 최종 답변에서 빠진다.
ANSWERING_MEMBERS = [m for m in members if m != "ExtractAgent"]     + ["ActionValidator", "ActionExecutor"]

options_for_next = ["FINISH", "FinalAnswerAgent"] + members
options_lower_map = {m.lower().replace("_", "").replace("-", ""): m for m in options_for_next}


class RouteResponse(TypedDict):
    next: Annotated[Literal[tuple(options_for_next)], "다음에 실행할 노드"]


# *************  [app 전용 — origin 에 없음]  *************
# needs-핸드오프 배분. Supervisor 소관이라 여기 둔다 (RouteResponse 와 같은 이유:
# origin 관례가 "스키마는 그것을 쓰는 파일에" 다).
#
# 입력은 ActionAgent 가 넘긴 원문 세 가지뿐이다.
#   question : ActionAgent 가 사용자에게 물은 것
#   answer   : 사용자가 실제로 답한 것 (원문)
#   fill     : 필요한 값의 이름
# 로스터를 보고 LLM 이 (1) 풀어 줄 워커와 (2) 그 워커에게 보낼 질의문을 고른다.
# 확신이 없으면 NONE — 그러면 사용자에게 직접 다시 묻는다.
#
# 워커를 새로 붙일 때 할 일은 로스터 프롬프트에 설명 한 줄을 더하는 것뿐이다.
# ActionAgent 도, 워커 본문도 건드리지 않는다.

class DispatchOut(BaseModel):
    agent: str = Field("NONE", description="도와줄 워커 이름. 없으면 NONE")
    query: str = Field("", description="그 워커에게 보낼 한 문장 질의")


def needs_dispatch(needs: dict, members: list, config=None,
                   model_name: str = None) -> dict:
    """상담 요청을 받아 도와줄 워커와 질의문을 고른다.

    반환: {"agent": 워커명 or None, "query": 질의문 or None}
    """
    answer = str(needs.get("answer") or "")

    try:
        out = _llm.structured_invoke(
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
        print(f"[AGENT] needs_dispatch llm 실패({e}) -> NONE (사용자에게 직접 질문)",
              flush=True)
        return {"agent": None, "query": None}
# *************


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
    # *************  [app — origin 에 없는 마지막 한 줄]
    # origin 은 MessagesPlaceholder 로 끝난다. 그런데 이 대화의 마지막은 거의
    # 항상 워커의 AIMessage(예: "[ExtractAgent] fab=M16 ...") 라서, 모델이
    # "다음 담당자를 고르는" 대신 그 문장을 이어 써 버린다. 실측하면 이 줄이
    # 없을 때 0/5, 있을 때 5/5 로 갈린다(tools/probe_supervisor.py).
    # 사람 차례로 끝내 주는 것뿐이라 사내 모델에도 무해하다.
    ("human", "위 대화 기준으로 다음에 일할 담당자를 고르시오. JSON 만 출력."),
    # *************
]).partial(
    options=str(options_for_next),
    members=", ".join(members),
)

# Supervisor 무한 순환 방지 상한 (한 user turn 내 노드 스텝)
MAX_SUPERVISOR_STEPS = 12

# needs-핸드오프에 별도 배분표는 없다.
#
# ActionAgent 는 "내가 이렇게 물었고(question) / 사용자가 이렇게 답했고
# (answer) / 나는 이 값이 필요하다(fill)" 원문만 넘긴다. 어느 워커가 그걸
# 풀 수 있는지는 Supervisor 가 평소 배분에 쓰는 것과 같은 로스터(members +
# 프롬프트의 워커 설명)를 보고 LLM 으로 판단한다(이 파일의 needs_dispatch).
#
# 그래서 워커를 새로 붙이면 — 로스터 프롬프트에 한 줄 설명을 더하는 순간 —
# needs 상담 대상에도 자동으로 편입된다. ActionAgent 는 아무것도 몰라도 된다.


def _model_of(state: _state.AgentState) -> str | None:
    """이번 턴에 쓸 모델명. 프론트가 고른 값이 state 에 실려 온다."""
    return state.get("model_name")


# ─────────────────────────────────────────────────────────────────────────
# Router / GeneralAgent — origin/_node.py 원문. app 추가분은 ************* 표시
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

    # *************  [app — 진행 중 HITL 액션이 있으면 분류 없이 Supervisor 고정.
    #  HITL 답변("STK102")이 general 로 오분류되면 진행 중 액션이 고아가 된다]
    if (state.get("action") or {}).get("phase"):
        print("[NODE] Router: 진행 중 액션 감지 -> Supervisor 고정", flush=True)
        return {"route": "supervisor", "handoff": True, "next": "Supervisor", "step": 1}
    # *************

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
# Supervisor
# ─────────────────────────────────────────────────────────────────────────

async def supervisor_node(state: _state.AgentState, config) -> _state.AgentState:
    """워커 배분. 결정적 규칙을 먼저 보고, 남으면 LLM 에게 묻는다.

    우선순위 (0~3 은 배관 — 판단이 아니라 파이프라인 연결)
      0) needs-핸드오프  : 상담 배분 / 헬퍼 답 회수 -> ActionAgent
      1) 턴 닫기         : 액션 노드가 질문을 던졌으면 END (사용자 대기)
      2) 재진입 배관     : confirming -> ActionExecutor,
                           collecting|awaiting_helper -> ActionAgent
      3) Extract 선행    : 신규 턴 재료 준비
      4) 스텝 상한 가드
      5) LLM 배분        : 신규 요청이 어느 담당자 일인지 (여기만 판단)
    """
    print("[NODE] Supervisor entered", flush=True)

    sc = state.get("action") or {}
    step = state.get("step", 0)
    messages = state.get("messages", []) or []

    # 0) needs-핸드오프 — ActionAgent 의 상담 요청 처리
    needs = sc.get("needs")
    if needs and not sc.get("needs_result"):

        # 0-b) 이미 워커에게 보냈고 답이 돌아온 상태 -> 답변 원문을 메일박스에
        #      실어 ActionAgent 로 돌려보낸다. 값 추출은 ActionAgent 가
        #      사용자 답변 읽듯 판독기로 한다 (여기서 파싱하지 않는다).
        dispatched = needs.get("dispatched_to")
        if dispatched:
            helper_msg = member_answered_this_turn(messages, [dispatched])
            out = dict(sc)
            if helper_msg:
                out["needs_result"] = {"text": str(helper_msg.content),
                                       "by": dispatched}
                print(f"[NODE] Supervisor: {dispatched} 답변 회수 -> ActionAgent",
                      flush=True)
            else:
                out["needs_result"] = {
                    "text": None, "by": dispatched,
                    "note": f"{dispatched} 가 답하지 않았습니다. 직접 입력해 주세요."}
                print(f"[NODE] Supervisor: {dispatched} 무응답 -> ActionAgent 반송",
                      flush=True)
            return {"action": out, "next": "ActionAgent", "step": step + 1}

        # 0-a) 첫 상담 -> 로스터를 보고 도와줄 워커를 고른다 (LLM/규칙).
        #      워커가 보낼 질의문도 여기서 함께 만든다. ActionAgent 는
        #      배분에 관여하지 않고, 워커도 needs 를 모른다.
        #      ExtractAgent 는 후보에서 뺀다 — 매 턴 선행 실행되는 재료 준비
        #      단계이지 참조를 풀어 주는 워커가 아니다. 로스터에 남겨 두면
        #      fill=eqp_id 라는 말에 끌려 "ID 추출" 로 오해하고 그걸 골라,
        #      아무것도 못 찾은 채 상담 왕복만 한 번 낭비한다(실측).
        helpers = [m for m in members if m != "ExtractAgent"]
        d = needs_dispatch(needs, helpers,
                                  config=config, model_name=_model_of(state))
        nxt = d.get("agent")

        if nxt in helpers:
            query = d.get("query") or needs.get("answer") or ""
            print(f"[NODE] Supervisor: needs 상담 -> {nxt} "
                  f"(fill={needs.get('fill')}, query='{query}')", flush=True)
            emit(config, "agent_status",
                 {"agent": "Supervisor", "detail": f"needs 상담 배분 -> {nxt}"})
            n2 = dict(needs)
            n2["dispatched_to"] = nxt
            out = dict(sc)
            out["needs"] = n2
            # 워커는 needs 계약을 모른다. 평소처럼 '이번 사용자 질의' 를 읽어
            # 일하도록, 조회 질의문을 대화에 실어 준다.
            return {"action": out,
                    "messages": [HumanMessage(content=query)],
                    "next": nxt, "step": step + 1}

        # 도와줄 워커가 없다 -> 빈 결과로 반송 (ActionAgent 가 직접 질문으로 강등)
        print("[NODE] Supervisor: needs 상담 — 처리할 워커 없음 -> ActionAgent 반송",
              flush=True)
        out = dict(sc)
        out["needs_result"] = {
            "text": None, "by": "Supervisor",
            "note": "이 답변을 해석할 수 있는 에이전트가 없습니다. 직접 입력해 주세요.",
        }
        return {"action": out, "next": "ActionAgent", "step": step + 1}

    # 1) 헬퍼가 값을 채워줬으면 ActionAgent 로 돌아간다
    if needs and sc.get("needs_result"):
        print("[NODE] Supervisor: 헬퍼 결과 도착 -> ActionAgent 재진입", flush=True)
        emit(config, "agent_status", {"agent": "Supervisor", "detail": "ActionAgent 재진입"})
        return {"next": "ActionAgent", "step": step + 1}

    # 1.5) ActionAgent 가 이번 턴에 사용자에게 질문을 던졌다 -> 턴 종료 (HITL 대기)
    #      awaiting 이 실려 있고, 그 질문 메시지가 이번 턴에 찍혀 있으면
    #      더 돌릴 게 없다. FinalAnswer 도 태우지 않고 그대로 끝낸다.
    #      (사용자의 답변은 다음 턴에 Router -> Supervisor 로 들어온다)
    if sc.get("awaiting") and member_answered_this_turn(messages, ACTION_NODES):
        print(f"[NODE] Supervisor: HITL 질문 발신({sc['awaiting'].get('type')}) "
              f"-> 턴 종료, 사용자 응답 대기", flush=True)
        emit(config, "agent_status",
             {"agent": "Supervisor", "detail": "사용자 응답 대기 — 턴 종료"})
        return {"next": "END", "step": step + 1}

    # 2) 진행 중 액션 재진입 배관 — 라우팅 판단이 아니라 파이프라인 배관이다.
    #    멈춘 단계(phase)가 재개 지점을 가리킨다. LLM 배분은 신규 요청만 탄다.
    #    (confirming 답변을 LLM 이 배분하게 두면 실행 단계를 건너뛰거나
    #     엉뚱한 워커로 새는 대참사가 가능하다 — 그래서 결정적으로 보낸다)
    if sc.get("phase") == "confirming":
        print("[NODE] Supervisor: 배관 confirming -> ActionExecutor", flush=True)
        return {"next": "ActionExecutor", "step": step + 1}
    if sc.get("phase") in ("collecting", "awaiting_helper"):
        print("[NODE] Supervisor: 배관 재진입 -> ActionAgent", flush=True)
        return {"next": "ActionAgent", "step": step + 1}

    # 3) ExtractAgent 선행 실행 — 라우팅 판단이 아니라 파이프라인 단계다.
    #    워커가 일하기 '전' 재료 준비이므로, 이번 턴에 이미 진행 중 액션이
    #    있거나(재료가 스크래치에 있음) 워커가 답을 낸 뒤라면 태우지 않는다.
    #    "다음에 누가 일하느냐" 는 아래 LLM 배분이 상황을 보고 정한다.
    in_flight = sc.get("phase") in ACTIVE_PHASES
    answered = member_answered_this_turn(messages, ANSWERING_MEMBERS)
    if (not in_flight and not answered
            and not agent_ran_this_turn(messages, "ExtractAgent")):
        print("[NODE] Supervisor: ExtractAgent 선행 실행", flush=True)
        emit(config, "agent_status",
             {"agent": "Supervisor", "detail": "ExtractAgent 선행 실행"})
        return {"next": "ExtractAgent", "step": step + 1}

    # 5) 스텝 상한 가드
    if step >= MAX_SUPERVISOR_STEPS:
        print(f"[WARN] Supervisor: step {step} >= {MAX_SUPERVISOR_STEPS} -> 강제 종료", flush=True)
        return {"next": "FinalAnswerAgent", "step": step + 1}

    # 6) LLM 기반 배분 — 여기부터는 origin/_node.py 의 supervisor_node 그대로다.
    #    모델이 뭘 뱉든 그래프가 죽으면 안 되므로 응답을 3단계로 방어한다.
    #      1. dict 인가  2. next 가 문자열인가  3. 아는 노드 이름인가(정규화 후 재시도)

    # 빈 content 메시지는 게이트웨이가 400 을 뱉는 경우가 있어 미리 걸러낸다
    raw_messages = state.get("messages", []) or []
    clean_messages = []
    for msg in raw_messages:
        content = getattr(msg, "content", None)
        if content is None or str(content).strip() == "":
            continue
        clean_messages.append(msg)

    # *************  [app — 상황을 하드코딩 분기 대신 LLM 에게 알려 준다]
    # 예전에는 "진행 중 액션 -> ActionAgent", "워커 답변 완료 -> FinalAnswer" 를
    # 코드가 강제했다. 지금은 그 사실만 대화에 실어 주고 배분은 LLM 이 한다.
    situ = []
    if answered:
        who = _util.agent_name_of(answered)
        situ.append(f"이번 턴에 {who} 가 이미 처리 결과를 냈다: "
                    f"{str(answered.content)[:120]}")
    if situ:
        clean_messages = clean_messages + [
            HumanMessage(content="[상황]\n" + "\n".join(f"- {s}" for s in situ))]
    # *************

    supervisor_chain = supervisor_prompt | _llm.get_llm(
        model_name=state.get("model_name"), temperature=0
    ).with_structured_output(RouteResponse, method="json_mode")

    next_node = "FinalAnswerAgent"

    # *************  [app — origin 은 여기서 final_message 를 만들어 반환의
    #  "messages" 에 실어 보낸다. app 은 싣지 않는다: messages 리듀서가 add 라
    #  기존 리스트를 되돌리면 대화가 통째로 중복 누적된다]
    # *************

    try:
        response = await supervisor_chain.ainvoke({"messages": clean_messages},
                                                  config=config)

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
                              f"'{raw_next}' Defaulting to FinalAnswerAgent.", flush=True)
                        next_node = "FinalAnswerAgent"
            else:
                print(f"[DEBUG] Supervisor returned a NON-STRING value for 'next': "
                      f"'{raw_next}' Defaulting to FinalAnswerAgent.", flush=True)
                next_node = "FinalAnswerAgent"
        else:
            print(f"[DEBUG] Supervisor response is NOT a dictionary: "
                  f"'{response}' Defaulting to FinalAnswerAgent.", flush=True)
            next_node = "FinalAnswerAgent"

    except Exception as e:
        print(f"[ERROR] Supervisor LLM invoke failed: {e}", flush=True)
        next_node = "FinalAnswerAgent"

    if next_node == "FINISH":
        next_node = "FinalAnswerAgent"

    # 정규화 맵을 타고 ExtractAgent 가 살아 돌아와도 차단한다 (위 순환 방지)
    if next_node == "ExtractAgent":
        print("[WARN] Supervisor: ExtractAgent 재배분 차단 -> FinalAnswerAgent", flush=True)
        next_node = "FinalAnswerAgent"

    print(f"[NODE] Supervisor -> {next_node}", flush=True)
    return {"next": next_node, "step": step + 1}


# ─────────────────────────────────────────────────────────────────────────
# 워커 노드들 (목업 스텁)
# ─────────────────────────────────────────────────────────────────────────

def location_node(state: _state.AgentState, config) -> _state.AgentState:
    """캐리어 위치 조회.

    needs-핸드오프 관련 코드는 없다. Supervisor 가 조회 질의문을 대화에
    실어 보내므로, 이 노드는 평소처럼 '이번 사용자 질의'만 처리하면 된다.
    """
    print("[NODE] LocationAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LocationAgent", "detail": "위치 조회"})

    text = last_user_text(state.get("messages", []))
    ids = _tool.params_extract_tool.invoke({"text": text}, config=config)

    lines, facts = [], {}
    for c in ids["carrier_ids"]:
        loc = mock_db.get_carrier_location(c)
        emit(config, "tool_call", {"agent": "LocationAgent", "tool": "location_search_tool",
                                   "args": {"carrier_id": c}, "result": {"eqp_id": loc}})
        if loc:
            lines.append(f"캐리어 {c} 는 현재 {loc} 에 있습니다.")
            facts[c] = {"eqp_id": loc}
        else:
            lines.append(f"캐리어 {c} 를 찾을 수 없습니다.")

    content = "\n".join(lines) or "질의에서 캐리어 ID 를 찾지 못했습니다."

    return {
        "messages": [AIMessage(content=content,
name="LocationAgent",
additional_kwargs={"agent_name": "LocationAgent"})],
        "facts": facts,
        "step": state.get("step", 0) + 1,
    }


def status_node(state: _state.AgentState, config) -> _state.AgentState:
    """캐리어/설비 상태 조회."""
    print("[NODE] StatusAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "StatusAgent", "detail": "상태 조회"})

    text = last_user_text(state.get("messages", []))
    ids = _tool.params_extract_tool.invoke({"text": text}, config=config)

    lines = []
    for c in ids["carrier_ids"]:
        emit(config, "tool_call", {"agent": "StatusAgent", "tool": "eqp_search_tool",
                                   "args": {"carrier_id": c}})
        info = mock_db.get_carrier(c)

        if info:
            lines.append(f"캐리어 {c}: 상태={info['status']}, "
                         f"위치={info['current_eqp']}, LOT={info['lot']}")
        else:
            lines.append(f"캐리어 {c} 를 찾을 수 없습니다.")

    content = "\n".join(lines) or "질의에서 캐리어 ID 를 찾지 못했습니다."

    return {
        "messages": [AIMessage(content=content,
name="StatusAgent",
additional_kwargs={"agent_name": "StatusAgent"})],
        "step": state.get("step", 0) + 1,
    }


def log_node(state: _state.AgentState, config) -> _state.AgentState:
    """반송 이력 분석.

    needs-핸드오프 관련 코드는 없다 (location_node 와 동일한 이유).
    """
    print("[NODE] LogAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LogAgent", "detail": "반송 이력 분석"})

    # 분석 대상 캐리어: 발화 내 ID > 전체
    text = last_user_text(state.get("messages", []))
    ids = _tool.params_extract_tool.invoke({"text": text}, config=config)
    carrier = ids["carrier_ids"][0] if ids["carrier_ids"] else None

    emit(config, "tool_call", {"agent": "LogAgent", "tool": "log_search_tool",
                               "args": {"carrier_id": carrier}})
    analysis = mock_db.analyze_transport_logs(carrier)

    combo_lines = [
        f"  - {c['eqp']} {c['reason']}/{c['description']} x{c['count']} (최초 {c['first_t']})"
        for c in analysis["combos"]
    ]
    content = "\n".join(
        [f"[LogAgent] 반송 이력 분석 결과 (carrier={carrier or '전체'})",
         f"- 에러 콤보 {len(analysis['combos'])}건:"]
        + combo_lines
        + [f"- 원인 장비: {analysis['cause_eqp']}",
           f"- 권장 대체 목적지: {analysis['recommended_dest']}"]
    )

    return {
        "messages": [AIMessage(content=content,
name="LogAgent",
additional_kwargs={"agent_name": "LogAgent"})],
        "facts": {"log_analysis": analysis},
        "step": state.get("step", 0) + 1,
    }


def extract_node(state: _state.AgentState, config) -> _state.AgentState:
    """FAB / 파라미터 추출.

    Supervisor 진입 후 항상 가장 먼저 실행된다(모든 워커에 선행).
    여기서 뽑은 ID 들이 뒤 단계의 재료가 된다.
    """
    print("[NODE] ExtractAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "ExtractAgent", "detail": "FAB/파라미터 추출"})

    text = last_user_text(state.get("messages", []))
    model = _model_of(state)

    # fab_extract_tool 상당 — 목업은 고정 FAB
    emit(config, "tool_call", {"agent": "ExtractAgent", "tool": "fab_extract_tool",
                               "args": {"text": text}, "result": {"fab": "M16"}})
    fab = "M16"

    # ID 판독기 — 진짜 툴을 호출하므로 트레이스는 on_tool_start/end 로 자동으로 뜬다
    # (emit 을 겹쳐 찍으면 프론트 트레이스에 같은 툴이 두 줄 난다)
    ids = _tool.params_extract_tool.invoke({"text": text}, config=config)

    carriers = ids.get("carrier_ids") or []
    eqps = ids.get("eqp_ids") or []

    # 게이트 판정은 하지 않는다.
    #   "ID 가 없으면 워커를 돌리지 말자" 는 발상이 틀렸다 — sys_admin_tool 처럼
    #   ID 가 아예 필요 없는 툴이 있어서, ID 유무로 워커 실행 여부를 정하면
    #   담당자 조회·패치 계획 같은 정상 질의가 통째로 막힌다.
    #   어느 워커로 보낼지는 원래대로 Supervisor 가 판단한다.

    content = (f"[ExtractAgent] fab={fab}, "
               f"carrier_ids={carriers or '없음'}, eqp_ids={eqps or '없음'}")

    return {
        "messages": [AIMessage(content=content,
name="ExtractAgent",
additional_kwargs={"agent_name": "ExtractAgent"})],
        # 추출 결과는 facts 에도 넣어둔다 (limiter 에 안 잘리는 공유 팩트)
        "facts": {"extracted": {"fab": fab, "carrier_ids": carriers,
                                "eqp_ids": eqps}},
        "step": state.get("step", 0) + 1,
    }


# *************  [app — origin 의 action_node(react agent 1개)를 3단 파이프라인으로 교체]  *************
#
# 왜 origin 과 다른가 (CLAUDE.md 규칙 3): origin 의 action_node 는 실행 툴을 문
# react agent 라 승인 전 실행을 막을 수 없다. HITL 은 수집→검증·승인→실행을
# 사람 확인으로 끊어야 하므로 최상위 노드 3개로 편다. interrupt 는 쓰지 않는다
# — 질문을 awaiting 에 싣고 턴을 정상 종료하면, 답변은 새 턴으로
# Router→Supervisor 를 거쳐 phase 배관을 타고 알맞은 단계로 재진입한다.
#
#   ActionAgent(판단) ──직접 엣지──> ActionValidator(검증·승인질문) ─> Supervisor
#        ▲                                                     승인 답변
#        └── Supervisor 배관(collecting|awaiting_helper)             │
#                                                                    ▼
#            Supervisor 배관(confirming) ─────────────────> ActionExecutor(판정·실행)
#
# 원칙: 툴은 에이전트가 부른다. 흐름(질문/상담/취소/이탈, 승인/거절)도
# 에이전트가 flow_tool / decide_tool 로 선언하고, 노드는 마지막 선언 인자를
# 읽어 턴만 닫는다. 노드가 직접 부르는 툴은 두 가지뿐이며 전부 ainvoke:
#   · 실행 직전 validate 1회 (안전 게이트 — 승인 대기 중 상태 변화 대비)
#   · {action}_execute_tool (명시적 승인 이후의 유일한 실행 지점)
#
# ActionValidator/ActionExecutor 는 Supervisor LLM 로스터·options_for_next 에
# 없다 — LLM 이 이름을 뱉어도 기각된다. 도달 경로는 직접 엣지와 배관뿐.

# 진행 중으로 취급하는 phase (배관·재진입 판정 기준)
ACTIVE_PHASES = {"collecting", "awaiting_helper", "confirming"}


def _parse_tool_result(content) -> dict:
    """ToolMessage.content 를 dict 로. JSON 도 파이썬 repr 도 받는다."""
    text = message_content_to_text(content)
    for parser in (json.loads, ast.literal_eval):
        try:
            data = parser(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def last_tool_call(result_messages, name: str):
    """에이전트 대화에서 이름이 name 인 **마지막** 툴 호출의 인자. 없으면 None.

    모델이 정식 tool_call 대신 텍스트 JSON({"name":..,"arguments":..})으로
    흉내 낸 것도 같은 판단으로 읽는다(실측: 소형 모델에서 발생).
    """
    found = None
    for m in result_messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            if tc.get("name") == name:
                found = tc.get("args") or {}
        pseudo = extract_json_object(
            message_content_to_text(getattr(m, "content", "")))
        if pseudo.get("name") == name:
            found = pseudo.get("arguments") or pseudo.get("args") or {}
    return found


def _say(text: str, who: str = "ActionAgent") -> list:
    return [AIMessage(content=text, name=who,
                      additional_kwargs={"agent_name": who})]


def _abandon(sc: dict, reason: str, step: int, who: str = "ActionAgent",
             tail: str = "") -> dict:
    """취소/거절/한도초과 종료. 스크래치 리셋 — 재시도 집착 금지."""
    catalog = _prompt.action_catalog()
    label = catalog[sc["action"]]["label"] if sc.get("action") in catalog else "명령"
    print(f"[ACTION abandon] {reason}", flush=True)
    body = f"🚫 {label} 을(를) 실행하지 않았습니다.\n- 사유: {reason}"
    if tail:
        body += f"\n{tail}"
    return {"messages": _say(body, who),
            "facts": {"last_action": {"action": sc.get("action"),
                                      "aborted": True, "reason": reason}},
            "action": {}, "next": "Supervisor", "step": step}


def _restart(text: str, step: int) -> dict:
    """맥락 이탈 — 진행 중 액션을 접고 새 발화로 다시 시작 (Router 부터)."""
    print(f"[ACTION restart] 새 질문으로 재시작: '{text}'", flush=True)
    return {"messages": [HumanMessage(content=text)],
            "action": {}, "next": "Supervisor", "step": step}


def _ask(sc: dict, field: str, prompt_text: str, step: int,
         who: str = "ActionAgent") -> dict:
    """⏸ 부족한 값을 묻고 턴 종료. 문구는 에이전트가 쓴 것(폴백: 카탈로그)."""
    sc["pending_field"] = field
    sc["phase"] = "collecting"
    sc["awaiting"] = {"type": "collect_param", "agent": "ActionAgent",
                      "action": sc.get("action"), "field": field,
                      "prompt": prompt_text, "params": sc.get("params", {}),
                      "missing": sc.get("missing", [])}
    print(f"[ACTION ask] ⏸ 질문 남기고 턴 종료 field={field}", flush=True)
    return {"action": sc, "messages": _say(prompt_text, who),
            "next": "Supervisor", "step": step}


def _fallback_ask_text(sc: dict, field: str) -> str:
    """에이전트가 질문 문구를 안 남겼을 때의 카탈로그 폴백 (빈 질문 방지)."""
    if field == "action" or not sc.get("action"):
        return _prompt.action_select_prompt()
    return _prompt.action_catalog()[sc["action"]]["param_prompts"].get(
        field, f"{field} 값을 알려주세요.")


async def _run_agent(create_fn, task: str, config, model_name):
    """에이전트 1회 실행. 툴을 아예 안 불렀거나 전부 인자 오류로 실패하면
    지시를 얹어 1회만 재시도한다(실측: text 인자 누락 등)."""
    agent = create_fn(model_name=model_name)
    out = await agent.ainvoke({"messages": [HumanMessage(content=task)]},
                              config=config)

    def _no_calls(result):
        return not any((getattr(m, "tool_calls", None) or [])
                       for m in result.get("messages", []))

    def _all_errors(result):
        tms = [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]
        return bool(tms) and all(
            message_content_to_text(m.content).startswith("Error invoking tool")
            for m in tms)

    if _no_calls(out) or _all_errors(out):
        print("[AGENT] 툴 호출 없음/실패 -> 1회 재시도", flush=True)
        nudge = ("반드시 도구를 호출해서 처리하라. 시스템 프롬프트의 "
                 "[도구 사용 순서]를 그대로 따르고, params_extract_tool 은 "
                 "text 인자에 발화 원문을 그대로 넣어라.")
        out = await agent.ainvoke(
            {"messages": [HumanMessage(content=task),
                          HumanMessage(content=nudge)]},
            config=config)
    return out.get("messages", [])


def _summary_of(result_messages) -> str:
    """에이전트의 마지막 답변(사용자 문구 재료). <think> 는 걷어낸다."""
    if not result_messages:
        return ""
    text = message_content_to_text(getattr(result_messages[-1], "content", ""))
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────────────────────
# 노드1 ActionAgent — 의도·파라미터 판단, 다음 흐름 선언
# ─────────────────────────────────────────────────────────────────────────

async def action_node(state: _state.AgentState, config) -> _state.AgentState:
    """판단 단계. 진입(신규 / 수집 답변 / 헬퍼 회수)마다 판단 에이전트를 1회
    돌리고, param_check 인자에서 (명령, 파라미터)를, flow_tool 에서 다음
    흐름과 사용자 문구를 읽어 턴을 닫는다. 필수값이 차면 검증 단계로 직행."""
    print("[NODE] ActionAgent entered", flush=True)
    sc = dict(state.get("action") or {})
    msgs = state.get("messages") or []
    awaiting = sc.get("awaiting")
    model_name = state.get("model_name")
    step = state.get("step", 0) + 1
    catalog = _prompt.action_catalog()
    user_text = last_user_text(msgs)

    # ── 진입 유형별 일감 문단 ────────────────────────────────────────────
    fx = (state.get("facts") or {}).get("extracted") or {}

    if sc.get("needs") and sc.get("needs_result"):
        res = sc.pop("needs_result") or {}
        needs = sc.pop("needs")
        sc["hops"] = sc.get("hops", 0) + 1
        print(f"[ACTION entry] 헬퍼({res.get('by')}) 결과 회수", flush=True)
        if res.get("text"):
            task = f"""진행 중인 명령: {sc.get('action')} (확보한 값: {sc.get('params')})
'{needs['fill']}' 값을 알아내려고 동료 {res.get('by')} 에게 물었고, 아래가 그 답변이다.
답변에서 {needs['fill']} 값을 판독해 param_check 까지 수행하라.
분석형 답변은 결론(권장값)이 마지막에 오는 경향이 있다.
동료의 답변:
{res['text']}"""
        else:
            task = f"""진행 중인 명령: {sc.get('action')} (확보한 값: {sc.get('params')})
'{needs['fill']}' 값을 동료에게 물었지만 얻지 못했다 ({res.get('note', '무응답')}).
param_check 로 상태를 확인하고, flow_tool(kind="ask_user", field="{needs['fill']}") 로
사용자에게 직접 물을 질문을 선언하라."""

    elif awaiting and msgs and isinstance(msgs[-1], HumanMessage):
        print(f"[ACTION entry] 수집 답변 수신: {user_text!r}", flush=True)
        sc.pop("awaiting", None)
        task = f"""진행 중인 명령: {sc.get('action') or '아직 정해지지 않음'} (확보한 값: {sc.get('params')})
사용자에게 물었던 것: {awaiting.get('prompt', '')}
사용자의 답변(원문): {user_text}
답변을 판독해 param_check 까지 수행하고, 필요한 흐름이 있으면 flow_tool 로 선언하라."""

    elif sc.get("phase") in ACTIVE_PHASES:
        # 새 발화 없는 재진입 — 하던 질문을 카탈로그 문구로 다시 조립한다
        field = sc.get("pending_field") or "action"
        print(f"[ACTION entry] 재진입 -> 질문 재조립 field={field}", flush=True)
        return _ask(sc, field, _fallback_ask_text(sc, field), step)

    else:
        print(f"[ACTION entry] 신규 진입: '{user_text}'", flush=True)
        sc = {"action": None, "params": {}, "missing": [],
              "collect_retries": 0, "validate_retries": 0, "hops": 0}
        task = user_text
        if fx.get("carrier_ids") or fx.get("eqp_ids"):
            # ExtractAgent 가 이번 턴에 선행 판독해 둔 결과 — 상류 에이전트의
            # 툴 작업 재사용이지, 노드가 툴을 대신 부르는 게 아니다
            task = (f"{user_text}\n[이번 턴 판독 결과] carrier_ids={fx.get('carrier_ids')}, "
                    f"eqp_ids={fx.get('eqp_ids')} — 이미 DB 로 확인된 값이다.")

    # ── 판단 에이전트 실행 + 결과 읽기 ──────────────────────────────────
    result = await _run_agent(_agent.create_action_agent, task, config, model_name)

    extract = None
    for m in result:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "params_extract_tool":
            data = _parse_tool_result(m.content)
            if "carrier_ids" in data:
                extract = data

    pc = last_tool_call(result, "param_check_tool")
    if pc is not None:
        got_action = (pc.get("action") or "").strip()
        if got_action in catalog and not sc.get("action"):
            sc["action"] = got_action
        got = {k: str(v).strip().upper() for k, v in (pc.get("params") or {}).items()
               if v and str(v).strip()}
        merged = dict(sc.get("params") or {})
        merged.update(got)          # 답변/정정의 새 값이 이긴다
        sc["params"] = merged

    # 명령이 아직 없으면 구조화 출력으로 명령 하나만 다시 묻는다
    # (자유 툴 호출은 흔들려도 tool_choice 고정은 안정적 — 실측)
    if not sc.get("action"):
        names = tuple(catalog.keys()) + ("unknown",)
        PickOut = create_model(
            "PickOut", action=(Literal[names],
                               Field(description="사용자가 요구한 명령. 모르면 unknown")))
        try:
            pick = _llm.structured_invoke(
                _llm.get_llm(model_name, temperature=0.0), PickOut,
                [SystemMessage(content=_prompt.action_agent_prompt().strip()),
                 HumanMessage(content=f"""아래 발화가 요구하는 명령 이름 하나만 답하라. 모르면 unknown.
발화: {user_text}""")],
                config=config)
            if pick.action != "unknown":
                sc["action"] = pick.action
                print(f"[ACTION pick] 구조화 폴백 -> {pick.action}", flush=True)
        except Exception as e:
            print(f"[ACTION pick] 폴백 실패({e}) -> 불명 유지", flush=True)

    # 판독 결과의 기계 대입 — 후보가 유일할 때만 (모호하면 판단의 영역이라 비워 둠)
    src = extract or fx
    if sc.get("action") and src:
        slot = {"carrier_id": src.get("carrier_ids") or [],
                "eqp_id": src.get("eqp_ids") or []}
        for field in catalog[sc["action"]]["required_params"]:
            pool = slot.get(field, [])
            if not sc["params"].get(field) and len(pool) == 1:
                sc["params"][field] = str(pool[0]).upper()

    summary = _summary_of(result)

    # ── flow_tool 선언 매핑 ─────────────────────────────────────────────
    flow = last_tool_call(result, "flow_tool") or {}
    kind = (flow.get("kind") or "").strip()
    field = (flow.get("field") or "").strip()
    message = (flow.get("message") or "").strip()
    if kind:
        print(f"[ACTION flow] {kind} field={field}", flush=True)

    if kind == "cancel":
        return _abandon(sc, message or "사용자 요청으로 명령을 취소했습니다.", step)

    if kind == "switch":
        return _restart(user_text, step)

    required = catalog[sc["action"]]["required_params"] if sc.get("action") else []
    sc["missing"] = ([p for p in required if not sc["params"].get(p)]
                     if sc.get("action") else ["action"])

    if kind == "consult" and sc.get("action"):
        fill = field if field in required else (sc["missing"][0] if sc["missing"] else "")
        if fill and sc.get("hops", 0) < cfg.MAX_HOPS:
            # needs 계약: fill / question(카탈로그) / answer(사용자 원문) / params
            sc["needs"] = {"fill": fill,
                           "question": _fallback_ask_text(sc, fill),
                           "answer": user_text,
                           "params": dict(sc.get("params") or {})}
            sc["phase"] = "awaiting_helper"
            print(f"[ACTION consult] Supervisor 상담 -> fill={fill}", flush=True)
            return {"action": sc, "next": "Supervisor", "step": step}
        # 상담 불가(홉 초과 등) -> 아래 직접 질문으로 강등

    # ask_user 선언, 또는 (흐름 선언 없이) 값이 아직 부족한 경우의 결정적 폴백
    if sc["missing"]:
        sc["collect_retries"] = sc.get("collect_retries", 0) + 1
        if sc["collect_retries"] > cfg.MAX_COLLECT:
            return _abandon(sc, "필수 파라미터를 수집하지 못해 요청을 종료합니다.", step)
        ask_field = sc["missing"][0]
        if kind == "ask_user" and field in (required or ["action"]):
            ask_field = field
        text = _fallback_ask_text(sc, ask_field)
        if kind == "ask_user" and message:
            text = message
        return _ask(sc, ask_field, text, step)

    # 필수값 충족 — 검증 단계로 직행 (직접 엣지)
    sc["phase"] = "validating"
    sc["agent_summary"] = summary
    print("[ACTION plan] 충족 -> ActionValidator 직행", flush=True)
    return {"action": sc, "next": "ActionValidator", "step": step}


# ─────────────────────────────────────────────────────────────────────────
# 노드2 ActionValidator — 검증 + 요약 + 승인 질문
# ─────────────────────────────────────────────────────────────────────────

async def action_validator_node(state: _state.AgentState, config) -> _state.AgentState:
    """검증 단계. 검증 에이전트가 {action}_validate_tool 을 부르고 결과를
    요약한다. 통과면 승인 질문(요약 + 코드가 박는 파라미터 명세 + 고정 문구)을
    발행하고, 실패면 문제 필드를 비우고 재질문을 발행한다."""
    print("[NODE] ActionValidator entered", flush=True)
    sc = dict(state.get("action") or {})
    step = state.get("step", 0) + 1
    model_name = state.get("model_name")
    action = sc.get("action")
    params = dict(sc.get("params") or {})

    if not action or not params:
        # 직접 엣지 외 경로로 들어올 일이 없지만, 들어왔다면 배분자에게 반송
        print("[NODE] ActionValidator: 전제 미충족 -> Supervisor 반송", flush=True)
        return {"next": "Supervisor", "step": step}

    task = f"""명령: {action}
파라미터: {params}
{action}_validate_tool 을 위 파라미터로 호출하고, 결과를 한국어로 정리하라."""
    result = await _run_agent(_agent.create_action_validator_agent,
                              task, config, model_name)

    v = None
    for m in result:
        if isinstance(m, ToolMessage) and getattr(m, "name", "").endswith("_validate_tool"):
            data = _parse_tool_result(m.content)
            if "ok" in data:
                v = data
    summary = _summary_of(result)

    if v is None:
        # 에이전트가 검증까지 못 갔다 — 실행 전 게이트와 같은 성격으로 직접 1회.
        # (툴 우회가 아니라 안전망. 사내 MCP 대비 ainvoke)
        print("[NODE] ActionValidator: 검증 결과 없음 -> 게이트 검증", flush=True)
        v = await getattr(_tool, f"{action}_validate_tool").ainvoke(
            {"params": params}, config=config)

    if v.get("ok"):
        head = summary or getattr(_tool, f"{action}_confirm_tool")(params)
        param_lines = "\n".join(f"- {k}: {val}" for k, val in params.items())
        guidance = f"{head}\n{param_lines}\n이 명령을 정말 실행할까요? (승인/거절)"
        sc["phase"] = "confirming"
        sc["awaiting"] = {"type": "confirm", "agent": "ActionAgent",
                          "action": action, "prompt": guidance, "params": params,
                          "options": ["승인", "거절"],
                          "asked_at": time.time()}   # 승인 TTL 기준 시각
        sc.pop("agent_summary", None)
        print("[ACTION validate] PASS -> ⏸ 승인 질문 남기고 턴 종료", flush=True)
        return {"action": sc, "messages": _say(guidance, "ActionValidator"),
                "next": "Supervisor", "step": step}

    # 검증 실패 — 문제 필드를 비우고 재질문 (답변은 배관 따라 노드1로)
    sc["validate_retries"] = sc.get("validate_retries", 0) + 1
    if sc["validate_retries"] >= cfg.MAX_VALIDATE:
        return _abandon(sc, f"유효성 검증에 반복 실패해 요청을 종료합니다. "
                            f"(사유: {v.get('reason')})", step, who="ActionValidator")
    for bad in v.get("bad_fields", []):
        sc["params"].pop(bad, None)
    required = _prompt.action_catalog()[action]["required_params"]
    remaining = [p for p in required if not sc["params"].get(p)]
    ask_field = remaining[0] if remaining else required[-1]
    text = summary or (f"검증 실패: {v.get('reason')}\n"
                       + _fallback_ask_text(sc, ask_field))
    print(f"[ACTION validate] FAIL({v.get('code')}) -> 재질문 field={ask_field}", flush=True)
    # G1: _ask 가 phase 를 collecting 으로 되돌린다 — confirming 으로 남으면
    # 재질문의 답이 배관을 타고 실행 단계로 가 버린다
    return _ask(sc, ask_field, text, step, who="ActionValidator")


# ─────────────────────────────────────────────────────────────────────────
# 노드3 ActionExecutor — 승인 판정 + 실행
# ─────────────────────────────────────────────────────────────────────────

async def action_executor_node(state: _state.AgentState, config) -> _state.AgentState:
    """실행 단계. 승인 답변 턴에만 일한다. decide_tool 판정이 approve=True 일
    때만 게이트 검증을 거쳐 실행하고, 그 외에는 무조건 종료한다(정정 루프 없음
    — 거절된 명령은 되살리지 않고, 사용자가 새로 요청한다)."""
    print("[NODE] ActionExecutor entered", flush=True)
    sc = dict(state.get("action") or {})
    msgs = state.get("messages") or []
    awaiting = sc.get("awaiting")
    step = state.get("step", 0) + 1
    model_name = state.get("model_name")

    # 진입 가드 — confirm 답변 소비 턴이 아니면 아무것도 하지 않는다.
    # Supervisor 반송이 아니라 FinalAnswer 로 보낸다(배관이 다시 여기로 보내는
    # 핑퐁 방지). 이 가드가 발동했다는 것은 배관 밖 경로로 들어왔다는 뜻이다.
    if not (awaiting and awaiting.get("type") == "confirm"
            and msgs and isinstance(msgs[-1], HumanMessage)):
        print("[NODE] ActionExecutor: 가드 발동 (confirm 답변 턴 아님) -> FinalAnswer", flush=True)
        return {"next": "FinalAnswerAgent", "step": step}

    answer = last_user_text(msgs)
    sc.pop("awaiting", None)
    action = sc.get("action")
    params = dict(sc.get("params") or {})

    # 승인 TTL — 내용과 무관하게 시간 초과면 만료 (판단이 아니라 시계다)
    asked_at = awaiting.get("asked_at")
    if asked_at is not None and time.time() - asked_at > cfg.CONFIRM_TTL_SEC:
        print("[ACTION confirm_ttl] 만료 -> abandon", flush=True)
        return _abandon(sc, f"승인 유효시간({cfg.CONFIRM_TTL_SEC}초)이 지나 "
                            "실행하지 않았습니다.", step, who="ActionExecutor",
                        tail="필요하면 명령을 다시 요청해 주세요.")

    task = f"""실행하려는 명령: {action} (파라미터: {params})
물었던 것: {awaiting.get('prompt', '')}
사용자의 답변(원문): {answer}
decide_tool 을 정확히 한 번 호출해 판정하라."""
    result = await _run_agent(_agent.create_action_confirm_agent,
                              task, config, model_name)
    decided = last_tool_call(result, "decide_tool") or {}
    approve = decided.get("approve") is True
    print(f"[ACTION decide] approve={approve} reason={decided.get('reason')!r}", flush=True)

    if not approve:
        reason = (decided.get("reason") or "").strip() \
            or "승인을 확인하지 못해 실행하지 않았습니다."
        # G5: 딴 주제 발화가 조용히 삼켜지지 않도록 원문 미처리를 알린다
        return _abandon(sc, reason, step, who="ActionExecutor",
                        tail="말씀하신 내용은 처리되지 않았습니다 — 필요하면 새로 요청해 주세요.")

    # 실행 직전 게이트 — 승인 대기 동안 설비 상태가 변했을 수 있다 [안전 게이트]
    v = await getattr(_tool, f"{action}_validate_tool").ainvoke(
        {"params": params}, config=config)
    if not v.get("ok"):
        return _abandon(sc, f"실행 직전 검증에 실패했습니다: {v.get('reason')}",
                        step, who="ActionExecutor",
                        tail="필요하면 명령을 다시 요청해 주세요.")

    result_exec = await getattr(_tool, f"{action}_execute_tool").ainvoke(
        {"params": params}, config=config)
    label = _prompt.action_catalog()[action]["label"]
    payload = result_exec.get("payload", {})
    lines = [f"✅ {label} 실행 완료",
             f"- Job ID: {result_exec.get('job_id')}",
             f"- 상태: {result_exec.get('status')}"]
    lines += [f"- {k}: {val}" for k, val in payload.items()]
    print(f"[ACTION execute] job={result_exec.get('job_id')}", flush=True)
    return {"messages": _say("\n".join(lines), "ActionExecutor"),
            "facts": {"last_action": {"action": action, "params": params,
                                      "result": result_exec}},
            "action": {}, "next": "Supervisor", "step": step}
# *************

# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 노드 (스트리밍)
# ─────────────────────────────────────────────────────────────────────────

async def final_node(state: _state.AgentState, config) -> _state.AgentState:
    """워커 결과를 받아 최종 답변을 만든다. (origin/_node.final_node 와 같은 틀)"""
    print("[NODE] FinalAnswer entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalAnswerAgent", "detail": "최종 응답 생성"})

    # *************  [app — 이번 턴에 워커가 낸 답을 근거로 실어 준다.
    #  origin 은 context 없이 state 만 넘긴다]
    member_msg = member_answered_this_turn(state.get("messages", []) or [],
                                           ANSWERING_MEMBERS)
    context = str(member_msg.content) if member_msg else ""
    # *************

    # create_final_agent 는 _ainvoke(state, config, context) 함수를 돌려준다
    out = await _agent.create_final_agent(
        model_name=_model_of(state))(state, config, context)

    msgs = out.get("messages", []) or []

    if msgs == []:
        print("[ERROR] 비어있는 FINAL 응답", flush=True)
    else:
        msg = msgs[0]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        finish = msg.response_metadata.get("finish_reason")

        # 빈 본문이거나 정상 종료(stop)가 아니면 폴백 문구로 교체한다
        is_empty = not content.strip()
        is_weird_finish = finish not in ("stop", None)

        if is_empty or is_weird_finish:
            print(f"[ERROR] 비정상 FINAL 응답 종료 "
                  f"content={content[:80]!r} finish_reason={finish}", flush=True)
            msgs = [AIMessage(content="응답 생성에 실패했습니다. 다시 시도해주세요.")]

    # *************  [app — 에이전트 이름표를 붙인다. Supervisor 의 '이번 턴에
    #  누가 답했나' 판정(_util.agent_name_of)이 이 값을 읽는다]
    msgs = [AIMessage(content=str(m.content), name="FinalAnswerAgent",
                      additional_kwargs={"agent_name": "FinalAnswerAgent"})
            for m in msgs]
    # *************

    return {
        "messages": msgs,
        "next": "END",
        "step": state.get("step", 0) + 1,
    }


async def final_general_node(state: _state.AgentState, config) -> _state.AgentState:
    """일반 대화의 최종 답변. final_node 와 같은 모양이다."""
    print("[NODE] FinalGeneral entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalGeneralAgent", "detail": "일반 응답 생성"})

    out = await _agent.create_final_general_agent(
        model_name=_model_of(state))(state, config)

    msgs = out.get("messages", []) or []

    if msgs == []:
        print("[ERROR] 비어있는 FINAL 응답", flush=True)
    else:
        msg = msgs[0]
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        finish = msg.response_metadata.get("finish_reason")

        is_empty = not content.strip()
        is_weird_finish = finish not in ("stop", None)

        if is_empty or is_weird_finish:
            print(f"[ERROR] 비정상 FINAL 응답 종료 "
                  f"content={content[:80]!r} finish_reason={finish}", flush=True)
            msgs = [AIMessage(content="응답 생성에 실패했습니다. 다시 시도해주세요.")]

    # *************  [app — 에이전트 이름표]
    msgs = [AIMessage(content=str(m.content), name="FinalGeneralAgent",
                      additional_kwargs={"agent_name": "FinalGeneralAgent"})
            for m in msgs]
    # *************

    return {
        "messages": msgs,
        "next": "END",
        "step": state.get("step", 0) + 1,
    }
