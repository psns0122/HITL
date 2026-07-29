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

# ExtractAgent 는 '답변'을 내는 워커가 아니라 재료를 뽑는 선행 단계다.
# 그래서 "이번 턴에 워커가 답을 냈나?" 판정에서는 빼야 한다.
# (안 빼면 Extract 가 돌자마자 턴이 끝나버린다.)
ANSWERING_MEMBERS = [m for m in members if m != "ExtractAgent"]

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

    우선순위
      0) needs-핸드오프 : ActionAgent 가 동료에게 값을 요청함
      1) 헬퍼 결과 도착 : ActionAgent 재진입
      2) 진행 중 액션   : ActionAgent 계속
      3) ExtractAgent   : 이번 턴에 아직 안 돌았으면 무조건 먼저
      4) 워커 응답 완료 : FinalAnswerAgent
      5) 스텝 상한      : 강제 종료
      6) 그 외          : LLM/규칙 배분
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
    if sc.get("awaiting") and member_answered_this_turn(messages, ["ActionAgent"]):
        print(f"[NODE] Supervisor: HITL 질문 발신({sc['awaiting'].get('type')}) "
              f"-> 턴 종료, 사용자 응답 대기", flush=True)
        emit(config, "agent_status",
             {"agent": "Supervisor", "detail": "사용자 응답 대기 — 턴 종료"})
        return {"next": "END", "step": step + 1}

    # 2) ExtractAgent 선행 실행 — 라우팅 판단이 아니라 파이프라인 단계다.
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
    if in_flight:
        aw = sc.get("awaiting") or {}
        situ.append(f"진행 중인 명령이 있다: action={sc.get('action')}, "
                    f"단계={sc.get('phase')}.")
        if aw:
            situ.append(f"사용자에게 '{aw.get('type')}' 질문을 던져 둔 상태이고, "
                        "방금 사용자 발화는 그 질문에 대한 답이다. "
                        "이 답은 그 명령을 진행하던 담당자가 이어받아야 한다.")
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


# *************  [app — origin 의 action_node 를 턴 기반 HITL 로 교체]  *************
#
# 원칙: 툴은 에이전트가 부른다.
#   판독(params_extract)·파라미터 확인(param_check)·검증(validate)은 전부
#   react agent(_agent.create_action_agent)가 스스로 호출한다. 이 노드는
#   에이전트를 돌리고, 그 툴 호출 기록에서 판단 결과를 읽어 턴을 열고 닫는
#   흐름만 잡는다. 노드가 직접 부르는 것은 두 가지뿐이다:
#     · {action}_execute_tool — 명시적 승인 이후의 유일한 실행 지점
#     · 실행 직전 validate 1회 — 안전 게이트 (승인 불변식과 같은 급)
#
# 판단(LLM)은 세 갈래다. 실패하면 지어내지 않고 재질문/미승인으로 떨어진다.
#   run_action_agent        : 명령·파라미터·검증 — 에이전트+툴
#   classify_collect_answer : 질문에 대한 답변의 종류 (값/상담/이탈/취소/무의미)
#   classify_confirm        : 승인 판정 (approve/reject/unclear)

# 진행 중으로 취급하는 phase (Supervisor 의 상황 컨텍스트·재진입 판정 기준)
ACTIVE_PHASES = {"param_check", "collecting", "awaiting_helper", "validating", "confirming"}


class CollectAnswerOut(BaseModel):
    """파라미터 질문에 대한 답변의 종류. 명령 종류와 무관하게 고정이다."""
    kind: Literal["value", "consult", "switch", "cancel", "empty"] = Field(
        description="답변이 무엇을 하려는 것인지")


class ConfirmOut(BaseModel):
    """승인 질문에 대한 판정."""
    verdict: Literal["approve", "reject", "unclear"] = Field(
        description="실행 승인 여부")


def classify_collect_answer(fieldname: str, answer, current_action: str | None,
                            question: str = None, config=None,
                            model_name: str = None) -> dict:
    """파라미터 질문에 대한 사용자 답변을 분류한다.

    반환: {"kind": value|consult|switch|cancel|empty, "text"/"note"...}
    """
    text = str(answer or "")

    # /chat/stop 등이 보내는 기계 센티널 — 모델에 물을 것도 없다
    if isinstance(answer, dict) and answer.get("aborted"):
        return {"kind": "cancel"}

    try:
        out = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            CollectAnswerOut,
            [SystemMessage(content=_prompt.action_collect_answer_prompt().strip()),
             HumanMessage(content=f"""진행 중인 명령: {current_action or '아직 정해지지 않음'}
사용자에게 물어본 것: {question or fieldname}
지금 받아야 하는 값: {fieldname}
사용자의 답변(원문): {text}""")],
            config=config)
        print(f"[AGENT] classify_collect_answer(llm) -> {out.kind}", flush=True)
        if out.kind in ("value", "consult", "switch"):
            return {"kind": out.kind, "text": text}
        if out.kind == "cancel":
            return {"kind": "cancel"}
        return {"kind": "empty", "note": f"답변에서 {fieldname} 값을 찾지 못했습니다."}

    except Exception as e:
        # 추측하지 않는다 — 다시 묻는 게 가장 안전하다
        print(f"[AGENT] classify_collect_answer llm 실패({e}) -> 재질문", flush=True)
        return {"kind": "empty", "note": "답변을 이해하지 못했습니다. 다시 알려주세요."}


def classify_confirm(answer, action: str = None, params: dict = None,
                     config=None, model_name: str = None) -> str:
    """승인 질문에 대한 답변 판정 -> approve | reject | unclear."""
    # /chat/stop 등이 보내는 기계 센티널 — 모델에 물을 것도 없다
    if isinstance(answer, dict):
        if answer.get("aborted"):
            return "reject"
        if "approved" in answer:
            return "approve" if answer["approved"] else "reject"

    try:
        out = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            ConfirmOut,
            [SystemMessage(content=_prompt.action_confirm_prompt().strip()),
             HumanMessage(content=f"""실행하려는 명령: {action or '?'} (파라미터: {params})
사용자의 답변(원문): {answer}""")],
            config=config)
        print(f"[AGENT] classify_confirm(llm) -> {out.verdict}", flush=True)
        return out.verdict

    except Exception as e:
        # 실행은 위험하다 — 판정 못 하면 절대 승인하지 않는다
        print(f"[AGENT] classify_confirm llm 실패({e}) -> unclear (미승인)", flush=True)
        return "unclear"


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


async def run_action_agent(task: str, config=None, model_name: str = None,
                           extracted: dict = None) -> dict:
    """react agent 를 한 번 돌리고 판단 결과를 읽는다.

    task 는 에이전트에게 줄 일감 문장 — 신규 발화 원문이거나, 재진입이면
    "진행 중인 명령 X 에서 사용자가 이렇게 답했다 / 동료가 이렇게 답했다"
    까지 담은 문단이다. 에이전트는 어느 경우든 똑같이 판독기 -> param_check
    -> validate 순서로 툴을 부르며 일한다.

    반환: {"action", "params", "missing", "valid", "summary"}
      action  : 에이전트가 판단한 명령 (없으면 None)
      params  : param_check 에 실은 파라미터 (normalized 결과 우선)
      missing : param_check 결과의 부족 목록 (param_check 미호출이면 None)
      valid   : validate 결과 dict (미호출이면 None)
      summary : 마지막 답변 텍스트 (<think> 제거) — 승인 질문 머리말로 재활용

    판단 결과는 자유 텍스트가 아니라 **툴 호출 인자/결과**에서 읽는다.
    모델이 툴을 정식으로 못 부르고 텍스트 JSON 을 뱉으면 그것도 읽고,
    그래도 명령이 비면 구조화 출력으로 명령만 한 번 더 묻는다(강제
    tool_choice 는 자유 호출보다 안정적이다 — 실측). 전부 LLM 판단이다.
    """
    from app._agent import create_action_agent   # _agent 가 _node 를 안 봐서 안전하지만 관례 유지

    def _made_tool_calls(result) -> bool:
        return any((getattr(m, "tool_calls", None) or [])
                   for m in result.get("messages", []))

    empty = {"action": None, "params": {}, "missing": None, "valid": None, "summary": ""}

    try:
        agent = create_action_agent(model_name=model_name)
        out = await agent.ainvoke({"messages": [HumanMessage(content=task)]},
                                  config=config)

        # 툴을 아예 안 불렀거나, 부른 게 전부 인자 오류로 실패했으면
        # (실측: params_extract_tool 을 text 인자 없이 호출) 지시를 얹어 1회 재시도
        def _all_tool_errors(result) -> bool:
            tms = [m for m in result.get("messages", []) if isinstance(m, ToolMessage)]
            return bool(tms) and all(
                message_content_to_text(m.content).startswith("Error invoking tool")
                for m in tms)

        if not _made_tool_calls(out) or _all_tool_errors(out):
            print("[AGENT] run_action_agent: 툴 호출 없음/실패 -> 1회 재시도", flush=True)
            nudge = ("반드시 도구를 호출해서 처리하라. 너에게 있는 도구는 "
                     "params_extract_tool / param_check_tool / "
                     "transport_validate_tool / dest_req_validate_tool 뿐이다. "
                     "params_extract_tool 을 부를 때는 text 인자에 사용자 발화 "
                     "원문을 그대로 넣어라. 로그 분석이나 위치 조회는 네 일이 "
                     "아니다 — 그런 값은 비워 두고 param_check 까지만 하라.")
            out = await agent.ainvoke(
                {"messages": [HumanMessage(content=task),
                              HumanMessage(content=nudge)]},
                config=config)

        catalog = _prompt.action_catalog()
        action, params, missing, valid = None, {}, None, None
        extract = None   # 마지막 params_extract_tool 결과 (DB 판독 사실)

        for m in out.get("messages", []):
            # 에이전트의 판단 = 툴 호출 인자
            for tc in (getattr(m, "tool_calls", None) or []):
                name = tc.get("name") or ""
                args = tc.get("args") or {}
                if name == "param_check_tool":
                    action = (args.get("action") or "").strip() or action
                    got = {k: str(v).strip().upper()
                           for k, v in (args.get("params") or {}).items()
                           if v and str(v).strip()}
                    if got:
                        params = got
                elif name.endswith("_validate_tool") and name[:-len("_validate_tool")] in catalog:
                    # param_check 를 건너뛰고 validate 부터 부르는 모델이 있다(실측).
                    # 그 툴을 골랐다는 것 자체가 명령 판단이므로 같은 근거로 읽는다.
                    action = name[:-len("_validate_tool")]
                    got = {k: str(v).strip().upper()
                           for k, v in (args.get("params") or {}).items()
                           if v and str(v).strip()}
                    if got:
                        params = got
            # 툴의 답 = 확인된 사실 (판독 결과 / missing / 검증 결과 / 정규화 값)
            if isinstance(m, ToolMessage):
                data = _parse_tool_result(m.content)
                tname = getattr(m, "name", "") or ""
                if tname == "params_extract_tool" and "carrier_ids" in data:
                    extract = data
                if tname == "param_check_tool" and "missing" in data:
                    missing = list(data.get("missing") or [])
                    norm = {k: str(v).strip().upper()
                            for k, v in (data.get("normalized") or {}).items()
                            if v and str(v).strip()}
                    if norm:
                        params = norm
                elif tname.endswith("_validate_tool") and "ok" in data:
                    valid = data

        # 모델이 툴 호출을 텍스트 JSON 으로 뱉은 경우도 같은 판단으로 읽는다
        if not action:
            for m in out.get("messages", []):
                pseudo = extract_json_object(
                    message_content_to_text(getattr(m, "content", "")))
                if pseudo.get("name") == "param_check_tool":
                    args = pseudo.get("arguments") or pseudo.get("args") or {}
                    action = (args.get("action") or "").strip() or action
                    got = {k: str(v).strip().upper()
                           for k, v in (args.get("params") or {}).items()
                           if v and str(v).strip()}
                    if got:
                        params = got

        # 카탈로그에 없는 명령을 지어냈으면 불명 처리
        if action and action not in catalog:
            print(f"[AGENT] run_action_agent: 모르는 명령 {action!r} -> 불명", flush=True)
            action = None

        # 그래도 명령이 비면 구조화 출력으로 명령 하나만 다시 묻는다
        if not action:
            names = tuple(catalog.keys()) + ("unknown",)
            PickOut = create_model(
                "PickOut",
                action=(Literal[names],
                        Field(description="사용자가 요구한 명령. 모르면 unknown")))
            pick = _llm.structured_invoke(
                _llm.get_llm(model_name, temperature=0.0), PickOut,
                [SystemMessage(content=_prompt.action_agent_prompt().strip()),
                 HumanMessage(content=f"""아래 발화가 요구하는 명령 이름 하나만 답하라. 모르면 unknown.
발화: {task}""")],
                config=config)
            if pick.action != "unknown":
                action = pick.action
                print(f"[AGENT] run_action_agent: 구조화 폴백 -> {action}", flush=True)

        # 에이전트가 판독까지만 하고 param_check 를 건너뛰거나 판독 호출 자체를
        # 망치면 판독 결과가 버려진다(실측). 판독기는 DB 조회 — 이번 실행의
        # 판독 결과가 없으면 ExtractAgent 가 이번 턴에 선행 판독해 둔 결과
        # (extracted)로 대신한다. 어느 쪽이든 **후보가 유일할 때만** 해당 자리에
        # 기계 대입한다. 후보가 여럿이면(어느 쪽이 대상인지는 판단의 영역)
        # 비워 두고 상담/질문 경로로 흘린다.
        src = extract or extracted
        if action and src:
            slot = {"carrier_id": src.get("carrier_ids") or [],
                    "eqp_id": src.get("eqp_ids") or []}
            for field in catalog[action]["required_params"]:
                pool = slot.get(field, [])
                if not params.get(field) and len(pool) == 1:
                    params[field] = str(pool[0]).upper()
                    # 에이전트의 검증은 이 값이 없던 시점의 결과다 — 무효화.
                    # (안 하면 낡은 실패 결과가 방금 채운 값을 도로 지운다. 실측)
                    valid = None

        summary = ""
        msgs = out.get("messages", [])
        if msgs:
            summary = message_content_to_text(getattr(msgs[-1], "content", ""))
            summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()

        r = {"action": action, "params": params, "missing": missing,
             "valid": valid, "summary": summary}
        print(f"[AGENT] run_action_agent -> action={action} params={params} "
              f"missing={missing} valid={'ok' if (valid or {}).get('ok') else valid and valid.get('code')}",
              flush=True)
        return r

    except Exception as e:
        # 추측하지 않는다 — 되묻는 게 안전하다
        print(f"[AGENT] run_action_agent 실패({e}) -> 불명 (사용자에게 묻는다)", flush=True)
        return empty


async def action_node(state: _state.AgentState, config) -> _state.AgentState:
    """명령 실행 — 턴 기반 HITL.

    interrupt() 를 쓰지 않는다. 사용자에게 물을 게 생기면 질문을
    action.awaiting 에 싣고 턴을 정상 종료하며, 답변은 새 턴으로 들어와
    Router -> Supervisor 를 거쳐 여기 재진입한다. 상태의 근거는 오직
    action 스크래치이고, execute 는 명시적 승인 이후에만 도달한다.

    진입 유형: ① 헬퍼 답변 회수 ② HITL 답변 ③ 재진입(질문 재조립) ④ 신규.
    어느 유형이든 값을 바꾸는 판단은 에이전트(run_action_agent)가 툴을
    불러서 하고, 이 함수는 그 결과로 턴의 끝(질문/상담/승인/실행/포기)만 정한다.
    """
    print("[NODE] ActionAgent entered", flush=True)
    sc = dict(state.get("action") or {})
    msgs = state.get("messages") or []
    awaiting = sc.get("awaiting")
    model_name = state.get("model_name")
    step = {"step": state.get("step", 0) + 1}

    # ── 턴을 닫는 조각들 (스크래치를 공유하는 지역 함수) ────────────────────

    def say(text: str) -> list:
        return [AIMessage(content=text, name="ActionAgent",
                          additional_kwargs={"agent_name": "ActionAgent"})]

    def ask_param(field: str) -> dict:
        """⏸ 부족한 값을 묻고 턴 종료."""
        if field == "action":
            prompt = _prompt.action_select_prompt()
        else:
            prompt = _prompt.action_catalog()[sc["action"]]["param_prompts"][field]
        note = sc.pop("last_parse_error", None)
        if note:
            prompt = f"{note}\n{prompt}"
        sc["pending_field"] = field
        sc["phase"] = "collecting"
        sc["awaiting"] = {"type": "collect_param", "agent": "ActionAgent",
                          "action": sc.get("action"), "field": field,
                          "prompt": prompt, "params": sc.get("params", {}),
                          "missing": sc.get("missing", [])}
        print(f"[ACTION ask_param] ⏸ 질문 남기고 턴 종료 field={field}", flush=True)
        return {"action": sc, "messages": say(prompt), "next": "Supervisor", **step}

    def ask_confirm() -> dict:
        """⏸ 실행 직전 승인 질문. 머리말은 에이전트 요약, 명세는 코드가 박는다."""
        summary = (sc.pop("agent_summary", "") or "").strip()
        if summary:
            param_lines = "\n".join(f"- {k}: {v}" for k, v in sc["params"].items())
            guidance = f"{summary}\n{param_lines}\n이 명령을 정말 실행할까요? (승인/거절)"
        else:
            guidance = getattr(_tool, f"{sc['action']}_confirm_tool")(sc["params"])
        sc["phase"] = "confirming"
        sc["awaiting"] = {"type": "confirm", "agent": "ActionAgent",
                          "action": sc["action"], "prompt": guidance,
                          "params": sc["params"], "options": ["승인", "거절"],
                          "asked_at": time.time()}   # 승인 TTL 기준 시각
        print(f"[ACTION ask_confirm] ⏸ 승인 질문 남기고 턴 종료", flush=True)
        return {"action": sc, "messages": say(guidance), "next": "Supervisor", **step}

    def abandon(reason: str) -> dict:
        """취소/거절/한도초과 종료. 스크래치 리셋 — 재시도 집착 금지."""
        catalog = _prompt.action_catalog()
        label = catalog[sc["action"]]["label"] if sc.get("action") in catalog else "명령"
        print(f"[ACTION abandon] {reason}", flush=True)
        return {"messages": say(f"🚫 {label} 을(를) 실행하지 않았습니다.\n- 사유: {reason}"),
                "facts": {"last_action": {"action": sc.get("action"),
                                          "aborted": True, "reason": reason}},
                "action": {}, "next": "Supervisor", **step}

    def restart(text: str) -> dict:
        """맥락 이탈 — 진행 중 액션을 접고 새 발화로 다시 시작."""
        print(f"[ACTION restart] 새 질문으로 재시작: '{text}'", flush=True)
        return {"messages": [HumanMessage(content=text)],
                "action": {}, "next": "Supervisor", **step}

    def needs_exit(fill: str) -> dict:
        """Supervisor 에게 상담하러 정상 종료. 발화 원문만 넘긴다."""
        if sc.get("action") and fill != "action":
            question = _prompt.action_catalog()[sc["action"]]["param_prompts"].get(fill, "")
        else:
            question = _prompt.action_select_prompt()
        sc["needs"] = {"fill": fill, "question": question,
                       "answer": sc.pop("consult_text"),
                       "params": dict(sc.get("params") or {})}
        sc["phase"] = "awaiting_helper"
        print(f"[ACTION needs_exit] Supervisor 상담 -> fill={fill}", flush=True)
        return {"action": sc, "next": "Supervisor", **step}

    async def execute() -> dict:
        """실행 — 명시적 승인 이후에만 도달하는 유일한 side-effect 지점.

        실행 직전 validate 를 한 번 더 태운다. 승인 대기 동안 설비 상태가
        변했을 수 있어서다. 이것은 에이전트 우회가 아니라 안전 게이트다.
        """
        v = await getattr(_tool, f"{sc['action']}_validate_tool").ainvoke(
            {"params": sc["params"]}, config=config)
        if not v.get("ok"):
            print(f"[ACTION execute] 게이트 검증 실패({v.get('code')}) -> 재수집", flush=True)
            for bad in v.get("bad_fields", []):
                sc["params"].pop(bad, None)
            sc["last_parse_error"] = f"검증 실패: {v.get('reason')}"
            missing = [p for p in _prompt.action_catalog()[sc["action"]]["required_params"]
                       if not sc["params"].get(p)]
            return ask_param(missing[0] if missing else "action")

        result = getattr(_tool, f"{sc['action']}_execute_tool")(sc["params"])
        emit(config, "tool_call", {"agent": "ActionAgent",
                                   "tool": f"{sc['action']}_execute_tool",
                                   "args": sc["params"], "result": result})
        label = _prompt.action_catalog()[sc["action"]]["label"]
        payload = result.get("payload", {})
        lines = [f"✅ {label} 실행 완료",
                 f"- Job ID: {result.get('job_id')}",
                 f"- 상태: {result.get('status')}"]
        lines += [f"- {k}: {v2}" for k, v2 in payload.items()]
        print(f"[ACTION execute] job={result.get('job_id')}", flush=True)
        return {"messages": say("\n".join(lines)),
                "facts": {"last_action": {"action": sc["action"],
                                          "params": sc.get("params"),
                                          "result": result}},
                "action": {}, "next": "Supervisor", **step}

    # ── 진입 판정 + 에이전트 실행 ────────────────────────────────────────────

    harvest = None   # run_action_agent 결과. 값을 바꾸는 진입이면 반드시 채워진다

    # ① 헬퍼 답변 회수 (needs-핸드오프 복귀)
    if sc.get("needs") and sc.get("needs_result"):
        res = sc.pop("needs_result") or {}
        needs = sc.pop("needs")
        sc.pop("consult_text", None)
        sc["hops"] = sc.get("hops", 0) + 1
        print(f"[ACTION entry] 헬퍼({res.get('by')}) 결과 회수", flush=True)
        if res.get("text"):
            harvest = await run_action_agent(
                f"""진행 중인 명령: {sc.get('action')} (지금까지 확보한 값: {sc.get('params')})
'{needs['fill']}' 값을 알아내려고 동료 {res.get('by')} 에게 물었고, 아래가 그 답변이다.
답변에서 {needs['fill']} 값을 판독해 param_check 까지 수행하라.
분석형 답변은 결론(권장값)이 마지막에 오는 경향이 있다.
동료의 답변:
{res['text']}""",
                config=config, model_name=model_name)
            merged = dict(harvest["params"])
            merged.update({k: v for k, v in (sc.get("params") or {}).items() if v})
            if harvest["params"].get(needs["fill"]):
                merged[needs["fill"]] = harvest["params"][needs["fill"]]
            sc["params"] = merged
        else:
            sc["collect_retries"] = sc.get("collect_retries", 0) + 1
            sc["last_parse_error"] = res.get("note") or "동료 답변에서 값을 찾지 못했습니다."

    # ② HITL 답변 (질문을 던져 두었고, 새 사용자 발화가 들어왔다)
    elif awaiting and msgs and isinstance(msgs[-1], HumanMessage):
        answer = last_user_text(msgs)
        print(f"[ACTION entry] HITL 답변 수신({awaiting['type']}): {answer!r}", flush=True)
        sc.pop("awaiting", None)

        if awaiting["type"] == "confirm":
            # 승인 TTL — 내용과 무관하게 시간 초과면 만료 (판단이 아니라 시계다)
            asked_at = awaiting.get("asked_at")
            if asked_at is not None and time.time() - asked_at > cfg.CONFIRM_TTL_SEC:
                print(f"[ACTION confirm_ttl] 만료 -> abandon", flush=True)
                return abandon(f"승인 유효시간({cfg.CONFIRM_TTL_SEC}초)이 지나 "
                               "실행하지 않았습니다. 필요하면 명령을 다시 요청해 주세요.")

            verdict = classify_confirm(answer, action=sc.get("action"),
                                       params=sc.get("params"),
                                       config=config, model_name=model_name)
            if verdict == "approve":
                return await execute()
            if verdict == "reject":
                return abandon("사용자가 실행을 거절해 명령을 종료합니다.")

            # 판정 불가 — 취소/이탈부터 가리고, 아니면 정정 시도로 본다
            kind = classify_collect_answer(
                "승인 여부", answer, sc.get("action"),
                question="이 명령을 정말 실행할까요? (승인/거절)",
                config=config, model_name=model_name)["kind"]
            if kind == "cancel":
                return abandon("사용자가 실행을 취소해 명령을 종료합니다.")
            if kind == "switch":
                return restart(answer)
            harvest = await run_action_agent(
                f"""진행 중인 명령: {sc.get('action')} (확보한 값: {sc.get('params')})
실행 승인을 물었더니 사용자가 승인 대신 이렇게 답했다 — 값을 고치려는 것으로 보인다:
{answer}
정정된 값을 판독해 param_check 와 validate 까지 수행하라.""",
                config=config, model_name=model_name)
            merged = dict(sc.get("params") or {})
            merged.update(harvest["params"])          # 정정이므로 새 값이 이긴다
            sc["params"] = merged

        else:   # collect_param
            field = awaiting.get("field")
            kind_r = classify_collect_answer(field, answer, sc.get("action"),
                                             question=awaiting.get("prompt"),
                                             config=config, model_name=model_name)
            kind = kind_r["kind"]
            if kind == "cancel":
                return abandon("사용자 요청으로 명령을 취소했습니다.")
            if kind == "switch":
                return restart(answer)
            if kind == "consult":
                sc["consult_text"] = answer
            elif kind == "empty":
                sc["collect_retries"] = sc.get("collect_retries", 0) + 1
                sc["last_parse_error"] = kind_r.get("note", "")
            elif field == "action":
                harvest = await run_action_agent(answer, config=config,
                                                 model_name=model_name)
                if harvest["action"]:
                    sc["action"] = harvest["action"]
                    merged = dict(harvest["params"])
                    merged.update({k: v for k, v in (sc.get("params") or {}).items() if v})
                    sc["params"] = merged
                else:
                    sc["collect_retries"] = sc.get("collect_retries", 0) + 1
                    sc["last_parse_error"] = "답변에서 명령을 알아내지 못했습니다."
            else:
                harvest = await run_action_agent(
                    f"""진행 중인 명령: {sc.get('action')} (확보한 값: {sc.get('params')})
'{field}' 값을 물었고 사용자가 이렇게 답했다:
{answer}
이 답에서 {field} 값을 판독해 param_check 와 validate 까지 수행하라.
판독되지 않는 ID 면 지어내지 말고 빈 값으로 보고하라.""",
                    config=config, model_name=model_name)
                merged = dict(sc.get("params") or {})
                merged.update(harvest["params"])      # 새로 받은 값이 이긴다
                sc["params"] = merged

    # ③ 진행 중 재진입 (새 발화 없음) — 하던 질문을 다시 조립한다
    elif sc.get("phase") in ACTIVE_PHASES:
        print(f"[ACTION entry] 재진입 (phase={sc.get('phase')})", flush=True)
        sc.pop("awaiting", None)

    # ④ 신규 진입
    else:
        text = last_user_text(msgs)
        print(f"[ACTION entry] 신규 진입: '{text}'", flush=True)
        # ExtractAgent 가 이번 턴에 선행 판독해 둔 결과를 재료로 넘긴다 —
        # 상류 에이전트의 툴 작업 재사용이지, 노드가 툴을 대신 부르는 게 아니다
        fx = (state.get("facts") or {}).get("extracted") or {}
        task = text
        if fx.get("carrier_ids") or fx.get("eqp_ids"):
            task = (f"{text}\n[이번 턴 판독 결과] carrier_ids={fx.get('carrier_ids')}, "
                    f"eqp_ids={fx.get('eqp_ids')} — 이 값은 이미 DB 로 확인된 것이다.")
        harvest = await run_action_agent(task, config=config, model_name=model_name,
                                         extracted=fx)
        sc = {"action": harvest["action"],
              "phase": "param_check",
              "params": dict(harvest["params"]),
              "missing": [],
              # 필수값이 비면 원문을 들고 무조건 Supervisor 상담부터 —
              # 간접 표현("~있는 위치로")의 해석은 Supervisor 소관이다.
              "consult_text": text,
              "collect_retries": 0, "validate_retries": 0, "hops": 0}

    if harvest and harvest.get("summary"):
        sc["agent_summary"] = harvest["summary"]

    # ── 턴의 끝 정하기 (단일 패스) ──────────────────────────────────────────
    #
    # 판단은 이미 끝났다. 여기서는 스크래치 사실만 보고 다음 한 수를 정한다.

    # 명령 불명 -> 상담 대상이 아니다(명령 종류는 사용자의 의도) -> 직접 질문
    if not sc.get("action"):
        sc.pop("consult_text", None)
        if sc.get("collect_retries", 0) >= cfg.MAX_COLLECT:
            return abandon("필수 파라미터를 수집하지 못해 요청을 종료합니다.")
        return ask_param("action")

    required = _prompt.action_catalog()[sc["action"]]["required_params"]
    sc["missing"] = [p for p in required if not (sc.get("params") or {}).get(p)]

    if sc["missing"]:
        fill = sc["missing"][0]
        if sc.get("consult_text") and sc.get("hops", 0) < cfg.MAX_HOPS:
            return needs_exit(fill)
        sc.pop("consult_text", None)
        if sc.get("collect_retries", 0) >= cfg.MAX_COLLECT:
            return abandon("필수 파라미터를 수집하지 못해 요청을 종료합니다.")
        return ask_param(fill)

    # 필수값 충족 — 검증 결과 확인.
    # 에이전트의 검증 결과는 '그때 그 값' 기준이다. 병합으로 값이 달라졌으면
    # 낡은 결과이므로 버리고 게이트를 태운다.
    v = (harvest or {}).get("valid")
    if v is not None and dict((harvest or {}).get("params") or {}) != dict(sc.get("params") or {}):
        v = None
    if v is None:
        # 이번 진입의 에이전트가 검증까지 못 갔다(수집 직후 재조립 등).
        # 실행 직전 게이트와 같은 성격으로 여기서 한 번 태운다 [안전 게이트].
        v = await getattr(_tool, f"{sc['action']}_validate_tool").ainvoke(
            {"params": sc["params"]}, config=config)

    if v.get("ok"):
        return ask_confirm()

    sc["validate_retries"] = sc.get("validate_retries", 0) + 1
    if sc["validate_retries"] >= cfg.MAX_VALIDATE:
        return abandon(f"유효성 검증에 반복 실패해 요청을 종료합니다. (사유: {v.get('reason')})")
    for bad in v.get("bad_fields", []):
        sc["params"].pop(bad, None)
    # 검증에서 튕긴 값은 직접 읽어 넣었던 값 — 원문 상담으로 캘 게 없다
    sc.pop("consult_text", None)
    sc["last_parse_error"] = f"검증 실패: {v.get('reason')}"
    remaining = [p for p in required if not sc["params"].get(p)]
    return ask_param(remaining[0] if remaining else "action")
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
