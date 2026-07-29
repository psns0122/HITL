"""그래프 노드들.

Router / Supervisor / Final* 은 사내 코드(shared_code.md) 형태를 유지하고,
Location / Status / Log / Extract 는 목업 스텁이다.
(단, needs-핸드오프 계약과 facts 적재는 실제로 동작한다.)

ActionAgent 는 actions/node.py 의 턴 기반 단일 노드가 담당한다.
"""
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from app import _agent, _llm, _prompt, _state
from app._util import (
    action_service,
    agent_ran_this_turn,
    emit,
    last_user_text,
    member_answered_this_turn,
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
# Router
# ─────────────────────────────────────────────────────────────────────────

async def router_node(state: _state.AgentState, config) -> dict:
    """일반 질의면 GeneralAgent, 업무 질의면 Supervisor 로 보낸다."""
    messages = state.get("messages", []) or []

    if messages and isinstance(messages[-1], HumanMessage):
        print(f"[USER] {messages[-1].content}", flush=True)

    print("[NODE] Router entered", flush=True)
    emit(config, "agent_status", {"agent": "Router", "detail": "의도 분류 중"})

    # [app 추가] HITL 진행 중이면 분류할 것도 없이 Supervisor 로 고정한다.
    # (HITL 답변이 general 로 오분류되면 진행 중 액션이 고아가 되기 때문)
    if (state.get("action") or {}).get("phase"):
        print("[NODE] Router: 진행 중 액션 감지 -> Supervisor 고정", flush=True)
        return {"route": "supervisor", "handoff": True, "next": "Supervisor", "step": 1}

    # 사내 원본과 동일: 에이전트가 route 를 판단하고,
    # 노드가 그 값을 노드 이름으로 바꿔 next 에 싣는다.
    result = await _agent.router_agent({
        "messages": state.get("messages", []),
        "model_name": state.get("model_name"),
    })

    route = result.get("route", "supervisor")
    if route == "supervisor":
        next_node = "Supervisor"
    else:
        next_node = "GeneralAgent"

    return {"route": route, "handoff": False, "next": next_node, "step": 1}


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent
# ─────────────────────────────────────────────────────────────────────────

async def general_node(state: _state.AgentState, config) -> dict:
    """일반 대화. 업무 질의로 재판정되면 Supervisor 로 넘긴다."""
    print("[NODE] GeneralAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "GeneralAgent", "detail": "일반 질의 처리"})

    result = await _agent.general_agent(state)

    # handoff=True 면 Supervisor 로, 아니면 FinalGeneralAgent 로 마무리
    next_node = "Supervisor" if result.get("handoff") else "FINISH"

    return {**result, "next": next_node, "step": state.get("step", 0) + 1}


# ─────────────────────────────────────────────────────────────────────────
# Supervisor
# ─────────────────────────────────────────────────────────────────────────

async def supervisor_node(state: _state.AgentState, config) -> dict:
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

    # 2) 진행 중인 액션이 있으면 계속 ActionAgent
    #    (awaiting 상태에서 새 사용자 발화가 들어온 경우도 여기로 온다 —
    #     Router 가 진행 중 액션을 보고 Supervisor 로 고정해 준다)
    if sc.get("phase") in ("param_check", "collecting", "validating", "confirming"):
        print("[NODE] Supervisor: 진행 중 액션 -> ActionAgent", flush=True)
        return {"next": "ActionAgent", "step": step + 1}

    # 3) 이번 턴에 워커가 이미 답을 냈으면 곧장 마무리
    #    (예: HITL 답변 턴에서 ActionAgent 가 finalize/abandon 을 낸 직후 —
    #     이때 Extract 선행을 태우는 건 낭비다)
    if member_answered_this_turn(messages, ANSWERING_MEMBERS):
        print("[NODE] Supervisor: member 응답 완료 -> FinalAnswerAgent", flush=True)
        return {"next": "FinalAnswerAgent", "step": step + 1}

    # 3-a) ExtractAgent 선행 실행
    #    사용자 질의가 들어오면 항상 Supervisor 부터 다시 시작하고,
    #    Supervisor 는 그 턴에 Extract 가 안 돌았으면 무조건 먼저 태운다.
    #    -> 뒤에 오는 워커들은 추출된 ID 를 재료로 쓸 수 있다.
    if not agent_ran_this_turn(messages, "ExtractAgent"):
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

def location_node(state: _state.AgentState, config, model_name: str = None) -> dict:
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


def status_node(state: _state.AgentState, config, model_name: str = None) -> dict:
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


def log_node(state: _state.AgentState, config, model_name: str = None) -> dict:
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


def extract_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """FAB / 파라미터 추출.

    Supervisor 진입 후 항상 가장 먼저 실행된다(모든 워커에 선행).
    여기서 뽑은 ID 들이 뒤 단계의 재료가 된다.
    """
    print("[NODE] ExtractAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "ExtractAgent", "detail": "FAB/파라미터 추출"})

    text = last_user_text(state.get("messages", []))
    model = model_name or _model_of(state)

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


# *************  [app — origin 의 action_node(react agent) 를 통째 교체]  *************
async def action_node(state: _state.AgentState, config) -> dict:
    """명령 실행 — 턴 기반 HITL 상태기계 (_util.ActionService 에 위임)."""
    return await action_service.action_node(state, config)
# *************


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 노드 (스트리밍)
# ─────────────────────────────────────────────────────────────────────────

async def final_node(state: _state.AgentState, config, model_name: str = None) -> dict:
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
        model_name=model_name or _model_of(state))(state, config, context)

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


async def final_general_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """일반 대화의 최종 답변. final_node 와 같은 모양이다."""
    print("[NODE] FinalGeneral entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalGeneralAgent", "detail": "일반 응답 생성"})

    out = await _agent.create_final_general_agent(
        model_name=model_name or _model_of(state))(state, config)

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
