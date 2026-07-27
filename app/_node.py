"""그래프 노드들.

Router / Supervisor / Final* 은 사내 코드(shared_code.md) 형태를 유지하고,
Location / Status / Log / Extract 는 목업 스텁이다.
(단, needs-핸드오프 계약과 facts 적재는 실제로 동작한다.)

ActionAgent 는 actions/graph.py 의 HITL 서브그래프가 담당한다.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app import _agent, _state
from app._util import (
    agent_ran_this_turn,
    emit,
    fake_llm_echo,
    last_user_text,
    member_answered_this_turn,
)
from app.actions import mock_db, resolvers

# Supervisor 밑에 붙는 워커들
members = ["StatusAgent", "LocationAgent", "LogAgent", "ActionAgent", "ExtractAgent"]

# ExtractAgent 는 '답변'을 내는 워커가 아니라 재료를 뽑는 선행 단계다.
# 그래서 "이번 턴에 워커가 답을 냈나?" 판정에서는 빼야 한다.
# (안 빼면 Extract 가 돌자마자 턴이 끝나버린다.)
ANSWERING_MEMBERS = [m for m in members if m != "ExtractAgent"]

options_for_next = members + ["FinalAnswerAgent", "FINISH"]
options_lower_map = {m.lower().replace("_", "").replace("-", ""): m for m in options_for_next}

# Supervisor 무한 순환 방지 상한 (한 user turn 내 노드 스텝)
MAX_SUPERVISOR_STEPS = 12

# needs-핸드오프에 별도 배분표는 없다.
#
# ActionAgent 는 "내가 이렇게 물었고(question) / 사용자가 이렇게 답했고
# (answer) / 나는 이 값이 필요하다(fill)" 원문만 넘긴다. 어느 워커가 그걸
# 풀 수 있는지는 Supervisor 가 평소 배분에 쓰는 것과 같은 로스터(members +
# 프롬프트의 워커 설명)를 보고 LLM 으로 판단한다(_agent.needs_dispatch).
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

    # HITL 재개가 아닌 새 질의만 여기로 온다.
    # 단, 진행 중 액션이 있으면 분류할 것도 없이 Supervisor 로 고정한다.
    if (state.get("action") or {}).get("phase"):
        print("[NODE] Router: 진행 중 액션 감지 -> Supervisor 고정", flush=True)
        return {"route": "supervisor", "handoff": False, "next": "Supervisor", "step": 1}

    result = await _agent.router_agent({
        "messages": messages,
        "model_name": _model_of(state),
    })

    route = result.get("route", "supervisor")
    next_node = "Supervisor" if route == "supervisor" else "GeneralAgent"

    return {"route": route, "handoff": False, "next": next_node, "step": 1}


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent
# ─────────────────────────────────────────────────────────────────────────

async def general_node(state: _state.AgentState, config) -> dict:
    """일반 대화. 업무 질의로 재판정되면 Supervisor 로 넘긴다."""
    print("[NODE] GeneralAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "GeneralAgent", "detail": "일반 질의 처리"})

    result = await _agent.general_agent({
        "messages": state.get("messages", []) or [],
        "model_name": _model_of(state),
    })

    # handoff=True 면 Supervisor 로, 아니면 FinalGeneralAgent 로 마무리
    if result.get("handoff"):
        return {"handoff": True, "route": "supervisor",
                "next": "Supervisor", "step": state.get("step", 0) + 1}

    return {"handoff": False, "route": "general",
            "next": "FINISH", "step": state.get("step", 0) + 1}


# ─────────────────────────────────────────────────────────────────────────
# Supervisor
# ─────────────────────────────────────────────────────────────────────────

def supervisor_node(state: _state.AgentState, config) -> dict:
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
        d = _agent.needs_dispatch(needs, members,
                                  config=config, model_name=_model_of(state))
        nxt = d.get("agent")

        if nxt in members:
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

    # 3-b) Extract 게이트: 추출된 게 없으면 워커를 안 돌리고 바로 Final (항목 6)
    #      진행 중 액션이 없을 때만. (Action 은 위 2)에서 이미 걸러졌다.)
    extracted = (state.get("facts") or {}).get("extracted") or {}
    if extracted and not extracted.get("gate_pass"):
        print("[NODE] Supervisor: Extract 게이트 실패 -> FinalAnswerAgent", flush=True)
        emit(config, "agent_status",
             {"agent": "Supervisor", "detail": "추출 결과 없음 -> 바로 응답"})
        return {"next": "FinalAnswerAgent", "step": step + 1}

    # 5) 스텝 상한 가드
    if step >= MAX_SUPERVISOR_STEPS:
        print(f"[WARN] Supervisor: step {step} >= {MAX_SUPERVISOR_STEPS} -> 강제 종료", flush=True)
        return {"next": "FinalAnswerAgent", "step": step + 1}

    # 6) LLM/규칙 기반 배분 (사내 supervisor_chain 자리)
    text = last_user_text(messages)
    next_node = "FinalAnswerAgent"

    try:
        raw_next = _agent.supervisor_agent(
            text, members, config=config, model_name=_model_of(state))

        clean_next = str(raw_next).strip().replace("'", "").replace('"', "")

        if clean_next in options_for_next:
            next_node = clean_next
        else:
            # 대소문자/구분자만 다른 경우를 구제한다
            normalized = clean_next.lower().replace("_", "").replace("-", "")
            if normalized in options_lower_map:
                next_node = options_lower_map[normalized]
            else:
                print(f"[WARN] Supervisor: unknown next '{raw_next}' -> FinalAnswerAgent",
                      flush=True)

    except Exception as e:
        print(f"[ERROR] Supervisor failed: {e}", flush=True)

    if next_node == "FINISH":
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

    model = model_name or _model_of(state)

    text = last_user_text(state.get("messages", []))
    ids = resolvers.extract_ids(text)

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
    fake_llm_echo("location", content, config=config, model_name=model)

    return {
        "messages": [AIMessage(content=content, name="LocationAgent")],
        "facts": facts,
        "step": state.get("step", 0) + 1,
    }


def status_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """캐리어/설비 상태 조회."""
    print("[NODE] StatusAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "StatusAgent", "detail": "상태 조회"})

    text = last_user_text(state.get("messages", []))
    ids = resolvers.extract_ids(text)
    model = model_name or _model_of(state)

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
    fake_llm_echo("status", content, config=config, model_name=model)

    return {
        "messages": [AIMessage(content=content, name="StatusAgent")],
        "step": state.get("step", 0) + 1,
    }


def log_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """반송 이력 분석.

    needs-핸드오프 관련 코드는 없다 (location_node 와 동일한 이유).
    """
    print("[NODE] LogAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LogAgent", "detail": "반송 이력 분석"})

    model = model_name or _model_of(state)

    # 분석 대상 캐리어: 발화 내 ID > 전체
    text = last_user_text(state.get("messages", []))
    ids = resolvers.extract_ids(text)
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
    fake_llm_echo("log", content, config=config, model_name=model)

    return {
        "messages": [AIMessage(content=content, name="LogAgent")],
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

    # params_extract_tool 상당 — 발화에서 ID 를 뽑는다
    ids = resolvers.extract_ids(text)
    emit(config, "tool_call", {"agent": "ExtractAgent", "tool": "params_extract_tool",
                               "args": {"text": text}, "result": ids})

    carriers = ids.get("carrier_ids") or []
    eqps = ids.get("eqp_ids") or []

    # 게이트 판정: 이후 워커가 쓸 재료가 하나라도 있는가.
    #   ID 가 있거나 / 명령 의도가 있거나 / 다른 에이전트 영역 키워드가 있으면 통과.
    #   아무것도 없으면 통과 실패 -> Supervisor 가 워커를 안 돌리고 바로 Final 로 보낸다(항목 6).
    gate_pass = bool(
        carriers or eqps
        or resolvers.detect_intent(text)
        or resolvers.CONTEXT_SWITCH_RE.search(text or "")
    )

    content = (f"[ExtractAgent] fab={fab}, "
               f"carrier_ids={carriers or '없음'}, eqp_ids={eqps or '없음'} "
               f"(gate={'통과' if gate_pass else '실패'})")
    fake_llm_echo("extract", content, config=config, model_name=model)

    return {
        "messages": [AIMessage(content=content, name="ExtractAgent")],
        # 추출 결과는 facts 에도 넣어둔다 (limiter 에 안 잘리는 공유 팩트)
        "facts": {"extracted": {"fab": fab, "carrier_ids": carriers,
                                "eqp_ids": eqps, "gate_pass": gate_pass}},
        "step": state.get("step", 0) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 노드 (스트리밍)
# ─────────────────────────────────────────────────────────────────────────

async def final_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """워커 결과를 받아 최종 답변을 스트리밍으로 만든다."""
    print("[NODE] FinalAnswer entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalAnswerAgent", "detail": "최종 응답 생성"})

    model = model_name or _model_of(state)
    messages = state.get("messages", []) or []

    # 이번 턴에 워커가 낸 답을 컨텍스트로 쓴다
    member_msg = member_answered_this_turn(messages, ANSWERING_MEMBERS)
    context = (str(member_msg.content) if member_msg
               else "처리 결과가 없습니다. 질문에 답할 수 있는 범위에서 안내하세요.")

    agent = _agent.create_final_agent(model_name=model)
    content = await agent(state, config=config, context=context)

    return {
        "messages": [AIMessage(content=content, name="FinalAnswerAgent")],
        "next": "END",
        "step": state.get("step", 0) + 1,
    }


async def final_general_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """일반 대화의 최종 답변. final_node 와 같은 모양이다."""
    print("[NODE] FinalGeneral entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalGeneralAgent", "detail": "일반 응답 생성"})

    model = model_name or _model_of(state)

    context = ("안녕하세요! AMHS 반송 시스템 챗봇입니다. "
               "캐리어 위치/상태 조회, 반송 이력 분석, "
               "반송요청명령·목적지요청 실행을 도와드릴 수 있어요.")

    agent = _agent.create_final_general_agent(model_name=model)
    content = await agent(state, config=config, context=context)

    return {
        "messages": [AIMessage(content=content, name="FinalGeneralAgent")],
        "next": "END",
        "step": state.get("step", 0) + 1,
    }
