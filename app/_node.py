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

    # 0) needs-핸드오프 — ActionAgent 가 요청한 동료에게 먼저 보낸다
    needs = sc.get("needs")
    if needs and not sc.get("needs_result"):
        nxt = needs.get("agent")
        if nxt in members:
            print(f"[NODE] Supervisor: needs 감지 -> {nxt} (fill={needs.get('fill')})", flush=True)
            emit(config, "agent_status",
                 {"agent": "Supervisor", "detail": f"needs 라우팅 -> {nxt}"})
            return {"next": nxt, "step": step + 1}

    # 1) 헬퍼가 값을 채워줬으면 ActionAgent 로 돌아간다
    if needs and sc.get("needs_result"):
        print("[NODE] Supervisor: 헬퍼 결과 도착 -> ActionAgent 재진입", flush=True)
        emit(config, "agent_status", {"agent": "Supervisor", "detail": "ActionAgent 재진입"})
        return {"next": "ActionAgent", "step": step + 1}

    # 2) 진행 중인 액션이 있으면 계속 ActionAgent
    if sc.get("phase") in ("param_check", "collecting", "validating", "confirming"):
        print("[NODE] Supervisor: 진행 중 액션 -> ActionAgent", flush=True)
        return {"next": "ActionAgent", "step": step + 1}

    # 3) ExtractAgent 선행 실행
    #    사용자 질의가 들어오면 항상 Supervisor 부터 다시 시작하고,
    #    Supervisor 는 그 턴에 Extract 가 안 돌았으면 무조건 먼저 태운다.
    #    -> 뒤에 오는 워커들은 추출된 ID 를 재료로 쓸 수 있다.
    if not agent_ran_this_turn(messages, "ExtractAgent"):
        print("[NODE] Supervisor: ExtractAgent 선행 실행", flush=True)
        emit(config, "agent_status",
             {"agent": "Supervisor", "detail": "ExtractAgent 선행 실행"})
        return {"next": "ExtractAgent", "step": step + 1}

    # 4) 이번 턴에 워커가 이미 답을 냈으면 마무리
    #    (ExtractAgent 는 답변 워커가 아니라 여기서 제외된다)
    if member_answered_this_turn(messages, ANSWERING_MEMBERS):
        print("[NODE] Supervisor: member 응답 완료 -> FinalAnswerAgent", flush=True)
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

def _serve_needs(sc: dict, agent: str, value, note: str = "") -> dict:
    """needs 메일박스를 채운다.

    ActionAgent 가 재진입할 때 param_check 가 이걸 회수해서 파라미터로 쓴다.
    """
    out = dict(sc)
    out["needs_result"] = {"value": value, "by": agent, "note": note}
    return out


def location_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """캐리어 위치 조회. needs-핸드오프 서비스도 겸한다."""
    print("[NODE] LocationAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LocationAgent", "detail": "위치 조회"})

    sc = state.get("action") or {}
    needs = sc.get("needs")
    model = model_name or _model_of(state)

    # --- needs-핸드오프 서비스 모드: ActionAgent 대신 값을 찾아준다
    if needs and needs.get("agent") == "LocationAgent":
        carrier = needs.get("carrier_id")

        emit(config, "tool_call", {"agent": "LocationAgent", "tool": "location_search_tool",
                                   "args": {"carrier_id": carrier}})
        loc = mock_db.get_carrier_location(carrier) if carrier else None

        note = "" if loc else f"캐리어 {carrier} 의 위치를 찾을 수 없습니다."
        content = (f"[LocationAgent] 캐리어 {carrier} 위치: {loc}" if loc
                   else f"[LocationAgent] {note}")
        fake_llm_echo("location", content, config=config, model_name=model)

        return {
            "messages": [AIMessage(content=content, name="LocationAgent")],
            "facts": {carrier: {"eqp_id": loc}} if loc else {},
            "action": _serve_needs(sc, "LocationAgent", loc, note),
            "step": state.get("step", 0) + 1,
        }

    # --- 일반 위치 질의 모드
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
    """반송 이력 분석. needs-핸드오프 서비스도 겸한다."""
    print("[NODE] LogAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LogAgent", "detail": "반송 이력 분석"})

    sc = state.get("action") or {}
    needs = sc.get("needs")
    model = model_name or _model_of(state)

    # 분석 대상 캐리어 결정: needs > 발화 내 ID > 전체
    if needs and needs.get("agent") == "LogAgent":
        carrier = needs.get("carrier_id") or (sc.get("params") or {}).get("carrier_id")
    else:
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

    out = {
        "messages": [AIMessage(content=content, name="LogAgent")],
        "facts": {"log_analysis": analysis},
        "step": state.get("step", 0) + 1,
    }

    # needs 로 불려온 경우 메일박스도 채워준다
    if needs and needs.get("agent") == "LogAgent":
        value = analysis["recommended_dest"]
        note = "" if value else "로그에서 권장 목적지를 도출하지 못했습니다."
        out["action"] = _serve_needs(sc, "LogAgent", value, note)

    return out


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

    content = (f"[ExtractAgent] fab={fab}, "
               f"carrier_ids={carriers or '없음'}, eqp_ids={eqps or '없음'}")
    fake_llm_echo("extract", content, config=config, model_name=model)

    return {
        "messages": [AIMessage(content=content, name="ExtractAgent")],
        # 추출 결과는 facts 에도 넣어둔다 (limiter 에 안 잘리는 공유 팩트)
        "facts": {"extracted": {"fab": fab, "carrier_ids": carriers, "eqp_ids": eqps}},
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
