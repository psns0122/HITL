"""그래프 노드들.

Router/Supervisor/Final* 은 사내 코드(shared_code.md) 형태를 유지하고,
Location/Status/Log/Extract 는 목업 스텁(단, needs-핸드오프 계약과 facts 적재는
실제로 동작)이다. ActionAgent 는 actions/graph.py 의 서브그래프가 담당한다.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app import _agent, _state
from app._util import emit, fake_llm_echo, last_human_text, member_answered_this_turn
from app.actions import mock_db, resolvers

members = ["StatusAgent", "LocationAgent", "LogAgent", "ActionAgent", "ExtractAgent"]
options_for_next = members + ["FinalAnswerAgent", "FINISH"]
options_lower_map = {m.lower().replace("_", "").replace("-", ""): m for m in options_for_next}

# Supervisor 무한 순환 방지 상한 (한 user turn 내 노드 스텝)
MAX_SUPERVISOR_STEPS = 12


# ── Router ────────────────────────────────────────────────────────────────

def router_node(state: _state.AgentState, config) -> dict:
    messages = state.get("messages", []) or []
    if messages and isinstance(messages[-1], HumanMessage):
        print(f"[USER] {messages[-1].content}", flush=True)

    print("[NODE] Router entered", flush=True)
    emit(config, "agent_status", {"agent": "Router", "detail": "의도 분류 중"})

    # HITL 재개(resume) 중이 아닌 새 질의만 여기로 온다.
    # 단, 진행 중 액션이 있으면 무조건 Supervisor (액션 이어가기).
    if (state.get("action") or {}).get("phase"):
        print("[NODE] Router: 진행 중 액션 감지 -> Supervisor 고정", flush=True)
        return {"route": "supervisor", "handoff": False, "next": "Supervisor", "step": 1}

    result = _agent.router_agent({"messages": messages, "config": config})
    route = result.get("route", "supervisor")
    next_node = "Supervisor" if route == "supervisor" else "GeneralAgent"

    return {"route": route, "handoff": False, "next": next_node, "step": 1}


# ── General ───────────────────────────────────────────────────────────────

def general_node(state: _state.AgentState, config) -> dict:
    print("[NODE] GeneralAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "GeneralAgent", "detail": "일반 질의 처리"})
    # 목업: 곧장 최종 일반 응답으로. (사내에선 툴 필요시 Supervisor 로 넘길 수 있음)
    return {"next": "FINISH", "step": state.get("step", 0) + 1}


# ── Supervisor ────────────────────────────────────────────────────────────

def supervisor_node(state: _state.AgentState, config) -> dict:
    print("[NODE] Supervisor entered", flush=True)
    sc = state.get("action") or {}
    step = state.get("step", 0)

    # 0) 결정적 우선순위 — LLM 판단보다 먼저 (needs-핸드오프 §1-E b)
    needs = sc.get("needs")
    if needs and not sc.get("needs_result"):
        nxt = needs.get("agent")
        if nxt in members:
            print(f"[NODE] Supervisor: needs 감지 -> {nxt} (fill={needs.get('fill')})", flush=True)
            emit(config, "agent_status",
                 {"agent": "Supervisor", "detail": f"needs 라우팅 -> {nxt}"})
            return {"next": nxt, "step": step + 1}
    if needs and sc.get("needs_result"):
        print("[NODE] Supervisor: 헬퍼 결과 도착 -> ActionAgent 재진입", flush=True)
        emit(config, "agent_status", {"agent": "Supervisor", "detail": "ActionAgent 재진입"})
        return {"next": "ActionAgent", "step": step + 1}
    if sc.get("phase") in ("param_check", "collecting", "validating", "confirming"):
        print("[NODE] Supervisor: 진행 중 액션 -> ActionAgent", flush=True)
        return {"next": "ActionAgent", "step": step + 1}

    # 1) 이번 턴에 member 가 이미 답을 냈으면 마무리
    if member_answered_this_turn(state.get("messages", []), members):
        print("[NODE] Supervisor: member 응답 완료 -> FinalAnswerAgent", flush=True)
        return {"next": "FinalAnswerAgent", "step": step + 1}

    # 2) 스텝 상한 가드
    if step >= MAX_SUPERVISOR_STEPS:
        print(f"[WARN] Supervisor: step {step} >= {MAX_SUPERVISOR_STEPS} -> 강제 종료", flush=True)
        return {"next": "FinalAnswerAgent", "step": step + 1}

    # 3) LLM/규칙 기반 배분 (사내 supervisor_chain 자리)
    text = last_human_text(state.get("messages", []))
    next_node = "FinalAnswerAgent"
    try:
        raw_next = _agent.supervisor_agent(text, members, config=config)
        clean_next = str(raw_next).strip().replace("'", "").replace('"', "")
        if clean_next in options_for_next:
            next_node = clean_next
        else:
            normalized = clean_next.lower().replace("_", "").replace("-", "")
            if normalized in options_lower_map:
                next_node = options_lower_map[normalized]
            else:
                print(f"[WARN] Supervisor: unknown next '{raw_next}' -> FinalAnswerAgent", flush=True)
    except Exception as e:
        print(f"[ERROR] Supervisor failed: {e}", flush=True)

    if next_node == "FINISH":
        next_node = "FinalAnswerAgent"
    print(f"[NODE] Supervisor -> {next_node}", flush=True)
    return {"next": next_node, "step": step + 1}


# ── 헬퍼/조회 member 스텁들 ───────────────────────────────────────────────

def _serve_needs(sc: dict, agent: str, value, note: str = "") -> dict:
    """needs 메일박스 채우기 — ActionAgent 재진입 시 param_check 가 회수."""
    out = dict(sc)
    out["needs_result"] = {"value": value, "by": agent, "note": note}
    return out


def location_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    print("[NODE] LocationAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LocationAgent", "detail": "위치 조회"})
    sc = state.get("action") or {}
    needs = sc.get("needs")

    # needs-핸드오프 서비스 모드
    if needs and needs.get("agent") == "LocationAgent":
        carrier = needs.get("carrier_id")
        emit(config, "tool_call", {"agent": "LocationAgent", "tool": "get_carrier_location",
                                   "args": {"carrier_id": carrier}})
        loc = mock_db.get_carrier_location(carrier) if carrier else None
        note = "" if loc else f"캐리어 {carrier} 의 위치를 찾을 수 없습니다."
        content = (f"[LocationAgent] 캐리어 {carrier} 위치: {loc}" if loc
                   else f"[LocationAgent] {note}")
        fake_llm_echo("location", content, config=config)
        return {
            "messages": [AIMessage(content=content, name="LocationAgent")],
            "facts": {carrier: {"eqp_id": loc}} if loc else {},
            "action": _serve_needs(sc, "LocationAgent", loc, note),
            "step": state.get("step", 0) + 1,
        }

    # 일반 위치 질의 모드
    text = last_human_text(state.get("messages", []))
    ids = resolvers.extract_ids(text)
    lines, facts = [], {}
    for c in ids["carrier_ids"]:
        loc = mock_db.get_carrier_location(c)
        emit(config, "tool_call", {"agent": "LocationAgent", "tool": "get_carrier_location",
                                   "args": {"carrier_id": c}, "result": {"eqp_id": loc}})
        if loc:
            lines.append(f"캐리어 {c} 는 현재 {loc} 에 있습니다.")
            facts[c] = {"eqp_id": loc}
        else:
            lines.append(f"캐리어 {c} 를 찾을 수 없습니다.")
    content = "\n".join(lines) or "질의에서 캐리어 ID 를 찾지 못했습니다."
    fake_llm_echo("location", content, config=config)
    return {"messages": [AIMessage(content=content, name="LocationAgent")],
            "facts": facts, "step": state.get("step", 0) + 1}


def status_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    print("[NODE] StatusAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "StatusAgent", "detail": "상태 조회"})
    text = last_human_text(state.get("messages", []))
    ids = resolvers.extract_ids(text)
    lines = []
    for c in ids["carrier_ids"]:
        info = mock_db.get_carrier(c)
        emit(config, "tool_call", {"agent": "StatusAgent", "tool": "get_carrier",
                                   "args": {"carrier_id": c}})
        lines.append(f"캐리어 {c}: 상태={info['status']}, 위치={info['current_eqp']}, "
                     f"LOT={info['lot']}" if info else f"캐리어 {c} 를 찾을 수 없습니다.")
    content = "\n".join(lines) or "질의에서 캐리어 ID 를 찾지 못했습니다."
    fake_llm_echo("status", content, config=config)
    return {"messages": [AIMessage(content=content, name="StatusAgent")],
            "step": state.get("step", 0) + 1}


def log_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    print("[NODE] LogAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LogAgent", "detail": "반송 이력 분석"})
    sc = state.get("action") or {}
    needs = sc.get("needs")

    # 분석 대상 캐리어: needs > 발화 내 ID > 전체
    if needs and needs.get("agent") == "LogAgent":
        carrier = needs.get("carrier_id") or (sc.get("params") or {}).get("carrier_id")
    else:
        text = last_human_text(state.get("messages", []))
        ids = resolvers.extract_ids(text)
        carrier = ids["carrier_ids"][0] if ids["carrier_ids"] else None

    emit(config, "tool_call", {"agent": "LogAgent", "tool": "analyze_transport_logs",
                               "args": {"carrier_id": carrier}})
    analysis = mock_db.analyze_transport_logs(carrier)

    combo_lines = [f"  - {c['eqp']} {c['reason']}/{c['description']} x{c['count']} "
                   f"(최초 {c['first_t']})" for c in analysis["combos"]]
    content = "\n".join(
        [f"[LogAgent] 반송 이력 분석 결과 (carrier={carrier or '전체'})",
         f"- 에러 콤보 {len(analysis['combos'])}건:"] + combo_lines +
        [f"- 원인 장비: {analysis['cause_eqp']}",
         f"- 권장 대체 목적지: {analysis['recommended_dest']}"])
    fake_llm_echo("log", content, config=config)

    out = {"messages": [AIMessage(content=content, name="LogAgent")],
           "facts": {"log_analysis": analysis},
           "step": state.get("step", 0) + 1}
    if needs and needs.get("agent") == "LogAgent":
        value = analysis["recommended_dest"]
        note = "" if value else "로그에서 권장 목적지를 도출하지 못했습니다."
        out["action"] = _serve_needs(sc, "LogAgent", value, note)
    return out


def extract_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    print("[NODE] ExtractAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "ExtractAgent", "detail": "추출(목업)"})
    content = "[ExtractAgent] 추출 기능은 목업입니다. (사내 구현으로 교체 지점)"
    fake_llm_echo("extract", content, config=config)
    return {"messages": [AIMessage(content=content, name="ExtractAgent")],
            "step": state.get("step", 0) + 1}


# ── Final answer 노드들 (shared_code.md 형태 유지, 스트리밍) ──────────────

def _final_guard(content: str, finish: str | None) -> str:
    is_empty = not (content or "").strip()
    is_weird_finish = finish not in ("stop", None)
    if is_empty or is_weird_finish:
        print("응답 비정상", flush=True)
        return "응답 생성이 실패했음"
    return content


async def final_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    print("[NODE] FinalAnswer entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalAnswerAgent", "detail": "최종 응답 생성"})
    question = last_human_text(state.get("messages", []))
    member_msg = member_answered_this_turn(state.get("messages", []), members)
    context = str(member_msg.content) if member_msg else \
        "처리 결과가 없습니다. 질문에 답할 수 있는 범위에서 안내하세요."

    content = await _agent.stream_final_answer(context, question, config=config, role="final")
    content = _final_guard(content, "stop")
    return {"messages": [AIMessage(content=content, name="FinalAnswerAgent")],
            "next": "END", "step": state.get("step", 0) + 1}


async def final_general_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    print("[NODE] FinalGeneral entered", flush=True)
    emit(config, "agent_status", {"agent": "FinalGeneralAgent", "detail": "일반 응답 생성"})
    question = last_human_text(state.get("messages", []))
    context = ("안녕하세요! AMHS 반송 시스템 챗봇입니다. 캐리어 위치/상태 조회, 반송 이력 분석, "
               "반송요청명령·목적지요청 실행을 도와드릴 수 있어요.")

    content = await _agent.stream_final_answer(context, question, config=config, role="final_general")
    content = _final_guard(content, "stop")
    return {"messages": [AIMessage(content=content, name="FinalGeneralAgent")],
            "next": "END", "step": state.get("step", 0) + 1}
