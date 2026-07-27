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

# ── needs-핸드오프 능력표 ────────────────────────────────────────────────
#
# ActionAgent 는 "carrier_location 이 필요하다" 고 종류만 말한다.
# 누가 처리하는지, 그 에이전트에게 어떻게 물어야 하는지, 답에서 값을 어떻게
# 꺼내는지는 전부 여기 한 곳에 모여 있다. 에이전트 본문에는 아무것도 없다.
#
#   agent : 담당 워커 (Supervisor 가 배분할 때 본다)
#   ask   : needs -> 그 워커가 평소 받는 형태의 질의문
#           (핸드오프 재개 시 사용자의 HITL 답변은 HumanMessage 로 남지 않아
#            대화록만 봐선 조회 대상을 알 수 없다. 그래서 어댑터가 needs 를
#            평범한 질의로 '번역'해 넣어준다 — 워커는 평소처럼 동작하면 된다)
#   pick  : 워커가 돌려준 dict -> 채울 값 (없으면 None)
#
# 새 참조 종류를 붙이거나 담당을 바꿀 때 이 표에만 줄을 추가하면 된다.
# 매핑이 없는 종류는 '처리할 동료 없음' 으로 ActionAgent 에 돌려보내고,
# ActionAgent 는 사용자에게 직접 묻는 쪽으로 강등한다.
NEEDS_CAPABILITIES = {
    "carrier_location": {
        "agent": "LocationAgent",
        "ask": lambda n: f"{n.get('carrier_id')} 위치 알려줘",
        "pick": lambda out, n: (
            (out.get("facts") or {}).get(n.get("carrier_id")) or {}).get("eqp_id"),
    },
    "log_analysis": {
        "agent": "LogAgent",
        "ask": lambda n: f"{n.get('carrier_id') or ''} 반송 로그 분석해줘".strip(),
        "pick": lambda out, n: (
            (out.get("facts") or {}).get("log_analysis") or {}).get("recommended_dest"),
    },
}

# kind -> 담당 워커 (Supervisor 배분용 뷰)
NEEDS_ROUTER = {k: v["agent"] for k, v in NEEDS_CAPABILITIES.items()}


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

    # 0) needs-핸드오프 — 요청된 '종류'를 보고 담당 동료를 Supervisor 가 고른다
    needs = sc.get("needs")
    if needs and not sc.get("needs_result"):
        kind = needs.get("kind")
        nxt = NEEDS_ROUTER.get(kind)

        if nxt in members:
            print(f"[NODE] Supervisor: needs({kind}) -> {nxt} "
                  f"(fill={needs.get('fill')})", flush=True)
            emit(config, "agent_status",
                 {"agent": "Supervisor", "detail": f"needs 배분 {kind} -> {nxt}"})
            return {"next": nxt, "step": step + 1}

        # 처리할 동료가 없다 -> 빈 결과를 채워 ActionAgent 에 돌려보낸다.
        # (ActionAgent 가 사용자에게 직접 묻는 쪽으로 강등한다)
        print(f"[NODE] Supervisor: needs({kind}) 담당 에이전트 없음 -> ActionAgent 반송",
              flush=True)
        out = dict(sc)
        out["needs_result"] = {
            "value": None, "by": "Supervisor",
            "note": "해당 조회를 처리할 에이전트가 없습니다. 직접 입력해 주세요.",
        }
        return {"action": out, "next": "ActionAgent", "step": step + 1}

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

    # 3-b) Extract 게이트: 추출된 게 없으면 워커를 안 돌리고 바로 Final (항목 6)
    #      진행 중 액션이 없을 때만. (Action 은 위 2)에서 이미 걸러졌다.)
    extracted = (state.get("facts") or {}).get("extracted") or {}
    if extracted and not extracted.get("gate_pass"):
        print("[NODE] Supervisor: Extract 게이트 실패 -> FinalAnswerAgent", flush=True)
        emit(config, "agent_status",
             {"agent": "Supervisor", "detail": "추출 결과 없음 -> 바로 응답"})
        return {"next": "FinalAnswerAgent", "step": step + 1}

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

def serve_needs(node_fn, agent_name: str):
    """기존 워커 노드를 감싸 needs-핸드오프를 대신 처리해 주는 어댑터.

    왜 어댑터인가
    -------------
    헬퍼 안에 `if needs["agent"] == "LocationAgent":` 같은 분기를 심으면
      - 워커가 자기 이름을 자기가 대조하게 되고(누가 부를지는 Supervisor 소관),
      - 헬퍼마다 같은 보일러플레이트를 복붙해야 하며,
      - 이미 잘 돌던 사내 에이전트 본문을 건드려야 한다.

    그래서 노드는 그대로 두고 배선 시점에 감싼다.

    하는 일은 둘뿐이다.
      1) needs 를 그 워커가 평소 받는 형태의 질의로 번역해 넣어준다(ask).
         핸드오프 재개 때는 사용자의 HITL 답변이 HumanMessage 로 남지 않아
         대화록만으로는 조회 대상을 알 수 없기 때문이다. 이 번역 메시지는
         노드에 넘기는 사본에만 넣고 상태로는 돌려주지 않는다.
      2) 워커가 돌려준 결과에서 값을 꺼내 메일박스를 채운다(pick).

    needs 가 없으면 원본을 그대로 통과시킨다. 사내 이식 시에도 에이전트
    본문 수정 0 줄이고, 배선에서 감싸고 능력표에 한 줄 추가하면 된다.

    Args:
        node_fn    : 원본 워커 노드 (동기 함수)
        agent_name : 메일박스에 남길 처리자 이름
    """
    def wrapped(state: _state.AgentState, config, **kwargs) -> dict:
        sc = state.get("action") or {}
        needs = sc.get("needs")

        # 평소 모드 — 원본 그대로
        if not needs or sc.get("needs_result"):
            return node_fn(state, config, **kwargs) or {}

        spec = NEEDS_CAPABILITIES.get(needs.get("kind")) or {}

        # 1) needs -> 평범한 질의로 번역해서 노드에 넘긴다 (상태에는 안 남긴다)
        call_state = state
        ask = spec.get("ask")
        if ask:
            try:
                question = ask(needs)
                print(f"[NEEDS] {agent_name} 에게 질의 번역: {question!r}", flush=True)
                call_state = {**state,
                              "messages": list(state.get("messages") or [])
                              + [HumanMessage(content=question)]}
            except Exception as e:
                print(f"[NEEDS] 질의 번역 실패: {e}", flush=True)

        out = node_fn(call_state, config, **kwargs) or {}

        # 2) 결과에서 값을 꺼내 메일박스를 채운다
        value = None
        pick = spec.get("pick")
        if pick:
            try:
                value = pick(out, needs)
            except Exception as e:          # 헬퍼가 어떻게 실패하든 그래프는 계속
                print(f"[NEEDS] {agent_name} 값 추출 실패: {e}", flush=True)

        # 실패 사유는 ActionAgent 가 그대로 사용자에게 보여준다.
        # 무엇을 조회하려다 실패했는지 짚어주고, 직접 입력을 안내한다.
        if value:
            note = ""
        else:
            target = needs.get("carrier_id")
            note = (f"{agent_name} 가 {target} 조회에 실패했습니다. 직접 입력해 주세요."
                    if target else
                    f"{agent_name} 가 값을 찾지 못했습니다. 직접 입력해 주세요.")
        print(f"[NEEDS] {agent_name} -> {needs.get('fill')}={value}", flush=True)

        merged = dict(sc)
        merged.update(out.get("action") or {})   # 원본이 action 을 건드렸으면 얹는다
        merged["needs_result"] = {"value": value, "by": agent_name, "note": note}

        out = dict(out)
        out["action"] = merged
        return out

    wrapped.__name__ = getattr(node_fn, "__name__", "wrapped")
    return wrapped


def location_node(state: _state.AgentState, config, model_name: str = None) -> dict:
    """캐리어 위치 조회.

    needs-핸드오프를 위한 분기는 여기 없다. 배선에서 serve_needs 로 감싸
    메일박스를 채우므로, 이 노드는 자기 일만 하면 된다.
    """
    print("[NODE] LocationAgent entered", flush=True)
    emit(config, "agent_status", {"agent": "LocationAgent", "detail": "위치 조회"})

    model = model_name or _model_of(state)

    # needs 로 불려온 경우엔 참조 대상 캐리어도 함께 조회해야 한다.
    # (발화에 그 캐리어가 들어 있으므로 평소 로직이 그대로 잡아낸다)
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

    needs-핸드오프 분기는 여기 없다 (배선의 serve_needs 가 처리).
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
