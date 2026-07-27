"""ActionAgent — HITL 서브그래프.

부모 그래프에서는 Supervisor 밑 member 노드 하나로 보이지만, 내부는
파라미터 수집(HITL 루프) → 검증(재시도 루프) → 실행 승인(HITL) → 실행의
결정적 상태 기계다.

interrupt 재실행 규칙(핵심):
- interrupt 된 노드는 resume 시 노드 맨 위부터 재실행된다.
- 따라서 interrupt 는 side-effect 없는 작은 노드(collect_param / confirm)에
  노드당 정확히 1개만 둔다. 실제 실행(execute)은 승인 이후에만 도달한다.

needs-핸드오프:
- 파라미터가 동료 에이전트의 분석/조회를 요구하면(action.needs 설정) interrupt 가
  아니라 '서브그래프 정상 종료'로 Supervisor 에게 양보하고, 헬퍼가
  action.needs_result 메일박스를 채워주면 재진입해 param_check 부터 재개한다.
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

import app.config as cfg
from app._agent import extract_intent
from app._state import AgentState
from app._util import emit, last_human_text
from app.actions import resolvers
from app.actions.registry import ACTION_REGISTRY, ACTION_SELECT_PROMPT
from app.actions.tools import id_lookup_tool, param_check_tool

# 진행 중으로 취급하는 phase (재진입 판정 기준)
ACTIVE_PHASES = {"param_check", "collecting", "awaiting_helper", "validating", "confirming"}


def _log(node: str, msg: str):
    print(f"[ACTION {node}] {msg}", flush=True)


def _scratch(state: AgentState) -> dict:
    return dict(state.get("action") or {})


# ── 노드들 ────────────────────────────────────────────────────────────────

def entry_node(state: AgentState, config) -> dict:
    """재진입 판정: 진행 중 스크래치가 있으면 infer_intent 를 건너뛴다."""
    sc = _scratch(state)
    reentry = sc.get("phase") in ACTIVE_PHASES
    _log("entry", f"enter (reentry={reentry}, phase={sc.get('phase')})")
    emit(config, "agent_status",
         {"agent": "ActionAgent", "detail": "재진입(수집 재개)" if reentry else "신규 진입"})
    sc["_route"] = "param_check" if reentry else "infer_intent"
    return {"action": sc}


def infer_intent_node(state: AgentState, config) -> dict:
    """LLM(또는 규칙)으로 의도/파라미터/참조 추출 — 새 액션 스크래치 구성."""
    text = last_human_text(state.get("messages", []))
    _log("infer_intent", f"enter text='{text}'")
    r = extract_intent(text, config=config)
    sc = {
        "action": r.action,
        "phase": "param_check",
        "params": {k: v.upper() for k, v in r.params.items()},
        "missing": [],
        "reference": r.reference,
        "collect_retries": 0,
        "validate_retries": 0,
        "hops": 0,
    }
    _log("infer_intent", f"-> action={r.action} params={sc['params']} ref={r.reference}")
    return {"action": sc}


def param_check_node(state: AgentState, config) -> dict:
    """필수 파라미터 충족 검사 + needs 메일박스 회수 + 다음 행선지 결정."""
    sc = _scratch(state)
    _log("param_check", f"enter params={sc.get('params')} needs={sc.get('needs')}")

    # 0) 헬퍼가 채워준 메일박스 회수 (needs-핸드오프 복귀 경로)
    if sc.get("needs"):
        res = sc.pop("needs_result", None)
        needs = sc.pop("needs")
        sc["hops"] = sc.get("hops", 0) + 1
        if res and res.get("value"):
            sc["params"][needs["fill"]] = str(res["value"]).upper()
            sc["reference"] = None
            _log("param_check", f"헬퍼({res.get('by')}) 결과 흡수: "
                                f"{needs['fill']}={res['value']}")
        else:
            # 헬퍼 실패 -> 참조 포기, 사용자에게 직접 묻기(HITL 강등)
            sc["reference"] = None
            sc["last_parse_error"] = (res or {}).get("note") or \
                f"{needs.get('agent')} 가 값을 찾지 못했습니다. 직접 입력해 주세요."
            _log("param_check", f"헬퍼 실패 -> HITL 강등: {sc['last_parse_error']}")

    check = param_check_tool(sc.get("action"), sc.get("params", {}))
    sc["params"] = check["normalized"] or sc.get("params", {})
    sc["missing"] = check["missing"]
    emit(config, "tool_call", {"agent": "ActionAgent", "tool": "param_check_tool",
                               "args": {"action": sc.get("action"), "params": sc["params"]},
                               "result": {"missing": sc["missing"]}})

    # 1) 참조형 파라미터 → 동료에게 위임 요청 (needs-핸드오프)
    #
    # 여기서는 '무엇이 필요한지'(kind)만 적는다. 누가 처리할지는 Supervisor 가
    # 정한다 — ActionAgent 는 동료 워커의 이름을 알지 못한다.
    # 처리할 수 있는 동료가 없으면 Supervisor 가 빈 결과를 채워 돌려보내고,
    # 그때 아래 0) 회수 분기가 사용자에게 직접 묻는 쪽으로 강등한다.
    ref = sc.get("reference")
    if ref and ref.get("fill") in sc["missing"]:
        if sc.get("hops", 0) >= cfg.MAX_HOPS:
            _log("param_check", f"MAX_HOPS({cfg.MAX_HOPS}) 초과 -> 참조 포기, 직접 질문")
            sc["reference"] = None
        else:
            sc["needs"] = {"fill": ref["fill"], "kind": ref["kind"],
                           "carrier_id": ref.get("carrier_id"),
                           "query": last_human_text(state.get("messages", []))}
            sc["phase"] = "awaiting_helper"
            sc["_route"] = "needs_exit"
            _log("param_check", f"needs-핸드오프 요청 (kind={ref['kind']}) -> Supervisor 가 배분")
            return {"action": sc}

    # 2) 미충족 → 수집 (한 번에 한 파라미터씩 질문)
    if sc["missing"]:
        if sc.get("collect_retries", 0) >= cfg.MAX_COLLECT:
            sc["abandon_reason"] = "필수 파라미터를 수집하지 못해 요청을 종료합니다."
            sc["_route"] = "abandon"
            _log("param_check", "MAX_COLLECT 초과 -> abandon")
            return {"action": sc}
        sc["pending_field"] = sc["missing"][0]
        sc["phase"] = "collecting"
        sc["_route"] = "collect_param"
        _log("param_check", f"미충족 -> collect '{sc['pending_field']}'")
        return {"action": sc}

    # 3) 충족 → 검증
    sc["phase"] = "validating"
    sc["_route"] = "validate"
    _log("param_check", "충족 -> validate")
    return {"action": sc}


def collect_param_node(state: AgentState, config) -> dict:
    """⏸ HITL #1 — 부족한 파라미터를 사용자에게 질문.

    주의: 이 노드는 resume 시 맨 위부터 재실행된다. interrupt 위에는
    로그/프롬프트 조립 외 어떤 side-effect 도 두지 않는다.
    """
    sc = _scratch(state)
    fieldname = sc.get("pending_field")
    if fieldname == "action":
        prompt = ACTION_SELECT_PROMPT
    else:
        prompt = ACTION_REGISTRY[sc["action"]].param_prompts[fieldname]
    note = sc.pop("last_parse_error", None)
    if note:
        prompt = f"{note}\n{prompt}"
    _log("collect_param", f"⏸ interrupt field={fieldname}")

    answer = interrupt({
        "type": "collect_param",
        "action": sc.get("action"),
        "field": fieldname,
        "prompt": prompt,
        "params": sc.get("params", {}),
        "missing": sc.get("missing", []),
    })

    _log("collect_param", f"resume answer={answer!r}")
    sc["pending_answer"] = answer
    return {"action": sc}


def _read_ids(text: str, config) -> dict:
    """ID 판독기 호출. 발화에서 캐리어/장비 ID 를 인식한다.

    두 단계로 나뉜다.
      1) id_candidates : ID 스러운 토큰을 형식 안 따지고 전부 후보로 (느슨)
      2) id_lookup_tool: 그게 캐리어인지 장비인지 아무것도 아닌지 조회 (권한)

    ★ 실무 교체 지점은 tools.id_lookup_tool 본문 한 곳이다.
      여기 인터페이스는 그대로 두고 그 함수만 사내 조회로 바꾸면
      infer_intent·merge_param·ExtractAgent 가 모두 따라온다.

    트레이스에 입력(text/후보)과 결과(ids)를 남긴다.
    """
    cands = resolvers.id_candidates(text)
    ids = id_lookup_tool(cands)
    emit(config, "tool_call", {
        "agent": "ActionAgent",
        "tool": "id_lookup_tool",          # ID 판독기
        "args": {"text": text, "candidates": cands},
        "result": ids,
    })
    return ids


def merge_param_node(state: AgentState, config) -> dict:
    """수집 답변을 판정해 처리한다.

    분기: 취소 / 참조(needs) / 맥락이탈(재시작) / 액션선택 / 값 / 재질문
    """
    sc = _scratch(state)
    answer = sc.pop("pending_answer", None)
    fieldname = sc.get("pending_field")
    _log("merge_param", f"enter field={fieldname}")

    r = resolvers.resolve_param_answer(fieldname, answer, sc.get("action"))

    # 취소
    if r["kind"] == "cancel":
        sc["abandon_reason"] = "사용자 요청으로 명령을 취소했습니다."
        sc["_route"] = "abandon"
        _log("merge_param", "취소 -> abandon")
        return {"action": sc}

    # 맥락 이탈 — 진행 중 액션을 접고 새 질문으로 재시작 (항목 12)
    if r["kind"] == "switch":
        sc["switch_text"] = r["text"]
        sc["_route"] = "restart"
        _log("merge_param", f"맥락 이탈 -> 재시작: '{r['text']}'")
        return {"action": sc}

    # 참조형 -> needs 핸드오프는 param_check 가 처리
    if r["kind"] == "reference":
        sc["reference"] = r["reference"]
        sc["_route"] = "param_check"
        _log("merge_param", f"참조형 답변 -> {r['reference']}")
        return {"action": sc}

    # 액션 선택 (action 을 묻던 중)
    if r["kind"] == "action":
        sc["action"] = r["value"]
        sc["_route"] = "param_check"
        _log("merge_param", f"액션 선택 -> {r['value']}")
        return {"action": sc}

    # 값 후보 -> ID 판독기 툴로 실제 인식·검증 (항목 11)
    if r["kind"] == "value":
        ids = _read_ids(r["text"], config)
        pool = ids["carrier_ids"] if fieldname == "carrier_id" else ids["eqp_ids"]

        if pool:
            sc["params"][fieldname] = pool[0].upper()
            # 같은 답변에 실려온 다른 ID 도 기회적으로 흡수
            if ids["carrier_ids"] and not sc["params"].get("carrier_id"):
                sc["params"]["carrier_id"] = ids["carrier_ids"][0]
            if ids["eqp_ids"] and not sc["params"].get("eqp_id"):
                sc["params"]["eqp_id"] = ids["eqp_ids"][0]
            sc["_route"] = "param_check"
            _log("merge_param", f"값 인식 -> params={sc['params']}")
            return {"action": sc}

        # 판독기가 값을 못 찾음 -> 재질문.
        # 후보로는 올라왔는데 조회에 안 걸린 경우엔 그 토큰을 짚어준다.
        sc["collect_retries"] = sc.get("collect_retries", 0) + 1
        if ids["unknown"]:
            sc["last_parse_error"] = (
                f"'{', '.join(ids['unknown'])}' 은(는) 조회되지 않는 ID 입니다.")
        else:
            sc["last_parse_error"] = f"입력하신 값에서 {fieldname} 를 찾지 못했습니다."
        sc["_route"] = "param_check"
        _log("merge_param", f"값 인식 실패(재시도 {sc['collect_retries']}): "
                            f"unknown={ids['unknown']}")
        return {"action": sc}

    # empty
    sc["collect_retries"] = sc.get("collect_retries", 0) + 1
    sc["last_parse_error"] = r.get("note", "")
    sc["_route"] = "param_check"
    _log("merge_param", f"재질문({sc['collect_retries']}): {r.get('note')}")
    return {"action": sc}


def validate_node(state: AgentState, config) -> dict:
    """액션별 validation 툴 호출. 실패 시 잘못된 파라미터만 비우고 수집 루프로."""
    sc = _scratch(state)
    spec = ACTION_REGISTRY[sc["action"]]
    _log("validate", f"enter action={sc['action']} params={sc['params']}")

    v = spec.validate(sc["params"])
    sc["validation"] = v
    emit(config, "tool_call", {"agent": "ActionAgent", "tool": f"{sc['action']}_validate_tool",
                               "args": sc["params"], "result": {"ok": v["ok"], "code": v["code"]}})

    if v["ok"]:
        sc["phase"] = "confirming"
        sc["_route"] = "confirm"
        _log("validate", "PASS -> confirm")
        return {"action": sc}

    sc["validate_retries"] = sc.get("validate_retries", 0) + 1
    if sc["validate_retries"] >= cfg.MAX_VALIDATE:
        sc["abandon_reason"] = f"유효성 검증에 반복 실패해 요청을 종료합니다. (사유: {v['reason']})"
        sc["_route"] = "abandon"
        _log("validate", f"MAX_VALIDATE 초과 -> abandon ({v['reason']})")
        return {"action": sc}

    # 문제가 된 파라미터만 비우고 재수집 (수렴 보장 지점)
    for bad in v.get("bad_fields", []):
        sc["params"].pop(bad, None)
    sc["last_parse_error"] = f"검증 실패: {v['reason']}"
    sc["phase"] = "param_check"
    sc["_route"] = "param_check"
    _log("validate", f"FAIL({v['code']}) -> {v.get('bad_fields')} 비우고 재수집")
    return {"action": sc}


def confirm_node(state: AgentState, config) -> dict:
    """⏸ HITL #2 — 실행 직전 최종 승인. interrupt 위 side-effect 금지."""
    sc = _scratch(state)
    spec = ACTION_REGISTRY[sc["action"]]
    guidance = spec.confirm_text(sc["params"])
    _log("confirm", f"⏸ interrupt guidance=\n{guidance}")

    decision = interrupt({
        "type": "confirm",
        "action": sc["action"],
        "prompt": guidance,
        "params": sc["params"],
        "options": ["승인", "거절"],
    })

    verdict = resolvers.detect_confirm_verdict(decision)
    _log("confirm", f"resume decision={decision!r} -> {verdict}")
    sc["confirm"] = verdict

    # 승인 — 유일하게 execute 로 가는 길
    if verdict == "approve":
        sc["phase"] = "executing"
        sc["_route"] = "execute"
        return {"action": sc}

    # 명시적 거절 — 종료
    if verdict == "reject":
        sc["phase"] = "abandoned"
        sc["abandon_reason"] = "사용자가 실행을 거절해 명령을 종료합니다."
        sc["_route"] = "abandon"
        return {"action": sc}

    # 판정 불가 — 승인/거절이 아니라 '파라미터를 고치려는 답변'일 수 있다.
    # 여기서 바로 접어버리면 그때까지 수집한 값이 통째로 날아가므로,
    # ID 후보가 실려 있으면 수집 루프로 되돌린다. (confirm 재질문이 아니라
    # 기존 param_check -> collect -> validate 경로를 그대로 다시 탄다)
    cands = resolvers.id_candidates(str(decision))
    if not cands:
        sc["phase"] = "abandoned"
        sc["abandon_reason"] = "승인 여부를 확인하지 못해 명령을 종료합니다."
        sc["_route"] = "abandon"
        _log("confirm", "판정 불가 + ID 후보 없음 -> abandon")
        return {"action": sc}

    ids = id_lookup_tool(cands)
    emit(config, "tool_call", {"agent": "ActionAgent", "tool": "id_lookup_tool",
                               "args": {"text": str(decision), "candidates": cands},
                               "result": ids})

    if ids["carrier_ids"] or ids["eqp_ids"]:
        # 조회되는 ID 를 줬다 -> 해당 파라미터만 교체하고 다시 검증·승인
        if ids["carrier_ids"]:
            sc["params"]["carrier_id"] = ids["carrier_ids"][0]
        if ids["eqp_ids"]:
            sc["params"]["eqp_id"] = ids["eqp_ids"][0]
        _log("confirm", f"파라미터 정정 -> params={sc['params']}")
    else:
        # 고치려 한 건 분명한데 조회가 안 되는 ID -> 그 자리만 비우고 다시 묻는다.
        # 어느 파라미터를 고치려는지 모를 땐 마지막 필수 파라미터로 본다
        # (transport 면 목적지 eqp_id — '바꿔줘'는 대개 목적지를 가리킨다).
        spec = ACTION_REGISTRY[sc["action"]]
        target = spec.required_params[-1]
        sc["params"].pop(target, None)
        sc["last_parse_error"] = (
            f"'{', '.join(ids['unknown'])}' 은(는) 조회되지 않는 ID 입니다.")
        _log("confirm", f"정정 실패 -> {target} 비우고 재수집 (unknown={ids['unknown']})")

    sc["phase"] = "param_check"
    sc["_route"] = "param_check"
    return {"action": sc}


def execute_node(state: AgentState, config) -> dict:
    """진짜 액션 수행 — 승인 이후에만 도달하는 유일한 side-effect 지점."""
    sc = _scratch(state)
    spec = ACTION_REGISTRY[sc["action"]]
    _log("execute", f"enter action={sc['action']} params={sc['params']}")
    result = spec.execute(sc["params"])
    sc["result"] = result
    sc["_route"] = "finalize"
    emit(config, "tool_call", {"agent": "ActionAgent", "tool": f"{sc['action']}_execute_tool",
                               "args": sc["params"], "result": result})
    return {"action": sc}


def finalize_node(state: AgentState, config) -> dict:
    """성공 종료: 결과 메시지 적재 + facts 기록 + 스크래치 리셋(필수)."""
    sc = _scratch(state)
    spec = ACTION_REGISTRY[sc["action"]]
    res = sc.get("result", {})
    payload = res.get("payload", {})
    lines = [f"✅ {spec.label} 실행 완료",
             f"- Job ID: {res.get('job_id')}",
             f"- 상태: {res.get('status')}"]
    lines += [f"- {k}: {v}" for k, v in payload.items()]
    content = "\n".join(lines)
    _log("finalize", f"job={res.get('job_id')}")

    return {
        "messages": [AIMessage(content=content, name="ActionAgent")],
        "facts": {"last_action": {"action": sc["action"], "params": sc.get("params"),
                                  "result": res}},
        "action": {},          # 스크래치 리셋 — 다음 요청 오염 방지
        "next": "Supervisor",
    }


def abandon_node(state: AgentState, config) -> dict:
    """취소/거절/한도초과 종료: 안내 메시지 + 스크래치 리셋. 재시도 집착 금지."""
    sc = _scratch(state)
    reason = sc.get("abandon_reason") or "요청을 종료했습니다."
    label = ACTION_REGISTRY[sc["action"]].label if sc.get("action") in ACTION_REGISTRY else "명령"
    content = f"🚫 {label} 을(를) 실행하지 않았습니다.\n- 사유: {reason}"
    _log("abandon", reason)

    return {
        "messages": [AIMessage(content=content, name="ActionAgent")],
        "facts": {"last_action": {"action": sc.get("action"), "aborted": True,
                                  "reason": reason}},
        "action": {},          # 스크래치 리셋
        "next": "Supervisor",
    }


def restart_node(state: AgentState, config) -> dict:
    """맥락 이탈 처리 — 진행 중 액션을 접고, 사용자의 새 발화로 다시 시작한다.

    사람들은 수집 도중에도 맥락을 벗어난 새 질문을 던진다.
    그럴 땐 스크래치를 리셋하고, 그 발화를 새 HumanMessage 로 넣어
    Supervisor 부터(ExtractAgent 선행 포함) 다시 태운다.
    """
    sc = _scratch(state)
    text = sc.get("switch_text", "")
    _log("restart", f"새 질문으로 재시작: '{text}'")
    emit(config, "agent_status",
         {"agent": "ActionAgent", "detail": "이전 작업 중단, 새 질문 처리"})

    return {
        # 새 발화를 대화에 넣어 뒤 단계가 이걸 '이번 질문'으로 읽게 한다
        "messages": [HumanMessage(content=text)],
        "action": {},          # 스크래치 리셋
        "next": "Supervisor",
    }


def needs_exit_node(state: AgentState, config) -> dict:
    """동료 에이전트에게 양보하며 서브그래프를 '정상 종료'(interrupt 아님).

    부모 엣지(ActionAgent -> Supervisor)를 타고 Supervisor 가 needs 를 읽어
    헬퍼로 라우팅한다.
    """
    sc = _scratch(state)
    _log("needs_exit", f"Supervisor 에 양보 -> needs={sc.get('needs')}")
    emit(config, "agent_status",
         {"agent": "ActionAgent",
          "detail": f"{sc['needs']['fill']} 조회 위임 요청 "
                    f"(kind={sc['needs']['kind']}) — 담당은 Supervisor 가 결정"})
    return {"action": sc, "next": "Supervisor"}


# ── 배선 ──────────────────────────────────────────────────────────────────

def _route(state: AgentState) -> str:
    return (state.get("action") or {}).get("_route", "infer_intent")


def build_action_graph():
    """ActionAgent 서브그래프. 체크포인터는 부모 것을 상속(별도 지정 금지)."""
    g = StateGraph(AgentState)

    g.add_node("action_entry", entry_node)
    g.add_node("infer_intent", infer_intent_node)
    g.add_node("param_check", param_check_node)
    g.add_node("collect_param", collect_param_node)
    g.add_node("merge_param", merge_param_node)
    g.add_node("validate", validate_node)
    g.add_node("confirm", confirm_node)
    g.add_node("execute", execute_node)
    g.add_node("finalize", finalize_node)
    g.add_node("abandon", abandon_node)
    g.add_node("restart", restart_node)
    g.add_node("needs_exit", needs_exit_node)

    g.add_edge(START, "action_entry")
    g.add_conditional_edges("action_entry", _route,
                            {"infer_intent": "infer_intent", "param_check": "param_check"})
    g.add_edge("infer_intent", "param_check")
    g.add_conditional_edges("param_check", _route,
                            {"collect_param": "collect_param", "validate": "validate",
                             "needs_exit": "needs_exit", "abandon": "abandon"})
    g.add_edge("collect_param", "merge_param")
    g.add_conditional_edges("merge_param", _route,
                            {"param_check": "param_check", "abandon": "abandon",
                             "restart": "restart"})
    g.add_conditional_edges("validate", _route,
                            {"confirm": "confirm", "param_check": "param_check",
                             "abandon": "abandon"})
    # confirm 은 승인/거절 외에 '파라미터 정정' 으로 수집 루프에 되돌아갈 수 있다
    g.add_conditional_edges("confirm", _route,
                            {"execute": "execute", "abandon": "abandon",
                             "param_check": "param_check"})
    g.add_edge("execute", "finalize")
    g.add_edge("finalize", END)
    g.add_edge("abandon", END)
    g.add_edge("restart", END)
    g.add_edge("needs_exit", END)

    return g.compile()


ACTION_SUBGRAPH_NODES = {
    "action_entry", "infer_intent", "param_check", "collect_param", "merge_param",
    "validate", "confirm", "execute", "finalize", "abandon", "restart", "needs_exit",
}
