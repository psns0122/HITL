"""ActionAgent 의 툴 계층.

기능별 툴 7종. LLM 이 임의 호출하는 ReAct 툴이 아니라, 서브그래프 노드가
결정적으로 호출하는 비즈니스 로직이다(순서 보장을 위해). 모든 툴은 진입과
진행 과정을 print 로 상세히 남긴다.
"""
from app.actions import mock_db


def _log(tool: str, msg: str):
    print(f"[TOOL {tool}] {msg}", flush=True)


# ── 0. ID 판독기 ─────────────────────────────────────────────────────────

def id_lookup_tool(candidates: list) -> dict:
    """★ ID 판독기 — 후보 토큰들이 실제로 무엇인지 조회해서 종류를 정한다.

    ★★ 사내 반입 시 이 함수 본문만 사내 조회 코드로 갈아끼우면 된다. ★★
       (여기서는 mock_db 로 땡 처리. 인터페이스는 그대로 유지한다.)

    설계 의도
    ---------
    캐리어/장비 ID 형식은 항상 정형화돼 있지 않다. 그래서 "8자 영숫자면
    캐리어" 같은 정규식으로 종류를 판정하면, 형식이 조금만 달라도 실제로
    존재하는 ID 를 못 알아보고 버린다.

    그래서 규칙을 뒤집는다.
      - 발화에서 ID '스러운' 토큰은 형식을 따지지 말고 전부 후보로 올리고,
      - 그게 캐리어인지 장비인지 아무것도 아닌지는 **판독기가 조회해서** 정한다.

    즉 유효성의 권한은 정규식이 아니라 이 함수에 있다.

    Returns:
        {"carrier_ids": [...], "eqp_ids": [...], "unknown": [...]}
        unknown = 후보로는 올라왔지만 조회에 걸리지 않은 것들.
                  (사용자에게 되물을 때 근거로 쓴다)
    """
    carrier_ids, eqp_ids, unknown = [], [], []

    for cand in candidates or []:
        if mock_db.get_carrier(cand):
            carrier_ids.append(cand)
        elif mock_db.get_equipment(cand):
            eqp_ids.append(cand)
        else:
            unknown.append(cand)

    _log("id_lookup", f"후보={list(candidates or [])} -> carrier={carrier_ids} "
                      f"eqp={eqp_ids} unknown={unknown}")
    return {"carrier_ids": carrier_ids, "eqp_ids": eqp_ids, "unknown": unknown}


# ── 1. (공용) param_check_tool ────────────────────────────────────────────

def param_check_tool(action: str | None, params: dict) -> dict:
    """의도별 필수 파라미터 충족 여부 확인. 미충족 -> HITL 루프의 근거가 된다."""
    from app.actions.registry import ACTION_REGISTRY   # 순환 import 회피

    _log("param_check", f"enter action={action} params={params}")
    if not action:
        _log("param_check", "action 미확정 -> missing=['action']")
        return {"satisfied": False, "missing": ["action"], "normalized": dict(params or {})}

    spec = ACTION_REGISTRY[action]
    normalized = {k: (v.upper() if isinstance(v, str) else v)
                  for k, v in (params or {}).items() if v}
    missing = [p for p in spec.required_params if not normalized.get(p)]
    _log("param_check", f"required={spec.required_params} -> missing={missing}")
    return {"satisfied": not missing, "missing": missing, "normalized": normalized}


# ── 2. validation tools (액션별) ──────────────────────────────────────────

def transport_validate_tool(params: dict) -> dict:
    """반송요청 유효성: 캐리어 존재 / 목적지 존재·온라인 / 현재 위치에서 도달 가능."""
    _log("transport_validate", f"enter params={params}")
    carrier_id = (params.get("carrier_id") or "").upper()
    eqp_id = (params.get("eqp_id") or "").upper()

    carrier = mock_db.get_carrier(carrier_id)
    if not carrier:
        _log("transport_validate", f"FAIL: carrier '{carrier_id}' 미존재")
        return {"ok": False, "code": "CARRIER_NOT_FOUND", "bad_fields": ["carrier_id"],
                "reason": f"캐리어 {carrier_id} 를 찾을 수 없습니다."}

    eqp = mock_db.get_equipment(eqp_id)
    if not eqp:
        _log("transport_validate", f"FAIL: eqp '{eqp_id}' 미존재")
        return {"ok": False, "code": "EQP_NOT_FOUND", "bad_fields": ["eqp_id"],
                "reason": f"장비 {eqp_id} 를 찾을 수 없습니다."}
    if not eqp["online"]:
        _log("transport_validate", f"FAIL: eqp '{eqp_id}' 오프라인")
        return {"ok": False, "code": "EQP_OFFLINE", "bad_fields": ["eqp_id"],
                "reason": f"장비 {eqp_id} 는 현재 오프라인이라 목적지로 지정할 수 없습니다."}

    cur = carrier["current_eqp"]
    if cur == eqp_id:
        _log("transport_validate", f"FAIL: 이미 {eqp_id} 에 위치")
        return {"ok": False, "code": "ALREADY_THERE", "bad_fields": ["eqp_id"],
                "reason": f"캐리어 {carrier_id} 는 이미 {eqp_id} 에 있습니다."}
    if not mock_db.is_reachable(cur, eqp_id):
        _log("transport_validate", f"FAIL: {cur} -> {eqp_id} 도달 불가")
        return {"ok": False, "code": "UNREACHABLE", "bad_fields": ["eqp_id"],
                "reason": f"현재 위치 {cur} 에서 {eqp_id} 로는 반송 경로가 없습니다."}

    _log("transport_validate", f"PASS: {carrier_id} {cur} -> {eqp_id}")
    return {"ok": True, "code": "OK", "bad_fields": [], "reason": ""}


def dest_req_validate_tool(params: dict) -> dict:
    """목적지요청 유효성: 캐리어 존재 / 요청 가능 상태(정책 목록 보유)."""
    _log("dest_req_validate", f"enter params={params}")
    carrier_id = (params.get("carrier_id") or "").upper()

    carrier = mock_db.get_carrier(carrier_id)
    if not carrier:
        _log("dest_req_validate", f"FAIL: carrier '{carrier_id}' 미존재")
        return {"ok": False, "code": "CARRIER_NOT_FOUND", "bad_fields": ["carrier_id"],
                "reason": f"캐리어 {carrier_id} 를 찾을 수 없습니다."}

    policy = mock_db.get_dest_policy(carrier_id)
    if not policy:
        _log("dest_req_validate", f"FAIL: '{carrier_id}' 목적지요청 불가 (status={carrier['status']})")
        return {"ok": False, "code": "NO_DEST_ALLOWED", "bad_fields": ["carrier_id"],
                "reason": f"캐리어 {carrier_id} 는 현재 상태({carrier['status']})에서 "
                          f"목적지요청이 허용되지 않습니다."}

    _log("dest_req_validate", f"PASS: {carrier_id} policy={policy}")
    return {"ok": True, "code": "OK", "bad_fields": [], "reason": ""}


# ── 3. 안내문(확인 질문) tools (액션별) ───────────────────────────────────

def transport_confirm_tool(params: dict) -> str:
    _log("transport_confirm", f"enter params={params}")
    carrier_id = params.get("carrier_id", "?")
    eqp_id = params.get("eqp_id", "?")
    cur = (mock_db.get_carrier(carrier_id) or {}).get("current_eqp", "?")
    text = (f"⚠️ 반송요청명령 실행 확인\n"
            f"- 캐리어: {carrier_id} (현재 위치 {cur})\n"
            f"- 목적지: {eqp_id}\n"
            f"이 명령을 정말 실행할까요? (승인/거절)")
    _log("transport_confirm", f"guidance rendered ({len(text)} chars)")
    return text


def dest_req_confirm_tool(params: dict) -> str:
    _log("dest_req_confirm", f"enter params={params}")
    carrier_id = params.get("carrier_id", "?")
    policy = mock_db.get_dest_policy(carrier_id)
    text = (f"⚠️ 목적지요청 실행 확인\n"
            f"- 캐리어: {carrier_id}\n"
            f"- 허용 목적지 후보: {', '.join(policy) if policy else '없음'}\n"
            f"이 명령을 정말 실행할까요? (승인/거절)")
    _log("dest_req_confirm", f"guidance rendered ({len(text)} chars)")
    return text


# ── 4. 실행 tools (액션별) ────────────────────────────────────────────────

def transport_execute_tool(params: dict) -> dict:
    """진짜 반송요청 수행(목업). 승인 interrupt 통과 후에만 호출되어야 한다."""
    _log("transport_execute", f"enter params={params}")
    carrier_id = (params.get("carrier_id") or "").upper()
    eqp_id = (params.get("eqp_id") or "").upper()
    job_id = mock_db.next_job_id("TJ")
    _log("transport_execute", f"dispatching job {job_id}: {carrier_id} -> {eqp_id}")
    mock_db.MOCK_DB["carriers"][carrier_id]["status"] = "TRANSFERRING"
    result = {"job_id": job_id, "status": "DISPATCHED",
              "payload": {"carrier_id": carrier_id, "dest": eqp_id}}
    _log("transport_execute", f"done -> {result}")
    return result


def dest_req_execute_tool(params: dict) -> dict:
    """진짜 목적지요청 수행(목업): 정책 1순위 목적지를 배정한다."""
    _log("dest_req_execute", f"enter params={params}")
    carrier_id = (params.get("carrier_id") or "").upper()
    policy = mock_db.get_dest_policy(carrier_id)
    assigned = policy[0]
    job_id = mock_db.next_job_id("DR")
    _log("dest_req_execute", f"assigning dest {assigned} for {carrier_id} (job {job_id})")
    result = {"job_id": job_id, "status": "ASSIGNED",
              "payload": {"carrier_id": carrier_id, "assigned_dest": assigned}}
    _log("dest_req_execute", f"done -> {result}")
    return result
