"""ActionAgent 의 툴 계층.

기능별 툴 7종. LLM 이 임의 호출하는 ReAct 툴이 아니라, 서브그래프 노드가
결정적으로 호출하는 비즈니스 로직이다(순서 보장을 위해). 모든 툴은 진입과
진행 과정을 print 로 상세히 남긴다.
"""
from app.actions import mock_db


# ID 판독기는 여기 없다 — ExtractAgent 소유의 공용 툴이다(app/id_reader.py).
# ActionAgent 는 예외적으로 그 모듈을 직접 import 해서 쓴다.


# ── 1. (공용) param_check_tool ────────────────────────────────────────────

def param_check_tool(action: str | None, params: dict) -> dict:
    """의도별 필수 파라미터 충족 여부 확인. 미충족 -> HITL 루프의 근거가 된다."""
    from app.actions.registry import ACTION_REGISTRY   # 순환 import 회피

    print(f"[TOOL param_check] enter action={action} params={params}", flush=True)
    if not action:
        print(f"[TOOL param_check] action 미확정 -> missing=['action']", flush=True)
        return {"satisfied": False, "missing": ["action"], "normalized": dict(params or {})}

    spec = ACTION_REGISTRY[action]
    normalized = {k: (v.upper() if isinstance(v, str) else v)
                  for k, v in (params or {}).items() if v}
    missing = [p for p in spec.required_params if not normalized.get(p)]
    print(f"[TOOL param_check] required={spec.required_params} -> missing={missing}", flush=True)
    return {"satisfied": not missing, "missing": missing, "normalized": normalized}


# ── 2. validation tools (액션별) ──────────────────────────────────────────

def transport_validate_tool(params: dict) -> dict:
    """반송요청 유효성: 캐리어 존재 / 목적지 존재·온라인 / 현재 위치에서 도달 가능."""
    print(f"[TOOL transport_validate] enter params={params}", flush=True)
    carrier_id = (params.get("carrier_id") or "").upper()
    eqp_id = (params.get("eqp_id") or "").upper()

    carrier = mock_db.get_carrier(carrier_id)
    if not carrier:
        print(f"[TOOL transport_validate] FAIL: carrier '{carrier_id}' 미존재", flush=True)
        return {"ok": False, "code": "CARRIER_NOT_FOUND", "bad_fields": ["carrier_id"],
                "reason": f"캐리어 {carrier_id} 를 찾을 수 없습니다."}

    eqp = mock_db.get_equipment(eqp_id)
    if not eqp:
        print(f"[TOOL transport_validate] FAIL: eqp '{eqp_id}' 미존재", flush=True)
        return {"ok": False, "code": "EQP_NOT_FOUND", "bad_fields": ["eqp_id"],
                "reason": f"장비 {eqp_id} 를 찾을 수 없습니다."}
    if not eqp["online"]:
        print(f"[TOOL transport_validate] FAIL: eqp '{eqp_id}' 오프라인", flush=True)
        return {"ok": False, "code": "EQP_OFFLINE", "bad_fields": ["eqp_id"],
                "reason": f"장비 {eqp_id} 는 현재 오프라인이라 목적지로 지정할 수 없습니다."}

    cur = carrier["current_eqp"]
    if cur == eqp_id:
        print(f"[TOOL transport_validate] FAIL: 이미 {eqp_id} 에 위치", flush=True)
        return {"ok": False, "code": "ALREADY_THERE", "bad_fields": ["eqp_id"],
                "reason": f"캐리어 {carrier_id} 는 이미 {eqp_id} 에 있습니다."}
    if not mock_db.is_reachable(cur, eqp_id):
        print(f"[TOOL transport_validate] FAIL: {cur} -> {eqp_id} 도달 불가", flush=True)
        return {"ok": False, "code": "UNREACHABLE", "bad_fields": ["eqp_id"],
                "reason": f"현재 위치 {cur} 에서 {eqp_id} 로는 반송 경로가 없습니다."}

    print(f"[TOOL transport_validate] PASS: {carrier_id} {cur} -> {eqp_id}", flush=True)
    return {"ok": True, "code": "OK", "bad_fields": [], "reason": ""}


def dest_req_validate_tool(params: dict) -> dict:
    """목적지요청 유효성: 캐리어 존재 / 요청 가능 상태(정책 목록 보유)."""
    print(f"[TOOL dest_req_validate] enter params={params}", flush=True)
    carrier_id = (params.get("carrier_id") or "").upper()

    carrier = mock_db.get_carrier(carrier_id)
    if not carrier:
        print(f"[TOOL dest_req_validate] FAIL: carrier '{carrier_id}' 미존재", flush=True)
        return {"ok": False, "code": "CARRIER_NOT_FOUND", "bad_fields": ["carrier_id"],
                "reason": f"캐리어 {carrier_id} 를 찾을 수 없습니다."}

    policy = mock_db.get_dest_policy(carrier_id)
    if not policy:
        print(f"[TOOL dest_req_validate] FAIL: '{carrier_id}' 목적지요청 불가 (status={carrier['status']})", flush=True)
        return {"ok": False, "code": "NO_DEST_ALLOWED", "bad_fields": ["carrier_id"],
                "reason": f"캐리어 {carrier_id} 는 현재 상태({carrier['status']})에서 "
                          f"목적지요청이 허용되지 않습니다."}

    print(f"[TOOL dest_req_validate] PASS: {carrier_id} policy={policy}", flush=True)
    return {"ok": True, "code": "OK", "bad_fields": [], "reason": ""}


# ── 3. 안내문(확인 질문) tools (액션별) ───────────────────────────────────

def transport_confirm_tool(params: dict) -> str:
    print(f"[TOOL transport_confirm] enter params={params}", flush=True)
    carrier_id = params.get("carrier_id", "?")
    eqp_id = params.get("eqp_id", "?")
    cur = (mock_db.get_carrier(carrier_id) or {}).get("current_eqp", "?")
    text = (f"⚠️ 반송요청명령 실행 확인\n"
            f"- 캐리어: {carrier_id} (현재 위치 {cur})\n"
            f"- 목적지: {eqp_id}\n"
            f"이 명령을 정말 실행할까요? (승인/거절)")
    print(f"[TOOL transport_confirm] guidance rendered ({len(text)} chars)", flush=True)
    return text


def dest_req_confirm_tool(params: dict) -> str:
    print(f"[TOOL dest_req_confirm] enter params={params}", flush=True)
    carrier_id = params.get("carrier_id", "?")
    policy = mock_db.get_dest_policy(carrier_id)
    text = (f"⚠️ 목적지요청 실행 확인\n"
            f"- 캐리어: {carrier_id}\n"
            f"- 허용 목적지 후보: {', '.join(policy) if policy else '없음'}\n"
            f"이 명령을 정말 실행할까요? (승인/거절)")
    print(f"[TOOL dest_req_confirm] guidance rendered ({len(text)} chars)", flush=True)
    return text


# ── 4. 실행 tools (액션별) ────────────────────────────────────────────────

def transport_execute_tool(params: dict) -> dict:
    """진짜 반송요청 수행(목업). 승인 interrupt 통과 후에만 호출되어야 한다."""
    print(f"[TOOL transport_execute] enter params={params}", flush=True)
    carrier_id = (params.get("carrier_id") or "").upper()
    eqp_id = (params.get("eqp_id") or "").upper()
    job_id = mock_db.next_job_id("TJ")
    print(f"[TOOL transport_execute] dispatching job {job_id}: {carrier_id} -> {eqp_id}", flush=True)
    mock_db.MOCK_DB["carriers"][carrier_id]["status"] = "TRANSFERRING"
    result = {"job_id": job_id, "status": "DISPATCHED",
              "payload": {"carrier_id": carrier_id, "dest": eqp_id}}
    print(f"[TOOL transport_execute] done -> {result}", flush=True)
    return result


def dest_req_execute_tool(params: dict) -> dict:
    """진짜 목적지요청 수행(목업): 정책 1순위 목적지를 배정한다."""
    print(f"[TOOL dest_req_execute] enter params={params}", flush=True)
    carrier_id = (params.get("carrier_id") or "").upper()
    policy = mock_db.get_dest_policy(carrier_id)
    assigned = policy[0]
    job_id = mock_db.next_job_id("DR")
    print(f"[TOOL dest_req_execute] assigning dest {assigned} for {carrier_id} (job {job_id})", flush=True)
    result = {"job_id": job_id, "status": "ASSIGNED",
              "payload": {"carrier_id": carrier_id, "assigned_dest": assigned}}
    print(f"[TOOL dest_req_execute] done -> {result}", flush=True)
    return result
