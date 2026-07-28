"""에이전트가 쓰는 툴 모음 (목업).

사내에서는 각 툴이 실제 DB/API 를 조회한다.
여기서는 mock_db 를 보고 같은 모양의 결과를 돌려주도록만 만들어 둔다.
→ 시그니처와 반환 형태를 유지한 채 본문만 갈아끼우면 사내에서 그대로 동작한다.

모든 툴은 진입 시점과 진행 과정을 print 로 상세히 남긴다.
(사내 요구사항: 별도 로깅 라이브러리 없이 print)
"""
from langchain_core.tools import tool

from app import _db as mock_db
from app.id_reader import extract_ids


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def general_tool(question: str) -> str:
    """일반적인 챗봇 사용 안내를 돌려준다. 사내 데이터가 필요 없는 질문에 쓴다."""
    print(f"[TOOL general] enter question={question!r}", flush=True)

    answer = (
        "AMHS 반송 시스템 챗봇입니다.\n"
        "- 캐리어 위치 / 상태 조회\n"
        "- 반송 이력 및 에러 원인 분석\n"
        "- 반송요청명령(transport), 목적지요청(dest_req) 실행"
    )

    print(f"[TOOL general] done", flush=True)
    return answer


@tool
def amhs_rag_tool(query: str) -> str:
    """AMHS 운영 문서에서 관련 내용을 찾아온다 (목업: 고정 문서 스니펫)."""
    print(f"[TOOL amhs_rag] enter query={query!r}", flush=True)

    # 사내 구현 자리: 벡터 검색 -> 상위 청크 반환
    snippet = (
        "[문서] 반송요청명령은 캐리어가 IDLE 상태이고 목적지 장비가 online 이며 "
        "현재 위치에서 도달 가능할 때만 수행됩니다."
    )

    print(f"[TOOL amhs_rag] 1건 반환 ({len(snippet)}자)", flush=True)
    return snippet


# ─────────────────────────────────────────────────────────────────────────
# StatusAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def queue_status_tool(fab: str = "") -> str:
    """반송 큐 적체 상태를 조회한다."""
    print(f"[TOOL queue_status] enter fab={fab!r}", flush=True)

    # 사내 구현 자리: 큐 테이블 조회
    result = "반송 큐: 대기 12건, 진행 3건, 평균 대기 42초 (임계치 이내)"

    print(f"[TOOL queue_status] result={result}", flush=True)
    return result


@tool
def server_status_tool(server: str = "") -> str:
    """AMHS 서버 프로세스 상태를 조회한다."""
    print(f"[TOOL server_status] enter server={server!r}", flush=True)

    result = "MCS-01 RUNNING / MCS-02 RUNNING / OHT-CTRL RUNNING (이상 없음)"

    print(f"[TOOL server_status] result={result}", flush=True)
    return result


@tool
def sysadmin_tool(command: str = "") -> str:
    """시스템 관리 정보를 조회한다 (읽기 전용 목업)."""
    print(f"[TOOL sysadmin] enter command={command!r}", flush=True)

    result = "디스크 61% / 메모리 48% / 최근 재기동 2026-07-20 03:10"

    print(f"[TOOL sysadmin] result={result}", flush=True)
    return result


@tool
def patch_plan_search_tool(keyword: str = "") -> str:
    """예정된 패치 계획을 검색한다."""
    print(f"[TOOL patch_plan_search] enter keyword={keyword!r}", flush=True)

    result = "2026-08-03 02:00~04:00 MCS 정기 패치 예정 (반송 일시 중단)"

    print(f"[TOOL patch_plan_search] result={result}", flush=True)
    return result


@tool
def eqp_search_tool(eqp_id: str = "") -> str:
    """장비 정보를 조회한다. 목업 DB 의 equipment 테이블을 본다."""
    print(f"[TOOL eqp_search] enter eqp_id={eqp_id!r}", flush=True)

    # eqp_id 를 안 주면 전체 목록을 돌려준다
    if not eqp_id:
        names = ", ".join(mock_db.MOCK_DB["equipment"].keys())
        print(f"[TOOL eqp_search] 전체 목록 {len(mock_db.MOCK_DB['equipment'])}건", flush=True)
        return f"등록 장비: {names}"

    info = mock_db.MOCK_DB["equipment"].get(eqp_id.upper())
    if not info:
        print(f"[TOOL eqp_search] 미존재", flush=True)
        return f"장비 {eqp_id} 를 찾을 수 없습니다."

    reachable = mock_db.MOCK_DB["reachable"].get(eqp_id.upper(), [])
    result = (f"{eqp_id.upper()}: type={info['type']}, online={info['online']}, "
              f"도달가능={reachable}")
    print(f"[TOOL eqp_search] result={result}", flush=True)
    return result


# ─────────────────────────────────────────────────────────────────────────
# LocationAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def location_search_tool(carrier_id: str) -> str:
    """캐리어가 현재 어느 장비에 있는지 조회한다."""
    print(f"[TOOL location_search] enter carrier_id={carrier_id!r}", flush=True)

    loc = mock_db.get_carrier_location(carrier_id)
    if not loc:
        print(f"[TOOL location_search] 위치 미확인", flush=True)
        return f"캐리어 {carrier_id} 의 위치를 찾을 수 없습니다."

    print(f"[TOOL location_search] result={loc}", flush=True)
    return f"캐리어 {carrier_id} 는 현재 {loc} 에 있습니다."


# ─────────────────────────────────────────────────────────────────────────
# LogAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def log_search_tool(carrier_id: str = "") -> str:
    """반송 이력과 에러 로그를 조회해 원인 장비까지 뽑아낸다."""
    print(f"[TOOL log_search] enter carrier_id={carrier_id!r}", flush=True)

    analysis = mock_db.analyze_transport_logs(carrier_id or None)

    # 같은 에러가 1분 내 반복되면 콤보로 묶여서 나온다
    combo_lines = [
        f"  - {c['eqp']} {c['reason']}/{c['description']} x{c['count']} (최초 {c['first_t']})"
        for c in analysis["combos"]
    ]
    print(f"[TOOL log_search] 콤보 {len(analysis['combos'])}건, "
          f"원인 장비={analysis['cause_eqp']}", flush=True)

    return "\n".join(
        [f"반송 이력 분석 (carrier={carrier_id or '전체'})",
         f"- 에러 콤보 {len(analysis['combos'])}건:"]
        + combo_lines
        + [f"- 원인 장비: {analysis['cause_eqp']}",
           f"- 권장 대체 목적지: {analysis['recommended_dest']}"]
    )


# ─────────────────────────────────────────────────────────────────────────
# ExtractAgent 용
#   이 두 툴은 다른 모든 에이전트보다 먼저 돌아서 뒤 단계가 쓸 재료를 만든다.
# ─────────────────────────────────────────────────────────────────────────

@tool
def fab_extract_tool(text: str) -> str:
    """질문에서 FAB 정보를 추출한다 (목업: 고정 FAB)."""
    print(f"[TOOL fab_extract] enter text={text!r}", flush=True)

    # 사내 구현 자리: 발화에서 FAB 코드 파싱
    fab = "M16"

    print(f"[TOOL fab_extract] result fab={fab}", flush=True)
    return f"fab={fab}"


@tool
def params_extract_tool(text: str) -> dict:
    """발화 속 정체불명 ID 가 실제로 무엇인지 판정한다 (ID 판독기).

    사내 params_extract_tool 과 같은 계약 — dict 를 돌려준다.
    ExtractAgent 의 핵심 툴이자, ActionAgent 의 HITL 수집 루프도
    사용자 답변을 읽을 때 이 툴을 그대로 쓴다.
    """
    print(f"[TOOL params_extract] enter text={text!r}", flush=True)

    ids = extract_ids(text)
    print(f"[TOOL params_extract] carrier_ids={ids['carrier_ids']} "
          f"eqp_ids={ids['eqp_ids']} unknown={ids['unknown']}", flush=True)

    return ids


# ─────────────────────────────────────────────────────────────────────────
# ActionAgent 용 — HITL 4단 툴 (param_check / validate / confirm / execute)
#   ★ 사내 반입 시 validate/execute 본문을 실제 DB/API 호출로 교체하는 지점
# ─────────────────────────────────────────────────────────────────────────

# ID 판독기는 여기 없다 — ExtractAgent 소유의 공용 툴이다(app/id_reader.py).
# ActionAgent 는 예외적으로 그 모듈을 직접 import 해서 쓴다.


# ── 1. (공용) param_check_tool ────────────────────────────────────────────

def param_check_tool(action: str | None, params: dict) -> dict:
    """의도별 필수 파라미터 충족 여부 확인. 미충족 -> HITL 루프의 근거가 된다."""
    from app import _prompt   # 액션 선언은 프롬프트 계층 소유

    print(f"[TOOL param_check] enter action={action} params={params}", flush=True)
    if not action:
        print(f"[TOOL param_check] action 미확정 -> missing=['action']", flush=True)
        return {"satisfied": False, "missing": ["action"], "normalized": dict(params or {})}

    required = _prompt.action_catalog()[action]["required_params"]
    normalized = {k: (v.upper() if isinstance(v, str) else v)
                  for k, v in (params or {}).items() if v}
    missing = [p for p in required if not normalized.get(p)]
    print(f"[TOOL param_check] required={required} -> missing={missing}", flush=True)
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
