"""에이전트가 쓰는 툴 모음 (목업).

사내에서는 각 툴이 실제 DB/API 를 조회한다.
여기서는 mock_db 를 보고 같은 모양의 결과를 돌려주도록만 만들어 둔다.
→ 시그니처와 반환 형태를 유지한 채 본문만 갈아끼우면 사내에서 그대로 동작한다.

모든 툴은 진입 시점과 진행 과정을 print 로 상세히 남긴다.
(사내 요구사항: 별도 로깅 라이브러리 없이 print)
"""
from langchain_core.tools import tool

from app.actions import mock_db, resolvers


def _log(tool_name: str, msg: str):
    """툴 로그 한 줄. flush 를 켜야 SSE 스트림과 순서가 섞이지 않는다."""
    print(f"[TOOL {tool_name}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def general_tool(question: str) -> str:
    """일반적인 챗봇 사용 안내를 돌려준다. 사내 데이터가 필요 없는 질문에 쓴다."""
    _log("general", f"enter question={question!r}")

    answer = (
        "AMHS 반송 시스템 챗봇입니다.\n"
        "- 캐리어 위치 / 상태 조회\n"
        "- 반송 이력 및 에러 원인 분석\n"
        "- 반송요청명령(transport), 목적지요청(dest_req) 실행"
    )

    _log("general", "done")
    return answer


@tool
def amhs_rag_tool(query: str) -> str:
    """AMHS 운영 문서에서 관련 내용을 찾아온다 (목업: 고정 문서 스니펫)."""
    _log("amhs_rag", f"enter query={query!r}")

    # 사내 구현 자리: 벡터 검색 -> 상위 청크 반환
    snippet = (
        "[문서] 반송요청명령은 캐리어가 IDLE 상태이고 목적지 장비가 online 이며 "
        "현재 위치에서 도달 가능할 때만 수행됩니다."
    )

    _log("amhs_rag", f"1건 반환 ({len(snippet)}자)")
    return snippet


# ─────────────────────────────────────────────────────────────────────────
# StatusAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def queue_status_tool(fab: str = "") -> str:
    """반송 큐 적체 상태를 조회한다."""
    _log("queue_status", f"enter fab={fab!r}")

    # 사내 구현 자리: 큐 테이블 조회
    result = "반송 큐: 대기 12건, 진행 3건, 평균 대기 42초 (임계치 이내)"

    _log("queue_status", f"result={result}")
    return result


@tool
def server_status_tool(server: str = "") -> str:
    """AMHS 서버 프로세스 상태를 조회한다."""
    _log("server_status", f"enter server={server!r}")

    result = "MCS-01 RUNNING / MCS-02 RUNNING / OHT-CTRL RUNNING (이상 없음)"

    _log("server_status", f"result={result}")
    return result


@tool
def sysadmin_tool(command: str = "") -> str:
    """시스템 관리 정보를 조회한다 (읽기 전용 목업)."""
    _log("sysadmin", f"enter command={command!r}")

    result = "디스크 61% / 메모리 48% / 최근 재기동 2026-07-20 03:10"

    _log("sysadmin", f"result={result}")
    return result


@tool
def patch_plan_search_tool(keyword: str = "") -> str:
    """예정된 패치 계획을 검색한다."""
    _log("patch_plan_search", f"enter keyword={keyword!r}")

    result = "2026-08-03 02:00~04:00 MCS 정기 패치 예정 (반송 일시 중단)"

    _log("patch_plan_search", f"result={result}")
    return result


@tool
def eqp_search_tool(eqp_id: str = "") -> str:
    """장비 정보를 조회한다. 목업 DB 의 equipment 테이블을 본다."""
    _log("eqp_search", f"enter eqp_id={eqp_id!r}")

    # eqp_id 를 안 주면 전체 목록을 돌려준다
    if not eqp_id:
        names = ", ".join(mock_db.MOCK_DB["equipment"].keys())
        _log("eqp_search", f"전체 목록 {len(mock_db.MOCK_DB['equipment'])}건")
        return f"등록 장비: {names}"

    info = mock_db.MOCK_DB["equipment"].get(eqp_id.upper())
    if not info:
        _log("eqp_search", "미존재")
        return f"장비 {eqp_id} 를 찾을 수 없습니다."

    reachable = mock_db.MOCK_DB["reachable"].get(eqp_id.upper(), [])
    result = (f"{eqp_id.upper()}: type={info['type']}, online={info['online']}, "
              f"도달가능={reachable}")
    _log("eqp_search", f"result={result}")
    return result


# ─────────────────────────────────────────────────────────────────────────
# LocationAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def location_search_tool(carrier_id: str) -> str:
    """캐리어가 현재 어느 장비에 있는지 조회한다."""
    _log("location_search", f"enter carrier_id={carrier_id!r}")

    loc = mock_db.get_carrier_location(carrier_id)
    if not loc:
        _log("location_search", "위치 미확인")
        return f"캐리어 {carrier_id} 의 위치를 찾을 수 없습니다."

    _log("location_search", f"result={loc}")
    return f"캐리어 {carrier_id} 는 현재 {loc} 에 있습니다."


# ─────────────────────────────────────────────────────────────────────────
# LogAgent 용
# ─────────────────────────────────────────────────────────────────────────

@tool
def log_search_tool(carrier_id: str = "") -> str:
    """반송 이력과 에러 로그를 조회해 원인 장비까지 뽑아낸다."""
    _log("log_search", f"enter carrier_id={carrier_id!r}")

    analysis = mock_db.analyze_transport_logs(carrier_id or None)

    # 같은 에러가 1분 내 반복되면 콤보로 묶여서 나온다
    combo_lines = [
        f"  - {c['eqp']} {c['reason']}/{c['description']} x{c['count']} (최초 {c['first_t']})"
        for c in analysis["combos"]
    ]
    _log("log_search", f"콤보 {len(analysis['combos'])}건, "
                       f"원인 장비={analysis['cause_eqp']}")

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
    _log("fab_extract", f"enter text={text!r}")

    # 사내 구현 자리: 발화에서 FAB 코드 파싱
    fab = "M16"

    _log("fab_extract", f"result fab={fab}")
    return f"fab={fab}"


@tool
def params_extract_tool(text: str) -> str:
    """질문에서 캐리어 ID / 장비 ID 를 추출한다.

    ExtractAgent 의 핵심 기능. 뒤에 붙는 에이전트들이 이 결과를 재료로 쓴다.
    """
    _log("params_extract", f"enter text={text!r}")

    ids = resolvers.extract_ids(text)
    carriers = ids.get("carrier_ids") or []
    eqps = ids.get("eqp_ids") or []

    _log("params_extract", f"carrier_ids={carriers} eqp_ids={eqps}")

    if not carriers and not eqps:
        return "추출된 ID 가 없습니다."

    parts = []
    if carriers:
        parts.append(f"carrier_ids={carriers}")
    if eqps:
        parts.append(f"eqp_ids={eqps}")
    return ", ".join(parts)
