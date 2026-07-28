"""에이전트가 쓰는 툴 모음.  [원본·추정]

툴 **본문**은 사내 DB/API 를 조회하는 코드라 받지 못했다.
여기 있는 건 `_agent.py` 가 참조하는 이름과 시그니처를 맞춰 둔 껍데기다.
**사내 실물로 덮어써 주세요.**

확실한 것은 툴 목록뿐이다 — 어떤 에이전트가 어떤 툴을 갖는지는
직접 제공된 `create_extract_agent` 와 공유해 준 에이전트 구성에서 나왔다.

모든 툴은 진입 시점과 진행 과정을 print 로 상세히 남긴다.
(사내 요구사항: 별도 로깅 라이브러리 없이 print)
"""
from langchain_core.tools import tool


def _log(tool_name: str, msg: str):
    """툴 로그 한 줄. flush 를 켜야 스트리밍 출력과 순서가 섞이지 않는다."""
    print(f"[TOOL {tool_name}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent
# ─────────────────────────────────────────────────────────────────────────

@tool
def general_tool(question: str) -> str:
    """일반적인 챗봇 사용 안내를 돌려준다. 사내 데이터가 필요 없는 질문에 쓴다."""
    _log("general", f"enter question={question!r}")
    raise NotImplementedError("사내 구현으로 교체")


@tool
def amhs_rag_tool(query: str) -> str:
    """AMHS 운영 문서에서 관련 내용을 찾아온다."""
    _log("amhs_rag", f"enter query={query!r}")
    raise NotImplementedError("사내 구현으로 교체 — 벡터 검색")


# ─────────────────────────────────────────────────────────────────────────
# StatusAgent
# ─────────────────────────────────────────────────────────────────────────

@tool
def queue_status_tool(fab: str = "") -> str:
    """반송 큐 적체 상태를 조회한다."""
    _log("queue_status", f"enter fab={fab!r}")
    raise NotImplementedError("사내 구현으로 교체 — 큐 테이블 조회")


@tool
def server_status_tool(server: str = "") -> str:
    """AMHS 서버 프로세스 상태를 조회한다."""
    _log("server_status", f"enter server={server!r}")
    raise NotImplementedError("사내 구현으로 교체")


@tool
def sysadmin_tool(command: str = "") -> str:
    """시스템 관리 정보를 조회한다 (읽기 전용)."""
    _log("sysadmin", f"enter command={command!r}")
    raise NotImplementedError("사내 구현으로 교체")


@tool
def patch_plan_search_tool(keyword: str = "") -> str:
    """예정된 패치 계획을 검색한다."""
    _log("patch_plan_search", f"enter keyword={keyword!r}")
    raise NotImplementedError("사내 구현으로 교체")


@tool
def eqp_search_tool(eqp_id: str = "") -> str:
    """장비 정보를 조회한다."""
    _log("eqp_search", f"enter eqp_id={eqp_id!r}")
    raise NotImplementedError("사내 구현으로 교체 — equipment 테이블 조회")


# ─────────────────────────────────────────────────────────────────────────
# LocationAgent
# ─────────────────────────────────────────────────────────────────────────

@tool
def location_search_tool(carrier_id: str) -> str:
    """캐리어가 현재 어느 장비에 있는지 조회한다."""
    _log("location_search", f"enter carrier_id={carrier_id!r}")
    raise NotImplementedError("사내 구현으로 교체")


# ─────────────────────────────────────────────────────────────────────────
# LogAgent
# ─────────────────────────────────────────────────────────────────────────

@tool
def log_search_tool(carrier_id: str = "") -> str:
    """반송 이력과 에러 로그를 조회해 원인 장비까지 뽑아낸다.

    psns0122/log 의 transport_job_timeline 분석이 이 툴의 사내 실물에 해당한다
    (createTransportJob 이력 -> 캐리어 잡 타임라인 -> (reason, description) 에러 콤보).
    """
    _log("log_search", f"enter carrier_id={carrier_id!r}")
    raise NotImplementedError("사내 구현으로 교체 — Logpresso 조회")


# ─────────────────────────────────────────────────────────────────────────
# ExtractAgent
# ─────────────────────────────────────────────────────────────────────────

@tool
def fab_extract_tool(text: str) -> str:
    """질문에서 FAB 정보를 추출한다."""
    _log("fab_extract", f"enter text={text!r}")
    raise NotImplementedError("사내 구현으로 교체")


@tool
def params_extract_tool(text: str) -> str:
    """질문에서 캐리어 ID / 장비 ID 를 추출한다."""
    _log("params_extract", f"enter text={text!r}")
    raise NotImplementedError("사내 구현으로 교체")


# ─────────────────────────────────────────────────────────────────────────
# ActionAgent
#   원본에서는 이 두 툴을 바로 부른다 — 검증도 승인도 없다.
#   app/actions/tools.py 에서 param_check / validate / confirm / execute
#   네 단계로 쪼개진 게 이번 작업의 결과다.
# ─────────────────────────────────────────────────────────────────────────

@tool
def transport_tool(carrier_id: str, eqp_id: str) -> str:
    """반송요청명령. 캐리어를 지정한 장비로 보낸다."""
    _log("transport", f"enter carrier_id={carrier_id!r} eqp_id={eqp_id!r}")
    raise NotImplementedError("사내 구현으로 교체 — createTransportJob")


@tool
def dest_req_tool(carrier_id: str) -> str:
    """목적지요청. 캐리어의 목적지를 시스템이 정하게 한다."""
    _log("dest_req", f"enter carrier_id={carrier_id!r}")
    raise NotImplementedError("사내 구현으로 교체")
