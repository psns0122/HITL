"""에이전트가 쓰는 툴 모음.  [원본·직접]

시그니처와 데코레이터는 사용자가 직접 제공한 것. 툴 **본문**은 사내 DB/MCP 를
치는 코드라 받지 못했으므로 `NotImplementedError` 로 비워 뒀다.

공통 규약 (예외 없음)
--------------------
1. 데코레이터 두 장을 겹쳐 쓴다.

       @tool("이름", description=_prompt.이름_description())
       @safe_tool
       def 이름(...): ...

   - 툴 이름과 설명은 **함수 밖**에서 정해진다. 설명은 docstring 이 아니라
     `_prompt.py` 의 `*_description()` 함수에서 온다 -> 문구를 고칠 때
     툴 코드를 건드리지 않는다.
   - docstring 은 사람이 읽는 용도로만 남는다.
2. 마지막 인자는 항상 `config: RunnableConfig = {}`.
3. 반환은 항상 `Dict[str, Any]` — 문자열이 아니다.
   에이전트 프롬프트가 이 dict 를 받아 해석하고 리포트로 다듬는다.
4. MCP 서버 툴을 직접 부르는 것들만 `async` 다
   (`eqp_search_tool`, `location_search_tool`, `params_extract_tool`).

모든 툴은 진입 시점과 진행 과정을 print 로 상세히 남긴다.
(사내 요구사항: 별도 로깅 라이브러리 없이 print)
"""
import asyncio
import functools
import traceback
from typing import Any, Dict, List, Optional, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from origin import _prompt


# ─────────────────────────────────────────────────────────────────────────
# safe_tool  [원본·추정]
#   이름과 붙는 위치(@tool 안쪽)만 확인됐고 본문은 받지 못했다.
#   툴이 던진 예외를 삼켜서 에러 dict 로 바꾸는 역할로 보고 그렇게 채웠다.
#   **사내 실물로 덮어쓸 것.**
# ─────────────────────────────────────────────────────────────────────────

def safe_tool(func):
    """툴 예외를 에러 dict 로 바꿔 돌려준다.

    react agent 는 툴이 예외를 던지면 루프가 통째로 깨진다. 조회 하나
    실패했다고 대화가 끝나면 안 되므로, 실패도 '결과' 로 만들어 돌려주고
    에이전트가 그 사실을 사용자에게 설명하게 둔다.

    @tool 이 시그니처를 보고 스키마를 만들기 때문에 functools.wraps 로
    원본 시그니처를 유지하는 게 필수다. (안 그러면 인자 추론이 깨진다)

    sync / async 툴 양쪽 다 감싼다.
    """

    def _error(e: Exception) -> dict:
        print(f"[TOOL ERROR] {func.__name__}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "tool": func.__name__}

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def _async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return _error(e)

        return _async_wrapper

    @functools.wraps(func)
    def _sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return _error(e)

    return _sync_wrapper


# ─────────────────────────────────────────────────────────────────────────
# GeneralAgent
# ─────────────────────────────────────────────────────────────────────────

@tool("general_tool", description=_prompt.general_tool_description())
@safe_tool
def general_tool(query: str, config: RunnableConfig = {}) -> Dict[str, Any]:
    """일반 지식, 범용 개념, 일반 프로그래밍/통계/수학 질문에 답변한다."""
    print(f"[TOOL general_tool] enter query={query!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체")


@tool("amhs_rag_tool", description=_prompt.amhs_rag_tool_description())
@safe_tool
def amhs_rag_tool(query: str, config: RunnableConfig = {}) -> Dict[str, Any]:
    """AMHS 내부 문서 기반 RAG 답변을 수행한다.

    이 툴은 별도 담당자가 구현한다. 이식 범위 밖.
    """
    print(f"[TOOL amhs_rag_tool] enter query={query!r}", flush=True)
    raise NotImplementedError("담당자 구현 — 이식 범위 밖")


# ─────────────────────────────────────────────────────────────────────────
# StatusAgent
#   전부 fabs(공장)를 첫 인자로 받는다. 조회 범위를 공장 단위로 자르는 구조.
# ─────────────────────────────────────────────────────────────────────────

@tool("queue_status_tool", description=_prompt.queue_status_tool_description())
@safe_tool
def queue_status_tool(
    fabs: str,
    from_dt: str = None,
    to_dt: str = None,
    user_query: str = "",
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """특정 공장의 반송현황을 DB 에서 조회해 리턴한다.

    리턴 이후 에이전트 프롬프트를 통해 해석되고 리포트 형태로 다듬어져
    사용자에게 제공된다.
    """
    print(f"[TOOL queue_status_tool] enter fabs={fabs!r} from={from_dt} to={to_dt}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — 반송현황 DB 조회")


@tool("server_status_search_tool",
      description=_prompt.server_status_search_tool_description())
@safe_tool
def server_status_search_tool(
    fabs: Union[str, List[str]],
    target_dt: str = None,
    user_query: str = "",
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """특정 공장의 서버(CPU) 점유율 등 상태를 DB 에서 모니터링해 응답한다."""
    print(f"[TOOL server_status_search_tool] enter fabs={fabs!r} target_dt={target_dt}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — 서버 상태 DB 조회")


@tool("sys_admin_tool", description=_prompt.sys_admin_tool_description())
@safe_tool
def sys_admin_tool(
    fabs: Union[str, List[str]] = None,
    systems: Union[str, List[str]] = None,
    user_query: str = "",
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """특정 공장이 사용하는 여러 시스템별 담당자를 DB 에서 조회해 응답한다."""
    print(f"[TOOL sys_admin_tool] enter fabs={fabs!r} systems={systems!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — 담당자 DB 조회")


@tool("patch_plan_search_tool",
      description=_prompt.patch_plan_search_tool_description())
@safe_tool
def patch_plan_search_tool(
    fabs: Union[str, List[str]],
    systems: Union[str, List[str]] = None,
    target_dt: str = None,
    user_query: str = "",
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """특정 공장의 시스템별 패치 계획을 DB 에서 조회해 응답한다."""
    print(f"[TOOL patch_plan_search_tool] enter fabs={fabs!r} systems={systems!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — 패치 계획 DB 조회")


@tool("eqp_search_tool", description=_prompt.eqp_search_tool_description())
@safe_tool
async def eqp_search_tool(
    machine_name: str,
    fab: str = None,
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """특정 장비의 등록 여부와 상태를 조회해 응답한다.

    MCP 서버의 툴을 직접 호출한다 (그래서 async).
    """
    print(f"[TOOL eqp_search_tool] enter machine_name={machine_name!r} fab={fab!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — MCP 툴 직접 호출")


# ─────────────────────────────────────────────────────────────────────────
# LocationAgent
# ─────────────────────────────────────────────────────────────────────────

@tool("location_search_tool", description=_prompt.location_search_tool_description())
@safe_tool
async def location_search_tool(
    identifier: str,
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """특정 id(캐리어 혹은 랏으로 짐작되는)의 위치를 DB 에서 찾아 알려준다.

    MCP 의 툴을 직접 호출한다 (그래서 async).
    인자 이름이 carrier_id 가 아니라 identifier 인 게 포인트다 —
    캐리어인지 랏인지 부르는 쪽이 확정하지 않는다.
    """
    print(f"[TOOL location_search_tool] enter identifier={identifier!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — MCP 툴 직접 호출")


# ─────────────────────────────────────────────────────────────────────────
# LogAgent
# ─────────────────────────────────────────────────────────────────────────

@tool("log_search_tool", description=_prompt.log_search_tool_description())
@safe_tool
def log_search_tool(
    carrier_id: str,
    time_inputs: Optional[List[str]] = None,
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """DB 에서 로그 데이터를 조회해 반환한다.

    반환 이후에는 에이전트 프롬프트를 통해 해석하고 리포트 형식으로 만들어져
    사용자에게 제공된다.

    psns0122/log 의 transport_job_timeline 분석이 이 툴 뒷단에 해당한다
    (createTransportJob 이력 -> 캐리어 잡 타임라인 -> (reason, description) 에러 콤보).
    """
    print(f"[TOOL log_search_tool] enter carrier_id={carrier_id!r} time_inputs={time_inputs}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — 로그 DB 조회")


# ─────────────────────────────────────────────────────────────────────────
# ExtractAgent
#   이 둘은 다른 모든 에이전트보다 먼저 돌아서 뒤 단계가 쓸 재료를 만든다.
# ─────────────────────────────────────────────────────────────────────────

@tool("fab_extract_tool", description=_prompt.fab_extract_tool_description())
@safe_tool
def fab_extract_tool(user_query: str, config: RunnableConfig = {}) -> Dict[str, Any]:
    """사용자 질문에서 FAB 형태의 문자열을 찾아 유효성 검사 후 돌려준다.

    DB 를 가지 않는다. 유효한 FAB 명과 각 FAB 의 암묵지 별칭이
    config.py 에 나열되어 있고, 그것만 보고 판정한다.
    """
    print(f"[TOOL fab_extract_tool] enter user_query={user_query!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — config.py 의 FAB 목록/별칭 대조")


@tool("params_extract_tool", description=_prompt.params_extract_tool_description())
@safe_tool
async def params_extract_tool(user_query: str, config: RunnableConfig = {}) -> Dict[str, Any]:
    """식별되지 않은 ID 문자열이 실제로 무엇인지 DB 로 판정한다.

    사용자가 형식만 보고는 정체를 알 수 없는 id 를 입력한다. 이 id 는
    (캐리어, 랏, 장비, 유닛, 포트, 존) 중 하나일 것으로 예상되고,
    최종적으로 그 예상지 중 하나에 정말 맞았는지 — 아니면 아무 데도
    해당사항이 없는지(언노운) — DB 를 통해 판단해 돌려준다.

    ※ 이 툴이 곧 'ID 판독기' 다. app/id_reader.py 는 이것을 몰라서 새로 만든
      중복 구현이므로, 사내 반입 시에는 id_reader 를 버리고 이 툴을 부를 것.
    """
    print(f"[TOOL params_extract_tool] enter user_query={user_query!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — ID 종류 판정 DB 조회")


# ─────────────────────────────────────────────────────────────────────────
# ActionAgent  [원본·추정]
#   여기만 시그니처를 받지 못했다. 사내 실물로 덮어쓸 것.
#   원본에서는 이 툴들을 바로 부른다 — 검증도 승인도 없다.
#   app/actions/tools.py 에서 param_check / validate / confirm / execute
#   네 단계로 쪼개진 게 이번 작업의 결과다.
# ─────────────────────────────────────────────────────────────────────────

@tool("transport_tool", description=_prompt.transport_tool_description())
@safe_tool
async def transport_tool(
    carrier_id: str,
    eqp_id: str,
    config: RunnableConfig = {},
) -> Dict[str, Any]:
    """반송요청명령. 캐리어를 지정한 장비로 보낸다."""
    print(f"[TOOL transport_tool] enter carrier_id={carrier_id!r} eqp_id={eqp_id!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체 — createTransportJob")


@tool("dest_req_tool", description=_prompt.dest_req_tool_description())
@safe_tool
async def dest_req_tool(carrier_id: str, config: RunnableConfig = {}) -> Dict[str, Any]:
    """목적지요청. 캐리어의 목적지를 시스템이 정하게 한다."""
    print(f"[TOOL dest_req_tool] enter carrier_id={carrier_id!r}", flush=True)
    raise NotImplementedError("사내 구현으로 교체")
