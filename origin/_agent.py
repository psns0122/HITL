"""에이전트 팩토리.  [원본·직접]

사용자가 직접 제공한 코드. 모든 create_*_agent 가 같은 세 줄 구조다.

    create_react_agent(
        model  = _llm.get_llm(model_name=model_name, temperature=0),
        tools  = disable_tool_caching([...]),
        prompt = _prompt.*_agent_prompt(),
    )

에이전트끼리 다른 건 tools 와 prompt 뿐이다.
새 에이전트를 붙일 때도 이 모양을 그대로 복사한다.
"""
from langgraph.prebuilt import create_react_agent

from origin import _llm, _prompt, _tool


def disable_tool_caching(tools_list):
    """툴 결과 캐싱을 끈다.

    사내 데이터는 조회 시점마다 값이 달라진다. 캐시가 남아 있으면
    직전 조회 결과를 그대로 돌려줘서 오답이 된다.
    """
    for t in tools_list:
        t.cache = False
    return tools_list


# ─────────────────────────────────────────────────────────────────────────
# 워커 에이전트
# ─────────────────────────────────────────────────────────────────────────

def create_extract_agent(model_name: str = None):
    """FAB / 파라미터 추출."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.fab_extract_tool,
            _tool.params_extract_tool,
        ]),
        prompt=_prompt.extract_agent_prompt(),
    )


def create_status_agent(model_name: str = None):
    """큐 / 서버 / 설비 상태, 패치 계획 조회."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.queue_status_tool,
            _tool.server_status_tool,
            _tool.sysadmin_tool,
            _tool.patch_plan_search_tool,
            _tool.eqp_search_tool,
        ]),
        prompt=_prompt.status_agent_prompt(),
    )


def create_location_agent(model_name: str = None):
    """캐리어 현재 위치 조회."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.location_search_tool,
        ]),
        prompt=_prompt.location_agent_prompt(),
    )


def create_log_agent(model_name: str = None):
    """반송 이력 / 에러 로그 분석."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.log_search_tool,
        ]),
        prompt=_prompt.log_agent_prompt(),
    )


def create_action_agent(model_name: str = None):
    """명령 실행 (반송요청명령 / 목적지요청).

    원본에서는 다른 워커와 완전히 같은 모양이다 — HITL 도, 파라미터 수집도,
    승인 절차도 없다. 툴을 그냥 부른다.
    이 자리를 턴 기반 HITL 노드로 바꾸는 게 이번 작업이다.
    (app/actions/node.py 와 비교)
    """
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.transport_tool,
            _tool.dest_req_tool,
        ]),
        prompt=_prompt.action_agent_prompt(),
    )


def create_general_agent(model_name: str = None):
    """업무 데이터 없이 답하는 일반 대화."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=disable_tool_caching([
            _tool.general_tool,
            _tool.amhs_rag_tool,
        ]),
        prompt=_prompt.general_agent_prompt(),
    )


# ─────────────────────────────────────────────────────────────────────────
# 최종 응답 에이전트
#   사용자가 토큰 스트리밍으로 보게 되는 건 이 둘의 출력뿐이다.
# ─────────────────────────────────────────────────────────────────────────

def create_final_agent(model_name: str = None):
    """워커 결과를 받아 최종 답변. 툴 없음."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=[],
        prompt=_prompt.final_agent_prompt(),
    )


def create_final_general_agent(model_name: str = None):
    """일반 대화의 최종 답변. 툴 없음."""
    return create_react_agent(
        model=_llm.get_llm(model_name=model_name, temperature=0),
        tools=[],
        prompt=_prompt.final_general_agent_prompt(),
    )
