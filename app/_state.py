"""AgentState 정의.

기존 사내 구조(messages 4턴 제한 리듀서)를 유지하되, HITL 액션 제어 상태는
limiter가 절대 못 건드리는 별도 필드(action / facts)에 둔다.

- action : ActionAgent 의 스크래치. annotation 없음 = last-write-wins(교체).
           finalize/abandon 에서 {} 로 리셋해야 다음 요청이 오염되지 않는다.
- facts  : 에이전트 간 공유 팩트(위치/로그분석 결과 등). dict 병합 리듀서.
"""
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


def create_limiter(max_turns: int = 4):
    def limit_messages(existing: list, new: list) -> list:
        # 최근 max_turns 턴(유저 기준)의 메시지만 유지하는 커스텀 리듀서.

        # 1. 기존 add_messages 로직 수행 (메세지 추가, 업데이트)
        combined = add_messages(existing, new)

        user_msg_count = 0
        sliced_messages = []

        # 2. 최근 메시지부터 거꾸로 순회
        for msg in reversed(combined):
            sliced_messages.insert(0, msg)  # 원래 순서를 유지하기 위해 맨 앞에 삽입
            if isinstance(msg, HumanMessage):
                user_msg_count += 1
            if user_msg_count >= max_turns:
                break

        return sliced_messages

    return limit_messages


def merge_dict(existing: dict, new: dict) -> dict:
    """facts 용 병합 리듀서. 노드는 바뀐 키만 돌려주면 된다."""
    return {**(existing or {}), **(new or {})}


class ActionScratch(TypedDict, total=False):
    """ActionAgent 의 진행 상태. messages 밖에 있어 limiter 영향을 받지 않는다."""

    action: str                # "transport" | "dest_req" | None(미확정)
    phase: str                 # infer/param_check/collecting/awaiting_helper/validating/confirming/executing/done/abandoned
    params: dict               # 수집·추출된 파라미터 {carrier_id, eqp_id}
    missing: list              # 아직 비어있는 필수 파라미터
    pending_field: str         # 지금 사용자에게 묻고 있는 파라미터
    pending_answer: object     # interrupt resume 로 받은 원본 답변
    reference: dict            # 참조형 파라미터 {kind, carrier_id, fill}
    needs: dict                # 동료 에이전트에게 위임한 조회 {agent, fill, query}
    needs_result: dict         # 헬퍼가 채워주는 메일박스 {value, by, note}
    validation: dict           # {ok, reason, code, bad_fields}
    confirm: str               # "approve" | "reject"
    result: dict               # {job_id, status, payload}
    collect_retries: int
    validate_retries: int
    hops: int                  # needs 왕복 횟수
    abandon_reason: str
    last_parse_error: str


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], create_limiter(max_turns=4)]
    route: Literal["general", "supervisor"]
    handoff: bool
    next: str
    step: int
    action: ActionScratch                    # 교체(last-write-wins)
    facts: Annotated[dict, merge_dict]       # 병합
