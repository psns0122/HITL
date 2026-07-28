"""AgentState 정의.  [원본·첨부]

messages 는 최근 4 user turn 만 남기는 커스텀 리듀서가 붙어 있다.
이 리듀서 때문에 HITL 문답을 messages 에만 두면 중간에 잘려 액션이 깨진다
— app/_state.py 가 action / facts 필드를 messages 밖에 따로 둔 이유다.

원본에는 그 두 필드도, model_name 도 없다.
"""
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


def create_limiter(max_turns: int = 4):
    """최근 max_turns 턴(유저 기준)의 메시지만 유지하는 커스텀 리듀서."""

    def limit_messages(existing: list, new: list) -> list:
        # 1. 기존 add_messages 로직 수행 (메세지 추가, 업데이트)
        combined = add_messages(existing, new)

        user_msg_count = 0
        sliced_messages = []

        # 2. 최근 메시지부터 거꾸로 순회
        for msg in reversed(combined):
            sliced_messages.insert(0, msg)   # 원래 순서를 유지하기 위해 맨 앞에 삽입
            if isinstance(msg, HumanMessage):
                user_msg_count += 1
            if user_msg_count >= max_turns:
                break

        return sliced_messages

    return limit_messages


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], create_limiter(max_turns=4)]
    route: Literal["general", "supervisor"]
    handoff: bool
    next: str
    step: int

    # 직접 제공된 노드 코드가 `state.get("model_name")` 로 참조하고 있어
    # 원본에도 이 키가 있는 것으로 봤다. (첨부 shared_code.md 에는 없었음)
    model_name: str
