"""액션 레지스트리 — 확장 지점(seam).

새 액션 추가 = ActionSpec 1개 + validate/confirm/execute 툴 3개 작성 후
ACTION_REGISTRY 에 등록. 서브그래프 배선은 건드릴 필요 없다.
"""
from dataclasses import dataclass, field
from typing import Callable

from app.actions import tools


@dataclass
class ActionSpec:
    name: str
    label: str                                   # 사용자 표시명
    required_params: list
    param_prompts: dict                          # field -> HITL 질문 문구
    validate: Callable[[dict], dict]
    confirm_text: Callable[[dict], str]
    execute: Callable[[dict], dict]
    # 파라미터별로 허용되는 참조 해석 종류 (§needs-핸드오프)
    referencable: dict = field(default_factory=dict)


ACTION_REGISTRY: dict[str, ActionSpec] = {
    "transport": ActionSpec(
        name="transport",
        label="반송요청명령",
        required_params=["carrier_id", "eqp_id"],
        param_prompts={
            "carrier_id": "반송할 캐리어 ID를 알려주세요. (예: 6PDMQ283)",
            "eqp_id": "목적지 장비 ID를 알려주세요. (예: STK102) "
                      "다른 캐리어가 있는 위치로 보내려면 '<캐리어ID> 위치로'라고 답하셔도 됩니다.",
        },
        validate=tools.transport_validate_tool,
        confirm_text=tools.transport_confirm_tool,
        execute=tools.transport_execute_tool,
        referencable={"eqp_id": ["carrier_location", "log_analysis"]},
    ),
    "dest_req": ActionSpec(
        name="dest_req",
        label="목적지요청",
        required_params=["carrier_id"],
        param_prompts={
            "carrier_id": "목적지요청할 캐리어 ID를 알려주세요. (예: 6PDMQ283)",
        },
        validate=tools.dest_req_validate_tool,
        confirm_text=tools.dest_req_confirm_tool,
        execute=tools.dest_req_execute_tool,
        referencable={},
    ),
}

# 액션 자체가 미확정일 때 묻는 질문
ACTION_SELECT_PROMPT = (
    "어떤 명령을 실행할까요?\n"
    "1) 반송요청명령(transport) — 캐리어를 특정 장비로 반송\n"
    "2) 목적지요청(dest_req) — 캐리어의 목적지 배정 요청\n"
    "('반송' 또는 '목적지'라고 답해주세요. 취소하려면 '취소')"
)

# 참조 해석 종류 -> 담당 헬퍼 에이전트 (needs-핸드오프 라우팅 표)
REFERENCE_AGENT = {
    "carrier_location": "LocationAgent",
    "log_analysis": "LogAgent",
}
