"""요청/응답 스키마."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat/stream 요청 본문."""

    query: str = Field(
        ...,
        description="사용자 입력. HITL 대기 중이면 그 질문에 대한 답변으로 해석된다.",
    )

    thread_id: str = Field(
        "local_test",
        description="채팅 세션 식별자. 세션당 하나를 계속 유지해야 대화 맥락이 이어진다. "
                    "기본값은 로컬 테스트용이라 실제 프론트는 항상 보낸다.",
    )

    model_name: str | None = Field(
        None,
        description="프론트에서 고른 모델. None 이면 .env 의 기본 모델을 쓴다.",
    )

    recursion_limit: int = Field(
        20,
        ge=1,
        le=200,
        description="LangGraph recursion limit. 노드가 이 횟수를 넘게 돌면 중단된다.",
    )


class ChatResponse(BaseModel):
    """최종 응답. 사내 원본과 동일하게 두 필드만 관리한다.

    route / step_history 같은 실행 내역은 응답으로 내보내지 않는다.
    프론트가 쓰지 않는 값을 응답에 실으면 계약만 넓어진다 — 그건 일별
    jsonl 로그에만 남는다.
    HITL 대기 여부도 여기가 아니라 스트림의 needs_input 제어 프레임으로 간다.
    """

    text: str = Field("", description="최종 답변 전문")
    thread_id: str = Field("", description="이 응답이 속한 세션")


class StopRequest(BaseModel):
    """POST /chat/stop 요청 본문."""

    thread_id: str = Field(..., description="중단할 세션")
