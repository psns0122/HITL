"""요청/응답 스키마."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat/stream 요청 본문."""

    query: str = Field(
        ...,
        description="사용자 입력. HITL 대기 중이면 그 질문에 대한 답변으로 해석된다.",
    )

    thread_id: str = Field(
        ...,
        description="채팅 세션 식별자. 세션당 하나를 계속 유지해야 대화 맥락이 이어진다.",
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
    """비스트리밍 응답 / 스트림 종료 후 최종 결과."""

    text: str = Field("", description="최종 답변 전문")
    thread_id: str = Field("", description="이 응답이 속한 세션")
    route: str | None = Field(None, description="general | supervisor")
    step_history: list = Field(default_factory=list, description="실행된 노드 순서")
    interrupted: bool = Field(False, description="HITL 로 멈춰서 추가 입력을 기다리는 중인지")


class StopRequest(BaseModel):
    """POST /chat/stop 요청 본문."""

    thread_id: str = Field(..., description="중단할 세션")
