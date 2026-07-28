"""요청/응답 스키마.  [원본·추정]

`ChatResponse` 에 `text` 필드가 있다는 것만 확실히 확인됐다.
나머지 필드는 routes.py 가 실제로 채우는 값에서 역산했다.
**사내 실물로 덮어써 주세요.**

app/api/schemas.py 와의 차이는 HITL 관련 두 가지다.
  - ChatRequest.model_name  : 원본에는 없다 (모델 선택이 추가되면서 생김)
  - ChatResponse.interrupted: 원본에는 없다 (멈출 일이 없다)
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat/stream 요청 본문."""

    query: str = Field(..., description="사용자 입력")

    thread_id: str = Field(
        ...,
        description="채팅 세션 식별자. 세션당 하나를 유지해야 대화 맥락이 이어진다.",
    )

    recursion_limit: int = Field(
        20,
        description="LangGraph recursion limit. 노드가 이 횟수를 넘게 돌면 중단된다.",
    )


class ChatResponse(BaseModel):
    """스트림 종료 후 최종 결과."""

    text: str = Field("", description="최종 답변 전문")
    thread_id: str = Field("", description="이 응답이 속한 세션")
    route: str | None = Field(None, description="general | supervisor")
    step_history: list = Field(default_factory=list, description="실행된 노드 순서")
