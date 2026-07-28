"""요청/응답 스키마.  [원본·추정]

`ChatResponse` 가 `text` 와 `thread_id` 둘만 관리한다는 것, `thread_id` 의
기본값이 "local_test" 라는 것은 확인됐다. 나머지는 routes.py 가 실제로
채우는 값에서 역산했다. **사내 실물로 덮어써 주세요.**

app/api/schemas.py 와의 차이는 HITL 관련 두 가지다.
  - ChatRequest.model_name  : 원본에는 없다 (모델 선택이 추가되면서 생김)
  - ChatResponse.interrupted: 원본에는 없다 (멈출 일이 없다)
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /chat/stream 요청 본문."""

    query: str = Field(..., description="사용자 입력")

    thread_id: str = Field(
        "local_test",
        description="채팅 세션 식별자. 세션당 하나를 유지해야 대화 맥락이 이어진다. "
                    "기본값은 로컬 테스트용이라 실제 프론트는 항상 보내야 한다.",
    )

    recursion_limit: int = Field(
        20,
        description="LangGraph recursion limit. 노드가 이 횟수를 넘게 돌면 중단된다.",
    )


class ChatResponse(BaseModel):
    """최종 응답.

    text 와 thread_id 만 관리한다. route / step_history 같은 실행 내역은
    응답으로 내보내지 않고 일별 jsonl 로그에만 남긴다 — 프론트가 쓰지 않는
    값을 응답에 실으면 계약만 넓어진다.
    """

    text: str = Field("", description="최종 답변 전문")
    thread_id: str = Field("", description="이 응답이 속한 세션")
