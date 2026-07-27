"""요청 스키마."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자 입력. HITL 대기 중이면 그 질문에 대한 답변으로 해석된다.")
    thread_id: str = Field(..., description="채팅 세션 식별자 (세션당 1개 유지)")
    model_name: str | None = None


class StopRequest(BaseModel):
    thread_id: str
