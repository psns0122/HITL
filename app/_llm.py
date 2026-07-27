"""LLM 래퍼.

- FAKE_LLM=0 : 사내 OpenAI 호환 endpoint (ChatOpenAI(base_url=...))
- FAKE_LLM=1 : FakeEchoChatModel — 프롬프트의 [ECHO] 뒤 내용을 그대로(스트리밍으로)
  돌려주는 목업. LLM 없이도 on_chat_model_stream 이벤트·usage_metadata 가 실제처럼
  발생해 SSE/토큰 원장 검증이 가능하다.
"""
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

import app.config as cfg

ECHO_MARKER = "[ECHO]"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


class FakeEchoChatModel(BaseChatModel):
    """[ECHO] 마커 뒤 텍스트를 그대로 반환/스트리밍하는 목업 챗모델."""

    chunk_size: int = 8

    @property
    def _llm_type(self) -> str:
        return "fake-echo"

    def _payload(self, messages: List[BaseMessage]) -> tuple[str, int]:
        joined = "\n".join(str(m.content) for m in messages)
        out = joined.split(ECHO_MARKER, 1)[1].strip() if ECHO_MARKER in joined else "OK"
        return out, _estimate_tokens(joined)

    def _usage(self, in_tok: int, out: str) -> dict:
        out_tok = _estimate_tokens(out)
        return {"input_tokens": in_tok, "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        out, in_tok = self._payload(messages)
        msg = AIMessage(content=out, usage_metadata=self._usage(in_tok, out),
                        response_metadata={"finish_reason": "stop"})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        out, in_tok = self._payload(messages)
        for i in range(0, len(out), self.chunk_size):
            piece = out[i:i + self.chunk_size]
            last = i + self.chunk_size >= len(out)
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content=piece,
                usage_metadata=self._usage(in_tok, out) if last else None,
                response_metadata={"finish_reason": "stop"} if last else {},
            ))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                       run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
                       **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        out, in_tok = self._payload(messages)
        for i in range(0, len(out), self.chunk_size):
            piece = out[i:i + self.chunk_size]
            last = i + self.chunk_size >= len(out)
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content=piece,
                usage_metadata=self._usage(in_tok, out) if last else None,
                response_metadata={"finish_reason": "stop"} if last else {},
            ))
            if run_manager:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def get_llm(model_name: str | None = None, temperature: float | None = None):
    """에이전트 공용 LLM 팩토리. (사내 코드의 _llm.llm_t1 자리)

    사내 게이트웨이는 OpenAI 호환 엔드포인트라 ChatOpenAI 로 붙는다.
    호출 패턴은 사내 기존 프로젝트(pptx-vision-rag/llm_client.py)와 동일하게 맞췄다.
    """
    if cfg.FAKE_LLM:
        return FakeEchoChatModel()
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=cfg.GATEWAY_BASE_URL,
        # 게이트웨이가 키를 요구하지 않아 빈 값이 와도 동작하도록 placeholder 사용
        # (OpenAI SDK 는 빈 키를 거부한다)
        api_key=cfg.GATEWAY_API_KEY or "EMPTY",
        model=model_name or cfg.CHAT_MODEL,
        temperature=cfg.TEMPERATURE if temperature is None else temperature,
        max_tokens=cfg.MAX_TOKENS,
        max_retries=cfg.LLM_MAX_RETRIES,
        timeout=cfg.LLM_TIMEOUT,      # 무한 대기 방지 — 멈춤 대신 명확한 타임아웃 에러
    )


# 사내 코드 호환용 별칭 (supervisor_chain 등이 llm_t1 을 참조하는 형태)
llm_t1 = get_llm()


def structured_invoke(llm, schema, messages, config=None):
    """구조화 출력: with_structured_output 우선, 실패 시 예외를 위임(호출부 폴백)."""
    runner = llm.with_structured_output(schema)
    return runner.invoke(messages, config=config)
