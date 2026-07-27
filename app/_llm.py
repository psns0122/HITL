"""LLM 래퍼.

프론트에서 모델을 골라 보내면 그 모델로 그래프를 돌린다.
model_name 이 None 이면 .env 의 기본 모델(LLM_CHAT_MODEL)을 쓴다.

동작 모드
- FAKE_LLM=0 : 사내 OpenAI 호환 게이트웨이 (ChatOpenAI)
- FAKE_LLM=1 : FakeEchoChatModel — 프롬프트의 [ECHO] 뒤 내용을 그대로 스트리밍으로
               돌려주는 목업. LLM 없이도 on_chat_model_stream 이벤트와
               usage_metadata 가 실제처럼 발생해 SSE/토큰 원장 검증이 가능하다.
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

# FakeEchoChatModel 이 "여기부터 응답" 이라고 인식하는 마커
ECHO_MARKER = "[ECHO]"


# ─────────────────────────────────────────────────────────────────────────
# 프론트에 노출할 모델 목록
#   게이트웨이의 /models 응답과 같은 모양으로 돌려준다.
#   {"object": "list", "data": [{"id": ..., "object": "model", ...}]}
# ─────────────────────────────────────────────────────────────────────────

# 프론트 드롭다운에 띄울 모델들. 첫 번째가 기본값이다.
# 사내에서 게이트웨이 /models 를 직접 조회하려면 list_models() 본문만 바꾸면 된다.
AVAILABLE_MODELS = [
    "GaiA-LLM-Latest",            # 기본값
    "gaia-GLM-5.2",
    "Qwen3.5-397B-A17B-FP8",
]


def default_model_name() -> str:
    """모델을 지정하지 않았을 때 쓸 기본 모델."""
    return cfg.CHAT_MODEL or AVAILABLE_MODELS[0]


def list_models() -> dict:
    """게이트웨이 /models 와 동일한 형태의 모델 목록.

    반환 예)
        {
          "object": "list",
          "data": [
            {"id": "GaiA-LLM-Latest", "object": "model", "owned_by": "in-house"},
            ...
          ]
        }
    """
    default = default_model_name()

    # 기본 모델이 목록에 없으면 맨 앞에 끼워 넣는다
    ids = list(AVAILABLE_MODELS)
    if default not in ids:
        ids.insert(0, default)

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "in-house",
                "is_default": model_id == default,
            }
            for model_id in ids
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
# 목업 챗모델
# ─────────────────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """대충 3글자 = 1토큰으로 잡는다. 목업 usage 용."""
    return max(1, len(text) // 3)


class FakeEchoChatModel(BaseChatModel):
    """[ECHO] 마커 뒤 텍스트를 그대로 반환/스트리밍하는 목업 챗모델."""

    chunk_size: int = 8

    @property
    def _llm_type(self) -> str:
        return "fake-echo"

    def bind_tools(self, tools, **kwargs):
        """create_react_agent 가 요구해서 뚫어둔 자리.

        목업 모델은 툴을 실제로 호출하지 않으므로 자기 자신을 그대로 돌려준다.
        (이게 없으면 BaseChatModel.bind_tools 가 NotImplementedError 를 던져
         FAKE_LLM 모드에서 react 에이전트를 만들 수 없다.)
        """
        return self

    def _payload(self, messages: List[BaseMessage]) -> tuple[str, int]:
        """프롬프트에서 응답으로 쓸 부분을 잘라내고 입력 토큰 수를 센다."""
        joined = "\n".join(str(m.content) for m in messages)

        if ECHO_MARKER in joined:
            out = joined.split(ECHO_MARKER, 1)[1].strip()
        else:
            out = "OK"

        return out, _estimate_tokens(joined)

    def _usage(self, in_tok: int, out: str) -> dict:
        out_tok = _estimate_tokens(out)
        return {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        out, in_tok = self._payload(messages)

        msg = AIMessage(
            content=out,
            usage_metadata=self._usage(in_tok, out),
            response_metadata={"finish_reason": "stop"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _make_chunk(self, piece: str, is_last: bool, in_tok: int, out: str):
        """스트리밍 청크 하나. usage 는 마지막 청크에만 싣는다."""
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=piece,
                usage_metadata=self._usage(in_tok, out) if is_last else None,
                response_metadata={"finish_reason": "stop"} if is_last else {},
            )
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        out, in_tok = self._payload(messages)

        for i in range(0, len(out), self.chunk_size):
            piece = out[i:i + self.chunk_size]
            is_last = i + self.chunk_size >= len(out)

            chunk = self._make_chunk(piece, is_last, in_tok, out)
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        out, in_tok = self._payload(messages)

        for i in range(0, len(out), self.chunk_size):
            piece = out[i:i + self.chunk_size]
            is_last = i + self.chunk_size >= len(out)

            chunk = self._make_chunk(piece, is_last, in_tok, out)
            if run_manager:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


# ─────────────────────────────────────────────────────────────────────────
# LLM 팩토리
# ─────────────────────────────────────────────────────────────────────────

def get_llm(model_name: str | None = None, temperature: float | None = None):
    """에이전트 공용 LLM 팩토리. (사내 코드의 _llm.llm_t1 자리)

    Args:
        model_name : 프론트에서 선택된 모델. None 이면 .env 기본 모델.
        temperature: 샘플링 온도. None 이면 .env 기본값.

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
        model=model_name or default_model_name(),
        temperature=cfg.TEMPERATURE if temperature is None else temperature,
        max_tokens=cfg.MAX_TOKENS,
        max_retries=cfg.LLM_MAX_RETRIES,
        timeout=cfg.LLM_TIMEOUT,   # 무한 대기 방지 — 멈춤 대신 명확한 타임아웃 에러
    )


# 사내 코드 호환용 별칭 (supervisor_chain 등이 llm_t1 을 참조하는 형태)
llm_t1 = get_llm()


def structured_invoke(llm, schema, messages, config=None):
    """구조화 출력. 실패 시 예외를 그대로 올려 호출부가 폴백하게 둔다."""
    runner = llm.with_structured_output(schema)
    return runner.invoke(messages, config=config)
