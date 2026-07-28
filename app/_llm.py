"""LLM 래퍼.

프론트에서 모델을 골라 보내면 그 모델로 그래프를 돌린다.
model_name 이 None 이면 .env 의 기본 모델(LLM_CHAT_MODEL)을 쓴다.

사내 OpenAI 호환 게이트웨이(ChatOpenAI)로 붙는다.
"""
import app.config as cfg


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
