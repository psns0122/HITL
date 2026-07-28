"""LLM 팩토리.  [원본·규약]

사내 게이트웨이는 OpenAI 호환 엔드포인트라 ChatOpenAI 로 붙는다.
호출 패턴은 사내 pptx-vision-rag/llm_client.py 와 동일하다.

원본에는 '프론트에서 모델 고르기' 가 없어 model_name 인자가 대부분
None 으로 들어온다. (app/_llm.py 에서 모델 목록 API 가 추가됐다)
"""
from langchain_openai import ChatOpenAI

from origin import config as cfg


def get_llm(model_name: str = None, temperature: float = 0):
    """에이전트 공용 LLM 팩토리."""
    return ChatOpenAI(
        base_url=cfg.LLM_GATEWAY_BASE_URL,
        # 게이트웨이가 키를 요구하지 않아 빈 값이 와도 동작하도록 placeholder 사용
        api_key=cfg.LLM_GATEWAY_API_KEY or "EMPTY",
        model=model_name or cfg.LLM_CHAT_MODEL,
        temperature=temperature,
        max_tokens=cfg.LLM_MAX_TOKENS,
        max_retries=cfg.LLM_MAX_RETRIES,
        timeout=cfg.LLM_TIMEOUT,   # 무한 대기 방지 — 멈춤 대신 명확한 타임아웃 에러
    )


# 기본 온도(=0) 인스턴스. 노드들이 이 별칭을 참조한다.
llm_t1 = get_llm()
