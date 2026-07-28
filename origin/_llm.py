"""LLM 팩토리.  [원본·직접]

사내 게이트웨이는 OpenAI 호환 엔드포인트라 ChatOpenAI 로 붙는다.

(model_name, temperature) 조합을 키로 인스턴스를 캐싱한다. 같은 조합이면
매번 새로 만들지 않고 재사용한다.
"""
from typing import Dict

import requests
from langchain_openai import ChatOpenAI

from origin import config as cfg

_llm_cache: Dict[str, ChatOpenAI] = {}

_DEFAULT_MODEL = "GaiA-LLM-Latest"


def getmodellist(api_base, output=False, name=None, k=None):
    """게이트웨이의 모델 목록을 조회해 모델명 하나를 골라 돌려준다.

    Args:
        api_base : 게이트웨이 구분자. API_BASE_TEMPLATE 에 끼워진다.
        output   : (제공된 코드에서 사용되지 않음)
        name     : 이 이름이 목록에 있으면 그것을 고른다.
        k        : name 으로 못 골랐을 때 목록의 k 번째를 고른다.

    못 고르면 None 을 돌려준다.
    """
    base_url = cfg.API_BASE_TEMPLATE.format(api_base=api_base)
    get_model_url = f"{base_url}{cfg.MODEL_LIST_ENDPOINT}"

    HEADERS = {
        "accept": "*/*",
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    response = requests.get(get_model_url, headers=HEADERS)
    selected_model = None

    if response.status_code == 200:
        raw_data = response.json()

        # 표준 OpenAI 형식은 {"data": [...]} 이지만 리스트로 바로 오는 경우도 있다
        if isinstance(raw_data, dict) and "data" in raw_data:
            models = raw_data["data"]
        else:
            models = raw_data

        # 1) 이름으로 찾기
        if name:
            for model in models:
                if isinstance(model, dict):
                    if model.get("id") == name:
                        selected_model = name
                        break
                elif model == name:
                    selected_model = name
                    break

        # 2) 이름으로 못 찾았으면 인덱스로 찾기
        if not selected_model and k is not None:
            try:
                target = models[k]
                selected_model = target.get("id", target) if isinstance(target, dict) else target
            except Exception:
                print("[ERROR] list index out of range")

    else:
        print(response.status_code, response.text)
        selected_model = None

    return selected_model


def get_llm(model_name: str = None, temperature: float = 0) -> ChatOpenAI:
    """에이전트 공용 LLM 팩토리. (model_name, temperature) 별로 캐싱한다."""
    if model_name is None:
        model_name = _DEFAULT_MODEL

    cache_key = f"{model_name}|{temperature}"

    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    print(f"[LLM] 새 인스턴스 생성 model={model_name} temperature={temperature}", flush=True)

    base_url = cfg.API_BASE_TEMPLATE.format(api_base="hcp")
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=cfg.api_key,
        model=model_name,
        temperature=temperature,
        streaming=True,
    )

    _llm_cache[cache_key] = llm
    return llm
