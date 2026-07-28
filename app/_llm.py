"""LLM 팩토리.

origin/_llm.py 와 동일하다. app 추가분은 ************* 로 표시.
"""
# *************  [app 전용 import — 아래 경고 억제에 쓴다]  *************
import warnings
# *************
from typing import Dict

import requests
from langchain_openai import ChatOpenAI

import app.config as cfg

_llm_cache: Dict[str, ChatOpenAI] = {}

_DEFAULT_MODEL = "GaiA-LLM-Latest"


def getmodellist(api_base, output=False, name=None, k=None):
    """게이트웨이의 모델 목록을 조회해 모델명 하나를 골라 돌려준다.

    Args:
        api_base : 게이트웨이 구분자. API_BASE_TEMPLATE 에 끼워진다.
        output   : True 면 조회된 모델 목록을 출력한다.
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

        # output=True 면 목록을 찍어 준다
        if output:
            print(f"[LLM] 모델 목록 {len(models)}건")
            for i, model in enumerate(models):
                model_id = model.get("id", model) if isinstance(model, dict) else model
                print(f"  [{i}] {model_id}")

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
        # *************  [app — 사내 밖에선 키가 비어 있어 placeholder 필요]
        api_key=cfg.api_key or "EMPTY",
        # *************
        model=model_name,
        temperature=temperature,
        streaming=True,
    )

    _llm_cache[cache_key] = llm
    return llm


# *************  [app 전용 — origin 에 없음]  *************

# 사내망 밖(집/로컬 ollama 등)에서 돌려볼 때만 쓰는 덮어쓰기.
#
# 위 origin 영역의 `_DEFAULT_MODEL = "GaiA-LLM-Latest"` 줄은 사내 원본과
# 한 글자도 다르지 않게 두고, 값만 여기서 갈아끼운다. get_llm 은 호출 시점에
# 모듈 전역을 읽으므로 이 재대입이 그대로 먹는다.
# .env 에 DEFAULT_MODEL 이 없으면 아무 일도 일어나지 않는다(사내 = 무변화).
# 사내 반입 시에는 이 [app 전용] 블록만 들어내면 원상복귀된다.
if cfg.DEFAULT_MODEL:
    print(f"[LLM] 기본 모델 덮어쓰기: {_DEFAULT_MODEL} -> {cfg.DEFAULT_MODEL}", flush=True)
    _DEFAULT_MODEL = cfg.DEFAULT_MODEL

# 프론트 드롭다운에 띄울 모델들. 첫 번째가 기본값이다.
# .env 의 AVAILABLE_MODELS 가 있으면 그걸 쓴다 (로컬 실행용).
AVAILABLE_MODELS = cfg.AVAILABLE_MODELS or [
    "GaiA-LLM-Latest",
    "gaia-GLM-5.2",
    "Qwen3.5-397B-A17B-FP8",
]


def default_model_name() -> str:
    """모델을 지정하지 않았을 때 쓸 기본 모델 (routes /health 등이 참조)."""
    return _DEFAULT_MODEL


def list_models() -> dict:
    """GET /models 응답 — 게이트웨이 /models 와 같은 형태."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "in-house",
                "is_default": model_id == _DEFAULT_MODEL,
            }
            for model_id in AVAILABLE_MODELS
        ],
    }


# 구조화 출력 때마다 콘솔에 뜨는 pydantic 직렬화 경고를 지운다.
#
#   PydanticSerializationUnexpectedValue(Expected `none` ...
#       [field_name='parsed', input_value=SupervisorOut(next='ActionAgent')])
#
# 에러가 아니다. langchain_openai 가 응답을 model_dump() 할 때 openai SDK 의
# ParsedChatCompletionMessage.parsed 가 제네릭 미지정(=None 타입)이라서 나는
# 상위 라이브러리 잡음이다. 판정 결과(SupervisorOut)는 정상으로 파싱돼 온다.
# 실행에는 영향이 없는데 콘솔에서는 에러처럼 보여 원인 추적을 방해하므로
# 이 문구만 좁게 막는다. (호출 방식을 바꿔 피하는 방법도 있지만, 게이트웨이가
#  지금 방식으로 잘 받고 있어 프로토콜은 건드리지 않는다)
warnings.filterwarnings("ignore",
                        message="Pydantic serializer warnings",
                        category=UserWarning)


def structured_invoke(llm, schema, messages, config=None):
    """구조화 출력 (HITL 판정용). 실패 시 예외를 그대로 올려 호출부가 폴백한다."""
    runner = llm.with_structured_output(schema)
    return runner.invoke(messages, config=config)

# *************  [app 전용 끝]  *************
