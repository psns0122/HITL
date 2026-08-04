"""그래프 빌드 캐시.

LangGraph 빌드는 무겁다. 매 요청마다 새로 만들면 안 되고,
그렇다고 하나만 만들어 두면 프론트에서 모델을 바꿨을 때 반영이 안 된다.

그래서 **모델명을 키로** 캐싱한다.
같은 모델이면 이미 만들어 둔 그래프를 그대로 쓰고,
처음 보는 모델이면 그때 한 번만 빌드한다.

체크포인터는 반대다 — 모델별로 나누면 안 되고 **하나를 공유**해야 한다.
HITL 진행 상태는 thread_id 에 달려 있지 모델에 달려 있지 않기 때문이다.
모델마다 따로 두면, 사용자가 승인 대기 중에 프론트에서 모델을 바꾸는 순간
그 답변이 '그 스레드를 본 적 없는' 체크포인터로 가서 신규 질문으로 오인되고,
원래 진행 중이던 명령은 영영 고아가 된다.

그래프를 여러 벌 만드는 게 이 모듈의 결정이므로, 그 여러 벌이 같은 대화
상태를 보게 하는 책임도 여기에 있다. (build_team_graph 는 자기가 몇 번
불릴지 모른다 — 그래서 checkpointer 를 인자로 받게 되어 있다)
"""
import asyncio
from typing import Dict

from langgraph.checkpoint.memory import MemorySaver

from origin._builder import build_team_graph

# 모델명 -> {"graph": ..., "checkpointer": ...}
_dynamic_graph_cache: Dict[str, dict] = {}

# 같은 모델을 동시에 두 번 빌드하지 않도록 막는 락
_lock = asyncio.Lock()

# 프로세스 공용 체크포인터. 모델이 몇 개든 이것 하나를 모든 그래프에 물린다.
_SHARED_CHECKPOINTER = MemorySaver()


async def get_team_graph(model_name: str = None):
    """모델명에 해당하는 (graph, checkpointer) 를 돌려준다.

    Args:
        model_name: 프론트에서 고른 모델. None 이면 "default" 키로 캐싱한다.

    graph 는 모델별로 다르지만 checkpointer 는 항상 같은 객체다.
    """
    cache_key = model_name if model_name else "default"

    # 1) 락 없이 빠르게 확인 — 대부분은 여기서 끝난다
    if cache_key in _dynamic_graph_cache:
        entry = _dynamic_graph_cache[cache_key]
        return entry["graph"], entry["checkpointer"]

    # 2) 없으면 락을 잡고 빌드
    async with _lock:
        # 락을 기다리는 동안 다른 요청이 이미 만들었을 수 있다
        if cache_key in _dynamic_graph_cache:
            entry = _dynamic_graph_cache[cache_key]
            return entry["graph"], entry["checkpointer"]

        print(f"[GRAPH] 빌드 (model={cache_key})", flush=True)
        graph, checkpointer = build_team_graph(_SHARED_CHECKPOINTER)

        _dynamic_graph_cache[cache_key] = {
            "graph": graph,
            "checkpointer": checkpointer,
        }

    return graph, checkpointer


def cached_models() -> list:
    """지금까지 빌드해 둔 모델 키 목록 (디버그/헬스체크용)."""
    return list(_dynamic_graph_cache.keys())
