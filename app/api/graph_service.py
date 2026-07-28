"""그래프 빌드 캐시.

LangGraph 빌드는 무겁다. 매 요청마다 새로 만들면 안 되고,
그렇다고 하나만 만들어 두면 프론트에서 모델을 바꿨을 때 반영이 안 된다.

그래서 **모델명을 키로** 캐싱한다.
같은 모델이면 이미 만들어 둔 그래프를 그대로 쓰고,
처음 보는 모델이면 그때 한 번만 빌드한다.

체크포인터도 그래프와 짝으로 같이 들고 있어야 한다.
(HITL 재개가 체크포인터에 붙어 있기 때문)
"""
import asyncio
from typing import Dict

from app._builder import build_team_graph

# 모델명 -> {"graph": ..., "checkpointer": ...}
_dynamic_graph_cache: Dict[str, dict] = {}

# 같은 모델을 동시에 두 번 빌드하지 않도록 막는 락
_lock = asyncio.Lock()


async def get_team_graph(model_name: str = None):
    """모델명에 해당하는 (graph, checkpointer) 를 돌려준다.

    Args:
        model_name: 프론트에서 고른 모델. None 이면 "default" 키로 캐싱한다.
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
        graph, checkpointer = build_team_graph()

        _dynamic_graph_cache[cache_key] = {
            "graph": graph,
            "checkpointer": checkpointer,
        }

    return graph, checkpointer


def cached_models() -> list:
    """지금까지 빌드해 둔 모델 키 목록 (디버그/헬스체크용)."""
    return list(_dynamic_graph_cache.keys())
