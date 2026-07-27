"""그래프 빌드 캐시 — 서버 기동 후 1회 빌드."""
import asyncio

from app._builder import build_team_graph

_graph = None
_checkpointer = None
_lock = asyncio.Lock()


async def get_team_graph():
    """(graph, checkpointer) 튜플 반환. 호출부에서 언패킹."""
    global _graph, _checkpointer
    if _graph is not None:
        return _graph, _checkpointer

    async with _lock:
        if _graph is None:
            _graph, _checkpointer = build_team_graph()
    return _graph, _checkpointer
