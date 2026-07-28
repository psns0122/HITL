"""그래프 빌드 캐시.

LangGraph 빌드는 무거워서 매 요청마다 새로 만들면 안 된다.
프로세스에 그래프 하나만 두고 재사용한다 (사내 원본과 동일).

모델별 캐시는 없다 — 모델명은 빌더가 아니라 state["model_name"] 으로
흐르고 노드가 실행 시점에 읽으므로, 그래프는 한 벌이면 된다.
"""
from app._builder import build_team_graph

_graph = None
_checkpointer = None


def get_team_graph():
    """프로세스 공용 (graph, checkpointer). 처음 호출될 때 한 번만 빌드한다."""
    global _graph, _checkpointer

    if _graph is None:
        print("[GRAPH] 빌드", flush=True)
        _graph, _checkpointer = build_team_graph()

    return _graph, _checkpointer
