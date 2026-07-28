"""그래프 빌드 캐시.  [원본·직접]

LangGraph 빌드는 무거워서 매 요청마다 새로 만들면 안 된다.
프로세스에 그래프 하나만 두고 재사용한다.

    graph, checkpointer = build_team_graph()

모델명을 빌더로 보내지 않는다 — 모델은 state["model_name"] 으로 흐르고
노드가 실행 시점에 읽는다.
"""
from origin._builder import build_team_graph

_graph = None
_checkpointer = None


def get_team_graph():
    """프로세스 공용 (graph, checkpointer). 처음 호출될 때 한 번만 빌드한다."""
    global _graph, _checkpointer

    if _graph is None:
        print("[GRAPH] 빌드", flush=True)
        _graph, _checkpointer = build_team_graph()

    return _graph, _checkpointer
