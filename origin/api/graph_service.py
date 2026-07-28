"""그래프 빌드 캐시.  [원본·추정]

LangGraph 빌드는 무거워서 매 요청마다 새로 만들면 안 된다.
원본은 모델이 하나뿐이라 프로세스에 그래프 하나만 두고 재사용한다.

app/api/graph_service.py 는 여기를 **모델명 키 캐시**로 바꾼 것이다
(프론트에서 모델을 고를 수 있게 되면서).
"""
from origin._builder import build_team_graph

_graph = None


def get_team_graph():
    """프로세스 공용 그래프. 처음 호출될 때 한 번만 빌드한다."""
    global _graph

    if _graph is None:
        print("[GRAPH] 빌드", flush=True)
        _graph = build_team_graph()

    return _graph
