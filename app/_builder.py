"""팀 그래프 빌더.

배선은 origin/_builder.py 와 동일하다. 다른 곳은 ************* 로 표시했다.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app import _node, _state

# *************  [app — origin 은 빌드마다 MemorySaver() 를 새로 만든다]
# 프로세스 공용 체크포인터. graph_service 가 모델별로 그래프를 여러 벌 만들어도
# 체크포인터는 하나여야 한다 — 안 그러면 승인 대기 중 모델을 바꾸는 순간
# 그 스레드의 HITL 진행 상태가 다른 체크포인터로 가서 고아가 된다.
SHARED_CHECKPOINTER = InMemorySaver()   # (구)MemorySaver — langgraph 1.x 표준명
# *************


def build_team_graph(checkpointer=None):
    """그래프와 체크포인터를 만들어 돌려준다.

    모델명은 받지 않는다. 노드가 실행 시점에 state["model_name"] 을 읽는다.
    """
    workflow = StateGraph(_state.AgentState)

    # --- 노드 등록
    workflow.add_node("Router", _node.router_node)
    workflow.add_node("GeneralAgent", _node.general_node)
    workflow.add_node("Supervisor", _node.supervisor_node)

    workflow.add_node("StatusAgent", _node.status_node)
    workflow.add_node("LocationAgent", _node.location_node)
    workflow.add_node("LogAgent", _node.log_node)
    workflow.add_node("ExtractAgent", _node.extract_node)
    workflow.add_node("ActionAgent", _node.action_node)

    workflow.add_node("FinalAnswerAgent", _node.final_node)
    workflow.add_node("FinalGeneralAgent", _node.final_general_node)

    # --- 배선
    # 워커는 실행 후 무조건 Supervisor 로 복귀한다
    for member in _node.members:
        workflow.add_edge(member, "Supervisor")

    workflow.add_edge(START, "Router")

    # Router -> 일반 / 업무 (router_node 가 next 에 노드 이름을 담아 준다)
    router_conditional_map = {
        "Supervisor": "Supervisor",
        "GeneralAgent": "GeneralAgent",
    }
    workflow.add_conditional_edges("Router", lambda s: s["next"], router_conditional_map)

    # GeneralAgent -> 일반 답변으로 끝내거나, 업무 질의면 Supervisor 로 handoff
    general_conditional_map = {
        "Supervisor": "Supervisor",
        "FINISH": "FinalGeneralAgent",
    }
    workflow.add_conditional_edges("GeneralAgent", lambda s: s["next"], general_conditional_map)

    # Supervisor -> 워커 또는 최종 답변
    supervisor_conditional_map = {m: m for m in _node.members}
    supervisor_conditional_map["FinalAnswerAgent"] = "FinalAnswerAgent"
    supervisor_conditional_map["FINISH"] = "FinalAnswerAgent"
    # *************  [app — HITL 질문을 던진 턴은 FinalAnswer 없이 그대로 끝난다]
    supervisor_conditional_map["END"] = END
    # *************
    workflow.add_conditional_edges("Supervisor", lambda s: s["next"], supervisor_conditional_map)

    workflow.add_edge("FinalAnswerAgent", END)
    workflow.add_edge("FinalGeneralAgent", END)

    checkpointer = checkpointer or SHARED_CHECKPOINTER
    graph = workflow.compile(checkpointer=checkpointer)

    return graph, checkpointer
