"""팀 그래프 빌더 — shared_code.md §2 의 배선을 유지하고 ActionAgent 자리에
HITL 서브그래프를 끼워 넣는다."""
import functools

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app import _node, _state
from app.actions.graph import build_action_graph


def build_team_graph(model_name: str = None):
    workflow = StateGraph(_state.AgentState)

    workflow.add_node("Router", _node.router_node)
    workflow.add_node("GeneralAgent", _node.general_node)
    workflow.add_node("Supervisor", _node.supervisor_node)
    workflow.add_node("StatusAgent",   functools.partial(_node.status_node,   model_name=model_name))
    workflow.add_node("LocationAgent", functools.partial(_node.location_node, model_name=model_name))
    workflow.add_node("LogAgent",      functools.partial(_node.log_node,      model_name=model_name))
    # ★ ActionAgent = HITL 서브그래프 (부모 입장에선 member 노드 하나)
    workflow.add_node("ActionAgent",   build_action_graph())
    workflow.add_node("ExtractAgent",  functools.partial(_node.extract_node,  model_name=model_name))
    workflow.add_node("FinalAnswerAgent",  functools.partial(_node.final_node,         model_name=model_name))
    workflow.add_node("FinalGeneralAgent", functools.partial(_node.final_general_node, model_name=model_name))

    for member in _node.members:
        workflow.add_edge(member, "Supervisor")
    workflow.add_edge(START, "Router")

    router_conditional_map = {"Supervisor": "Supervisor", "GeneralAgent": "GeneralAgent"}
    workflow.add_conditional_edges("Router", lambda s: s["next"], router_conditional_map)

    general_conditional_map = {"Supervisor": "Supervisor", "FINISH": "FinalGeneralAgent"}
    workflow.add_conditional_edges("GeneralAgent", lambda s: s["next"], general_conditional_map)

    supervisor_conditional_map = {m: m for m in _node.members}
    supervisor_conditional_map["FinalAnswerAgent"] = "FinalAnswerAgent"
    supervisor_conditional_map["FINISH"] = "FinalAnswerAgent"
    workflow.add_conditional_edges("Supervisor", lambda s: s["next"], supervisor_conditional_map)

    workflow.add_edge("FinalAnswerAgent", END)
    workflow.add_edge("FinalGeneralAgent", END)

    checkpointer = InMemorySaver()   # (구)MemorySaver — langgraph 1.x 표준명
    graph = workflow.compile(checkpointer=checkpointer)

    return graph, checkpointer
