"""팀 그래프 빌더.

사내 배선(shared_code.md §2)을 그대로 유지하고,
ActionAgent 자리에만 HITL 서브그래프를 끼워 넣는다.

흐름 요약
    START -> Router
      Router -> GeneralAgent    (일반 질의)
      Router -> Supervisor      (업무 질의)

      GeneralAgent -> FINISH     -> FinalGeneralAgent -> END
      GeneralAgent -> Supervisor (handoff: 업무 질의로 재판정된 경우)

      Supervisor -> 워커 노드 -> 다시 Supervisor
      Supervisor -> FINISH       -> FinalAnswerAgent -> END

워커는 실행 후 항상 Supervisor 로 돌아온다.
그래서 사용자 질의가 들어오면 언제나 Supervisor 부터 다시 판단하게 되고,
Supervisor 가 그 턴의 첫 워커로 ExtractAgent 를 태운다.
"""
import functools

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app import _node, _state
from app.actions.graph import build_action_graph


def build_team_graph(model_name: str = None):
    """그래프와 체크포인터를 만들어 돌려준다.

    Args:
        model_name: 프론트에서 고른 모델. 노드가 partial 로 받아 쓴다.
                    (실행 시점에는 state["model_name"] 이 우선한다)
    """
    workflow = StateGraph(_state.AgentState)

    # --- 노드 등록
    workflow.add_node("Router", _node.router_node)
    workflow.add_node("GeneralAgent", _node.general_node)
    workflow.add_node("Supervisor", _node.supervisor_node)

    workflow.add_node("StatusAgent",
                      functools.partial(_node.status_node, model_name=model_name))
    workflow.add_node("LocationAgent",
                      functools.partial(_node.location_node, model_name=model_name))
    workflow.add_node("LogAgent",
                      functools.partial(_node.log_node, model_name=model_name))
    workflow.add_node("ExtractAgent",
                      functools.partial(_node.extract_node, model_name=model_name))

    # ActionAgent = HITL 서브그래프 (부모 입장에선 member 노드 하나)
    workflow.add_node("ActionAgent", build_action_graph())

    workflow.add_node("FinalAnswerAgent",
                      functools.partial(_node.final_node, model_name=model_name))
    workflow.add_node("FinalGeneralAgent",
                      functools.partial(_node.final_general_node, model_name=model_name))

    # --- 배선
    # 워커는 실행 후 무조건 Supervisor 로 복귀한다
    for member in _node.members:
        workflow.add_edge(member, "Supervisor")

    workflow.add_edge(START, "Router")

    # Router -> 일반 / 업무
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
    workflow.add_conditional_edges("Supervisor", lambda s: s["next"], supervisor_conditional_map)

    workflow.add_edge("FinalAnswerAgent", END)
    workflow.add_edge("FinalGeneralAgent", END)

    # HITL 은 체크포인터가 있어야 동작한다 (interrupt 후 재개)
    checkpointer = InMemorySaver()   # (구)MemorySaver — langgraph 1.x 표준명
    graph = workflow.compile(checkpointer=checkpointer)

    return graph, checkpointer
