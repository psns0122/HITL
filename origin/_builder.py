"""팀 그래프 빌더.  [원본·첨부]

배선은 단순하다.

    START -> Router
      Router -> GeneralAgent    (일반 질의)
      Router -> Supervisor      (업무 질의)

      GeneralAgent -> FINISH     -> FinalGeneralAgent -> END
      GeneralAgent -> Supervisor (handoff)

      Supervisor -> 워커 -> 다시 Supervisor
      Supervisor -> FINISH       -> FinalAnswerAgent -> END

워커는 실행 후 항상 Supervisor 로 돌아온다.

app/_builder.py 와의 차이는 두 줄이다.
  1. 체크포인터가 없다 — HITL 이 없으니 재개할 상태도 없다.
  2. Supervisor 분기표에 END 가 없다 — 턴이 질문으로 끝나는 경우가 없다.
"""
from langgraph.graph import END, START, StateGraph

from origin import _node, _state


def build_team_graph():
    """그래프를 만들어 돌려준다."""
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
    workflow.add_edge(START, "Router")

    # 워커는 실행 후 무조건 Supervisor 로 복귀한다
    for member in _node.members:
        workflow.add_edge(member, "Supervisor")

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

    return workflow.compile()
