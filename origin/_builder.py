"""팀 그래프 빌더.  [원본·첨부]

흐름 요약
    START -> Router
      Router -> GeneralAgent    (일반 질의)
      Router -> Supervisor      (업무 질의)

      GeneralAgent -> FINISH     -> FinalGeneralAgent -> END
      GeneralAgent -> Supervisor (handoff: 업무 질의로 재판정된 경우)

      Supervisor -> 워커 노드 -> 다시 Supervisor
      Supervisor -> FINISH       -> FinalAnswerAgent -> END

워커는 실행 후 항상 Supervisor 로 돌아온다.
그래서 사용자 질의가 들어오면 언제나 Supervisor 부터 다시 판단하게 된다.

build_team_graph 는 매개변수를 받지 않는다 — 모델명은 빌더가 아니라
state["model_name"] 으로 흐른다 (app 도 동일하게 맞춰져 있다).

app/_builder.py 와의 차이는 하나다: ActionAgent 자리가 평범한 워커 노드다.
(app 은 여기에 턴 기반 HITL 단일 노드를 끼운다 — 그게 이번 작업)
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from origin import _node, _state

# ── 프로세스 공용 체크포인터 ──────────────────────────────────────────────
# HITL 진행 상태가 아니라 '대화 맥락' 이 여기 달려 있다. 채팅 세션 하나가
# thread_id 하나이고, 그 스레드의 messages 를 체크포인터가 들고 있다.
SHARED_CHECKPOINTER = MemorySaver()


def build_team_graph():
    """그래프와 체크포인터를 만들어 돌려준다.

    매개변수는 없다. 모델명은 빌더가 아니라 state["model_name"] 으로 흐르고,
    체크포인터는 프로세스 공용 SHARED_CHECKPOINTER 하나다.
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

    # Router -> 일반 / 업무
    # Router 는 next 에 노드 이름이 아니라 route 값을 담아 준다
    router_conditional_map = {
        "supervisor": "Supervisor",
        "general": "GeneralAgent",
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

    graph = workflow.compile(checkpointer=SHARED_CHECKPOINTER)

    return graph, SHARED_CHECKPOINTER
