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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app import _node, _state
from app.actions.node import build_action_node

# ── 프로세스 공용 체크포인터 ──────────────────────────────────────────────
# 그래프는 모델별로 따로 빌드되지만, 체크포인터는 절대 모델별로 나누면 안 된다.
#
# HITL 진행 상태는 thread_id 에 달려 있지 모델에 달려 있지 않다. 모델마다
# 체크포인터를 따로 두면, 사용자가 승인 대기 중에 프론트에서 모델을 바꾸는
# 순간 그 답변이 '그 스레드를 본 적 없는' 체크포인터로 가서 신규 질문으로
# 오인되고, 원래 인터럽트는 영영 고아가 된다.
# -> 하나를 공유해서 모델을 바꿔도 같은 스레드가 이어지게 한다.
SHARED_CHECKPOINTER = InMemorySaver()   # (구)MemorySaver — langgraph 1.x 표준명


def build_team_graph():
    """그래프와 체크포인터를 만들어 돌려준다.

    매개변수는 없다 (사내 원본과 동일).
    - 모델명은 빌더가 아니라 state["model_name"] 으로 흐른다.
      노드가 실행 시점에 읽으므로 그래프를 모델별로 만들 이유가 없다.
    - 체크포인터는 프로세스 공용 SHARED_CHECKPOINTER 하나다.
    """
    workflow = StateGraph(_state.AgentState)

    # --- 노드 등록
    workflow.add_node("Router", _node.router_node)
    workflow.add_node("GeneralAgent", _node.general_node)
    workflow.add_node("Supervisor", _node.supervisor_node)

    # 워커에는 needs-핸드오프 관련 래핑이 전혀 없다. 상담 배분과 답변 회수는
    # 전부 Supervisor 가 하고, 워커는 대화에 실려 온 질의만 평소처럼 처리한다.
    # 모델명도 안 넘긴다 — 노드가 state["model_name"] 을 직접 읽는다.
    workflow.add_node("StatusAgent", _node.status_node)
    workflow.add_node("LocationAgent", _node.location_node)
    workflow.add_node("LogAgent", _node.log_node)
    workflow.add_node("ExtractAgent", _node.extract_node)

    # ActionAgent = 턴 기반 HITL 단일 노드 (actions/node.py)
    workflow.add_node("ActionAgent", build_action_node())

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
    # HITL 질문을 던진 턴은 FinalAnswer 없이 그대로 끝난다 (사용자 응답 대기)
    supervisor_conditional_map["END"] = END
    workflow.add_conditional_edges("Supervisor", lambda s: s["next"], supervisor_conditional_map)

    workflow.add_edge("FinalAnswerAgent", END)
    workflow.add_edge("FinalGeneralAgent", END)

    # HITL 은 체크포인터가 있어야 동작한다 (진행 상태 보관/재개)
    graph = workflow.compile(checkpointer=SHARED_CHECKPOINTER)

    return graph, SHARED_CHECKPOINTER
