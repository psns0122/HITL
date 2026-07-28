"""공용 헬퍼.  [원본·첨부]

핵심은 agent_node — 모든 워커 노드가 이 한 줄에 위임하므로,
워커 노드는 "에이전트를 만들고 넘긴다" 외에 할 일이 없다.

여기 없는 함수는 임의로 만들지 않는다. 이 파일에 무엇이 더 있는지는
사내 실물 확인 대기. (safe_tool 은 _util 이 아니라 _tool.py 에 있다)
"""
from langchain_core.messages import AIMessage, HumanMessage


async def agent_node(state, agent, name: str):
    """react agent 를 실행하고 결과를 AgentState 조각으로 돌려준다.  [원본·추정]

    호출 형태만 확인됐다 — `await _util.agent_node(state, agent, "ExtractAgent")`.
    본문은 받지 못했으므로 아래는 추정이다. **사내 실물로 덮어쓸 것.**

    Args:
        state : 현재 그래프 상태
        agent : create_react_agent 로 만든 실행기
        name  : 에이전트 이름. 결과 AIMessage 의 name 으로 박힌다.

    결과에 name 을 박아 두는 게 중요하다. Supervisor 가 "이번 턴에 누가
    답했는지" 를 이 name 으로 판정하기 때문이다.
    """
    print(f"[NODE] {name} entered", flush=True)

    # react agent 는 {"messages": [...]} 를 받아 같은 모양으로 돌려준다.
    result = await agent.ainvoke({"messages": state.get("messages", []) or []})

    # 마지막 메시지가 그 에이전트의 최종 답변이다.
    last = result["messages"][-1]
    content = getattr(last, "content", "") or ""

    print(f"[NODE] {name} done ({len(str(content))}자)", flush=True)

    return {
        "messages": [AIMessage(content=content, name=name)],
        "step": state.get("step", 0) + 1,
    }


def last_human_text(messages: list) -> str:
    """가장 최근 사용자 발화. 없으면 빈 문자열.  [원본·첨부]"""
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""
