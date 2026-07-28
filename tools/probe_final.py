"""FinalAnswerAgent 가 처리 결과의 Job ID·사유를 그대로 옮기는지 확인한다.

    python tools/probe_final.py

시나리오 테스트(A/B/E)는 최종 답변에 "TJ-" 나 "실행하지 않았습니다" 가 들어있는지
본다. 전체 그래프를 돌리면 케이스당 수 분이 걸려 반복 확인이 어려우므로,
최종 응답 단계만 떼어 여러 번 돌려 본다.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage

from app import _agent

N = 3

EXECUTED = ("✅ 반송요청명령 실행 완료\n"
            "- Job ID: TJ-20260729-0003\n"
            "- 상태: QUEUED\n"
            "- carrier_id: 6PDMQ283\n"
            "- eqp_id: STK102")

REJECTED = ("🚫 목적지요청 을(를) 실행하지 않았습니다.\n"
            "- 사유: 사용자가 실행을 거절해 명령을 종료합니다.")


def state(user: str, worker: str):
    return {"messages": [
        HumanMessage(content=user),
        AIMessage(content="[ExtractAgent] fab=M16, carrier_ids=['6PDMQ283']",
                  additional_kwargs={"agent_name": "ExtractAgent"}),
        AIMessage(content=worker, additional_kwargs={"agent_name": "ActionAgent"}),
    ]}


async def probe(label, user, worker, must_have):
    agent = _agent.create_final_agent()
    ok = 0
    for _ in range(N):
        out = await agent(state(user, worker), None, worker)
        text = str(out["messages"][0].content)
        if must_have in text:
            ok += 1
        else:
            print(f"    누락! -> {text[:120]!r}")
    print(f"  {label:28s} {ok}/{N}  ({must_have!r} 포함)")


async def main():
    print("FinalAnswerAgent 근거 보존 확인\n")
    await probe("실행 완료 -> Job ID", "6PDMQ283 를 STK102 로 반송해줘", EXECUTED, "TJ-")
    await probe("거절 -> 사유", "9ZXCV456 목적지 요청해줘", REJECTED, "실행하지 않았습니다")


if __name__ == "__main__":
    asyncio.run(main())
