"""Supervisor 가 발화별로 올바른 워커를 고르는지 확인한다.

    python tools/probe_supervisor.py

두 가지를 본다.

1) 프롬프트가 JSON 을 내는가
   origin 의 supervisor_prompt 는 MessagesPlaceholder 로 끝난다. 그런데 대화의
   마지막은 거의 항상 워커의 AIMessage 라서, 작은 모델은 담당자를 고르는 대신
   그 문장을 이어 써 버린다. 프롬프트 맨 끝에 사람 차례 한 줄을 넣으면 해결된다
   (app/_node.py 의 supervisor_prompt 에 그 줄이 있다).

2) 실행 요청을 조회 워커로 새게 하지 않는가
   "6PDMQ283 를 ZZZZ9999 있는 위치로 반송해줘" 는 반송 명령이므로 ActionAgent 가
   받아야 한다. '위치' 라는 단어에 끌려 LocationAgent 로 가면 명령이 사라진다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage

from app import _llm
from app._node import RouteResponse, supervisor_prompt

N = 3

# (발화, 기대 워커) — ExtractAgent 는 이미 선행 실행됐다고 가정한 대화 모양
CASES = [
    # 실행 요청 — 참조 표현이 섞여도 ActionAgent 여야 한다
    ("6PDMQ283 반송해줘", "ActionAgent"),
    ("6PDMQ283 를 STK102 로 반송해줘", "ActionAgent"),
    ("6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘", "ActionAgent"),
    ("6PDMQ283 를 ZZZZ9999 있는 위치로 반송해줘", "ActionAgent"),
    ("로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘", "ActionAgent"),
    ("9ZXCV456 목적지 요청", "ActionAgent"),
    ("7HITL001 반송 부탁해", "ActionAgent"),
    # 순수 조회·분석 — 실행 담당자로 새면 안 된다
    ("6PDMQ283 지금 어디 있어?", "LocationAgent"),
    ("6PDMQ283 위치 알려줘", "LocationAgent"),
    ("6PDMQ283 반송 이력 분석해줘", "LogAgent"),
    ("6PDMQ283 왜 반송 실패했어?", "LogAgent"),
    ("M16 큐 상태 어때?", "StatusAgent"),
]


def messages_for(text):
    return [
        HumanMessage(content=text),
        AIMessage(content="[ExtractAgent] fab=M16, carrier_ids=['6PDMQ283'], eqp_ids=없음",
                  additional_kwargs={"agent_name": "ExtractAgent"}),
    ]


def main():
    print(f"모델 {_llm._DEFAULT_MODEL} — 케이스당 {N}회\n")
    chain = supervisor_prompt | _llm.get_llm(temperature=0).with_structured_output(
        RouteResponse, method="json_mode")

    total_ok = 0
    for text, expect in CASES:
        got, ok = [], 0
        for _ in range(N):
            try:
                out = chain.invoke({"messages": messages_for(text)})
                nxt = out.get("next") if isinstance(out, dict) else None
            except Exception as e:
                nxt = f"{type(e).__name__}"
            got.append(nxt)
            if nxt == expect:
                ok += 1
        total_ok += ok
        mark = "OK " if ok == N else ("~  " if ok else "NG ")
        print(f"  {mark} {text[:34]:36s} 기대={expect:14s} {ok}/{N} {got}")

    print(f"\n  합계 {total_ok}/{len(CASES) * N}")


if __name__ == "__main__":
    main()
