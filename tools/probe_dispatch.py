"""needs-핸드오프 배분이 '올바른 대상'을 조회하는지 확인한다.

    python tools/probe_dispatch.py

ActionAgent 는 값을 못 읽으면 Supervisor 에 상담한다. Supervisor 는 (1) 어느
워커가 풀 수 있는지와 (2) 그 워커에게 보낼 질의문을 만든다.

여기서 틀리기 쉬운 것: 이미 확정된 파라미터(= 명령의 대상)를 조회해 버리는 것.
"6PDMQ283 를 ZZZZ9999 있는 위치로 반송해줘" 에서 6PDMQ283 의 위치를 물으면
자기가 지금 있는 자리를 목적지로 잡아 검증에서 튕긴다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _prompt
from app import _node as _agent
members = _agent.members

N = 3
HELPERS = [m for m in members if m != "ExtractAgent"]

Q_EQP = ("목적지 장비 ID를 알려주세요. (예: STK102) "
         "다른 캐리어가 있는 위치로 보내려면 '<캐리어ID> 위치로'라고 답하셔도 됩니다.")

# (라벨, needs, 기대 워커, query 에 있어야 하는 문자열, query 에 없어야 하는 문자열)
CASES = [
    ("참조 캐리어 존재",
     {"fill": "eqp_id", "question": Q_EQP,
      "answer": "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘",
      "params": {"carrier_id": "6PDMQ283"}},
     "LocationAgent", "9ZXCV456", "6PDMQ283"),

    ("참조 캐리어 없음",
     {"fill": "eqp_id", "question": Q_EQP,
      "answer": "6PDMQ283 를 ZZZZ9999 있는 위치로 반송해줘",
      "params": {"carrier_id": "6PDMQ283"}},
     "LocationAgent", "ZZZZ9999", "6PDMQ283"),

    ("로그 분석 참조",
     {"fill": "eqp_id", "question": Q_EQP,
      "answer": "로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘",
      "params": {"carrier_id": "6PDMQ283"}},
     "LogAgent", "6PDMQ283", None),

    ("HITL 답변 중 참조",
     {"fill": "eqp_id", "question": Q_EQP,
      "answer": "9ZXCV456 위치로",
      "params": {"carrier_id": "7HITL001"}},
     "LocationAgent", "9ZXCV456", "7HITL001"),
]


def main():
    print("needs_dispatch 배분 확인\n")
    print("--- 프롬프트 중괄호 렌더링 확인 ---")
    for line in _prompt.needs_dispatch_prompt(HELPERS).splitlines():
        if "carrier_id" in line or "AAAA" in line or "BBBB" in line:
            print("   ", line.strip())
    print()

    total = 0
    for label, needs, exp_agent, must, must_not in CASES:
        ok = 0
        seen = []
        for _ in range(N):
            d = _agent.needs_dispatch(needs, HELPERS)
            q = str(d.get("query") or "")
            seen.append((d.get("agent"), q))
            good = d.get("agent") == exp_agent and must in q
            if must_not:
                good = good and must_not not in q
            ok += bool(good)
        total += ok
        mark = "OK " if ok == N else ("~  " if ok else "NG ")
        print(f"  {mark} {label:16s} 기대={exp_agent:14s} {ok}/{N}")
        if ok < N:
            for a, q in seen:
                print(f"        {a} / {q!r}")

    print(f"\n  합계 {total}/{len(CASES) * N}")


if __name__ == "__main__":
    main()
