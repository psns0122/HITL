"""run_action_agent 가 명령·파라미터를 제대로 읽는지 확인한다.

    python tools/probe_intent.py

ActionAgent 는 react agent 다 — 프롬프트가 가르친 순서대로
params_extract -> param_check -> validate 툴을 스스로 부르고,
우리는 마지막 param_check_tool 호출의 인자에서 (action, params) 를 읽는다.

주의할 것.
  · 옮길 대상(carrier_id)과 위치 기준으로 언급된 다른 캐리어가 뒤집히기 쉽다.
  · 어느 명령인지 불명한 발화를 둘 중 하나로 찍으면 안 된다 -> action=None
    이어야 ActionAgent 가 사용자에게 되물을 수 있다.
  · 툴 호출 자체를 안 하는 모델(텍스트로만 답함)이면 전부 None 이 나온다 —
    그건 모델의 tool-calling 미지원 문제다 (ollama 의 glm4:9b 가 그랬다).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app._node import run_action_agent

N = 3

# (발화, 기대 action, 기대 params)
CASES = [
    ("6PDMQ283 반송해줘", "transport", {"carrier_id": "6PDMQ283"}),
    ("7HITL001 반송 부탁해", "transport", {"carrier_id": "7HITL001"}),
    ("6PDMQ283 를 STK102 로 반송해줘", "transport",
     {"carrier_id": "6PDMQ283", "eqp_id": "STK102"}),
    ("6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘", "transport",
     {"carrier_id": "6PDMQ283"}),
    ("로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘", "transport",
     {"carrier_id": "6PDMQ283"}),
    ("9ZXCV456 목적지 요청", "dest_req", {"carrier_id": "9ZXCV456"}),
    # 액션 불명 -> None 이어야 한다 (찍으면 안 된다)
    ("6PDMQ283 에 명령 실행해줘", None, {"carrier_id": "6PDMQ283"}),
]


async def main():
    print(f"run_action_agent 확인 — 케이스당 {N}회\n")
    total = 0
    for text, exp_action, exp_params in CASES:
        ok, seen = 0, []
        for _ in range(N):
            r = await run_action_agent(text)
            got_params = {k: v.upper() for k, v in r["params"].items()}
            good = r["action"] == exp_action and got_params == exp_params
            seen.append((r["action"], got_params))
            ok += bool(good)
        total += ok
        mark = "OK " if ok == N else ("~  " if ok else "NG ")
        print(f"  {mark} {text[:34]:36s} {ok}/{N}")
        if ok < N:
            print(f"        기대: {exp_action} {exp_params}")
            for s in seen:
                print(f"        실제: {s}")

    print(f"\n  합계 {total}/{len(CASES) * N}")


if __name__ == "__main__":
    asyncio.run(main())
