"""extract_intent 가 액션·대상·참조를 제대로 가르는지 확인한다.

    python tools/probe_intent.py

주의할 두 가지.
  · 옮길 대상(carrier_id) 과 위치 기준(reference_carrier_id) 이 뒤집히기 쉽다.
  · 어느 명령인지 불명한 발화를 둘 중 하나로 찍어 버리기 쉽다 -> unknown 이어야
    ActionAgent 가 사용자에게 "반송이냐 목적지요청이냐" 를 되물을 수 있다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _agent

N = 3

# (발화, 기대 action, 기대 params, 기대 reference.kind)
CASES = [
    ("6PDMQ283 반송해줘", "transport", {"carrier_id": "6PDMQ283"}, None),
    ("7HITL001 반송 부탁해", "transport", {"carrier_id": "7HITL001"}, None),
    ("6PDMQ283 를 STK102 로 반송해줘", "transport",
     {"carrier_id": "6PDMQ283", "eqp_id": "STK102"}, None),
    ("6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘", "transport",
     {"carrier_id": "6PDMQ283"}, "carrier_location"),
    ("로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘", "transport",
     {"carrier_id": "6PDMQ283"}, "log_analysis"),
    ("9ZXCV456 목적지 요청", "dest_req", {"carrier_id": "9ZXCV456"}, None),
    # 액션 불명 -> unknown 이어야 한다 (찍으면 안 된다)
    ("6PDMQ283 에 명령 실행해줘", None, {"carrier_id": "6PDMQ283"}, None),
]


def main():
    print(f"extract_intent 확인 — 케이스당 {N}회\n")
    total = 0
    for text, exp_action, exp_params, exp_ref in CASES:
        ok, seen = 0, []
        for _ in range(N):
            r = _agent.extract_intent(text)
            ref = (r["reference"] or {}).get("kind")
            good = (r["action"] == exp_action
                    and r["params"] == exp_params
                    and ref == exp_ref)
            seen.append((r["action"], r["params"], ref))
            ok += bool(good)
        total += ok
        mark = "OK " if ok == N else ("~  " if ok else "NG ")
        print(f"  {mark} {text[:34]:36s} {ok}/{N}")
        if ok < N:
            print(f"        기대: {exp_action} {exp_params} ref={exp_ref}")
            for s in seen:
                print(f"        실제: {s}")

    print(f"\n  합계 {total}/{len(CASES) * N}")


if __name__ == "__main__":
    main()
