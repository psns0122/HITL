"""HITL 답변 분류(classify_collect_answer / classify_confirm) 를 확인한다.

    python tools/probe_collect.py

여기가 흔들리면 증상이 눈에 잘 띈다.
  · 값을 줬는데 empty 로 보면 같은 질문을 반복한다.
  · 딴 얘기를 value 로 보면 대화에서 못 빠져나온다.
  · 정정 시도를 reject 로 보면 지금까지 모은 값이 통째로 버려진다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import _prompt
from app import _node as _agent

N = 3

# 질문 문구는 **실제 코드가 쓰는 것을 그대로** 가져온다.
# 여기서 축약한 문구를 쓰면 probe 는 통과하는데 앱은 실패한다(실제로 겪었다:
# eqp_id 질문 뒤에 붙는 "'<캐리어ID> 위치로'라고 답하셔도 됩니다" 한 문장이
# 분류 결과를 바꿨는데, 축약 probe 가 그걸 못 잡았다).
Q_ACTION = _prompt.action_select_prompt()
Q_EQP = _prompt.action_catalog()["transport"]["param_prompts"]["eqp_id"]

# (필드, 질문, 답변, 기대 kind, 기대 value/None)
COLLECT = [
    # action 을 묻는 중 — 낱말만 와도 action 이어야 한다
    ("action", Q_ACTION, "반송으로", "action", "transport"),
    ("action", Q_ACTION, "반송", "action", "transport"),
    ("action", Q_ACTION, "목적지요청", "action", "dest_req"),
    ("action", Q_ACTION, "1번", "action", "transport"),
    # eqp_id 를 묻는 중
    ("eqp_id", Q_EQP, "STK102", "value", None),
    # consult/value 둘 다 정답이다. value 로 와도 _util 이 판독기로 ID 종류를
    # 보고("eqp 를 물었는데 carrier 가 왔다") 상담으로 돌린다.
    ("eqp_id", Q_EQP, "9ZXCV456 위치로", ("consult", "value"), None),
    # 명령형 어미가 붙어도 action_* 로 새면 안 된다 (지금은 eqp_id 를 묻는 중)
    ("eqp_id", Q_EQP, "9ZXCV456 있는 위치로 채워줘", ("consult", "value"), None),
    ("eqp_id", Q_EQP, "STK102 로 해줘", ("value", "consult"), None),
    ("eqp_id", Q_EQP, "그만할래", "cancel", None),
    ("eqp_id", Q_EQP, "6PDMQ283 지금 어디 있어?", "switch", None),
    ("eqp_id", Q_EQP, "ㅋㅋ", "empty", None),
]

# (답변, 기대 verdict)
CONFIRM = [
    ("승인", "approve"),
    ("ㄱㄱ", "approve"),
    ("아니 거절", "reject"),
    ("STK103 으로 바꿔줘", "unclear"),
]


def main():
    print(f"HITL 답변 분류 확인 — 케이스당 {N}회\n")
    total = done = 0

    print("[classify_collect_answer]")
    for field, q, ans, exp_kind, exp_val in COLLECT:
        ok, seen = 0, []
        for _ in range(N):
            r = _agent.classify_collect_answer(
                field, ans, "transport" if field != "action" else None, question=q)
            allowed = exp_kind if isinstance(exp_kind, tuple) else (exp_kind,)
            good = r.get("kind") in allowed
            if exp_val:
                good = good and r.get("value") == exp_val
            seen.append((r.get("kind"), r.get("value")))
            ok += bool(good)
        total += N
        done += ok
        mark = "OK " if ok == N else ("~  " if ok else "NG ")
        label = "|".join(exp_kind) if isinstance(exp_kind, tuple) else exp_kind
        print(f"  {mark} {field:8s} {ans[:22]:24s} 기대={label:14s} {ok}/{N}"
              + ("" if ok == N else f"  실제={seen}"))

    print("\n[classify_confirm]")
    for ans, exp in CONFIRM:
        ok, seen = 0, []
        for _ in range(N):
            v = _agent.classify_confirm(ans, action="transport",
                                        params={"carrier_id": "6PDMQ283",
                                                "eqp_id": "STK102"})
            seen.append(v)
            ok += (v == exp)
        total += N
        done += ok
        mark = "OK " if ok == N else ("~  " if ok else "NG ")
        print(f"  {mark} {ans[:22]:24s} 기대={exp:8s} {ok}/{N}"
              + ("" if ok == N else f"  실제={seen}"))

    print(f"\n  합계 {done}/{total}")


if __name__ == "__main__":
    main()
