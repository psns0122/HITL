"""json_mode + TypedDict 구조화 출력이 현재 엔드포인트에서 되는지 실측한다.

    python tools/probe_jsonmode.py

origin 은 `.with_structured_output(RouteResponse, method="json_mode")` 를 쓴다.
app 을 origin 관례로 되돌리기 전에, 지금 붙어 있는 게이트웨이(사내든 로컬
ollama 든)가 그 방식을 실제로 받아주는지 확인하는 용도다.
"""
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from app import _llm

options_for_next = ["FINISH", "FinalAnswerAgent", "StatusAgent", "LocationAgent",
                    "LogAgent", "ActionAgent", "ExtractAgent"]


class RouteResponse(TypedDict):
    next: Annotated[Literal[tuple(options_for_next)], "다음에 실행할 노드"]


class Verdict(TypedDict):
    verdict: Annotated[Literal["approve", "reject", "unclear"], "승인 판정"]


def probe(label, schema, method, system, user):
    llm = _llm.get_llm(temperature=0.0)
    try:
        if method:
            runner = llm.with_structured_output(schema, method=method)
        else:
            runner = llm.with_structured_output(schema)
        out = runner.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        print(f"  [OK] {label}: {out!r}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 64)
    print(f"  {_llm._DEFAULT_MODEL}")
    print("=" * 64)

    sup_sys = ("당신은 Supervisor 입니다. 다음 에이전트를 하나 고르세요.\n"
               f"{options_for_next}\n"
               "You MUST respond in JSON format with a single key 'next'. "
               'Example: {"next": "LocationAgent"}')
    sup_user = "6PDMQ283 캐리어 지금 어디 있어?"

    con_sys = ("승인 판정기입니다. approve / reject / unclear 중 하나로 판정하세요.\n"
               "You MUST respond in JSON with a single key 'verdict'.")
    con_user = "사용자의 답변: 응 실행해"

    print("\n[TypedDict + json_mode]  <- origin 방식")
    a1 = probe("RouteResponse", RouteResponse, "json_mode", sup_sys, sup_user)
    a2 = probe("Verdict", Verdict, "json_mode", con_sys, con_user)

    print("\n[TypedDict + 기본(function_calling)]")
    b1 = probe("RouteResponse", RouteResponse, None, sup_sys, sup_user)
    b2 = probe("Verdict", Verdict, None, con_sys, con_user)

    print("\n[TypedDict + json_schema]")
    c1 = probe("RouteResponse", RouteResponse, "json_schema", sup_sys, sup_user)
    c2 = probe("Verdict", Verdict, "json_schema", con_sys, con_user)

    print("\n" + "=" * 64)
    print(f"  json_mode           : {'OK' if a1 and a2 else 'FAIL'}   <- 이게 되면 origin 그대로")
    print(f"  function_calling    : {'OK' if b1 and b2 else 'FAIL'}")
    print(f"  json_schema         : {'OK' if c1 and c2 else 'FAIL'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
