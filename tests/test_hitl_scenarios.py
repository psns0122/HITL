"""HITL 시나리오 E2E 테스트 (FAKE_LLM 모드, 그래프 직접 invoke).

실행: python3 tests/test_hitl_scenarios.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app._builder import build_team_graph


async def main():
    graph, cp = build_team_graph()

    def C(t):
        return {"configurable": {"thread_id": t}}

    # A: 파라미터 수집 -> 승인 -> 실행
    cfg = C("A")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 반송해줘")]}, cfg)
    snap = await graph.aget_state(cfg)
    assert snap.interrupts[0].value["type"] == "collect_param", snap.interrupts
    await graph.ainvoke(Command(resume="STK102로 보내줘"), cfg)
    snap = await graph.aget_state(cfg)
    assert snap.interrupts[0].value["type"] == "confirm", snap.interrupts
    r = await graph.ainvoke(Command(resume="승인"), cfg)
    assert "TJ-" in r["messages"][-1].content, r["messages"][-1].content
    print("A) collect->confirm->execute PASS")

    # B: 파라미터 완비 -> confirm 거절 -> abandon
    cfg = C("B")
    await graph.ainvoke({"messages": [HumanMessage("9ZXCV456 목적지 요청해줘")]}, cfg)
    snap = await graph.aget_state(cfg)
    assert snap.interrupts[0].value["type"] == "confirm", snap.interrupts
    r = await graph.ainvoke(Command(resume="아니 거절"), cfg)
    assert "실행하지 않았습니다" in r["messages"][-1].content
    print("B) dest_req reject PASS")

    # C: 검증 실패(오프라인 장비) -> bad field 재수집 -> 실행
    cfg = C("C")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 를 DFF401 로 반송")]}, cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "collect_param" and "검증 실패" in iv["prompt"], iv
    await graph.ainvoke(Command(resume="PHT201"), cfg)
    snap = await graph.aget_state(cfg)
    assert snap.interrupts[0].value["type"] == "confirm"
    r = await graph.ainvoke(Command(resume="ㄱㄱ"), cfg)
    assert "TJ-" in r["messages"][-1].content
    print("C) validation-fail loop -> execute PASS")

    # D: 수집 중 자연어 취소
    cfg = C("D")
    await graph.ainvoke({"messages": [HumanMessage("7HITL001 반송 부탁해")]}, cfg)
    r = await graph.ainvoke(Command(resume="아 그냥 취소해줘"), cfg)
    assert "실행하지 않았습니다" in r["messages"][-1].content
    print("D) cancel mid-collect PASS")

    # E: 참조형 최초 질의 -> LocationAgent needs-핸드오프 -> 재진입 -> 실행
    cfg = C("E")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘")]}, cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "confirm" and iv["params"]["eqp_id"] == "STK102", iv
    r = await graph.ainvoke(Command(resume="승인"), cfg)
    assert "TJ-" in r["messages"][-1].content
    print("E) needs-handoff LocationAgent PASS")

    # F: 분석형 질의 -> LogAgent needs-핸드오프 (원인 장비/현재 위치 제외 권장지)
    cfg = C("F")
    await graph.ainvoke({"messages": [HumanMessage("로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘")]}, cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "confirm" and iv["params"] == {
        "carrier_id": "6PDMQ283", "eqp_id": "STK102"}, iv
    print("F) needs-handoff LogAgent PASS")

    # G: HITL 질문에 참조형 답변
    cfg = C("G")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 반송해줘")]}, cfg)
    await graph.ainvoke(Command(resume="9ZXCV456 있는 위치로 채워줘"), cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "confirm" and iv["params"]["eqp_id"] == "STK102", iv
    print("G) reference answer during HITL PASS")

    # H: 일반 질의 -> FinalGeneralAgent
    cfg = C("H")
    r = await graph.ainvoke({"messages": [HumanMessage("안녕!")]}, cfg)
    assert r["messages"][-1].name == "FinalGeneralAgent", r["messages"][-1]
    print("H) general route PASS")

    # I: /chat/stop 스타일 abort 센티널
    cfg = C("I")
    await graph.ainvoke({"messages": [HumanMessage("7HITL001 반송해줘")]}, cfg)
    r = await graph.ainvoke(Command(resume={"aborted": True}), cfg)
    assert "실행하지 않았습니다" in r["messages"][-1].content
    print("I) abort sentinel PASS")

    # J: 존재하지 않는 캐리어 위치 참조 -> 헬퍼 실패 -> HITL 강등
    cfg = C("J")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 를 ZZZZ9999 있는 위치로 반송해줘")]}, cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "collect_param" and iv["field"] == "eqp_id", iv
    assert "찾을 수 없" in iv["prompt"] or "직접" in iv["prompt"], iv["prompt"]
    print("J) helper-fail -> HITL downgrade PASS")

    # K: 수집 중 의도 전환 transport -> dest_req
    cfg = C("K")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 반송해줘")]}, cfg)
    await graph.ainvoke(Command(resume="아냐 그냥 목적지 요청으로 바꿔줘"), cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "confirm" and iv["action"] == "dest_req", iv
    print("K) intent flip mid-collect PASS")

    # L: 액션 불명 질의 -> action 자체를 HITL 로 질문
    cfg = C("L")
    await graph.ainvoke({"messages": [HumanMessage("6PDMQ283 에 명령 실행해줘")]}, cfg)
    snap = await graph.aget_state(cfg)
    iv = snap.interrupts[0].value
    assert iv["type"] == "collect_param" and iv["field"] == "action", iv
    await graph.ainvoke(Command(resume="반송으로"), cfg)
    snap = await graph.aget_state(cfg)
    assert snap.interrupts[0].value["field"] == "eqp_id", snap.interrupts[0].value
    print("L) unknown action -> HITL PASS")

    print("\nALL SCENARIO TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
