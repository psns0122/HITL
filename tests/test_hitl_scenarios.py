"""HITL 시나리오 E2E 테스트 — 그래프를 직접 invoke 한다.

사내 LLM 게이트웨이가 붙어 있어야 돈다 (목업 LLM 은 없다).

턴 기반 HITL: 모든 사용자 입력(최초 질문·HITL 답변)이 똑같이 새 턴으로
들어가 Router → Supervisor 를 경유한다. interrupt/resume 은 쓰지 않는다.
HITL 대기 여부는 state["action"]["awaiting"] 으로 판정한다.

실행: python3 tests/test_hitl_scenarios.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage

from app._builder import build_team_graph
from tests._preflight import require_gateway


async def main():
    require_gateway()

    graph, cp = build_team_graph()

    def C(t):
        return {"configurable": {"thread_id": t}}

    async def turn(cfg, text):
        """모든 입력은 새 턴 — HITL 답변도 예외 없다."""
        return await graph.ainvoke({"messages": [HumanMessage(text)]}, cfg)

    async def awaiting(cfg):
        """HITL 대기 payload. 없으면 None."""
        snap = await graph.aget_state(cfg)
        return ((snap.values or {}).get("action") or {}).get("awaiting")

    # A: 파라미터 수집 -> 승인 -> 실행
    cfg = C("A")
    await turn(cfg, "6PDMQ283 반송해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "collect_param", aw
    await turn(cfg, "STK102로 보내줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm", aw
    r = await turn(cfg, "승인")
    assert "TJ-" in r["messages"][-1].content, r["messages"][-1].content
    assert not await awaiting(cfg)
    print("A) collect->confirm->execute PASS")

    # B: 파라미터 완비 -> confirm 거절 -> abandon
    cfg = C("B")
    await turn(cfg, "9ZXCV456 목적지 요청해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm", aw
    r = await turn(cfg, "아니 거절")
    assert "실행하지 않았습니다" in r["messages"][-1].content
    print("B) dest_req reject PASS")

    # C: 검증 실패(오프라인 장비) -> bad field 재수집 -> 실행
    cfg = C("C")
    await turn(cfg, "6PDMQ283 를 DFF401 로 반송")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "collect_param" and "검증 실패" in aw["prompt"], aw
    await turn(cfg, "PHT201")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm", aw
    r = await turn(cfg, "ㄱㄱ")
    assert "TJ-" in r["messages"][-1].content
    print("C) validation-fail loop -> execute PASS")

    # D: 수집 중 자연어 취소
    cfg = C("D")
    await turn(cfg, "7HITL001 반송 부탁해")
    r = await turn(cfg, "아 그냥 취소해줘")
    assert "실행하지 않았습니다" in r["messages"][-1].content
    print("D) cancel mid-collect PASS")

    # E: 참조형 최초 질의 -> needs 상담 -> LocationAgent -> 재진입 -> 실행
    cfg = C("E")
    await turn(cfg, "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm" and aw["params"]["eqp_id"] == "STK102", aw
    r = await turn(cfg, "승인")
    assert "TJ-" in r["messages"][-1].content
    print("E) needs-handoff LocationAgent PASS")

    # F: 분석형 질의 -> needs 상담 -> LogAgent (원인 장비/현재 위치 제외 권장지)
    cfg = C("F")
    await turn(cfg, "로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm" and aw["params"] == {
        "carrier_id": "6PDMQ283", "eqp_id": "STK102"}, aw
    print("F) needs-handoff LogAgent PASS")

    # G: HITL 질문에 참조형 답변 (답변이 새 턴으로 들어와도 상담이 돈다)
    cfg = C("G")
    await turn(cfg, "6PDMQ283 반송해줘")
    await turn(cfg, "9ZXCV456 있는 위치로 채워줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm" and aw["params"]["eqp_id"] == "STK102", aw
    print("G) reference answer during HITL PASS")

    # H: 일반 질의 -> FinalGeneralAgent
    cfg = C("H")
    r = await turn(cfg, "안녕!")
    assert r["messages"][-1].name == "FinalGeneralAgent", r["messages"][-1]
    print("H) general route PASS")

    # I: /chat/stop 방식 정리 — 턴 기반이라 스크래치 리셋이 전부다
    cfg = C("I")
    await turn(cfg, "7HITL001 반송해줘")
    assert await awaiting(cfg)
    await graph.aupdate_state(cfg, {"action": {}})     # /chat/stop 이 하는 일
    assert not await awaiting(cfg)
    r = await turn(cfg, "안녕!")                        # 다음 질문은 오염 없이 일반 라우팅
    assert r["messages"][-1].name == "FinalGeneralAgent", r["messages"][-1]
    print("I) stop -> scratch reset PASS")

    # J: 존재하지 않는 캐리어 위치 참조 -> 헬퍼 실패 -> HITL 강등
    cfg = C("J")
    await turn(cfg, "6PDMQ283 를 ZZZZ9999 있는 위치로 반송해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "collect_param" and aw["field"] == "eqp_id", aw
    assert "찾을 수 없" in aw["prompt"] or "직접" in aw["prompt"], aw["prompt"]
    print("J) helper-fail -> HITL downgrade PASS")

    # K: 수집 중 맥락 이탈 -> 진행 중 액션 접고 새 질문으로 재시작
    cfg = C("K")
    await turn(cfg, "6PDMQ283 반송해줘")
    aw = await awaiting(cfg)
    assert aw and aw["field"] == "eqp_id", aw          # eqp 묻는 중
    await turn(cfg, "9ZXCV456 목적지 요청해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm" and aw["action"] == "dest_req", aw
    assert aw["params"]["carrier_id"] == "9ZXCV456", aw
    print("K) context-switch restart mid-collect PASS")

    # M: 수집 중 완전히 다른 에이전트 질의(위치) -> 재시작해 위치로 응답
    cfg = C("M")
    await turn(cfg, "6PDMQ283 반송해줘")
    r = await turn(cfg, "아 3KWQ7712 지금 어디 있어?")
    assert not await awaiting(cfg)                     # 대기 없이 바로 답
    assert "PHT201" in r["messages"][-1].content, r["messages"][-1].content
    print("M) context-switch to LocationAgent PASS")

    # L: 액션 불명 질의 -> action 자체를 HITL 로 질문
    cfg = C("L")
    await turn(cfg, "6PDMQ283 에 명령 실행해줘")
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "collect_param" and aw["field"] == "action", aw
    await turn(cfg, "반송으로")
    aw = await awaiting(cfg)
    assert aw and aw["field"] == "eqp_id", aw
    print("L) unknown action -> HITL PASS")

    # N: 모든 입력이 Router/Supervisor 를 탔는지 — HITL 답변 턴의 노드 이력 검증
    cfg = C("N")
    await turn(cfg, "6PDMQ283 반송해줘")
    seen = []
    async for ev in graph.astream_events(
            {"messages": [HumanMessage("STK102")]}, cfg, version="v2"):
        md = ev.get("metadata") or {}
        node = md.get("langgraph_node")
        if node and (not seen or seen[-1] != node):
            seen.append(node)
    assert "Router" in seen and "Supervisor" in seen, seen
    assert seen.index("Router") < seen.index("Supervisor"), seen
    aw = await awaiting(cfg)
    assert aw and aw["type"] == "confirm", aw
    print(f"N) HITL answer passes Router->Supervisor PASS (nodes={seen[:6]}...)")

    print("\nALL SCENARIO TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
