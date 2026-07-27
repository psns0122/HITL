"""SSE API E2E 테스트 — 실제 HTTP(ASGI) 로 HITL 왕복을 돌린다.

실행: python3 tests/test_api_sse.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.main import app

BASE = "http://test/llm/api"


async def stream(client, thread_id: str, query: str) -> list[dict]:
    """/chat/stream 을 호출하고 수신한 SSE 이벤트를 리스트로 돌려준다."""
    events = []
    async with client.stream("POST", f"{BASE}/chat/stream",
                             json={"query": query, "thread_id": thread_id}) as r:
        assert r.status_code == 200, r.status_code
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def kinds(events):
    return [e["type"] for e in events]


def first(events, t):
    return next((e for e in events if e["type"] == t), None)


async def main():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                 timeout=60) as client:

        h = await client.get(f"{BASE}/health")
        assert h.status_code == 200 and h.json()["ok"], h.text
        print("health PASS")

        # ── 1) 신규 턴 → 파라미터 부족으로 HITL 인터럽트
        ev = await stream(client, "sse-A", "6PDMQ283 반송해줘")
        ni = first(ev, "needs_input")
        assert ni, kinds(ev)
        assert ni["kind"] == "collect_param", ni
        assert ni.get("field") == "eqp_id", ni
        assert first(ev, "done")["reason"] == "interrupted", ev[-1]
        assert first(ev, "node_enter"), "노드 트레이스 이벤트 없음"
        assert first(ev, "tool_call"), "툴 트레이스 이벤트 없음"
        print("1) new turn -> needs_input(collect_param/eqp_id) PASS")

        # ── 2) 같은 엔드포인트로 답변 → 재개 → 승인 요청 인터럽트
        ev = await stream(client, "sse-A", "STK102 로 보내줘")
        ni = first(ev, "needs_input")
        assert ni and ni["kind"] == "confirm" and ni.get("action") == "transport", ni
        assert ni.get("params", {}).get("eqp_id") == "STK102", ni
        assert "승인" in (ni.get("options") or []), ni
        assert first(ev, "done")["reason"] == "interrupted"
        print("2) resume -> needs_input(confirm) PASS")

        # ── 3) 승인 → 최종 답변 토큰 스트리밍 + usage + 완료
        ev = await stream(client, "sse-A", "승인")
        toks = [e for e in ev if e["type"] == "token"]
        assert toks, kinds(ev)
        answer = "".join(t["text"] for t in toks)
        assert "TJ-" in answer, answer
        assert all(t["agent"] == "FinalAnswerAgent" for t in toks)
        usage = first(ev, "usage")
        assert usage and usage["total_tokens"] > 0, usage
        assert usage["hitl_rounds"] == 2, usage           # 파라미터 1회 + 승인 1회
        assert usage["stream_calls"] == 3, usage
        assert usage["ttft_ms"] is not None and usage["compute_ms"] >= 0, usage
        assert usage["per_agent"], usage
        assert first(ev, "done")["reason"] == "complete"
        print(f"3) approve -> token stream + usage PASS "
              f"(tokens={usage['total_tokens']}, agents={list(usage['per_agent'])})")

        # ── 4) 일별 jsonl 로그가 기록됐는지
        from app.api.routes import _get_log_dir, kst_date_str, read_log_records
        logf = Path(_get_log_dir()) / f"{kst_date_str()}.jsonl"
        assert logf.exists(), f"로그 파일 없음: {logf}"
        recs = read_log_records(logf)
        rec = next(r for r in reversed(recs) if r["thread_id"] == "sse-A")
        assert rec["token_cost"]["total_tokens"] > 0, rec
        assert rec["time_cost"]["ttft_ms"] is not None, rec
        assert rec["hitl"]["rounds"] == 2, rec
        assert rec["outcome"] == "complete"
        assert rec["step_history"], rec
        assert rec["time_cost"]["human_wait_ms"] >= 0
        print(f"4) daily jsonl log PASS ({logf.name}, steps={len(rec['step_history'])})")

        # ── 5) 거절 경로
        ev = await stream(client, "sse-B", "9ZXCV456 목적지 요청")
        assert first(ev, "needs_input")["type"] == "needs_input"
        ev = await stream(client, "sse-B", "아니 하지마")
        answer = "".join(e["text"] for e in ev if e["type"] == "token")
        assert "실행하지 않았습니다" in answer, answer
        assert first(ev, "done")["reason"] == "complete"
        print("5) reject path PASS")

        # ── 6) /chat/stop 이 interrupt 대기 스레드를 정리하는가
        ev = await stream(client, "sse-C", "7HITL001 반송해줘")
        assert first(ev, "needs_input"), kinds(ev)
        r = await client.post(f"{BASE}/chat/stop", json={"thread_id": "sse-C"})
        assert r.json()["mode"] == "aborted_interrupt", r.text
        # 정리 후에는 인터럽트가 남아있지 않아야 다음 질문이 답변으로 오인되지 않는다
        from app.api.graph_service import get_team_graph
        from app.api.routes import _collect_interrupts
        graph, _ = await get_team_graph()
        snap = await graph.aget_state({"configurable": {"thread_id": "sse-C"}})
        assert not _collect_interrupts(snap), snap.interrupts
        print("6) /chat/stop on interrupted thread PASS")

        # ── 7) needs-핸드오프가 SSE 로도 보이는가 (동료 에이전트 연계)
        ev = await stream(client, "sse-D", "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘")
        ni = first(ev, "needs_input")
        assert ni and ni.get("params", {}).get("eqp_id") == "STK102", ni
        nodes = [e["agent"] for e in ev if e["type"] == "node_enter"]
        assert "LocationAgent" in nodes and "ActionAgent" in nodes, nodes
        print(f"7) needs-handoff visible in SSE PASS (nodes={nodes})")

        # ── 8) 일반 질의
        ev = await stream(client, "sse-E", "안녕!")
        toks = [e for e in ev if e["type"] == "token"]
        assert toks and all(t["agent"] == "FinalGeneralAgent" for t in toks), kinds(ev)
        assert first(ev, "done")["reason"] == "complete"
        print("8) general route streams from FinalGeneralAgent PASS")

    print("\nALL SSE API TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
