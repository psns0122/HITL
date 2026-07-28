"""API E2E 테스트 — 실제 HTTP(ASGI) 로 HITL 왕복을 돌린다.

사내 LLM 게이트웨이가 붙어 있어야 돈다 (목업 LLM 은 없다).

스트림은 최종 답변을 raw text 로 흘리고, 제어 정보만 \\x1e 로 시작하는
JSON 한 줄로 보낸다. 아래 parse_stream 이 그걸 갈라낸다.

실행: python3 tests/test_api_sse.py
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx

from app.api.main import app
from tests._preflight import require_gateway

BASE = "http://test/llm/api"
EVENT_PREFIX = "\x1e"


async def stream(client, thread_id: str, query: str, **kw) -> tuple[str, list]:
    """/chat/stream 호출 -> (최종 답변 텍스트, 제어 이벤트 리스트)."""
    payload = {"query": query, "thread_id": thread_id}
    payload.update(kw)

    text_parts, events, buffer = [], [], ""

    async with client.stream("POST", f"{BASE}/chat/stream", json=payload) as r:
        assert r.status_code == 200, r.status_code

        async for raw in r.aiter_text():
            buffer += raw

            # 제어 프레임을 하나씩 떼어낸다
            while EVENT_PREFIX in buffer:
                head, _, rest = buffer.partition(EVENT_PREFIX)
                if head:
                    text_parts.append(head)

                if "\n" not in rest:
                    buffer = EVENT_PREFIX + rest
                    break

                line, _, remainder = rest.partition("\n")
                events.append(json.loads(line))
                buffer = remainder
            else:
                if buffer:
                    text_parts.append(buffer)
                    buffer = ""

    if buffer:
        text_parts.append(buffer)

    return "".join(text_parts), events


def first(events, t):
    return next((e for e in events if e["type"] == t), None)


def kinds(events):
    return [e["type"] for e in events]


async def main():
    require_gateway()

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                 timeout=120) as client:

        # ── 0) health / models
        h = await client.get(f"{BASE}/health")
        assert h.status_code == 200 and h.json()["ok"], h.text
        assert h.json()["default_model"] == "GaiA-LLM-Latest", h.json()

        m = await client.get(f"{BASE}/models")
        body = m.json()
        assert body["object"] == "list", body
        ids = [d["id"] for d in body["data"]]
        assert "GaiA-LLM-Latest" in ids and "gaia-GLM-5.2" in ids, ids
        assert "Qwen3.5-397B-A17B-FP8" in ids, ids
        print(f"0) health + models PASS (models={ids})")

        # ── 1) 신규 턴 -> 파라미터 부족으로 HITL 인터럽트
        _, ev = await stream(client, "sse-A", "6PDMQ283 반송해줘")
        ni = first(ev, "needs_input")
        assert ni, kinds(ev)
        assert ni["kind"] == "collect_param" and ni.get("field") == "eqp_id", ni
        assert first(ev, "done")["reason"] == "interrupted"

        # ExtractAgent 가 워커 중 가장 먼저 돌았는지
        nodes = [e["agent"] for e in ev if e["type"] == "node_enter"]
        assert "ExtractAgent" in nodes, nodes
        assert nodes.index("ExtractAgent") < nodes.index("ActionAgent"), nodes
        print(f"1) 신규 턴 -> HITL, ExtractAgent 선행 PASS (nodes={nodes})")

        # ── 2) 답변 재개 -> 승인 요청
        _, ev = await stream(client, "sse-A", "STK102 로 보내줘")
        ni = first(ev, "needs_input")
        assert ni and ni["kind"] == "confirm", ni
        assert ni.get("params", {}).get("eqp_id") == "STK102", ni
        print("2) resume -> confirm PASS")

        # ── 3) 승인 -> 최종 답변이 raw text 로 흘러야 한다
        answer, ev = await stream(client, "sse-A", "승인")
        assert "TJ-" in answer, repr(answer)
        assert "실행 완료" in answer, repr(answer)

        usage = first(ev, "usage")
        assert usage and usage["total_tokens"] > 0, usage
        assert usage["hitl_rounds"] == 2, usage
        assert usage["stream_calls"] == 3, usage
        assert usage["ttft_ms"] is not None, usage
        assert first(ev, "done")["reason"] == "complete"
        print(f"3) 승인 -> raw text 스트리밍 + usage PASS "
              f"(tokens={usage['total_tokens']})")

        # ── 4) 일별 jsonl 로그
        from app.api.routes import _get_log_dir, kst_date_str, read_log_records
        logf = Path(_get_log_dir()) / f"{kst_date_str()}.jsonl"
        assert logf.exists(), f"로그 파일 없음: {logf}"

        rec = next(r for r in reversed(read_log_records(logf))
                   if r["thread_id"] == "sse-A")
        assert rec["outcome"] == "complete", rec
        assert rec["model_name"] == "GaiA-LLM-Latest", rec
        assert rec["token_cost"]["total_tokens"] > 0, rec
        assert rec["hitl"]["rounds"] == 2, rec
        logged_nodes = [s["node"] for s in rec["step_history"]]
        assert "ExtractAgent" in logged_nodes, logged_nodes
        print(f"4) 일별 jsonl 로그 PASS (steps={logged_nodes})")

        # ── 5) 거절 경로
        _, ev = await stream(client, "sse-B", "9ZXCV456 목적지 요청")
        assert first(ev, "needs_input")["kind"] == "confirm"
        answer, ev = await stream(client, "sse-B", "아니 하지마")
        assert "실행하지 않았습니다" in answer, repr(answer)
        print("5) 거절 경로 PASS")

        # ── 6) /chat/stop 이 HITL 대기(진행 중 액션)를 정리하는가
        _, ev = await stream(client, "sse-C", "7HITL001 반송해줘")
        assert first(ev, "needs_input"), kinds(ev)

        r = await client.post(f"{BASE}/chat/stop", json={"thread_id": "sse-C"})
        assert r.json()["mode"] == "aborted_action", r.text

        from app.api.graph_service import get_team_graph
        graph, _ = await get_team_graph(None)
        snap = await graph.aget_state({"configurable": {"thread_id": "sse-C"}})
        assert not ((snap.values or {}).get("action") or {}), (snap.values or {}).get("action")
        print("6) /chat/stop (액션 스크래치 정리) PASS")

        # ── 7) needs-핸드오프
        _, ev = await stream(client, "sse-D", "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘")
        ni = first(ev, "needs_input")
        assert ni and ni.get("params", {}).get("eqp_id") == "STK102", ni
        nodes = [e["agent"] for e in ev if e["type"] == "node_enter"]
        assert "LocationAgent" in nodes and "ActionAgent" in nodes, nodes
        print(f"7) needs-핸드오프 PASS (nodes={nodes})")

        # ── 8) 일반 질의
        answer, ev = await stream(client, "sse-E", "안녕!")
        assert answer.strip(), repr(answer)
        assert first(ev, "done")["reason"] == "complete"
        print("8) 일반 질의 PASS")

        # ── 9) 모델을 바꾸면 그래프가 모델별로 캐싱되는가
        _, ev = await stream(client, "sse-F", "안녕!", model_name="gaia-GLM-5.2")
        assert first(ev, "done")["reason"] == "complete", kinds(ev)

        from app.api.graph_service import cached_models
        assert "gaia-GLM-5.2" in cached_models(), cached_models()
        print(f"9) 모델별 그래프 캐싱 PASS (cached={cached_models()})")

        # ── 10) recursion_limit 이 요청대로 먹는가 (1 이면 즉시 한도 초과)
        _, ev = await stream(client, "sse-G", "6PDMQ283 위치 알려줘", recursion_limit=1)
        err = first(ev, "error")
        assert err and "recursion" in err["message"].lower(), ev
        assert first(ev, "done")["reason"] == "error"
        print("10) recursion_limit 반영 PASS")

        # ── 11) 스키마 검증: 범위 밖 recursion_limit 은 거부
        r = await client.post(f"{BASE}/chat/stream",
                              json={"query": "x", "thread_id": "sse-H",
                                    "recursion_limit": 999})
        assert r.status_code == 422, r.status_code
        print("11) recursion_limit 범위 검증 PASS")

        # ── 12) 맥락 이탈: 수집 중 다른 명령 -> 재시작
        _, ev = await stream(client, "sse-K", "6PDMQ283 반송해줘")
        assert first(ev, "needs_input")["field"] == "eqp_id"
        _, ev = await stream(client, "sse-K", "9ZXCV456 목적지 요청해줘")
        ni = first(ev, "needs_input")
        assert ni and ni["kind"] == "confirm" and ni["action"] == "dest_req", ni
        assert ni["params"]["carrier_id"] == "9ZXCV456", ni
        print("12) 맥락 이탈 재시작 PASS")

    print("\nALL API TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
