"""FastAPI 라우터 — SSE 스트리밍 + HITL 재개 + 강제 종료 + 일별 jsonl 로그.

핵심 흐름 (/chat/stream 한 번의 호출):
 1. aget_state(thread_id) 로 이 스레드가 지금 interrupt 로 멈춰 있는지 검사
      멈춤 없음 → 신규 턴  : {"messages": [HumanMessage(query)]}
      멈춤 있음 → 재개     : Command(resume=query)
 2. astream_events(v2) 를 돌며
      - on_chat_model_stream : FinalAnswerAgent/FinalGeneralAgent 토큰만 token 이벤트로 송출
      - on_chain_start       : 노드 진입 → node_enter (트레이스 창)
      - on_tool_start/end    : 툴 호출 → tool_call
      - on_custom_event      : 노드가 emit() 한 상세 트레이스
      - on_chat_model_end    : usage_metadata 를 에이전트별로 누적
      - 중단 플래그가 서면 즉시 루프 탈출
 3. 루프 종료 후 aget_state 재검사 (★ interrupt 는 이벤트로 안 오고 상태로 확인하는 게 정석)
      interrupt 있음 → needs_input + done(interrupted). 로그는 아직 안 쓴다.
      없음           → usage + done(complete) + 일별 jsonl 1회 기록
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.config as cfg
from app._node import members
from app.api import usage_store
from app.api.graph_service import get_team_graph
from app.api.schemas import ChatRequest, StopRequest
from app.api.sse import sse_pack

router = APIRouter()

# 최종 답변 토큰을 사용자 화면으로 흘려보낼 에이전트
FINAL_AGENTS = {"FinalAnswerAgent", "FinalGeneralAgent"}
# 트레이스 창에 노출할 부모 그래프 노드
TRACE_NODES = set(members) | FINAL_AGENTS | {"Router", "Supervisor", "GeneralAgent"}

# thread_id -> True 이면 진행 중인 스트림이 다음 이벤트에서 빠져나온다
_STOP_FLAGS: Dict[str, bool] = {}


# ── 로그 (shared_code.md §6 포맷 그대로) ─────────────────────────────────

def kst_date_str() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")


def kst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _get_log_dir() -> str:
    """logs/{.env 의 LOG_DIR 폴더명}/{YYYY-MM}"""
    env_folder = os.path.basename(cfg.LOG_DIR.rstrip(os.sep)) or "devLogs"
    monthly = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m")
    return os.path.join(PROJECT_ROOT, "logs", env_folder, monthly)


def save_formatted_log(record: Dict[str, Any]):
    """최상위 키는 개행 구분, 값은 compact 한 줄. 경로: logs/{env}/{month}/{date}.jsonl"""
    target_dir = _get_log_dir()
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, f"{kst_date_str()}.jsonl")

    lines = []
    for key, value in record.items():
        compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        lines.append(f'  "{key}":{compact}')
    json_str = "{\n" + ",\n".join(lines) + "\n}"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json_str + "\n")
    print(f"[saved] {path}", flush=True)


def read_log_records(path) -> list:
    """일별 로그를 레코드 리스트로 읽는다.

    한 레코드가 여러 줄에 걸친 pretty JSON 이라 줄 단위 파싱이 안 된다.
    raw_decode 로 객체 경계를 따라가며 읽는다.
    """
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    out, idx = [], 0
    while idx < len(text):
        while idx < len(text) and text[idx] in " \r\n\t":
            idx += 1
        if idx >= len(text):
            break
        obj, idx = decoder.raw_decode(text, idx)
        out.append(obj)
    return out


def _write_turn_log(thread_id: str, reason: str):
    """턴 종료 시 1회 기록. 원장을 닫고 꺼내 쓴다."""
    snap = usage_store.finish(thread_id)
    if not snap:
        return
    t = snap.get("totals", {})
    save_formatted_log({
        "timestamp": kst_now_iso(),
        "env": os.path.basename(cfg.LOG_DIR.rstrip(os.sep)),
        "thread_id": thread_id,
        "query": snap.get("query"),
        "route": snap.get("route"),
        "outcome": reason,
        "step_history": snap.get("step_history"),
        "tool_calls": snap.get("tool_calls"),
        "final_answer": snap.get("final_answer"),
        "token_cost": {
            "user_input_tokens": t.get("user_input_tokens"),
            "per_agent": t.get("per_agent"),
            "agent_input_tokens": t.get("agent_input_tokens"),
            "agent_output_tokens": t.get("agent_output_tokens"),
            "total_tokens": t.get("total_tokens"),
        },
        "time_cost": {
            "ttft_ms": t.get("ttft_ms"),
            "elapsed_ms": t.get("elapsed_ms"),
            "human_wait_ms": t.get("human_wait_ms"),
            "compute_ms": t.get("compute_ms"),
        },
        "hitl": {"rounds": t.get("hitl_rounds"), "stream_calls": t.get("stream_calls")},
    })


# ── 상태/인터럽트 유틸 ────────────────────────────────────────────────────

def _collect_interrupts(snap) -> list:
    """버전 관용적 인터럽트 수집.

    langgraph 0.3+ 는 StateSnapshot.interrupts 를 주고, 구버전은 tasks[].interrupts
    에만 있다. 둘 다 본다.
    """
    if snap is None:
        return []
    found = list(getattr(snap, "interrupts", None) or [])
    if found:
        return found
    for task in (getattr(snap, "tasks", None) or []):
        found.extend(list(getattr(task, "interrupts", None) or []))
    return found


def _interrupt_payload(intr) -> dict:
    val = getattr(intr, "value", intr)
    return val if isinstance(val, dict) else {"type": "collect_param", "prompt": str(val)}


def _agent_of(event: dict) -> str:
    md = event.get("metadata") or {}
    return md.get("langgraph_node") or event.get("name") or "?"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


# ── /chat/stream ──────────────────────────────────────────────────────────

async def _event_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    thread_id = req.thread_id
    graph, _ = await get_team_graph()
    config = {"configurable": {"thread_id": thread_id}}
    seq = 0

    def pack(payload: dict) -> str:
        nonlocal seq
        seq += 1
        return sse_pack(payload, seq)

    _STOP_FLAGS.pop(thread_id, None)

    # 1) 신규 턴인가, HITL 재개인가 — 체크포인터 상태가 판정한다
    snap = await graph.aget_state(config)
    pending = _collect_interrupts(snap)
    resuming = bool(pending)

    led = usage_store.begin_turn(thread_id, req.query, resuming)
    if not resuming:
        usage_store.set_input_tokens(thread_id, _estimate_tokens(req.query))

    if resuming:
        kind = _interrupt_payload(pending[0]).get("type", "collect_param")
        graph_input = Command(resume=req.query)
        print(f"[STREAM] thread={thread_id} RESUME({kind}) <- {req.query!r}", flush=True)
        yield pack({"type": "agent_status", "agent": "HITL",
                    "detail": f"사용자 응답 수신({kind}) — 실행 재개"})
    else:
        graph_input = {"messages": [HumanMessage(content=req.query)]}
        print(f"[STREAM] thread={thread_id} NEW TURN <- {req.query!r}", flush=True)

    stopped = False
    try:
        async for ev in graph.astream_events(graph_input, config, version="v2"):
            if _STOP_FLAGS.get(thread_id):
                stopped = True
                print(f"[STREAM] thread={thread_id} 중단 플래그 감지 -> 루프 탈출", flush=True)
                break

            etype = ev.get("event")
            agent = _agent_of(ev)

            if etype == "on_chat_model_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                text = getattr(chunk, "content", "") or ""
                if not text:
                    continue
                if agent in FINAL_AGENTS:
                    usage_store.mark_first_token(thread_id)
                    usage_store.append_answer(thread_id, text)
                    yield pack({"type": "token", "agent": agent, "text": text})
                elif cfg.SHOW_THINKING_TOKENS:
                    # 기본 off — 켜면 중간 에이전트 토큰도 트레이스로 흐른다
                    yield pack({"type": "thinking", "agent": agent, "text": text})

            elif etype == "on_chat_model_end":
                out = (ev.get("data") or {}).get("output")
                usage = getattr(out, "usage_metadata", None)
                if usage:
                    usage_store.add_usage(thread_id, agent, dict(usage))

            elif etype == "on_chain_start":
                node = ev.get("name")
                if node in TRACE_NODES:
                    usage_store.add_step(thread_id, node)
                    yield pack({"type": "node_enter", "agent": node, "node": node})

            elif etype == "on_tool_start":
                payload = {"agent": agent, "tool": ev.get("name"),
                           "args": (ev.get("data") or {}).get("input")}
                usage_store.add_tool_call(thread_id, payload)
                print(f"[TOOL-EVENT] {payload['tool']} args={payload['args']}", flush=True)
                yield pack({"type": "tool_call", **payload})

            elif etype == "on_custom_event":
                name = ev.get("name")
                data = ev.get("data") or {}
                if name == "tool_call":
                    usage_store.add_tool_call(thread_id, data)
                yield pack({"type": name, **data})

    except asyncio.CancelledError:
        raise
    except Exception as e:
        print(f"[ERROR] stream failed: {e}", flush=True)
        yield pack({"type": "error", "message": str(e)})
        yield pack({"type": "done", "reason": "error"})
        _write_turn_log(thread_id, "error")
        return

    # 2) 라우트 기록
    try:
        snap = await graph.aget_state(config)
        usage_store.set_route(thread_id, (snap.values or {}).get("route"))
    except Exception:
        snap = None

    if stopped:
        yield pack({"type": "done", "reason": "stopped"})
        _write_turn_log(thread_id, "stopped")
        return

    # 3) ★ 인터럽트 감지는 이벤트가 아니라 스트림 종료 후 상태 검사로 한다
    pending = _collect_interrupts(snap)
    if pending:
        payload = _interrupt_payload(pending[0])
        usage_store.mark_paused(thread_id)
        token = None
        try:
            token = (snap.config or {}).get("configurable", {}).get("checkpoint_id")
        except Exception:
            pass
        print(f"[STREAM] thread={thread_id} ⏸ interrupt: {payload.get('type')}", flush=True)
        # 인터럽트 payload 의 'type'(collect_param/confirm)은 SSE 봉투의 'type' 과
        # 충돌하므로 'kind' 로 옮겨 싣는다.
        body = {k: v for k, v in payload.items() if k != "type"}
        yield pack({"type": "needs_input", "kind": payload.get("type"),
                    "resume_token": token, **body})
        yield pack({"type": "done", "reason": "interrupted"})
        return

    # 4) 정상 완료 — 집계 후 로그 1회
    yield pack({"type": "usage", **usage_store.totals(thread_id)})
    yield pack({"type": "done", "reason": "complete"})
    _write_turn_log(thread_id, "complete")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """사용자 질문 → SSE 스트리밍 응답. HITL 대기 중이면 같은 엔드포인트가 재개를 처리한다."""
    return StreamingResponse(
        _event_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# ── /chat/stop ────────────────────────────────────────────────────────────

@router.post("/chat/stop")
async def chat_stop(req: StopRequest):
    """실행 중인 스레드 강제 종료.

    두 경우를 구분한다.
    - 실행 중      : 중단 플래그를 세워 스트림 루프를 탈출시킨다.
    - interrupt 중 : 사실 '실행 중'이 아니다(멈춰 있음). 그냥 두면 인터럽트가
                     남아 다음 질문이 답변으로 오인되므로, abort 센티널로 재개해
                     abandon 경로를 태워 깨끗이 정리한다.
    """
    thread_id = req.thread_id
    graph, _ = await get_team_graph()
    config = {"configurable": {"thread_id": thread_id}}

    snap = await graph.aget_state(config)
    pending = _collect_interrupts(snap)

    if pending:
        print(f"[STOP] thread={thread_id} interrupt 대기 중 -> abort 센티널로 정리", flush=True)
        try:
            await graph.ainvoke(Command(resume={"aborted": True}), config)
        except Exception as e:
            print(f"[STOP] abort 재개 실패: {e}", flush=True)
        _write_turn_log(thread_id, "aborted")
        return {"ok": True, "mode": "aborted_interrupt", "thread_id": thread_id}

    running = bool(getattr(snap, "next", ()) or ())
    _STOP_FLAGS[thread_id] = True
    print(f"[STOP] thread={thread_id} 중단 플래그 설정 (running={running})", flush=True)
    return {"ok": True, "mode": "stop_flag" if running else "idle", "thread_id": thread_id}


@router.get("/health")
async def health():
    return {"ok": True, "fake_llm": cfg.FAKE_LLM, "model": cfg.MODEL_NAME,
            "time": kst_now_iso()}
