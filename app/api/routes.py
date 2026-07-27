"""API 라우터.

  POST /chat/stream : 사용자 질문 -> 스트리밍 응답 (HITL 재개도 같은 엔드포인트)
  POST /chat/stop   : 실행 중인 스레드 강제 종료
  GET  /models      : 프론트 모델 선택용 목록
  GET  /health      : 헬스체크


스트림 포맷
-----------
최종 답변 토큰은 **가공 없는 raw text** 로 그대로 흘린다(사내 현행 방식).
제어 정보(HITL 질문, 노드 트레이스, 토큰 집계)는 텍스트에 섞이면 안 되므로
구분자 \\x1e(RS) 로 시작하는 JSON 한 줄로 보낸다.

    안녕하세요 반송을...        <- 그냥 텍스트
    \\x1e{"type":"needs_input",...}\\n   <- 제어 프레임

\\x1e 는 일반 텍스트에 나올 일이 없는 제어문자라 안전하게 갈라낼 수 있다.
(RFC 7464 JSON Text Sequences 와 같은 방식)


HITL 판정
---------
같은 엔드포인트가 신규 질문과 HITL 답변을 모두 받는다.
서버가 체크포인터 상태를 보고 어느 쪽인지 정한다.

    인터럽트 없음 -> {"messages": [HumanMessage(query)]} 로 새 턴
    인터럽트 있음 -> Command(resume=query) 로 재개

인터럽트는 이벤트로 안 오기 때문에, astream_events 루프가 끝난 뒤
aget_state() 를 다시 봐서 판정한다. 이게 정석이다.
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.config as cfg
from app import _llm
from app._node import members
from app.api import limits, usage_store
from app.api.graph_service import cached_models, get_team_graph
from app.api.schemas import ChatRequest, ChatResponse, StopRequest

router = APIRouter()


# 제어 프레임 구분자 (위 docstring 참고)
EVENT_PREFIX = "\x1e"

# 최종 답변 토큰을 사용자 화면으로 흘려보낼 노드
FINAL_AGENTS = ("FinalAnswerAgent", "FinalGeneralAgent")

# step_history 에 기록할 노드들
AGENT_NODES = list(members) + [
    "Router",
    "Supervisor",
    "GeneralAgent",
    "FinalAnswerAgent",
    "FinalGeneralAgent",
]


def _event(payload: dict) -> str:
    """제어 프레임 한 줄을 만든다."""
    return EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, default=str) + "\n"


# ─────────────────────────────────────────────────────────────────────────
# 로그 (shared_code.md §6 포맷 유지)
# ─────────────────────────────────────────────────────────────────────────

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
    """최상위 키는 개행 구분, 값은 compact 한 줄.

    경로: logs/{env}/{month}/{date}.jsonl
    """
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


def _write_turn_log(thread_id: str, reason: str, model_name: str = None):
    """턴 종료 시 1회 기록. 원장을 닫고 꺼내 쓴다."""
    snap = usage_store.finish(thread_id)
    if not snap:
        return

    t = snap.get("totals", {})

    save_formatted_log({
        "timestamp": kst_now_iso(),
        "env": os.path.basename(cfg.LOG_DIR.rstrip(os.sep)),
        "thread_id": thread_id,
        "model_name": model_name or _llm.default_model_name(),
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
        "hitl": {
            "rounds": t.get("hitl_rounds"),
            "stream_calls": t.get("stream_calls"),
        },
    })


# ─────────────────────────────────────────────────────────────────────────
# 상태 / 인터럽트 유틸
# ─────────────────────────────────────────────────────────────────────────

def _collect_interrupts(snap) -> list:
    """버전 관용적 인터럽트 수집.

    langgraph 0.3+ 는 StateSnapshot.interrupts 를 주고,
    구버전은 tasks[].interrupts 에만 있다. 둘 다 본다.
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
    """Interrupt 객체에서 우리가 실어 보낸 dict 를 꺼낸다."""
    val = getattr(intr, "value", intr)
    if isinstance(val, dict):
        return val
    return {"type": "collect_param", "prompt": str(val)}


def _node_of(event: dict) -> str:
    """이벤트를 발생시킨 그래프 노드 이름.

    on_chat_model_stream 의 ev["name"] 은 모델 클래스명(ChatOpenAI)이라
    쓸 수 없다. metadata.langgraph_node 를 봐야 한다.
    """
    md = event.get("metadata") or {}
    return md.get("langgraph_node") or event.get("name") or ""


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def _stop_flags(request: Request) -> dict:
    """앱 전역 중단 플래그 저장소. 없으면 만든다."""
    if not hasattr(request.app.state, "stop_flags"):
        request.app.state.stop_flags = {}
    return request.app.state.stop_flags


# ─────────────────────────────────────────────────────────────────────────
# POST /chat/stream
# ─────────────────────────────────────────────────────────────────────────

async def _generate(req: ChatRequest, stop_flags: dict) -> AsyncGenerator[str, None]:
    """실제 스트리밍 본체."""
    thread_id = req.thread_id
    effective_model_name = req.model_name

    # 모델별로 캐싱된 그래프를 가져온다
    team_graph, _ = await get_team_graph(model_name=effective_model_name)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "model_name": effective_model_name,
        },
        # LangGraph 순환 상한. 프론트가 조절할 수 있다.
        "recursion_limit": req.recursion_limit,
    }

    # --- 신규 턴인가, HITL 재개인가 (체크포인터 상태가 판정한다)
    snap = await team_graph.aget_state(config)
    pending = _collect_interrupts(snap)
    resuming = bool(pending)

    usage_store.begin_turn(thread_id, req.query, resuming)
    if not resuming:
        usage_store.set_input_tokens(thread_id, _estimate_tokens(req.query))

    if resuming:
        kind = _interrupt_payload(pending[0]).get("type", "collect_param")
        inputs = Command(resume=req.query)
        print(f"[STREAM] thread={thread_id} RESUME({kind}) <- {req.query!r}", flush=True)
        yield _event({"type": "agent_status", "agent": "HITL",
                      "detail": f"사용자 응답 수신({kind}) — 실행 재개"})
    else:
        inputs = {
            "messages": [HumanMessage(content=req.query)],
            "model_name": effective_model_name,
        }
        print(f"[STREAM] thread={thread_id} NEW TURN <- {req.query!r}", flush=True)

    # --- 스트리밍 상태 변수
    step_history: list = []      # 노드 실행 이력
    last_recorded_node = None    # 같은 노드 연속 기록 방지
    printed_any = False          # 최종 토큰을 하나라도 내보냈는지
    stopped = False              # 사용자 중단 플래그

    # 동시 실행 슬롯을 잡는다. 사용자(thread)별 슬롯이라 남의 작업이 내 걸 안 막는다.
    # 자기 상한을 넘겨 요청하면 '자기 자신'만 대기한다(거절 아님).
    slot_wait_start = time.time()
    async with limits.concurrency_slot(thread_id):
        waited = time.time() - slot_wait_start
        if waited > 0.5:
            yield _event({"type": "agent_status", "agent": "system",
                          "detail": f"대기 후 실행 시작 ({waited:.1f}s 대기)"})

        async for gen_item in _run_graph(team_graph, inputs, config, thread_id,
                                         stop_flags, step_history):
            yield gen_item


async def _run_graph(team_graph, inputs, config, thread_id, stop_flags, step_history):
    """세마포어 슬롯을 잡은 상태에서 실제 그래프 스트림을 돈다."""
    effective_model_name = config["configurable"].get("model_name")
    last_recorded_node = None
    printed_any = False
    stopped = False

    try:
        async for ev in team_graph.astream_events(inputs, config, version="v2"):

            # 매 이벤트마다 중단 플래그 확인 후 즉시 루프 탈출
            if stop_flags.get(thread_id):
                stopped = True
                print(f"[STREAM] thread={thread_id} 중단 플래그 감지 -> 루프 탈출", flush=True)
                break

            event_type = ev.get("event")
            node = _node_of(ev)

            # --- step_history 갱신
            if node and node in AGENT_NODES and node != last_recorded_node:
                if not step_history or step_history[-1] != node:
                    step_history.append(node)
                    usage_store.add_step(thread_id, node)
                    yield _event({"type": "node_enter", "agent": node, "node": node})
                last_recorded_node = node

            # --- 최종 답변 토큰: raw text 로 그대로 흘린다
            if event_type == "on_chat_model_stream":
                chunk = (ev.get("data") or {}).get("chunk")
                if not chunk:
                    continue

                if node in FINAL_AGENTS:
                    text = getattr(chunk, "content", None)
                    if text:
                        usage_store.mark_first_token(thread_id)
                        usage_store.append_answer(thread_id, text)
                        printed_any = True
                        print(text, end="", flush=True)
                        yield text

                elif cfg.SHOW_THINKING_TOKENS:
                    # 기본 off. 켜면 중간 에이전트 토큰도 트레이스로 흐른다.
                    text = getattr(chunk, "content", None)
                    if text:
                        yield _event({"type": "thinking", "agent": node, "text": text})

            # --- 토큰 사용량 누적
            elif event_type == "on_chat_model_end":
                out = (ev.get("data") or {}).get("output")
                usage = getattr(out, "usage_metadata", None)
                if usage:
                    usage_store.add_usage(thread_id, node or "?", dict(usage))

            # --- 툴 호출 로깅
            elif event_type == "on_tool_start":
                payload = {
                    "agent": node,
                    "tool": ev.get("name"),
                    "args": (ev.get("data") or {}).get("input"),
                }
                usage_store.add_tool_call(thread_id, payload)
                print(f"[TOOL-EVENT] {payload['tool']} args={payload['args']}", flush=True)
                yield _event({"type": "tool_call", **payload})

            # --- 툴 결과 (실제 LLM 모드의 ReAct 툴). 입력은 위 start 에서, 결과는 여기서.
            elif event_type == "on_tool_end":
                out = (ev.get("data") or {}).get("output")
                result = getattr(out, "content", out)   # ToolMessage 면 content
                yield _event({"type": "tool_call", "agent": node,
                              "tool": ev.get("name"), "result": result})

            # --- 노드가 emit() 한 상세 트레이스
            elif event_type == "on_custom_event":
                name = ev.get("name")
                data = ev.get("data") or {}
                if name == "tool_call":
                    usage_store.add_tool_call(thread_id, data)
                yield _event({"type": name, **data})

    except asyncio.CancelledError:
        # 클라이언트가 연결을 끊은 경우
        print(f"[STREAM] thread={thread_id} 클라이언트 연결 종료", flush=True)
        raise

    except Exception as e:
        print(f"\n[ERROR] stream failed: {e}", flush=True)
        yield _event({"type": "error", "message": str(e)})
        yield _event({"type": "done", "reason": "error"})
        _write_turn_log(thread_id, "error", effective_model_name)
        return

    finally:
        # 다음 턴에 영향 주지 않도록 플래그를 정리한다
        stop_flags.pop(thread_id, None)

    if printed_any:
        print("", flush=True)   # 콘솔 줄바꿈

    # --- 라우트 기록
    try:
        snap = await team_graph.aget_state(config)
        usage_store.set_route(thread_id, (snap.values or {}).get("route"))
    except Exception:
        snap = None

    if stopped:
        yield _event({"type": "done", "reason": "stopped"})
        _write_turn_log(thread_id, "stopped", effective_model_name)
        return

    # --- 인터럽트 감지는 이벤트가 아니라 스트림 종료 후 상태 검사로
    pending = _collect_interrupts(snap)
    if pending:
        payload = _interrupt_payload(pending[0])
        usage_store.mark_paused(thread_id)

        checkpoint_id = None
        try:
            checkpoint_id = (snap.config or {}).get("configurable", {}).get("checkpoint_id")
        except Exception:
            pass

        print(f"[STREAM] thread={thread_id} ⏸ interrupt: {payload.get('type')}", flush=True)

        # 인터럽트 payload 의 'type'(collect_param/confirm)은 프레임의 'type' 과
        # 충돌하므로 'kind' 로 옮겨 싣는다.
        body = {k: v for k, v in payload.items() if k != "type"}
        yield _event({
            "type": "needs_input",
            "kind": payload.get("type"),
            "resume_token": checkpoint_id,
            "step_history": step_history,
            **body,
        })
        yield _event({"type": "done", "reason": "interrupted"})
        return

    # --- 정상 완료
    yield _event({"type": "usage", **usage_store.totals(thread_id)})
    yield _event({"type": "done", "reason": "complete", "step_history": step_history})
    _write_turn_log(thread_id, "complete", effective_model_name)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """사용자 질문 -> 스트리밍 응답.

    HITL 대기 중인 스레드면 같은 엔드포인트가 재개를 처리한다.
    """
    # 이번 스레드의 중단 플래그를 초기화한다
    flags = _stop_flags(request)
    flags[req.thread_id] = False

    return StreamingResponse(
        _generate(req, flags),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────────────────
# POST /chat/stop
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/stop")
async def chat_stop(req: StopRequest, request: Request):
    """실행 중인 스레드 강제 종료.

    두 경우를 구분한다.

    실행 중
        중단 플래그를 세워 스트림 루프를 다음 이벤트에서 탈출시킨다.

    인터럽트 대기 중
        사실 '실행 중'이 아니라 멈춰 있는 상태다. 그냥 두면 인터럽트가 남아
        다음 질문이 "답변"으로 오인되므로, abort 센티널로 재개해
        abandon 경로를 태워 깨끗이 정리한다.
    """
    thread_id = req.thread_id
    flags = _stop_flags(request)

    # 어느 모델로 돌던 스레드인지 모르므로, 캐시된 그래프를 모두 뒤져
    # 인터럽트가 걸린 쪽을 찾는다.
    for key in (cached_models() or ["default"]):
        model_name = None if key == "default" else key
        graph, _ = await get_team_graph(model_name=model_name)
        config = {"configurable": {"thread_id": thread_id, "model_name": model_name}}

        snap = await graph.aget_state(config)
        pending = _collect_interrupts(snap)

        if pending:
            print(f"[STOP] thread={thread_id} interrupt 대기 중 -> abort 센티널로 정리",
                  flush=True)
            try:
                await graph.ainvoke(Command(resume={"aborted": True}), config)
            except Exception as e:
                print(f"[STOP] abort 재개 실패: {e}", flush=True)

            _write_turn_log(thread_id, "aborted", model_name)
            return {"ok": True, "mode": "aborted_interrupt", "thread_id": thread_id}

    # 인터럽트가 없으면 실행 중이거나 이미 끝난 스레드다
    flags[thread_id] = True
    print(f"[STOP] thread={thread_id} 중단 플래그 설정", flush=True)
    return {"ok": True, "mode": "stop_flag", "thread_id": thread_id}


# ─────────────────────────────────────────────────────────────────────────
# GET /models, GET /health
# ─────────────────────────────────────────────────────────────────────────

@router.get("/models")
async def models():
    """프론트 모델 선택 드롭다운용 목록.

    게이트웨이 /models 와 같은 형태로 돌려준다.
        {"object": "list", "data": [{"id": ..., "object": "model", ...}]}
    """
    return _llm.list_models()


@router.get("/health")
async def health():
    return {
        "ok": True,
        "fake_llm": cfg.FAKE_LLM,
        "default_model": _llm.default_model_name(),
        "cached_graphs": cached_models(),
        "concurrency": limits.concurrency_state(),
        "rate_limit_per_min": cfg.RATE_LIMIT_PER_MIN,
        "time": kst_now_iso(),
    }
