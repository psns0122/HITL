"""API 라우터.

  POST /chat/stream : 사용자 질문 -> 스트리밍 응답 (HITL 재개도 같은 엔드포인트)
  POST /chat/stop   : 실행 중인 스레드 강제 종료
  GET  /models      : 프론트 모델 선택용 목록
  GET  /health      : 헬스체크


스트림 포맷 — 정식 SSE (text/event-stream)
------------------------------------------
모든 것이 표준 SSE 프레임으로 나간다. 답변 토큰도 예외 없다.

    event: token
    data: {"text": "안녕하세요 반송을"}

    event: needs_input
    data: {"type": "needs_input", "kind": "confirm", ...}

- 프레임 = `event:` 한 줄 + `data:` 한 줄(JSON) + 빈 줄.
- 답변 토큰은 event 이름 `token`, 제어 정보는 type 값이 그대로 event 이름.
  data JSON 안에도 type 을 남겨 두므로 이벤트 이름 없이 data 만 파싱해도 된다.
- 토큰 text 를 JSON 으로 감싸는 이유: SSE 의 data 줄은 개행을 담을 수 없어서
  raw 로 흘리면 답변 속 개행이 프레임 경계와 섞인다.
- 주의: 브라우저 내장 EventSource 는 GET 전용이라 이 POST 스트림에는 못 붙는다.
  fetch/httpx 로 받아서 빈 줄 기준으로 프레임을 갈라 파싱한다(아래 클라이언트 참고).


HITL 판정 (턴 기반 — interrupt 없음)
------------------------------------
모든 입력은 **항상 새 턴**이다. HITL 답변도 예외 없이
{"messages": [HumanMessage(query)]} 로 들어가 Router → Supervisor 를 타고,
진행 중 액션이 있으면 Supervisor 가 ActionAgent 로 보낸다.
("모든 사용자 입력은 Router/Supervisor 를 경유한다"는 불변식)

HITL 대기 여부는 스트림이 끝난 뒤 aget_state() 로 action.awaiting 을
확인해서 판정한다. awaiting 이 있으면 needs_input 프레임을 내보낸다.
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.config as cfg
from app import _llm
from app._node import members
from app.api import usage_store
from app.api.graph_service import cached_models, get_team_graph
from app.api.schemas import ChatRequest, ChatResponse, StopRequest

router = APIRouter()


# (구) \x1e 프레임은 폐기 — 정식 SSE 로 전환됐다 (위 docstring 참고)

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
    """제어 프레임 하나를 SSE 형식으로 만든다. event 이름 = payload["type"]."""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {payload.get('type', 'message')}\ndata: {data}\n\n"


def _token(text: str) -> str:
    """답변 토큰 하나를 SSE 형식으로 만든다."""
    data = json.dumps({"text": text}, ensure_ascii=False)
    return f"event: token\ndata: {data}\n\n"


def _trace(agent, tool=None, args=None, result=None) -> str:
    """트레이스 프레임. 노드 진입과 툴 실행을 하나의 이벤트로 합쳤다.

    필드는 항상 4개 전부 실린다 (없으면 null).
      tool == null : 노드 진입
      tool != null : 툴 실행 (args=입력, result=결과 — 시작/종료에 나눠 옴)
    """
    return _event({"type": "trace", "agent": agent,
                   "tool": tool, "args": args, "result": result})


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
# 상태 유틸
# ─────────────────────────────────────────────────────────────────────────

def _awaiting_of(snap) -> dict | None:
    """스냅샷에서 HITL 대기 payload(action.awaiting)를 꺼낸다. 없으면 None."""
    if snap is None:
        return None
    sc = (getattr(snap, "values", None) or {}).get("action") or {}
    return sc.get("awaiting") or None


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

    # --- HITL 대기 중이었는지 (토큰 원장의 라운드 집계용 — 입력 경로는 동일하다)
    snap = await team_graph.aget_state(config)
    awaiting_before = _awaiting_of(snap)
    resuming = bool(awaiting_before)

    usage_store.begin_turn(thread_id, req.query, resuming)
    if not resuming:
        usage_store.set_input_tokens(thread_id, _estimate_tokens(req.query))

    # 신규 질문이든 HITL 답변이든 **똑같이 새 턴**이다.
    # Router 가 진행 중 액션을 보고 Supervisor 로 고정하고, Supervisor 가
    # ActionAgent 로 보낸다 — 모든 입력이 Router/Supervisor 를 경유한다.
    inputs = {
        "messages": [HumanMessage(content=req.query)],
        "model_name": effective_model_name,
    }
    if resuming:
        kind = awaiting_before.get("type", "collect_param")
        print(f"[STREAM] thread={thread_id} HITL 답변({kind}) <- {req.query!r}", flush=True)
    else:
        print(f"[STREAM] thread={thread_id} NEW TURN <- {req.query!r}", flush=True)

    # --- 스트리밍 상태 변수
    step_history: list = []      # 노드 실행 이력
    last_recorded_node = None    # 같은 노드 연속 기록 방지
    printed_any = False          # 최종 토큰을 하나라도 내보냈는지
    stopped = False              # 사용자 중단 플래그

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
                    yield _trace(node)
                last_recorded_node = node

            # --- 최종 답변 토큰: event "token" 으로 흘린다
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
                        yield _token(text)


            # --- 토큰 사용량 누적
            elif event_type == "on_chat_model_end":
                out = (ev.get("data") or {}).get("output")
                usage = getattr(out, "usage_metadata", None)
                if usage:
                    usage_store.add_usage(thread_id, node or "?", dict(usage))

            # --- 툴 시작: 입력을 trace 로
            elif event_type == "on_tool_start":
                tool = ev.get("name")
                args = (ev.get("data") or {}).get("input")
                usage_store.add_tool_call(thread_id, {"agent": node, "tool": tool,
                                                      "args": args})
                print(f"[TOOL-EVENT] {tool} args={args}", flush=True)
                yield _trace(node, tool=tool, args=args)

            # --- 툴 종료: 결과를 trace 로 (같은 tool 의 입력 줄에 프론트가 병합)
            elif event_type == "on_tool_end":
                out = (ev.get("data") or {}).get("output")
                result = getattr(out, "content", out)   # ToolMessage 면 content
                yield _trace(node, tool=ev.get("name"), result=result)

            # --- 노드가 emit() 한 상세 트레이스.
            #     tool_call 만 trace 로 내보내고 나머지(agent_status 등)는
            #     콘솔 로그 전용으로 삼켜 프레임 수를 줄인다.
            elif event_type == "on_custom_event":
                name = ev.get("name")
                data = ev.get("data") or {}
                if name == "tool_call":
                    usage_store.add_tool_call(thread_id, data)
                    yield _trace(data.get("agent"), tool=data.get("tool"),
                                 args=data.get("args"), result=data.get("result"))

    except asyncio.CancelledError:
        # 클라이언트가 연결을 끊은 경우
        print(f"[STREAM] thread={thread_id} 클라이언트 연결 종료", flush=True)
        raise

    except Exception as e:
        print(f"\n[ERROR] stream failed: {e}", flush=True)
        yield _event({"type": "done", "reason": "error", "message": str(e)})
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
        yield _event({"type": "done", "reason": "stopped", "message": None})
        _write_turn_log(thread_id, "stopped", effective_model_name)
        return

    # --- HITL 대기 감지: 스트림 종료 후 action.awaiting 확인
    payload = _awaiting_of(snap)
    if payload:
        usage_store.mark_paused(thread_id)
        print(f"[STREAM] thread={thread_id} ⏸ HITL 대기: {payload.get('type')}", flush=True)

        # 프론트에 필요한 세 필드만 싣는다. (질문 문구에 이미 파라미터가
        # 다 들어 있으므로 action/params 원본은 보내지 않는다)
        yield _event({
            "type": "needs_input",
            "kind": payload.get("type"),          # collect_param | confirm
            "agent": payload.get("agent"),        # 질문 주체 (예: ActionAgent)
            "prompt": payload.get("prompt"),
            "field": payload.get("field"),        # confirm 이면 null
        })
        # HITL 로 멈춘 턴에도 지금까지의 토큰/시간 집계를 보낸다.
        # (원장은 닫지 않는다 — 다음 턴 답변까지 이어서 누적)
        yield _event({"type": "usage", **usage_store.totals(thread_id)})
        yield _event({"type": "done", "reason": "interrupted", "message": None})
        return

    # --- 정상 완료
    yield _event({"type": "usage", **usage_store.totals(thread_id)})
    yield _event({"type": "done", "reason": "complete", "message": None})
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
        media_type="text/event-stream",
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

    HITL 대기 중 (진행 중 액션)
        턴 기반이라 그래프는 이미 멈춰 있고, action 스크래치만 남아 있다.
        스크래치를 리셋하면 끝 — abort 센티널 재개 같은 곡예가 필요 없다.
        (체크포인터는 프로세스 공용이라 어느 그래프로 봐도 같은 스레드다)
    """
    thread_id = req.thread_id
    flags = _stop_flags(request)

    graph, _ = await get_team_graph(None)
    config = {"configurable": {"thread_id": thread_id}}

    snap = await graph.aget_state(config)
    sc = (getattr(snap, "values", None) or {}).get("action") or {}

    if sc.get("awaiting") or sc.get("phase"):
        print(f"[STOP] thread={thread_id} 진행 중 액션 -> 스크래치 리셋", flush=True)
        await graph.aupdate_state(config, {"action": {}})
        _write_turn_log(thread_id, "aborted", None)
        return {"ok": True, "mode": "aborted_action", "thread_id": thread_id}

    # 진행 중 액션이 없으면 실행 중이거나 이미 끝난 스레드다
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
        "default_model": _llm.default_model_name(),
        "cached_graphs": cached_models(),
        "time": kst_now_iso(),
    }
