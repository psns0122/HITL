"""API 라우터.  [원본·직접]

  POST /chat/stream : 사용자 질문 -> 최종 답변 토큰 스트리밍

사용자가 직접 붙여 준 사내 실물 코드다. 구조·변수명·주석을 그대로 두었고,
`app.*` 를 참조하던 import 만 `origin.*` 으로 바꿨다
(origin ↔ app 은 서로 import 하지 않는다 — CLAUDE.md 규칙 4).

이전 스냅샷과 달라진 점
-----------------------
- HITL 이 들어왔다. interrupt 를 감지해 thread_id 별로 pending 을 들고 있다가,
  다음 사용자 입력이 승인/거절이면 Command(resume=...) 로 재개한다.
- 로그 레코드 구조가 바뀌었다 (agent_data / token_cost / time_cost).
- media_type 이 text/event-stream 이다.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.types import Command

from origin import _node
from origin.api.schemas import ChatRequest
from origin.api.graph_service import get_team_graph
from origin import config as cfg


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


router = APIRouter()


# =============================================================================
# 기본 유틸
# =============================================================================

def _sse_pack(data: str, event: str = "message") -> str:
    """
    SSE 포맷으로 내보낼 때 사용하는 함수.
    현재 Streamlit이 requests.post(..., stream=True)로 raw chunk를 읽고 있다면
    gen() 내부에서는 그대로 yield text 해도 된다.

    EventSource 기반으로 바꾸는 경우:
        yield _sse_pack(text)
    형태로 사용한다.
    """
    safe = (data or "").replace("\r", "").replace("\n", "\\n")
    return f"event: {event}\ndata: {safe}\n\n"


def kst_date_str() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y-%m-%d")


def kst_now_iso() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _get_log_dir() -> str:
    env_folder = os.path.basename(cfg.LOG_DIR.rstrip(os.sep))
    kst = timezone(timedelta(hours=9))
    monthly_folder = datetime.now(kst).strftime("%Y-%m")

    return os.path.join(PROJECT_ROOT, "logs", env_folder, monthly_folder)


def save_formatted_log(record: Dict[str, Any], base_dir: str = cfg.LOG_DIR) -> None:
    target_dir = _get_log_dir()
    os.makedirs(target_dir, exist_ok=True)

    path = os.path.join(target_dir, f"{kst_date_str()}.jsonl")

    formatted_lines = []
    for key, value in record.items():
        compact_value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        formatted_lines.append(f'  "{key}": {compact_value}')

    json_str = "{\n" + ",\n".join(formatted_lines) + "\n}"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json_str + "\n")


# =============================================================================
# Human-in-the-loop helper
# =============================================================================

def ensure_pending_interrupt_store(request: Request) -> Dict[str, Any]:
    """
    thread_id별 interrupt 대기 상태를 저장한다.

    개발/단일 프로세스 환경:
        request.app.state.pending_interrupts 사용 가능

    운영/멀티 worker 환경:
        Redis, DB 등 외부 저장소로 바꾸는 것이 안전하다.
    """
    if not hasattr(request.app.state, "pending_interrupts"):
        request.app.state.pending_interrupts = {}

    return request.app.state.pending_interrupts


def parse_approval_text(text: str) -> Optional[bool]:
    """
    사용자의 다음 입력이 승인/거절인지 판단한다.

    반환:
        True  -> 승인 / 실행
        False -> 거절 / 취소
        None  -> 승인/거절로 해석 불가
    """
    normalized = (text or "").strip().lower()

    approve_words = {
        "실행", "승인", "확인", "진행", "해", "해줘",
        "yes", "y", "ok", "approve", "approved"
    }

    reject_words = {
        "거절", "취소", "중단", "아니", "하지마",
        "no", "n", "cancel", "reject", "rejected"
    }

    if normalized in approve_words:
        return True

    if normalized in reject_words:
        return False

    return None


def find_interrupts(obj: Any) -> List[Any]:
    """
    astream_events(version='v2') event 안에서 interrupt 정보를 찾는다.

    LangGraph 버전/stream mode/subgraph 여부에 따라 interrupt가 들어오는 위치가
    다를 수 있으므로 재귀적으로 찾는다.
    """
    found: List[Any] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("__interrupt__", "interrupts"):
                if isinstance(value, (list, tuple)):
                    found.extend(value)
                else:
                    found.append(value)
            else:
                found.extend(find_interrupts(value))

    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.extend(find_interrupts(item))

    return found


def get_interrupt_value(interrupt_obj: Any) -> Any:
    """
    Interrupt 객체 또는 dict에서 실제 payload를 꺼낸다.
    """
    value = getattr(interrupt_obj, "value", None)
    if value is not None:
        return value

    if isinstance(interrupt_obj, dict):
        return interrupt_obj.get("value", interrupt_obj)

    return interrupt_obj


def format_action_confirm_payload(payload: Any) -> str:
    """
    ActionAgent의 interrupt payload를 Streamlit에 보여줄 텍스트로 변환한다.
    """
    if not isinstance(payload, dict):
        return (
            "\n[실행 확인 필요]\n\n"
            f"{payload}\n\n"
            "실행하려면 `실행` 또는 `승인`을 입력하세요.\n"
            "취소하려면 `거절` 또는 `취소`를 입력하세요."
        )

    message = payload.get(
        "message",
        "검증 결과 실행 가능합니다. 실제 반송 명령을 전송하시겠습니까?"
    )

    request_id = payload.get("request_id")
    command_type = payload.get("command_type", "")
    command_label = payload.get("command_label", "")
    command_preview = payload.get("command_preview", {}) or {}

    lines = []
    lines.append("")
    lines.append("[실행 확인 필요]")
    lines.append("")
    lines.append(message)
    lines.append("")

    if request_id:
        lines.append(f"- 요청 ID: {request_id}")

    if command_label or command_type:
        lines.append(f"- 명령 유형: {command_label or command_type}")

    if command_preview:
        lines.append("- 실행 예정 파라미터:")
        for key, value in command_preview.items():
            lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("실행하려면 `실행` 또는 `승인`을 입력하세요.")
    lines.append("취소하려면 `거절` 또는 `취소`를 입력하세요.")

    return "\n".join(lines)


def build_graph_input_for_request(
    req: ChatRequest,
    request: Request,
    inputs: Dict[str, Any],
) -> Any:
    """
    이번 요청을 graph에 어떻게 넣을지 결정한다.

    1. 해당 thread_id에 pending interrupt가 있고,
       req.query가 실행/거절이면 Command(resume=...)로 변환

    2. 해당 thread_id에 pending interrupt가 있는데,
       req.query가 실행/거절이 아니면 graph를 재개하지 않고 local response 반환

    3. pending interrupt가 없으면 일반 신규 입력으로 처리
    """
    pending_interrupts = ensure_pending_interrupt_store(request)

    pending_payload = pending_interrupts.get(req.thread_id)
    approval = parse_approval_text(req.query)

    if pending_payload is not None and approval is not None:
        # resume 시작 시 pending 상태 제거
        pending_interrupts.pop(req.thread_id, None)

        return Command(resume={
            "approved": approval,
            "approved_by": "streamlit_user",
            "reason": req.query,
        })

    if pending_payload is not None and approval is None:
        return {
            "__local_response__": (
                "현재 실행 승인 대기 중입니다.\n\n"
                "실행하려면 `실행` 또는 `승인`을 입력하세요.\n"
                "취소하려면 `거절` 또는 `취소`를 입력하세요."
            )
        }

    return inputs


# =============================================================================
# Logging helper
# =============================================================================

def extract_agent_data(turn_msgs: List[BaseMessage]) -> Dict[str, Any]:
    agent_data: Dict[str, Any] = {}
    agent_tools: Dict[str, Any] = {}

    for msg in turn_msgs:
        if isinstance(msg, AIMessage):
            agent_name = msg.additional_kwargs.get("agent_name")

            if agent_name and getattr(msg, "tool_calls", None):
                if agent_name not in agent_tools:
                    agent_tools[agent_name] = []

                for tc in msg.tool_calls:
                    tool_name = tc.get("name")
                    tool_args = tc.get("args", {})

                    if tool_name:
                        agent_tools[agent_name].append({
                            "name": tool_name,
                            "args": tool_args,
                        })

            content = msg.content or ""
            if agent_name and "STATUS:" in content:
                tools = agent_tools.get(agent_name, [])
                agent_data[agent_name] = {
                    "result": content,
                    "tool": tools,
                }

    return agent_data


def create_log_record(
    req: ChatRequest,
    full_state: Dict[str, Any],
    turn_msgs: List[BaseMessage],
    final_answer: str,
    step_history: List[str],
    token_accumulator: Dict[str, int],
    requested_model_name: str,
    time_cost: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    agent_data = extract_agent_data(turn_msgs)

    record = {
        "timestamp": kst_now_iso(),
        "thread_id": getattr(req, "thread_id", None),
        "user_query": getattr(req, "query", None),
        "route": full_state.get("route"),
        "handoff": full_state.get("handoff"),
        "step": full_state.get("step", 0) + 1,
        "step_history": step_history or [],
        **agent_data,
        "FinalAnswerAgent": final_answer,
        "action_result": full_state.get("action_result"),
        "token_cost": token_accumulator,
        "time_cost": time_cost or {},
        "requested_model": requested_model_name,
    }

    return record


# =============================================================================
# Model helper
# =============================================================================

def get_effective_model_name(req: ChatRequest) -> Optional[str]:
    if req.model_name:
        return req.model_name

    llm_configs = getattr(cfg, "LLM_CONFIGS", {}) or {}

    default_cfg = (
        llm_configs.get("default")
        or llm_configs.get("defualt")  # 기존 오타 호환
        or {}
    )

    return default_cfg.get("model_name")


# =============================================================================
# Endpoint
# =============================================================================

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    effective_model_name = get_effective_model_name(req)

    team_graph, _ = await get_team_graph(model_name=effective_model_name)

    # LangGraph checkpointer는 configurable.thread_id를 사용한다.
    config = {
        "configurable": {
            "thread_id": req.thread_id,
            "model_name": effective_model_name,
        },
        "recursion_limit": req.recursion_limit,
    }

    if not hasattr(request.app.state, "stop_flags"):
        request.app.state.stop_flags = {}

    request.app.state.stop_flags[req.thread_id] = False

    async def gen() -> AsyncGenerator[str, None]:
        total_start_time = time.time()
        first_yield_time = None
        is_first_yield = True

        start_idx = 0

        try:
            snap = team_graph.get_state(config)
            start_idx = len((snap.values or {}).get("messages", []) or [])
        except Exception:
            start_idx = 0

        # 기존 inputs 구성 유지
        inputs = {
            "messages": [HumanMessage(content=req.query)],
            "model_name": effective_model_name,
        }

        # 현재 요청이 일반 신규 질의인지, HITL resume인지 결정
        graph_input = build_graph_input_for_request(
            req=req,
            request=request,
            inputs=inputs,
        )

        if isinstance(graph_input, dict) and "__local_response__" in graph_input:
            yield graph_input["__local_response__"]
            return

        printed_any = False
        end_node_event = None

        token_accumulator = {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        processed_msg_ids = set()

        step_history: List[str] = []
        last_recorded_step = -1

        final_answer_parts: List[str] = []
        is_stopped_by_user = False

        AGENT_NODES = {
            "Router",
            "GeneralAgent",
            "Supervisor",
            "LocationAgent",
            "StatusAgent",
            "LogAgent",
            "ActionAgent",
            "ExtractAgent",
            "FinalAnswerAgent",
            "FinalGeneralAgent",
        }

        try:
            async for ev in team_graph.astream_events(
                graph_input,
                config=config,
                version="v2",
            ):
                if getattr(request.app.state, "stop_flags", {}).get(req.thread_id, False):
                    is_stopped_by_user = True
                    break

                # -----------------------------------------------------------------
                # 1. Human-in-the-loop interrupt 먼저 감지
                # -----------------------------------------------------------------
                interrupts = find_interrupts(ev)

                if interrupts:
                    interrupt_payload = get_interrupt_value(interrupts[0])

                    pending_interrupts = ensure_pending_interrupt_store(request)
                    pending_interrupts[req.thread_id] = interrupt_payload

                    confirm_text = format_action_confirm_payload(interrupt_payload)

                    yield confirm_text
                    printed_any = True

                    print("\n[HITL INTERRUPT DETECTED]")
                    print(json.dumps(interrupt_payload, ensure_ascii=False, indent=2, default=str))

                    # interrupt 발생 시 graph는 pause 상태다.
                    # 여기서 종료하고, 다음 사용자 입력을 Command(resume=...)로 받아야 한다.
                    return

                # -----------------------------------------------------------------
                # 2. 기존 event parsing
                # -----------------------------------------------------------------
                md = ev.get("metadata", {}) or {}
                node = md.get("langgraph_node") or ev.get("name")
                event_type = ev.get("event")
                current_step = md.get("langgraph_step", -1)

                if event_type == "on_tool_start":
                    tool_name = ev.get("name")
                    tool_input = ev.get("data", {}).get("input")
                    # 필요 시 tool log 처리
                    # print(f"[TOOL START] {tool_name}: {tool_input}")

                # step history 기록
                if node and current_step > last_recorded_step:
                    if node in AGENT_NODES:
                        if not step_history or step_history[-1] != node:
                            step_history.append(node)
                            last_recorded_step = current_step

                # -----------------------------------------------------------------
                # 3. LLM stream chunk 처리
                # -----------------------------------------------------------------
                if event_type == "on_chat_model_stream":
                    chunk = ev.get("data", {}).get("chunk")

                    if not chunk:
                        continue

                    # token usage 수집
                    usage_metadata = getattr(chunk, "usage_metadata", None)

                    if usage_metadata:
                        msg_id = getattr(chunk, "id", None)

                        if msg_id and msg_id not in processed_msg_ids:
                            input_t = usage_metadata.get("input_tokens", 0) or 0
                            output_t = usage_metadata.get("output_tokens", 0) or 0

                            if node == "Router" and token_accumulator["input_tokens"] == 0:
                                token_accumulator["input_tokens"] = input_t
                                token_accumulator["output_tokens"] += output_t
                            else:
                                token_accumulator["output_tokens"] += output_t

                                if input_t > 0:
                                    token_accumulator["input_tokens"] += input_t

                            processed_msg_ids.add(msg_id)

                    # 최종 답변 agent만 사용자에게 stream
                    if node in ("FinalGeneralAgent", "FinalAnswerAgent"):
                        text = getattr(chunk, "content", None)

                        if text:
                            if is_first_yield:
                                first_yield_time = time.time()
                                is_first_yield = False

                            yield text
                            final_answer_parts.append(text)
                            printed_any = True

                # -----------------------------------------------------------------
                # 4. Final node 종료 감지
                # -----------------------------------------------------------------
                if event_type == "on_chain_end" and node in ("FinalGeneralAgent", "FinalAnswerAgent"):
                    end_node_event = ev
                    break

            if printed_any:
                print()

        except Exception as e:
            error_text = f"오류가 발생했습니다: {e}"
            yield error_text
            print(f"[ERROR][chat_stream] {e}")
            return

        # -------------------------------------------------------------------------
        # 5. 최종 state / log 저장
        # -------------------------------------------------------------------------
        try:
            snap2 = team_graph.get_state(config)
            full_state = snap2.values or {}
        except Exception:
            full_state = {}

        all_msgs = full_state.get("messages", []) or []
        turn_msgs = all_msgs[start_idx:] if start_idx <= len(all_msgs) else all_msgs

        final_answer = "".join(final_answer_parts)

        if is_stopped_by_user:
            final_answer = "사용자에 의해 프로세스가 중단되었습니다."

        elif not final_answer and end_node_event and "data" in end_node_event:
            output_msg = end_node_event["data"].get("output")

            if isinstance(output_msg, AIMessage):
                final_answer = output_msg.content or ""

            elif isinstance(output_msg, dict):
                # graph final output이 dict로 오는 경우 방어 처리
                maybe_messages = output_msg.get("messages") or []
                if maybe_messages:
                    last_msg = maybe_messages[-1]
                    final_answer = getattr(last_msg, "content", "") or ""

        end_time = time.time()

        time_cost = {
            "stream": (first_yield_time - total_start_time) if first_yield_time else 0,
            "end": end_time - total_start_time,
        }

        try:
            record = create_log_record(
                req=req,
                full_state=full_state,
                turn_msgs=turn_msgs,
                final_answer=final_answer,
                step_history=step_history,
                token_accumulator=token_accumulator,
                requested_model_name=effective_model_name,
                time_cost=time_cost,
            )
            save_formatted_log(record)

        except Exception as e:
            print(f"[WARN][chat_stream] failed to save log: {e}")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
    )
