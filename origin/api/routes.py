"""API 라우터.  [원본·첨부]

  POST /chat/stream : 사용자 질문 -> 최종 답변 토큰 스트리밍

스트림 포맷
-----------
최종 답변 토큰을 **가공 없는 raw text** 로 그대로 흘린다.
원본에는 제어 프레임이 없다 — 흘릴 게 답변뿐이기 때문이다.
(app/api/routes.py 에서 \\x1e 제어 프레임이 추가됐다. 트레이스와 HITL 질문을
 텍스트에 섞지 않고 보내려고.)

로그
----
`logs/{env}/{YYYY-MM}/{YYYY-MM-DD}.jsonl` 에 턴당 1회 기록한다.
최상위 키는 개행으로 나누고 값은 compact 한 줄로 쓰는 포맷이라,
사람이 눈으로 읽으면서도 grep 이 되게 되어 있다.
`save_formatted_log` 는 사내 원문 그대로다.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from origin import config as cfg
from origin._node import members
from origin.api.graph_service import get_team_graph
from origin.api.schemas import ChatRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

router = APIRouter()

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


# ─────────────────────────────────────────────────────────────────────────
# 로그
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


# ─────────────────────────────────────────────────────────────────────────
# POST /chat/stream
# ─────────────────────────────────────────────────────────────────────────

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """질문을 받아 최종 답변 토큰을 흘린다."""
    graph = get_team_graph()

    config = {
        "configurable": {"thread_id": req.thread_id},
        "recursion_limit": req.recursion_limit,
    }

    async def generate() -> AsyncGenerator[str, None]:
        started = time.perf_counter()
        ttft_ms = None

        route = None
        step_history = []
        answer_parts = []

        # 토큰/시간 원장 — 에이전트별로 나눠 담는다
        per_agent: Dict[str, Dict[str, int]] = {}

        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=req.query)]},
            config,
            version="v2",
        ):
            kind = event.get("event")
            metadata = event.get("metadata") or {}
            node = metadata.get("langgraph_node")

            # 1) 노드 진입 이력
            if kind == "on_chain_start" and node in AGENT_NODES:
                if not step_history or step_history[-1] != node:
                    step_history.append(node)

            # 2) 라우팅 결과 기록
            if kind == "on_chain_end" and node == "Router":
                output = (event.get("data") or {}).get("output") or {}
                if isinstance(output, dict):
                    route = output.get("route") or route

            # 3) 최종 답변 토큰만 사용자에게 흘린다
            if kind == "on_chat_model_stream" and node in FINAL_AGENTS:
                chunk = (event.get("data") or {}).get("chunk")
                text = getattr(chunk, "content", "") or ""
                if text:
                    if ttft_ms is None:
                        ttft_ms = int((time.perf_counter() - started) * 1000)
                    answer_parts.append(text)
                    yield text

            # 4) 에이전트별 토큰 집계
            if kind == "on_chat_model_end" and node:
                output = (event.get("data") or {}).get("output")
                usage = getattr(output, "usage_metadata", None) or {}
                bucket = per_agent.setdefault(node, {"input": 0, "output": 0})
                bucket["input"] += usage.get("input_tokens", 0)
                bucket["output"] += usage.get("output_tokens", 0)

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        # 턴이 끝났으니 로그를 1회 기록한다
        save_formatted_log({
            "timestamp": kst_now_iso(),
            "env": os.path.basename(cfg.LOG_DIR.rstrip(os.sep)),
            "thread_id": req.thread_id,
            "query": req.query,
            "route": route,
            "step_history": step_history,
            "final_answer": "".join(answer_parts),
            "token_cost": {
                "per_agent": per_agent,
                "agent_input_tokens": sum(v["input"] for v in per_agent.values()),
                "agent_output_tokens": sum(v["output"] for v in per_agent.values()),
                "total_tokens": sum(v["input"] + v["output"] for v in per_agent.values()),
            },
            "time_cost": {
                "ttft_ms": ttft_ms,
                "elapsed_ms": elapsed_ms,
            },
        })

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
