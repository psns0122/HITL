"""토큰/시간 원장 (thread_id 키, in-memory).

HITL 때문에 한 번의 논리적 액션이 여러 번의 /chat/stream 호출에 걸쳐 진행된다.
(최초 질의 → interrupt → 답변 resume → interrupt → 승인 resume → 완료)
그래서 토큰·시간은 호출마다 리셋하면 안 되고 thread_id 별로 **누적**해야 한다.

- 누적 대상 : 에이전트별 input/output 토큰, step_history, 툴 호출, 라운드 수
- 시간      : TTFT(최초 입력 → 첫 최종답변 토큰), 총 소요, 그리고 사람이 생각한
              시간(human_wait_ms)을 뺀 순수 연산 시간
- flush     : 완료(done:complete/aborted) 시 1회만 jsonl 로 기록하고 원장 삭제

주의: LangGraph 는 interrupt 된 노드를 resume 시 맨 위부터 재실행하므로,
interrupt 위쪽에서 LLM 을 호출하면 그 토큰은 이중 계상된다. 본 프로젝트는
interrupt 노드(collect_param/confirm)에 LLM 호출을 두지 않아 구조적으로 회피한다.
"""
import time

_LEDGERS: dict[str, dict] = {}


def _new(thread_id: str, query: str) -> dict:
    return {
        "thread_id": thread_id,
        "query": query,                # 이 액션을 시작시킨 최초 사용자 질의
        "route": None,
        "per_agent": {},               # {agent: {input, output, calls}}
        "input_tokens": 0,             # 최초 사용자 질의의 입력 토큰
        "step_history": [],            # [{step, node}] 노드 실행 이력
        "tool_calls": [],              # [{agent, tool, args, result}]
        "hitl_rounds": 0,              # 사용자에게 되물은 횟수
        "n_stream_calls": 0,           # /chat/stream 호출 횟수(최초 + resume)
        "t_start": time.time(),        # 최초 입력 시각
        "t_first_token": None,         # 첫 최종답변 토큰 시각 (TTFT 기준)
        "t_completed": None,
        "human_wait_ms": 0,            # interrupt 로 멈춰 사람을 기다린 총 시간
        "_t_paused_at": None,          # 마지막 interrupt 발생 시각
        "final_answer": "",
    }


def begin_turn(thread_id: str, query: str, resuming: bool) -> dict:
    """스트림 호출 시작. 신규 턴이면 원장을 새로 만들고, resume 이면 이어쓴다."""
    led = _LEDGERS.get(thread_id)
    if led is None or not resuming:
        led = _new(thread_id, query)
        _LEDGERS[thread_id] = led
    led["n_stream_calls"] += 1

    if resuming:
        led["hitl_rounds"] += 1
        # 멈춰 있던 동안(사람이 답을 고민한 시간)은 연산 시간에서 제외한다
        if led.get("_t_paused_at"):
            led["human_wait_ms"] += int((time.time() - led["_t_paused_at"]) * 1000)
            led["_t_paused_at"] = None
    return led


def get(thread_id: str) -> dict | None:
    return _LEDGERS.get(thread_id)


def add_usage(thread_id: str, agent: str, usage: dict):
    """on_chat_model_end 에서 잡은 usage_metadata 를 에이전트별로 누적."""
    led = _LEDGERS.get(thread_id)
    if not led or not usage:
        return
    slot = led["per_agent"].setdefault(agent, {"input": 0, "output": 0, "calls": 0})
    slot["input"] += int(usage.get("input_tokens") or 0)
    slot["output"] += int(usage.get("output_tokens") or 0)
    slot["calls"] += 1


def set_input_tokens(thread_id: str, n: int):
    """최초 사용자 입력의 토큰 수 (한 번만 기록)."""
    led = _LEDGERS.get(thread_id)
    if led and not led["input_tokens"]:
        led["input_tokens"] = int(n)


def add_step(thread_id: str, node: str):
    led = _LEDGERS.get(thread_id)
    if not led:
        return
    # 같은 노드가 연속 중복 기록되는 것만 억제(재실행 노이즈)
    if led["step_history"] and led["step_history"][-1]["node"] == node:
        return
    led["step_history"].append({"step": len(led["step_history"]) + 1, "node": node})


def add_tool_call(thread_id: str, payload: dict):
    led = _LEDGERS.get(thread_id)
    if led:
        led["tool_calls"].append(payload)


def set_route(thread_id: str, route: str):
    led = _LEDGERS.get(thread_id)
    if led and not led["route"]:
        led["route"] = route


def mark_first_token(thread_id: str):
    led = _LEDGERS.get(thread_id)
    if led and led["t_first_token"] is None:
        led["t_first_token"] = time.time()
        # HITL 액션은 첫 최종답변 토큰이 여러 라운드 뒤에야 나온다.
        # 그때까지 사람이 답을 고민한 시간은 TTFT 가 아니므로, 이 시점까지
        # 누적된 대기를 기억해 뒀다가 totals 에서 뺀다. (안 빼면 승인에
        # 1분 고민한 턴의 "첫 응답"이 80초처럼 보인다)
        led["_human_wait_at_first_token"] = led["human_wait_ms"]


def append_answer(thread_id: str, text: str):
    led = _LEDGERS.get(thread_id)
    if led:
        led["final_answer"] += text


def mark_paused(thread_id: str):
    """interrupt 로 멈춤 — 여기서부터 사람 대기 시간 측정 시작."""
    led = _LEDGERS.get(thread_id)
    if led:
        led["_t_paused_at"] = time.time()


def totals(thread_id: str) -> dict:
    """SSE usage 이벤트 / 로그에 실을 집계값."""
    led = _LEDGERS.get(thread_id)
    if not led:
        return {}
    tin = sum(a["input"] for a in led["per_agent"].values())
    tout = sum(a["output"] for a in led["per_agent"].values())
    now = led.get("t_completed") or time.time()
    elapsed_ms = int((now - led["t_start"]) * 1000)
    ttft_ms = (int((led["t_first_token"] - led["t_start"]) * 1000)
               - led.get("_human_wait_at_first_token", 0)
               if led["t_first_token"] else None)
    return {
        "user_input_tokens": led["input_tokens"],
        "per_agent": led["per_agent"],
        "agent_input_tokens": tin,
        "agent_output_tokens": tout,
        "total_tokens": tin + tout,
        "hitl_rounds": led["hitl_rounds"],
        "stream_calls": led["n_stream_calls"],
        "ttft_ms": ttft_ms,                                  # 최초 입력 → 첫 최종토큰 (사람 대기 제외)
        "elapsed_ms": elapsed_ms,                            # 최초 입력 → 종료(사람 대기 포함)
        "human_wait_ms": led["human_wait_ms"],
        "compute_ms": max(0, elapsed_ms - led["human_wait_ms"]),  # 사람 대기 제외
    }


def finish(thread_id: str) -> dict:
    """완료 처리 후 원장을 꺼내고 삭제한다(다음 액션은 새 원장)."""
    led = _LEDGERS.get(thread_id)
    if not led:
        return {}
    led["t_completed"] = time.time()
    snapshot = {**led, "totals": totals(thread_id)}
    snapshot.pop("_t_paused_at", None)
    snapshot.pop("_human_wait_at_first_token", None)
    _LEDGERS.pop(thread_id, None)
    return snapshot
