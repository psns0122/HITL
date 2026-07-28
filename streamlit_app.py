"""Streamlit 채팅 UI.

실행:
    uvicorn app.api.main:app --reload --port 8000   # 터미널 1
    streamlit run streamlit_app.py                  # 터미널 2

스트림 파싱 — 정식 SSE
----------------------
백엔드는 모든 것을 SSE 프레임(`event:` + `data:` JSON + 빈 줄)으로 보낸다.
답변 토큰은 event 이름 `token`, 나머지는 type 값이 그대로 event 이름이다.
POST 스트림이라 브라우저 EventSource 는 못 쓰고 httpx 로 받아 직접 파싱한다.

HITL 렌더링
-----------
    kind=collect_param : 채팅창에 값을 입력하면 그대로 재개 답변이 된다.
    kind=confirm       : [승인] / [거절] 버튼을 띄운다 (자연어 입력도 허용).

채팅 세션 하나당 thread_id 하나를 유지해 대화 맥락을 이어간다.
"""
import json
import uuid

import httpx
import streamlit as st

import app.config as cfg

API = cfg.API_BASE_URL



# 프론트 모델 선택지. 첫 번째가 기본값이다.
MODEL_CHOICES = [
    "GaiA-LLM-Latest",
    "gaia-GLM-5.2",
    "Qwen3.5-397B-A17B-FP8",
]

st.set_page_config(page_title="AMHS HITL Chatbot", page_icon="🚚", layout="centered")


# ─────────────────────────────────────────────────────────────────────────
# 세션 상태
# ─────────────────────────────────────────────────────────────────────────

def _init_state():
    ss = st.session_state
    ss.setdefault("thread_id", f"ui-{uuid.uuid4().hex[:8]}")
    ss.setdefault("history", [])       # [{role, content, trace?, usage?}]
    ss.setdefault("pending", None)     # 마지막 needs_input payload
    ss.setdefault("outbox", None)      # 버튼으로 예약한 전송


_init_state()


# ─────────────────────────────────────────────────────────────────────────
# 백엔드 호출
# ─────────────────────────────────────────────────────────────────────────

def stream_chat(query: str, model_name: str, recursion_limit: int):
    """/chat/stream 을 호출하고 (종류, 값) 튜플을 하나씩 yield 한다.

    yield 형태
        ("text",  "…")   최종 답변 조각
        ("event", {...}) 제어 프레임
    """
    payload = {
        "query": query,
        "thread_id": st.session_state.thread_id,
        "model_name": model_name,
        "recursion_limit": recursion_limit,
    }

    buffer = ""

    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{API}/chat/stream", json=payload) as r:
            r.raise_for_status()

            for raw in r.iter_text():
                if not raw:
                    continue

                buffer += raw

                # SSE 프레임은 빈 줄로 끝난다. 완성된 프레임만 하나씩 파싱한다.
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)

                    event_name, data_lines = "message", []
                    for line in block.splitlines():
                        if line.startswith("event:"):
                            event_name = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].lstrip())

                    if not data_lines:
                        continue

                    try:
                        data = json.loads("\n".join(data_lines))
                    except json.JSONDecodeError:
                        continue

                    if event_name == "token":
                        yield ("text", data.get("text", ""))
                    else:
                        yield ("event", data)


def call_stop():
    """실행 중단 / 진행 중인 액션 취소."""
    try:
        r = httpx.post(f"{API}/chat/stop",
                       json={"thread_id": st.session_state.thread_id}, timeout=30)
        st.toast(f"중단 요청: {r.json().get('mode')}")
    except Exception as e:
        st.toast(f"중단 실패: {e}")

    st.session_state.pending = None


# ─────────────────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("모델")
    model_name = st.selectbox(
        "사용할 모델",
        MODEL_CHOICES,
        index=0,
        help="프론트에서 고른 모델로 그래프가 빌드/캐싱됩니다.",
    )

    recursion_limit = st.number_input(
        "recursion_limit",
        min_value=1,
        max_value=200,
        value=20,
        step=5,
        help="LangGraph 순환 상한. 노드가 이 횟수를 넘게 돌면 중단됩니다.",
    )

    st.divider()
    st.subheader("세션")
    st.code(st.session_state.thread_id, language=None)
    st.caption("채팅 세션 하나당 thread_id 하나 — 대화 맥락이 이어집니다.")

    if st.button("🆕 새 대화", use_container_width=True):
        st.session_state.thread_id = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.history = []
        st.session_state.pending = None
        st.rerun()

    if st.button("⛔ 실행 중단 / 액션 취소", use_container_width=True):
        call_stop()
        st.rerun()

    st.divider()
    st.subheader("목업 데이터")
    st.caption("Carrier")
    st.code("6PDMQ283  3KWQ7712\n9ZXCV456  7HITL001", language=None)
    st.caption("Equipment")
    st.code("STK101  STK102  PHT201\nETC301  CLN501  DFF401(offline)", language=None)

    st.divider()
    st.subheader("예시 질의")
    examples = [
        "6PDMQ283 반송해줘",
        "9ZXCV456 목적지 요청",
        "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘",
        "로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex-{ex}"):
            st.session_state.outbox = ex
            st.rerun()


st.title("🚚 AMHS HITL Chatbot")
st.caption("Router → Supervisor → ExtractAgent → 워커 → FinalAnswerAgent")


# ─────────────────────────────────────────────────────────────────────────
# 렌더 헬퍼
# ─────────────────────────────────────────────────────────────────────────

def render_trace(trace: list, expanded: bool = False):
    """노드/툴 실행 트레이스를 접이식으로.

    카드에 담는 것:
      ▶ 노드              : 어느 노드에 들어갔는지
      - 툴               : 무엇을 어떤 입력으로 실행해 어떤 결과가 나왔는지
      ⏸ 사용자 입력 대기  : HITL 로 멈춘 턴이면 마지막 줄로 (질문 주체 표기)
    """
    if not trace:
        return

    with st.expander(f"실행 트레이스 ({len(trace)} step)", expanded=expanded):
        for line in trace:
            st.markdown(line)


def _fmt_secs(ms) -> str:
    """ms -> 초 문자열. None 이면 '-'."""
    if ms is None:
        return "-"
    return f"{ms / 1000:.2f}s"


def render_usage(u: dict):
    """토큰/시간/HITL 요약을 하단 접이식 카드로 숨긴다(항목 13·14)."""
    if not u:
        return

    with st.expander("응답 상세 (토큰 · 시간 · HITL)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("첫 응답", _fmt_secs(u.get("ttft_ms")))
        c2.metric("총 시간", _fmt_secs(u.get("elapsed_ms")))
        c3.metric("HITL", f"{u.get('hitl_rounds', 0)}회")

        st.caption(
            f"총 토큰 {u.get('total_tokens')} "
            f"(사용자 입력 {u.get('user_input_tokens')} · "
            f"에이전트 in {u.get('agent_input_tokens')} / out {u.get('agent_output_tokens')})　"
            f"연산 시간 {_fmt_secs(u.get('compute_ms'))} "
            f"(사람 대기 {_fmt_secs(u.get('human_wait_ms'))} 제외)"
        )

        per_agent = u.get("per_agent") or {}
        if per_agent:
            st.caption("에이전트별 토큰")
            st.table({
                "에이전트": list(per_agent.keys()),
                "in": [v.get("input", 0) for v in per_agent.values()],
                "out": [v.get("output", 0) for v in per_agent.values()],
                "calls": [v.get("calls", 0) for v in per_agent.values()],
            })


# 지난 대화 렌더 — **마지막 턴만** 펼치고 그 이전 턴은 접어 둔다.
# 전부 펼치면 두세 턴만 쌓여도 화면이 트레이스로 가득 차 최신 답변이
# 밀려나고, 전부 접으면 방금 끝난 턴의 트레이스가 사라져 보인다.
_last = len(st.session_state.history) - 1
for _i, turn in enumerate(st.session_state.history):
    with st.chat_message(turn["role"]):
        if turn.get("trace"):
            render_trace(turn["trace"], expanded=(_i == _last))
        st.markdown(turn["content"])
        render_usage(turn.get("usage"))


# ─────────────────────────────────────────────────────────────────────────
# 전송 처리
# ─────────────────────────────────────────────────────────────────────────

# 진입점만 아이콘. 툴은 아이콘 없이 `- ...` 로 통일 (항목 9)
NODE_ICON = "▶"


def _fmt_val(v, limit: int = 300) -> str:
    """툴 입력/결과를 카드 한 줄에 들어가게 다듬는다 (개행 제거 + 길이 제한)."""
    text = " ".join(str(v).split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text


def send(query: str):
    """질문을 보내고 스트림을 화면에 그린다."""
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    trace: list[str] = []
    answer = ""
    usage = None
    needs = None
    final_agent = None    # FinalAnswerAgent / FinalGeneralAgent 중 누가 답했는지

    with st.chat_message("assistant"):
        status = st.status("에이전트 실행 중…", expanded=True)
        trace_box = status.empty()      # 트레이스는 통째로 다시 그린다 (병합 반영)
        answer_box = st.empty()

        def redraw_trace():
            if trace:
                trace_box.markdown("\n\n".join(trace))

        try:
            for kind, item in stream_chat(query, model_name, int(recursion_limit)):

                # --- 최종 답변 조각
                if kind == "text":
                    answer += item
                    answer_box.markdown(answer)
                    continue

                # --- 제어 프레임
                ev = item
                t = ev.get("type")

                # 트레이스는 이벤트 하나(trace)로 온다.
                #   tool == null : 노드 진입  -> `▶ agent`
                #   tool != null : 툴 실행    -> `- tool 입력 → 결과` 한 줄
                if t == "trace":
                    tool = ev.get("tool")

                    if not tool:
                        # 노드 진입
                        trace.append(f"{NODE_ICON} **{ev['agent']}**")
                        redraw_trace()
                        status.update(label=f"{ev['agent']} 실행 중…", expanded=True)
                        if ev["agent"] in ("FinalAnswerAgent", "FinalGeneralAgent"):
                            final_agent = ev["agent"]
                        continue

                    # 툴 실행. 입력(시작)과 결과(종료)가 두 프레임으로 나뉘어
                    # 오므로, 결과만 온 프레임은 같은 툴의 '결과 없는 줄' 을
                    # 뒤에서부터 찾아 이어 붙인다. 카드는 trace 리스트를 통째로
                    # 다시 그리므로(redraw_trace) 이미 그린 줄도 제자리에서 갱신된다.
                    args = ev.get("args")
                    result = ev.get("result")

                    merged = False
                    if result is not None and args is None:
                        for i in range(len(trace) - 1, -1, -1):
                            if (trace[i].startswith(f"- {tool} ")
                                    or trace[i] == f"- {tool}") \
                                    and "→ 결과:" not in trace[i]:
                                trace[i] += f" → 결과: `{_fmt_val(result)}`"
                                merged = True
                                break

                    if not merged:
                        line = f"- {tool}"
                        if args is not None:
                            line += f" 입력: `{_fmt_val(args)}`"
                        if result is not None:
                            line += f" → 결과: `{_fmt_val(result)}`"
                        trace.append(line)
                    redraw_trace()

                elif t == "needs_input":
                    needs = ev

                elif t == "usage":
                    usage = ev

                elif t == "done" and ev.get("reason") == "error":
                    st.error(ev.get("message"))

        except Exception as e:
            status.update(label="연결 실패", state="error")
            st.error(
                f"백엔드 호출 실패: {e}\n\n"
                "`uvicorn app.api.main:app --port 8000` 이 떠 있는지 확인하세요."
            )
            return

        # 종료 라벨 — 방금 끝난 턴의 카드는 펼친 채 둔다.
        # (rerun 후에는 히스토리의 마지막 턴으로서 같은 라벨·펼침으로 다시 그려진다)
        if needs:
            asker = needs.get("agent") or "에이전트"
            # 대기 상태를 트레이스의 마지막 줄로 남긴다 — 질문 주체 노드를
            # 진입점(▶)으로 찍고, 대기 상태는 툴과 같은 `- ` 표기로 통일
            trace.append(f"{NODE_ICON} **{asker}**")
            trace.append("- 사용자 입력 대기")
            redraw_trace()
            status.update(label=f"사용자 입력 대기 ⏸ · {asker}",
                          state="complete", expanded=True)
            answer = needs.get("prompt", "추가 입력이 필요합니다.")
            answer_box.markdown(answer)
        else:
            # Supervisor 다음에 어느 최종 에이전트가 답했는지 라벨에 남긴다
            label = f"완료 · {final_agent}" if final_agent else "완료"
            status.update(label=label, state="complete", expanded=True)

        render_usage(usage)

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "trace": trace,
        "usage": usage,
    })
    st.session_state.pending = needs


# 사이드바 예시 / 승인 버튼이 예약한 전송
if st.session_state.outbox:
    q = st.session_state.outbox
    st.session_state.outbox = None
    send(q)
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────
# HITL 위젯
# ─────────────────────────────────────────────────────────────────────────

pending = st.session_state.pending

if pending and pending.get("kind") == "confirm":
    st.warning("실행 승인이 필요합니다.")

    c1, c2 = st.columns(2)
    if c1.button("✅ 승인", use_container_width=True, type="primary"):
        st.session_state.outbox = "승인"
        st.rerun()
    if c2.button("🚫 거절", use_container_width=True):
        st.session_state.outbox = "거절"
        st.rerun()

# 입력창 placeholder 를 현재 HITL 상태에 맞춰 바꾼다
placeholder = "메시지를 입력하세요"
if pending:
    if pending.get("kind") == "collect_param":
        placeholder = f"{pending.get('field')} 값을 입력하세요 (취소하려면 '취소')"
    elif pending.get("kind") == "confirm":
        placeholder = "승인 / 거절 (버튼 대신 직접 입력해도 됩니다)"

if q := st.chat_input(placeholder):
    send(q)
    st.rerun()
