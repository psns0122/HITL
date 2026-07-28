"""Streamlit 채팅 UI.

실행:
    uvicorn app.api.main:app --reload --port 8000   # 터미널 1
    streamlit run streamlit_app.py                  # 터미널 2

스트림 파싱
-----------
백엔드는 최종 답변을 raw text 로 흘리고, 제어 정보만 \\x1e 로 시작하는
JSON 한 줄로 보낸다. 그래서 청크를 받을 때마다 \\x1e 기준으로 갈라서
텍스트는 답변 버블에, JSON 은 트레이스/HITL 처리에 쓴다.

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

# 제어 프레임 구분자 (routes.py 의 EVENT_PREFIX 와 같아야 한다)
EVENT_PREFIX = "\x1e"

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

                # 제어 프레임이 섞여 있으면 갈라낸다
                while EVENT_PREFIX in buffer:
                    text_part, _, rest = buffer.partition(EVENT_PREFIX)

                    if text_part:
                        yield ("text", text_part)

                    # 제어 프레임은 개행으로 끝난다. 아직 안 왔으면 다음 청크를 기다린다.
                    if "\n" not in rest:
                        buffer = EVENT_PREFIX + rest
                        break

                    line, _, remainder = rest.partition("\n")
                    try:
                        yield ("event", json.loads(line))
                    except json.JSONDecodeError:
                        pass
                    buffer = remainder
                else:
                    # 제어 프레임이 없으면 통째로 텍스트
                    if buffer:
                        yield ("text", buffer)
                        buffer = ""

    if buffer:
        yield ("text", buffer)


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

def render_trace(trace: list, expanded: bool = True):
    """노드/툴 실행 트레이스를 접이식으로. 기본은 펼침 — 접는 건 사용자 선택.

    카드에는 두 가지만 담는다.
      ▶ 노드   : 어느 노드에 들어갔는지
      - 툴    : 무엇을 어떤 입력으로 실행해 어떤 결과가 나왔는지
    에이전트의 중간 응답/상태 문구는 싣지 않는다 — 시끄럽기만 하다.
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


# 지난 대화 렌더
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn.get("trace"):
            render_trace(turn["trace"])
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

                # 카드에 담는 건 node_enter / tool_call 뿐이다.
                # agent_status(상태 문구)와 thinking(중간 응답 토큰)은 버린다.
                if t == "node_enter":
                    trace.append(f"{NODE_ICON} **{ev['agent']}**")
                    redraw_trace()
                    status.update(label=f"{ev['agent']} 실행 중…", expanded=True)

                elif t == "tool_call":
                    # 툴 하나 = 한 줄: `- tool 입력: … → 결과: …`
                    # ReAct 툴은 입력(on_tool_start)과 결과(on_tool_end)가 두 이벤트로
                    # 나뉘어 오므로, 결과만 온 이벤트는 직전 같은 툴 줄에 이어 붙인다.
                    tool = ev.get("tool")
                    args = ev.get("args")
                    result = ev.get("result")

                    if (result is not None and args is None and trace
                            and trace[-1].startswith(f"- `{tool}`")
                            and "→ 결과:" not in trace[-1]):
                        trace[-1] += f" → 결과: `{_fmt_val(result)}`"
                    else:
                        line = f"- `{tool}`"
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

                elif t == "error":
                    st.error(ev.get("message"))

        except Exception as e:
            status.update(label="연결 실패", state="error")
            st.error(
                f"백엔드 호출 실패: {e}\n\n"
                "`uvicorn app.api.main:app --port 8000` 이 떠 있는지 확인하세요."
            )
            return

        # HITL 로 멈춘 경우: 질문을 답변 버블에 띄운다
        if needs:
            status.update(label="사용자 입력 대기 ⏸", state="complete", expanded=True)
            answer = needs.get("prompt", "추가 입력이 필요합니다.")
            answer_box.markdown(answer)
        else:
            status.update(label="완료", state="complete", expanded=True)

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
