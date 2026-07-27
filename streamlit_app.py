"""Streamlit 채팅 UI — FastAPI SSE 백엔드에 붙는 프론트.

실행:
    uvicorn app.main:app --reload --port 8000     # 터미널 1
    streamlit run streamlit_app.py                # 터미널 2

HITL 렌더링 규칙
- kind=collect_param : 아래 채팅창에 값을 입력하면 그대로 재개 답변이 된다.
- kind=confirm       : [승인] / [거절] 버튼을 띄운다(자연어 입력도 허용).
- 채팅 세션 하나당 thread_id 하나를 유지해 대화 맥락(자아)을 이어간다.
"""
import json
import uuid

import httpx
import streamlit as st

import app.config as cfg

API = cfg.API_BASE_URL

st.set_page_config(page_title="AMHS HITL Chatbot", page_icon="🚚", layout="centered")


# ── 상태 초기화 ───────────────────────────────────────────────────────────
def _init():
    ss = st.session_state
    ss.setdefault("thread_id", f"ui-{uuid.uuid4().hex[:8]}")
    ss.setdefault("history", [])      # [{role, content, trace?, usage?}]
    ss.setdefault("pending", None)    # 마지막 needs_input payload
    ss.setdefault("outbox", None)     # 버튼으로 넣은 전송 예약


_init()


# ── SSE 클라이언트 ────────────────────────────────────────────────────────
def sse_iter(query: str):
    """/chat/stream 을 호출하고 이벤트(dict)를 하나씩 yield."""
    with httpx.Client(timeout=None) as client:
        with client.stream("POST", f"{API}/chat/stream",
                           json={"query": query,
                                 "thread_id": st.session_state.thread_id}) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])


def call_stop():
    try:
        r = httpx.post(f"{API}/chat/stop",
                       json={"thread_id": st.session_state.thread_id}, timeout=30)
        st.toast(f"중단 요청: {r.json().get('mode')}")
    except Exception as e:
        st.toast(f"중단 실패: {e}")
    st.session_state.pending = None


# ── 사이드바 ──────────────────────────────────────────────────────────────
with st.sidebar:
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
    for ex in ["6PDMQ283 반송해줘",
               "9ZXCV456 목적지 요청",
               "6PDMQ283 를 9ZXCV456 있는 위치로 반송해줘",
               "로그 분석해서 원인 장비 피해서 6PDMQ283 반송해줘"]:
        if st.button(ex, use_container_width=True, key=f"ex-{ex}"):
            st.session_state.outbox = ex
            st.rerun()


st.title("🚚 AMHS HITL Chatbot")
st.caption("Router → Supervisor → ActionAgent(HITL) → FinalAnswerAgent")


# ── 지난 대화 렌더 ────────────────────────────────────────────────────────
def render_trace(trace: list, expanded: bool = False):
    if not trace:
        return
    with st.expander(f"🧠 실행 트레이스 ({len(trace)} step)", expanded=expanded):
        for line in trace:
            st.markdown(line)


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        if turn.get("trace"):
            render_trace(turn["trace"])
        st.markdown(turn["content"])
        if turn.get("usage"):
            u = turn["usage"]
            st.caption(
                f"🔢 총 {u.get('total_tokens')} tok "
                f"(입력 {u.get('user_input_tokens')} / 에이전트 in {u.get('agent_input_tokens')} "
                f"· out {u.get('agent_output_tokens')})　"
                f"⏱ 첫 응답 {u.get('ttft_ms')}ms · 총 {u.get('elapsed_ms')}ms "
                f"(사람대기 {u.get('human_wait_ms')}ms 제외 시 {u.get('compute_ms')}ms)　"
                f"🙋 HITL {u.get('hitl_rounds')}회"
            )


# ── 전송 처리 ─────────────────────────────────────────────────────────────
ICON = {"node_enter": "▶️", "tool_call": "🔧", "agent_status": "💬", "thinking": "🧠"}


def send(query: str):
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    trace: list[str] = []
    answer = ""
    usage = None
    needs = None

    with st.chat_message("assistant"):
        status = st.status("에이전트 실행 중…", expanded=True)
        answer_box = st.empty()
        try:
            for ev in sse_iter(query):
                t = ev.get("type")

                if t == "token":
                    answer += ev["text"]
                    answer_box.markdown(answer)

                elif t == "node_enter":
                    line = f"{ICON[t]} **{ev['agent']}** 진입"
                    trace.append(line)
                    status.write(line)
                    status.update(label=f"{ev['agent']} 실행 중…")

                elif t == "tool_call":
                    detail = f"{ICON[t]} `{ev.get('tool')}`"
                    if ev.get("args"):
                        detail += f" · args={ev['args']}"
                    if ev.get("result"):
                        detail += f" → {ev['result']}"
                    trace.append(detail)
                    status.write(detail)

                elif t == "agent_status":
                    line = f"{ICON[t]} **{ev.get('agent')}** — {ev.get('detail')}"
                    trace.append(line)
                    status.write(line)

                elif t == "thinking":
                    status.write(f"{ICON[t]} {ev.get('agent')}: {ev.get('text')}")

                elif t == "needs_input":
                    needs = ev

                elif t == "usage":
                    usage = ev

                elif t == "error":
                    st.error(ev.get("message"))

        except Exception as e:
            status.update(label="연결 실패", state="error")
            st.error(f"백엔드 호출 실패: {e}\n\n`uvicorn app.main:app --port 8000` 이 떠 있는지 확인하세요.")
            return

        # HITL 로 멈춘 경우: 질문을 답변 버블에 띄운다
        if needs:
            status.update(label="사용자 입력 대기 ⏸", state="complete")
            answer = needs.get("prompt", "추가 입력이 필요합니다.")
            answer_box.markdown(answer)
        else:
            status.update(label="완료", state="complete")

        if usage:
            st.caption(
                f"🔢 총 {usage.get('total_tokens')} tok "
                f"(입력 {usage.get('user_input_tokens')} / 에이전트 in {usage.get('agent_input_tokens')} "
                f"· out {usage.get('agent_output_tokens')})　"
                f"⏱ 첫 응답 {usage.get('ttft_ms')}ms · 총 {usage.get('elapsed_ms')}ms "
                f"(사람대기 {usage.get('human_wait_ms')}ms 제외 시 {usage.get('compute_ms')}ms)　"
                f"🙋 HITL {usage.get('hitl_rounds')}회"
            )

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "trace": trace, "usage": usage})
    st.session_state.pending = needs


# 사이드바 예시 버튼 / 승인 버튼이 예약한 전송
if st.session_state.outbox:
    q = st.session_state.outbox
    st.session_state.outbox = None
    send(q)
    st.rerun()


# ── HITL 위젯 ─────────────────────────────────────────────────────────────
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

placeholder = "메시지를 입력하세요"
if pending:
    if pending.get("kind") == "collect_param":
        placeholder = f"{pending.get('field')} 값을 입력하세요 (취소하려면 '취소')"
    elif pending.get("kind") == "confirm":
        placeholder = "승인 / 거절 (버튼 대신 직접 입력해도 됩니다)"

if q := st.chat_input(placeholder):
    send(q)
    st.rerun()
