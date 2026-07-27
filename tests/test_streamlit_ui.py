"""Streamlit UI E2E 테스트 — 실제 위젯을 조작해 HITL 왕복을 확인한다.

AppTest 가 streamlit_app.py 를 그대로 실행하고, 그 UI 가 백그라운드로 띄운
실제 uvicorn 서버에 SSE 로 붙는다. 즉 UI → HTTP → 그래프 → HITL → UI 전 구간을 탄다.

실행: python3 tests/test_streamlit_ui.py
"""
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx
import uvicorn

import app.config as cfg
from app.main import app as fastapi_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int):
    server = uvicorn.Server(uvicorn.Config(fastapi_app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}/llm/api"
    for _ in range(60):                      # 최대 30초 대기
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                return base
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("API 서버 기동 실패")


def main():
    from streamlit.testing.v1 import AppTest

    base = _start_server(_free_port())
    cfg.API_BASE_URL = base                  # streamlit_app 이 매 실행마다 다시 읽는다
    print(f"API up: {base}")

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value.endswith("AMHS HITL Chatbot"), at.title[0].value
    print("초기 렌더 PASS")

    # 1) 파라미터가 부족한 질의 → HITL 질문이 화면에 떠야 한다
    at.chat_input[0].set_value("6PDMQ283 반송해줘").run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    md = "\n".join(m.value for m in at.markdown)
    assert "목적지 장비 ID" in md, md[-600:]
    print("1) 질의 -> HITL 파라미터 질문 렌더 PASS")

    # 2) 값을 답하면 승인/거절 버튼이 떠야 한다
    at.chat_input[0].set_value("STK102").run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    assert any("승인" in l for l in labels), labels
    assert any("거절" in l for l in labels), labels
    assert at.warning, "승인 경고 배너가 없음"
    print("2) 답변 -> 승인/거절 버튼 렌더 PASS")

    # 3) 승인 클릭 → 실제 실행 결과가 렌더돼야 한다
    next(b for b in at.button if "승인" in b.label).click().run()
    assert not at.exception, at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "TJ-" in md and "실행 완료" in md, md[-800:]
    caps = "\n".join(c.value for c in at.caption)
    assert "tok" in caps and "HITL" in caps, caps[-400:]
    print("3) 승인 클릭 -> 실행 결과 + 토큰/시간 캡션 PASS")

    # 4) 트레이스(노드/툴 단계)가 쌓였는지
    assert any("ActionAgent" in m.value for m in at.markdown), "트레이스에 ActionAgent 없음"
    print("4) 실행 트레이스 렌더 PASS")

    # 5) 거절 경로도 UI 에서 동작하는지
    at.chat_input[0].set_value("9ZXCV456 목적지 요청").run()
    assert not at.exception, at.exception
    assert any("거절" in b.label for b in at.button), [b.label for b in at.button]
    next(b for b in at.button if "거절" in b.label).click().run()
    assert not at.exception, at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "실행하지 않았습니다" in md, md[-600:]
    print("5) 거절 버튼 -> 취소 안내 렌더 PASS")

    print("\nSTREAMLIT UI E2E PASS")


if __name__ == "__main__":
    main()
