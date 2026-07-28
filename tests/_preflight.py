"""테스트 선행 점검 — 사내 LLM 게이트웨이가 붙어 있는지 확인한다.

이 프로젝트의 모든 판단(라우팅·의도 추출·답변 분류·승인 판정)은 실제
모델이 하므로, 시나리오 테스트도 게이트웨이가 있어야 돌아간다.
사내망 밖에서 돌리면 여기서 이유를 알려주고 멈춘다.
"""
import sys

import app.config as cfg


def require_gateway():
    """게이트웨이에 실제로 한 번 붙어 보고, 안 되면 이유를 찍고 종료한다."""
    from app import _llm
    print(f"[preflight] gateway={cfg.API_BASE_TEMPLATE} "
          f"model={_llm._DEFAULT_MODEL}", flush=True)

    try:
        from langchain_core.messages import HumanMessage

        from app._llm import get_llm

        # 가장 싼 호출로 왕복만 확인한다
        get_llm(temperature=0.0).invoke([HumanMessage(content="ping")])

    except Exception as e:
        print("\n[preflight] 게이트웨이에 붙지 못했습니다.", flush=True)
        print(f"           사유: {type(e).__name__}: {e}", flush=True)
        print("\n이 테스트들은 실제 LLM 이 있어야 돕니다.", flush=True)
        print(".env 의 LLM_GATEWAY_BASE_URL / LLM_GATEWAY_API_KEY 를 확인하고,", flush=True)
        print("사내망에서 다시 실행해 주세요.", flush=True)
        sys.exit(1)

    print("[preflight] OK\n", flush=True)
