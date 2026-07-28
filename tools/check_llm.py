"""LLM 엔드포인트 적합성 점검 — 앱을 띄우기 전에 먼저 돌려 보세요.

    python tools/check_llm.py

.env 에 설정된 엔드포인트(사내 게이트웨이든 로컬 ollama 든)가
이 프로젝트가 요구하는 세 가지를 실제로 하는지 확인합니다.

  1) 모델 목록 조회      GET  {base}/models
  2) 일반 대화           POST {base}/chat/completions
  3) 구조화 출력         POST ... response_format=json_object   ← 제일 중요
                         (origin 관례: with_structured_output(..., "json_mode"))

3번이 이 프로젝트의 급소입니다. 라우팅·의도추출·답변분류·승인판정이 전부
구조화 출력을 쓰기 때문에, 모델이 이걸 못 하면 아무것도 동작하지 않습니다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage

import app.config as cfg
from app import _llm


def main():
    base = cfg.API_BASE_TEMPLATE.format(api_base="hcp")
    model = _llm._DEFAULT_MODEL

    print("=" * 60)
    print(f"  엔드포인트 : {base}")
    print(f"  기본 모델  : {model}")
    print(f"  드롭다운   : {_llm.AVAILABLE_MODELS}")
    print("=" * 60)

    # ── 1) 모델 목록
    print("\n[1/3] 모델 목록 조회...")
    try:
        picked = _llm.getmodellist("hcp", output=True, name=model, k=0)
        if picked is None:
            print("  ❌ 목록을 못 받았습니다. base_url 을 확인하세요.")
            return 1
        if picked != model:
            print(f"  ⚠️  기본 모델 '{model}' 이 목록에 없습니다. 서버가 가진 건 '{picked}'.")
            print(f"     .env 에  DEFAULT_MODEL={picked}  로 맞춰 주세요.")
        else:
            print(f"  ✅ '{model}' 확인")
    except Exception as e:
        print(f"  ❌ 조회 실패: {type(e).__name__}: {e}")
        print("     서버가 떠 있는지, base_url 이 맞는지 확인하세요.")
        print("     (ollama 라면: ollama serve / ollama list)")
        return 1

    llm = _llm.get_llm(temperature=0.0)

    # ── 2) 일반 대화
    print("\n[2/3] 일반 대화 호출...")
    try:
        r = llm.invoke([HumanMessage(content="한 단어로만 답하세요: 안녕")])
        print(f"  ✅ 응답: {str(r.content)[:60]!r}")
    except Exception as e:
        print(f"  ❌ 실패: {type(e).__name__}: {e}")
        return 1

    # ── 3) 구조화 출력 (급소)
    print("\n[3/3] 구조화 출력(with_structured_output, method=json_mode)...")
    try:
        from app._node import RouteResponse
        runner = llm.with_structured_output(RouteResponse, method="json_mode")
        out = runner.invoke([HumanMessage(
            content="캐리어를 반송하라는 요청이다. ActionAgent 를 고르세요.\n"
                    '반드시 JSON 만 출력: {"next": "ActionAgent"}')])
        print(f"  ✅ 구조화 응답: {out!r}")
    except Exception as e:
        print(f"  ❌ 실패: {type(e).__name__}: {e}")
        print()
        print("  이 모델/서버는 OpenAI 구조화 출력(json_schema)을 지원하지 않습니다.")
        print("  이 프로젝트는 모든 판단에 이걸 쓰므로 이대로는 못 돕니다.")
        print()
        print("  해볼 것:")
        print("   - ollama 를 최신 버전으로 (구조화 출력은 0.5+ 부터)")
        print("   - 도구/함수 호출을 지원하는 모델로 교체")
        print("     (ollama show <model> 로 capabilities 확인)")
        return 1

    print("\n" + "=" * 60)
    print("  ✅ 전부 통과 — 앱을 띄워도 됩니다.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
