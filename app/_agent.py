"""에이전트 결정 로직 (사내 _agent.py 자리의 목업/구현).

FAKE_LLM=1 : 규칙 기반 결정 + FakeEcho 모델로 usage 기록
FAKE_LLM=0 : ChatOpenAI 구조화 출력(json) — 실패 시 규칙 기반 폴백
"""
import json

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal, Optional

import app.config as cfg
from app._llm import get_llm, structured_invoke
from app._util import fake_llm_echo, last_human_text
from app.actions import resolvers
from app.actions.registry import ACTION_REGISTRY


def _log(msg: str):
    print(f"[AGENT] {msg}", flush=True)


# ── Router ────────────────────────────────────────────────────────────────

_SUPERVISOR_HINT = ("반송", "이송", "옮겨", "이동", "목적지", "위치", "어디", "상태",
                    "로그", "이력", "에러", "원인", "캐리어", "장비", "추출", "명령")


class RouteOut(BaseModel):
    route: Literal["general", "supervisor"]


def router_agent(inputs: dict) -> dict:
    """사용자 질의 -> general | supervisor."""
    text = last_human_text(inputs.get("messages", []))
    if cfg.FAKE_LLM:
        hit = any(k in text for k in _SUPERVISOR_HINT) or bool(resolvers.extract_ids(text)["carrier_ids"])
        route = "supervisor" if hit else "general"
        fake_llm_echo("router", json.dumps({"route": route}, ensure_ascii=False),
                      config=inputs.get("config"))
        _log(f"router(fake) -> {route}")
        return {"route": route}
    try:
        out = structured_invoke(
            get_llm(), RouteOut,
            [HumanMessage(content=(
                "너는 AMHS 챗봇의 라우터다. 아래 질의가 설비/캐리어/반송/목적지/로그 등 "
                "업무 질의면 supervisor, 일반 잡담이면 general 로 분류하라.\n"
                f"질의: {text}"))],
            config=inputs.get("config"))
        _log(f"router(llm) -> {out.route}")
        return {"route": out.route}
    except Exception as e:
        _log(f"router llm 실패({e}) -> 규칙 폴백")
        hit = any(k in text for k in _SUPERVISOR_HINT)
        return {"route": "supervisor" if hit else "general"}


# ── Supervisor ────────────────────────────────────────────────────────────

class SupervisorOut(BaseModel):
    next: str


def supervisor_agent(text: str, members: list, config=None) -> str:
    """키워드/LLM 기반 member 선택. (needs/재진입 등 결정적 우선순위는
    supervisor_node 쪽에서 이미 처리된 뒤에만 호출된다.)"""
    if cfg.FAKE_LLM:
        if resolvers.detect_intent(text) or any(k in text for k in ("명령", "실행", "요청")):
            nxt = "ActionAgent"
        elif "위치" in text or "어디" in text:
            nxt = "LocationAgent"
        elif "상태" in text:
            nxt = "StatusAgent"
        elif any(k in text for k in ("로그", "이력", "에러", "원인")):
            nxt = "LogAgent"
        elif "추출" in text:
            nxt = "ExtractAgent"
        else:
            nxt = "FinalAnswerAgent"
        fake_llm_echo("supervisor", json.dumps({"next": nxt}, ensure_ascii=False), config=config)
        _log(f"supervisor(fake) -> {nxt}")
        return nxt
    try:
        out = structured_invoke(
            get_llm(), SupervisorOut,
            [HumanMessage(content=(
                "너는 멀티에이전트 챗봇의 Supervisor 다. 질의를 처리할 다음 에이전트를 "
                f"다음 중에서 골라라: {members + ['FinalAnswerAgent']}\n"
                "- 반송/목적지 등 '명령 실행' 은 ActionAgent\n"
                "- 위치 조회는 LocationAgent, 상태 조회는 StatusAgent, 로그 분석은 LogAgent\n"
                f"질의: {text}"))],
            config=config)
        nxt = out.next if out.next in members + ["FinalAnswerAgent"] else "FinalAnswerAgent"
        _log(f"supervisor(llm) -> {nxt}")
        return nxt
    except Exception as e:
        _log(f"supervisor llm 실패({e}) -> FinalAnswerAgent 폴백")
        return "FinalAnswerAgent"


# ── ActionAgent 의도 추출 ─────────────────────────────────────────────────

class IntentOut(BaseModel):
    action: Literal["transport", "dest_req", "unknown"] = "unknown"
    carrier_id: Optional[str] = Field(None, description="명령 대상 캐리어 ID (8자 영숫자)")
    eqp_id: Optional[str] = Field(None, description="목적지 장비 ID (영문3+숫자3)")
    reference_kind: Optional[Literal["carrier_location", "log_analysis"]] = Field(
        None, description="eqp_id 가 리터럴이 아니라 참조로 표현된 경우 그 종류")
    reference_carrier_id: Optional[str] = Field(
        None, description="carrier_location 참조의 대상 캐리어 ID")
    cancel: bool = False


def extract_intent(text: str, config=None) -> resolvers.IntentResult:
    """자연어 -> (액션, 파라미터, 참조, 취소) 구조화 추출."""
    if cfg.FAKE_LLM:
        r = resolvers.parse_intent(text)
        fake_llm_echo("action_intent",
                      json.dumps({"action": r.action, "params": r.params}, ensure_ascii=False),
                      config=config)
        return r
    try:
        spec_desc = "\n".join(
            f"- {s.name}({s.label}): 필수 {s.required_params}" for s in ACTION_REGISTRY.values())
        out: IntentOut = structured_invoke(
            get_llm(), IntentOut,
            [HumanMessage(content=(
                "너는 AMHS 명령 파라미터 추출기다. 사용자 발화에서 액션과 파라미터를 추출하라.\n"
                f"{spec_desc}\n"
                "eqp_id 가 '다른 캐리어가 있는 위치' 로 표현되면 reference_kind=carrier_location,\n"
                "'로그를 분석해 원인 장비로' 처럼 표현되면 reference_kind=log_analysis 로 표시하라.\n"
                f"발화: {text}"))],
            config=config)
        r = resolvers.IntentResult(
            action=None if out.action == "unknown" else out.action,
            params={k: v for k, v in
                    {"carrier_id": out.carrier_id, "eqp_id": out.eqp_id}.items() if v},
            reference=({"kind": out.reference_kind, "fill": "eqp_id",
                        "carrier_id": out.reference_carrier_id}
                       if out.reference_kind else None),
            cancel=out.cancel)
        _log(f"extract_intent(llm) -> {r}")
        return r
    except Exception as e:
        _log(f"extract_intent llm 실패({e}) -> 규칙 폴백")
        return resolvers.parse_intent(text)


# ── Final answer 스트리밍 ─────────────────────────────────────────────────

async def stream_final_answer(context: str, question: str, config=None,
                              role: str = "final") -> str:
    """최종 응답을 llm.astream 으로 생성 — on_chat_model_stream 이벤트가
    FinalAnswerAgent/FinalGeneralAgent 노드 이름으로 발생한다."""
    llm = get_llm()
    if cfg.FAKE_LLM:
        from app._llm import ECHO_MARKER
        prompt = f"[ROLE:{role}]\n{ECHO_MARKER}{context}"
    else:
        prompt = (
            "너는 AMHS 반송 시스템 챗봇의 최종 응답자다. 아래 처리 결과를 바탕으로 "
            "사용자 질문에 한국어로 간결하고 정확하게 답하라. 결과에 없는 내용은 지어내지 마라.\n"
            f"[처리 결과]\n{context}\n[사용자 질문]\n{question}")
    parts = []
    async for chunk in llm.astream([HumanMessage(content=prompt)], config=config):
        if chunk.content:
            parts.append(str(chunk.content))
    return "".join(parts)
