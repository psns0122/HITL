"""공용 헬퍼.

LLM 응답 파싱, 메시지 훑기, SSE 트레이스 이벤트 발행처럼
여러 모듈이 함께 쓰는 잡동사니를 모아둔다.
"""
import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# *************  [app 전용 import — 판단부와 ActionService 가 쓴다]  *************
import time
from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field, create_model

import app.config as cfg
from app import _llm, _prompt, _tool
from app._state import AgentState
from app._tool import param_check_tool
# *************



# ─────────────────────────────────────────────────────────────────────────
# SSE 트레이스
# ─────────────────────────────────────────────────────────────────────────

def emit(config, name: str, data: dict):
    """SSE 트레이스용 커스텀 이벤트 발행.

    astream_events 의 on_custom_event 로 잡힌다.
    이벤트 발행이 실패해도 그래프 실행이 깨지면 안 되므로 조용히 무시한다.
    """
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data, config=config)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# 메시지 훑기
# ─────────────────────────────────────────────────────────────────────────

def message_content_to_text(content) -> str:
    """LLM 응답의 content 를 문자열로 정규화한다.

    content 는 모델/버전에 따라 세 가지 형태로 온다.
      1) 문자열                      -> 그대로
      2) 블록 리스트                 -> text 블록만 이어붙임
      3) 그 외(None, dict 등)        -> str() 로 강제
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    # 멀티모달/블록 형태: [{"type": "text", "text": "..."}, ...]
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)

    return str(content)


def last_user_text(source) -> str:
    """가장 최근 사용자 발화. 없으면 빈 문자열.

    호출부가 messages 리스트를 주기도 하고 state 를 통째로 주기도 한다
    (사내 코드에서 Router 는 state, GeneralAgent 는 messages 를 넘긴다).
    둘 다 받는다.
    """
    messages = source.get("messages", []) if isinstance(source, dict) else source

    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return message_content_to_text(msg.content)
    return ""


def agent_name_of(msg) -> str | None:
    """메시지를 만든 에이전트 이름.

    사내 규약은 additional_kwargs["agent_name"] 이다 (_util.agent_node 가 박는다).
    name= 은 이 저장소가 함께 싣는 폴백 — 판독은 사내 규약을 먼저 본다.
    """
    if not isinstance(msg, AIMessage):
        return None
    return (msg.additional_kwargs or {}).get("agent_name") or getattr(msg, "name", None)


def member_answered_this_turn(messages: list, members: list) -> AIMessage | None:
    """이번 user turn 안에서 member 에이전트가 남긴 마지막 AIMessage.

    직전 HumanMessage 를 만나면 멈춘다 = 이전 턴 결과는 보지 않는다.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return None
        if agent_name_of(msg) in members:
            return msg
    return None


def agent_ran_this_turn(messages: list, agent_name: str) -> bool:
    """이번 턴에 특정 에이전트가 이미 실행됐는지.

    ExtractAgent 를 턴당 한 번만 태우는 판정에 쓴다.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return False
        if agent_name_of(msg) == agent_name:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# LLM 출력 파싱
# ─────────────────────────────────────────────────────────────────────────

# ```json ... ``` 같은 코드펜스로 감싸 오는 모델이 많아서 먼저 벗겨낸다
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """응답 문자열에서 JSON 객체 하나를 꺼낸다. 실패하면 빈 dict.

    모델이 JSON 앞뒤에 설명을 붙이거나 코드펜스로 감싸는 경우가 잦아
    세 단계로 시도한다.
      1) 통째로 파싱
      2) 코드펜스 안쪽만 파싱
      3) 첫 '{' ~ 마지막 '}' 구간만 파싱
    """
    if not text:
        return {}

    raw = text.strip()

    # 1) 통째로
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    # 2) 코드펜스 안쪽
    fence = _FENCE_RE.search(raw)
    if fence:
        try:
            parsed = json.loads(fence.group(1).strip())
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    # 3) 중괄호 구간만
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

    return {}


def normalize_route_label(content: str) -> str:
    """라우터 응답을 'general' | 'supervisor' 로 정규화한다.

    JSON 으로 왔으면 route 키를 보고, 아니면 본문에서 단어를 찾는다.
    판단이 안 되면 supervisor 로 보낸다 — 조회를 놓치는 것보다
    불필요하게 조회하는 편이 낫기 때문.
    """
    parsed = extract_json_object(content)
    candidate = str(parsed.get("route", "")) if parsed else ""

    # JSON 이 아니면 본문 전체를 후보로 본다
    if not candidate:
        candidate = content or ""

    lowered = candidate.lower()

    if "general" in lowered:
        return "general"
    if "supervisor" in lowered:
        return "supervisor"

    return "supervisor"


# *************  [app 전용 — origin 에 없음]  *************
# ActionAgent — 판단부 + 턴 기반 HITL 상태기계.
# _node.action_node 가 여기 위임한다. 원 설계 설명은 ActionService 도크스트링.

# 진행 중으로 취급하는 phase (재진입 판정 기준)
ACTIVE_PHASES = {"param_check", "collecting", "awaiting_helper", "validating", "confirming"}

# 수집/검증 내부 루프 폭주 방지 (정상적으로는 MAX_COLLECT/MAX_VALIDATE 에서 먼저 걸린다)
MAX_LOOP_TURNS = 40


# ─────────────────────────────────────────────────────────────────────────
# 판단 (LLM)
#
# 사내 규칙: 판단은 전부 LLM 이 한다.
#   run_action_agent        : 명령·파라미터 판단 — react agent 가 툴을 직접 부른다
#   classify_collect_answer : 파라미터 질문에 대한 답변의 종류
#   classify_confirm        : 승인 판정
# ActionService 는 이 결과에 따라 흐름만 잡는다. 실패하면 값을 지어내지 않고
# 안전한 쪽(재질문 / 미승인)으로 떨어진다.
# 예외: ID 인식은 LLM 이 아니라 판독기 툴(params_extract_tool)의 DB 조회다.
# ─────────────────────────────────────────────────────────────────────────

class CollectAnswerOut(BaseModel):
    """파라미터 질문에 대한 답변의 종류. 명령 종류와 무관하게 고정이다."""
    kind: Literal["value", "consult", "switch", "cancel", "empty"] = Field(
        description="답변이 무엇을 하려는 것인지")


class ConfirmOut(BaseModel):
    """승인 질문에 대한 판정."""
    verdict: Literal["approve", "reject", "unclear"] = Field(
        description="실행 승인 여부")


async def run_action_agent(text: str, config=None, model_name: str = None) -> dict:
    """react agent(_agent.create_action_agent)를 돌려 명령과 파라미터를 판단한다.

    반환: {"action": str|None, "params": dict}

    에이전트는 프롬프트가 가르친 순서대로 params_extract -> param_check ->
    validate 툴을 직접 호출한다. 여기서는 그 대화에서 **마지막
    param_check_tool 호출의 인자**를 읽는다 — 그것이 에이전트가 판단한
    명령(action)과 파라미터(params)다. 툴 호출 인자는 구조화된 dict 라서
    자유 텍스트 답변을 파싱하는 것보다 훨씬 안전하다.

    param_check 호출이 아예 없으면(에이전트가 헛돌았으면) 명령 불명으로
    돌려주고, ActionService 가 사용자에게 되묻는다.
    """
    from app._agent import create_action_agent   # _agent 가 _util 을 import — 순환 회피

    def _made_tool_calls(result) -> bool:
        return any((getattr(m, "tool_calls", None) or [])
                   for m in result.get("messages", []))

    try:
        agent = create_action_agent(model_name=model_name)
        out = await agent.ainvoke({"messages": [HumanMessage(content=text)]},
                                  config=config)

        # 모델이 툴을 하나도 안 부르고 텍스트로만 답하는 턴이 가끔 있다(실측).
        # 판단 근거가 없으니 1회만 다시 돌린다 — 최종 응답의 빈 응답 재시도와
        # 같은 선례다. 두 번째도 안 부르면 불명으로 떨어져 사용자에게 묻는다.
        if not _made_tool_calls(out):
            # 재시도는 같은 요청 + 지시 한 줄. 발화에 '로그 분석' 같은 말이
            # 섞이면 모델이 자기에게 없는 도구를 텍스트로 흉내 내다 깨지는
            # 일이 있어(실측: 최종 답변이 "{" 한 글자), 가진 도구만 쓰라고
            # 못 박아 다시 시킨다.
            print("[AGENT] run_action_agent: 툴 호출 없음 -> 1회 재시도", flush=True)
            nudge = ("반드시 도구를 호출해서 처리하라. 너에게 있는 도구는 "
                     "params_extract_tool / param_check_tool / "
                     "transport_validate_tool / dest_req_validate_tool 뿐이다. "
                     "로그 분석이나 위치 조회는 네 일이 아니다 — 그런 값은 "
                     "비워 두고 param_check 까지만 하라.")
            out = await agent.ainvoke(
                {"messages": [HumanMessage(content=text),
                              HumanMessage(content=nudge)]},
                config=config)

        # 1순위: param_check_tool 호출 인자. 에이전트가 순서를 건너뛰고
        # {action}_validate_tool 부터 부르는 일이 있어(실측: qwen2.5 가
        # dest_req 발화에서 그랬다) validate 호출도 같은 근거로 읽는다 —
        # 그 툴을 골랐다는 것 자체가 에이전트의 명령 판단이다.
        action, params = None, {}
        catalog = _prompt.action_catalog()
        for msg in out.get("messages", []):
            for tc in (getattr(msg, "tool_calls", None) or []):
                name = tc.get("name") or ""
                args = tc.get("args") or {}
                if name == "param_check_tool":
                    action = (args.get("action") or "").strip() or None
                    params = {k: str(v).strip() for k, v in (args.get("params") or {}).items()
                              if v and str(v).strip()}
                elif name.endswith("_validate_tool"):
                    guessed = name[:-len("_validate_tool")]
                    if guessed in catalog:
                        action = guessed
                        params = {k: str(v).strip() for k, v in (args.get("params") or {}).items()
                                  if v and str(v).strip()}

        # 모델이 툴 호출을 정식 tool_call 이 아니라 텍스트 JSON 으로 뱉는
        # 일이 있다(실측: 최종 답변이 '{"name": "param_check_tool", ...}').
        # 직렬화만 다를 뿐 같은 판단이므로 그것도 읽는다.
        if not action:
            for msg in out.get("messages", []):
                pseudo = extract_json_object(
                    message_content_to_text(getattr(msg, "content", "")))
                if pseudo.get("name") == "param_check_tool":
                    args = pseudo.get("arguments") or pseudo.get("args") or {}
                    action = (args.get("action") or "").strip() or action
                    got = {k: str(v).strip() for k, v in (args.get("params") or {}).items()
                           if v and str(v).strip()}
                    params = got or params

        # 에이전트가 카탈로그에 없는 명령을 지어냈으면 불명 처리
        if action and action not in _prompt.action_catalog():
            print(f"[AGENT] run_action_agent: 모르는 명령 {action!r} -> 불명", flush=True)
            action = None

        # 그래도 명령이 비었으면, 같은 프롬프트로 명령 하나만 강제 구조화
        # 출력으로 다시 묻는다. 자유 툴 호출은 흔들려도 tool_choice 를 고정한
        # 구조화 출력은 안정적이다(실측). 판단 주체는 그대로 LLM 이다.
        if not action:
            names = tuple(_prompt.action_catalog().keys()) + ("unknown",)
            PickOut = create_model(
                "PickOut",
                action=(Literal[names],
                        Field(description="사용자가 요구한 명령. 모르면 unknown")))
            pick = _llm.structured_invoke(
                _llm.get_llm(model_name, temperature=0.0), PickOut,
                [SystemMessage(content=_prompt.action_agent_prompt().strip()),
                 HumanMessage(content=f"""아래 발화가 요구하는 명령 이름 하나만 답하라. 모르면 unknown.
발화: {text}""")],
                config=config)
            if pick.action != "unknown":
                action = pick.action
                print(f"[AGENT] run_action_agent: 구조화 폴백 -> {action}", flush=True)

        # 에이전트의 마지막 답변(검증 결과 요약) — 승인 질문에 재활용한다.
        # qwen 계열이 섞어 내는 <think> 블록은 사용자에게 보일 글이 아니므로 걷어낸다.
        final_text = ""
        msgs = out.get("messages", [])
        if msgs:
            final_text = message_content_to_text(getattr(msgs[-1], "content", ""))
            final_text = re.sub(r"<think>.*?</think>", "", final_text,
                                flags=re.DOTALL).strip()

        r = {"action": action, "params": params, "final_text": final_text}
        print(f"[AGENT] run_action_agent -> action={action} params={params} "
              f"summary={len(final_text)}자", flush=True)
        return r

    except Exception as e:
        # 추측하지 않는다 — 되묻는 게 안전하다
        print(f"[AGENT] run_action_agent 실패({e}) -> 불명 (사용자에게 묻는다)", flush=True)
        return {"action": None, "params": {}, "final_text": ""}


def classify_collect_answer(fieldname: str, answer, current_action: str | None,
                            question: str = None, config=None,
                            model_name: str = None) -> dict:
    """파라미터 질문에 대한 사용자 답변을 분류한다.

    반환: {"kind": value|consult|switch|cancel|empty, "text"/"note"...}
    """
    text = str(answer or "")

    # /chat/stop 등이 보내는 기계 센티널 — 모델에 물을 것도 없다
    if isinstance(answer, dict) and answer.get("aborted"):
        return {"kind": "cancel"}

    try:
        out = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            CollectAnswerOut,
            [SystemMessage(content=_prompt.action_collect_answer_prompt().strip()),
             HumanMessage(content=f"""진행 중인 명령: {current_action or '아직 정해지지 않음'}
사용자에게 물어본 것: {question or fieldname}
지금 받아야 하는 값: {fieldname}
사용자의 답변(원문): {text}""")],
            config=config)
        print(f"[AGENT] classify_collect_answer(llm) -> {out.kind}", flush=True)

        if out.kind == "cancel":
            return {"kind": "cancel"}
        if out.kind == "consult":
            return {"kind": "consult", "text": text}
        if out.kind == "switch":
            return {"kind": "switch", "text": text}
        if out.kind == "value":
            return {"kind": "value", "text": text}
        return {"kind": "empty", "note": f"답변에서 {fieldname} 값을 찾지 못했습니다."}

    except Exception as e:
        # 추측하지 않는다 — 다시 묻는 게 가장 안전하다
        print(f"[AGENT] classify_collect_answer llm 실패({e}) -> 재질문", flush=True)
        return {"kind": "empty", "note": "답변을 이해하지 못했습니다. 다시 알려주세요."}


def classify_confirm(answer, action: str = None, params: dict = None,
                     config=None, model_name: str = None) -> str:
    """승인 질문에 대한 답변 판정 -> approve | reject | unclear."""
    # /chat/stop 등이 보내는 기계 센티널 — 모델에 물을 것도 없다
    if isinstance(answer, dict):
        if answer.get("aborted"):
            return "reject"
        if "approved" in answer:
            return "approve" if answer["approved"] else "reject"

    try:
        out = _llm.structured_invoke(
            _llm.get_llm(model_name, temperature=0.0),
            ConfirmOut,
            [SystemMessage(content=_prompt.action_confirm_prompt().strip()),
             HumanMessage(content=f"""실행하려는 명령: {action or '?'} (파라미터: {params})
사용자의 답변(원문): {answer}""")],
            config=config)
        print(f"[AGENT] classify_confirm(llm) -> {out.verdict}", flush=True)
        return out.verdict

    except Exception as e:
        # 실행은 위험하다 — 판정 못 하면 절대 승인하지 않는다
        print(f"[AGENT] classify_confirm llm 실패({e}) -> unclear (미승인)", flush=True)
        return "unclear"


class ActionService:
    """ActionAgent — 턴 기반 HITL, 단일 노드 버전.

판단과 흐름의 분리 (사내 규칙)
-----------------------------
이 노드는 판단하지 않는다. 의도 추출·답변 분류·승인 판정 같은 결정은 전부
위의 판단 함수들(run_action_agent 등)이 LLM 으로 한다. 노드는
그 결과에 따라 수집 루프를 돌리고 턴을 닫는 흐름만 담당한다.
(LLM 호출이 실패하면 값을 지어내지 않고 "다시 물어본다" 는 안전한 기본값으로
 떨어진다. 예외는 ID 인식 — 이것은 LLM 판단이 아니라 판독기 툴의 DB 조회다.)

hitl_new(서브그래프 13노드)의 동작을 그대로 유지하면서 노드 함수 하나로 접었다.
부모 그래프에서는 똑같이 Supervisor 밑 member 노드 하나다.

왜 단일 노드로 접어도 되는가
---------------------------
서브그래프를 쪼갰던 원래 이유는 interrupt 재실행 격리였다(멈춘 노드만 재실행).
턴 기반은 interrupt 를 아예 쓰지 않으므로 그 이유가 사라진다. 남는 것은
"흐름이 엣지에 보인다" 는 가독성뿐인데, 이 규모의 상태 기계는 위에서 아래로
읽히는 함수 하나가 오히려 짧고 명확하다. (671줄 -> 이 파일)

턴 기반 HITL 규칙(핵심)
----------------------
- LangGraph interrupt() 를 쓰지 않는다. 사용자에게 물을 게 생기면 질문을
  action.awaiting 에 싣고 **턴을 정상 종료**한다. Supervisor 가 awaiting 을
  보고 턴을 닫는다.
- 사용자의 답변은 항상 **새 턴**으로 들어와 Router → Supervisor 를 거쳐
  여기 재진입한다. 진입부가 awaiting + 새 HumanMessage 를 보고 그 발화를
  답으로 소비한다.
  => "모든 사용자 입력은 Router 와 Supervisor 를 탄다"는 불변식이
     HITL 답변에도 그대로 성립한다.
- 상태의 근거는 오직 action 스크래치다. 어느 턴에 다시 들어와도 스크래치만
  보고 이어간다. 실제 실행(execute)은 명시적 승인 이후에만 도달한다.

needs-핸드오프
--------------
사용자의 답변에 값이 간접적으로 실려 있으면(직접 판독 불가) Supervisor 에게
상담하러 정상 종료한다. ActionAgent 가 아는 것은 세 가지 원문뿐이다:
  · 내가 사용자에게 무엇을 물었는지 (question)
  · 사용자가 무엇이라 답했는지     (answer)
  · 내가 어떤 값이 필요한지        (fill)
어느 동료가 풀 수 있는지는 Supervisor 가 로스터를 보고 정한다. 헬퍼의
자연어 답이 action.needs_result 메일박스로 돌아오면, 사용자 답변을 읽던
것과 똑같이 ID 판독기로 읽는다.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 종료 형태들 — 노드가 부모 그래프에 돌려주는 최종 dict 를 만든다
    # ─────────────────────────────────────────────────────────────────────────

    def _ask_param(self, sc: dict) -> dict:
        """⏸ HITL #1 — 부족한 파라미터를 질문하고 턴을 끝낸다 (interrupt 아님)."""
        fieldname = sc.get("pending_field")
        if fieldname == "action":
            prompt = _prompt.action_select_prompt()
        else:
            prompt = _prompt.action_catalog()[sc["action"]]["param_prompts"][fieldname]
        note = sc.pop("last_parse_error", None)
        if note:
            prompt = f"{note}\n{prompt}"

        sc["awaiting"] = {
            "type": "collect_param",
            "agent": "ActionAgent",          # 질문 주체 — needs_input 프레임에 실린다
            "action": sc.get("action"),
            "field": fieldname,
            "prompt": prompt,
            "params": sc.get("params", {}),
            "missing": sc.get("missing", []),
        }
        print(f"[ACTION ask_param] ⏸ 질문 남기고 턴 종료 field={fieldname}", flush=True)

        return {
            "action": sc,
            # 질문을 대화에도 남긴다 — Supervisor 의 '방금 물었음' 판정과
            # 다음 턴 컨텍스트의 근거가 된다.
            "messages": [AIMessage(content=prompt,
    name="ActionAgent",
    additional_kwargs={"agent_name": "ActionAgent"})],
            "next": "Supervisor",
        }


    def _ask_confirm(self, sc: dict) -> dict:
        """⏸ HITL #2 — 실행 직전 최종 승인 질문을 남기고 턴을 끝낸다.

        질문 머리말은 이번 턴 에이전트가 쓴 검증 요약을 재활용한다(LLM 서술).
        단 사용자가 승인하는 근거인 파라미터 명세는 코드가 sc["params"] 에서
        직접 박는다 — 요약이 틀려도 이 줄들이 진실이다. 에이전트가 안 돈
        턴(수집 후 재검증, 승인 재질문)은 {action}_confirm_tool 템플릿 폴백.
        """
        summary = (sc.pop("agent_summary", "") or "").strip()
        if summary:
            param_lines = "\n".join(f"- {k}: {v}" for k, v in sc["params"].items())
            guidance = f"{summary}\n{param_lines}\n이 명령을 정말 실행할까요? (승인/거절)"
        else:
            # 툴 바인딩은 네이밍 규칙 — _tool.py 의 {action}_confirm_tool
            guidance = getattr(_tool, f"{sc['action']}_confirm_tool")(sc["params"])

        sc["awaiting"] = {
            "type": "confirm",
            "agent": "ActionAgent",          # 질문 주체 — needs_input 프레임에 실린다
            "action": sc["action"],
            "prompt": guidance,
            "params": sc["params"],
            "options": ["승인", "거절"],
            "asked_at": time.time(),         # 승인 TTL 기준 시각 (CONFIRM_TTL_SEC)
        }
        print(f"[ACTION ask_confirm] ⏸ 승인 질문 남기고 턴 종료\n{guidance}", flush=True)

        return {
            "action": sc,
            "messages": [AIMessage(content=guidance,
    name="ActionAgent",
    additional_kwargs={"agent_name": "ActionAgent"})],
            "next": "Supervisor",
        }


    def _needs_exit(self, sc: dict, config) -> dict:
        """Supervisor 에게 상담하러 정상 종료 (interrupt 아님)."""
        print(f"[ACTION needs_exit] Supervisor 에 상담 -> needs={sc.get('needs')}", flush=True)
        emit(config, "agent_status",
             {"agent": "ActionAgent",
              "detail": f"{sc['needs']['fill']} 값을 사용자 답변에서 못 읽음 "
                        f"— Supervisor 에 상담"})
        return {"action": sc, "next": "Supervisor"}


    def _execute_and_finalize(self, sc: dict, config) -> dict:
        """진짜 액션 수행 — 명시적 승인 이후에만 도달하는 유일한 side-effect 지점."""
        catalog = _prompt.action_catalog()
        label = catalog[sc["action"]]["label"] if sc.get("action") in catalog else "명령"
        print(f"[ACTION execute] enter action={sc['action']} params={sc['params']}", flush=True)
        result = getattr(_tool, f"{sc['action']}_execute_tool")(sc["params"])
        sc["result"] = result
        emit(config, "tool_call", {"agent": "ActionAgent", "tool": f"{sc['action']}_execute_tool",
                                   "args": sc["params"], "result": result})

        res = sc.get("result", {})
        payload = res.get("payload", {})
        lines = [f"✅ {label} 실행 완료",
                 f"- Job ID: {res.get('job_id')}",
                 f"- 상태: {res.get('status')}"]
        lines += [f"- {k}: {v}" for k, v in payload.items()]
        print(f"[ACTION finalize] job={res.get('job_id')}", flush=True)

        return {
            "messages": [AIMessage(content="\n".join(lines),
    name="ActionAgent",
    additional_kwargs={"agent_name": "ActionAgent"})],
            "facts": {"last_action": {"action": sc["action"], "params": sc.get("params"),
                                      "result": res}},
            "action": {},          # 스크래치 리셋 — 다음 요청 오염 방지
            "next": "Supervisor",
        }


    def _abandon(self, sc: dict) -> dict:
        """취소/거절/한도초과 종료: 안내 메시지 + 스크래치 리셋. 재시도 집착 금지."""
        reason = sc.get("abandon_reason") or "요청을 종료했습니다."
        catalog = _prompt.action_catalog()
        label = catalog[sc["action"]]["label"] if sc.get("action") in catalog else "명령"
        print(f"[ACTION abandon] {reason}", flush=True)
        return {
            "messages": [AIMessage(content=f"🚫 {label} 을(를) 실행하지 않았습니다.\n- 사유: {reason}",
    name="ActionAgent",
    additional_kwargs={"agent_name": "ActionAgent"})],
            "facts": {"last_action": {"action": sc.get("action"), "aborted": True,
                                      "reason": reason}},
            "action": {},          # 스크래치 리셋
            "next": "Supervisor",
        }


    def _restart(self, text: str, config) -> dict:
        """맥락 이탈 — 진행 중 액션을 접고 사용자의 새 발화로 다시 시작한다.

        사람들은 수집 도중에도 맥락을 벗어난 새 질문을 던진다. 그 발화를 새
        HumanMessage 로 넣어 Supervisor 부터(ExtractAgent 선행 포함) 다시 태운다.
        """
        print(f"[ACTION restart] 새 질문으로 재시작: '{text}'", flush=True)
        emit(config, "agent_status",
             {"agent": "ActionAgent", "detail": "이전 작업 중단, 새 질문 처리"})
        return {
            "messages": [HumanMessage(content=text)],
            "action": {},          # 스크래치 리셋
            "next": "Supervisor",
        }


    # ─────────────────────────────────────────────────────────────────────────
    # 답변 처리 — 진입부가 소비한 사용자 답을 스크래치에 반영한다
    #   반환값이 dict 면 그대로 턴 종료(abandon/restart), None 이면 수집 루프 계속
    # ─────────────────────────────────────────────────────────────────────────

    async def _consume_param_answer(self, sc: dict, answer, config, model_name=None, question=None):
        """수집 질문(collect_param)에 대한 답변 처리.

        분기: 취소 / 맥락이탈(재시작) / 상담 / 액션선택 / 값 / 재질문
        ★ 판단은 노드가 하지 않는다 — classify_collect_answer(LLM)가 분류하고,
          노드는 그 결과에 따라 흐름만 잡는다.
        """
        fieldname = sc.get("pending_field")
        print(f"[ACTION merge_param] enter field={fieldname}", flush=True)

        r = classify_collect_answer(
            fieldname, answer, sc.get("action"),
            question=question, config=config, model_name=model_name)

        # 취소
        if r["kind"] == "cancel":
            sc["abandon_reason"] = "사용자 요청으로 명령을 취소했습니다."
            print(f"[ACTION merge_param] 취소 -> abandon", flush=True)
            return self._abandon(sc)

        # 맥락 이탈 — 진행 중 액션을 접고 새 질문으로 재시작
        if r["kind"] == "switch":
            print(f"[ACTION merge_param] 맥락 이탈 -> 재시작: '{r['text']}'", flush=True)
            return self._restart(r["text"], config)

        # 상담형 -> 답변 원문을 들고 수집 루프가 Supervisor 상담을 요청한다
        if r["kind"] == "consult":
            sc["consult_text"] = r["text"]
            print(f"[ACTION merge_param] 상담형 답변 -> '{r['text']}'", flush=True)
            return None

        # 액션 선택 (action 을 묻던 중) — 답을 에이전트에 다시 태워
        # 어느 명령인지 고르게 한다. 파라미터가 함께 실려 있으면 흡수한다.
        if r["kind"] == "value" and fieldname == "action":
            r2 = await run_action_agent(r["text"], config=config, model_name=model_name)
            if r2["action"]:
                sc["action"] = r2["action"]
                sc["agent_summary"] = r2["final_text"]
                for k, v in r2["params"].items():
                    sc["params"].setdefault(k, v.upper())
                print(f"[ACTION merge_param] 액션 선택 -> {r2['action']}", flush=True)
                return None
            sc["collect_retries"] = sc.get("collect_retries", 0) + 1
            sc["last_parse_error"] = "답변에서 명령을 알아내지 못했습니다."
            print(f"[ACTION merge_param] 액션 불명 재질문({sc['collect_retries']})", flush=True)
            return None

        # 값 후보 -> ID 판독기 툴로 실제 인식·검증
        if r["kind"] == "value":
            ids = _tool.params_extract_tool.invoke({"text": r["text"]}, config=config)
            pool = ids["carrier_ids"] if fieldname == "carrier_id" else ids["eqp_ids"]

            if pool:
                sc["params"][fieldname] = pool[0].upper()
                # 같은 답변에 실려온 다른 ID 도 기회적으로 흡수
                if ids["carrier_ids"] and not sc["params"].get("carrier_id"):
                    sc["params"]["carrier_id"] = ids["carrier_ids"][0]
                if ids["eqp_ids"] and not sc["params"].get("eqp_id"):
                    sc["params"]["eqp_id"] = ids["eqp_ids"][0]
                print(f"[ACTION merge_param] 값 인식 -> params={sc['params']}", flush=True)
                return None

            # 묻는 종류의 값은 없는데 '다른 종류'의 ID 가 실려 있다
            # ("carrier 를 물었는데 → 그건 STK101 장비에 있어").
            # 값을 간접적으로 준 것일 수 있으니 답변 원문을 들고 Supervisor 상담.
            if ids["carrier_ids"] or ids["eqp_ids"]:
                sc["consult_text"] = r["text"]
                print(f"[ACTION merge_param] 다른 종류 ID 감지 -> Supervisor 상담: '{r['text']}'", flush=True)
                return None

            # ID 스러운 토큰이 있었는데 조회에 안 걸림 -> 그 토큰을 짚어 재질문.
            # (존재하지 않는 ID 는 동료가 대신 만들어 줄 수 없다 — 상담 안 간다)
            if ids["unknown"]:
                sc["collect_retries"] = sc.get("collect_retries", 0) + 1
                sc["last_parse_error"] = (
                    f"'{', '.join(ids['unknown'])}' 은(는) 조회되지 않는 ID 입니다.")
                print(f"[ACTION merge_param] 미조회 ID 재질문(재시도 "
                      f"{sc['collect_retries']}): {ids['unknown']}", flush=True)
                return None

            # ID 판독과 무관한 답변인데 정보가 실린 것 같다 -> Supervisor 상담.
            # ("그 스토커로", "아까 장애 났던 데 말고")
            #
            # 여기서 LLM 에게 한 번 더 묻지 않는다. 이 자리에 오려면 이미
            # classify_collect_answer 가 이 답을 value(=값이 실려 있다)로
            # 분류했다는 뜻이고, 진짜 노이즈였다면 empty 로 왔을 것이다.
            # 그 판단을 신뢰한다 — 같은 걸 두 번 묻는 LLM 호출을 없앴다.
            sc["consult_text"] = r["text"]
            print(f"[ACTION merge_param] 판독 불가·정보성 답변 -> Supervisor 상담: '{r['text']}'", flush=True)
            return None

        # empty
        sc["collect_retries"] = sc.get("collect_retries", 0) + 1
        sc["last_parse_error"] = r.get("note", "")
        print(f"[ACTION merge_param] 재질문({sc['collect_retries']}): {r.get('note')}", flush=True)
        return None


    def _consume_confirm_answer(self, sc: dict, decision, config, model_name=None):
        """승인 질문(confirm)에 대한 답변 처리.

        execute 로 가는 유일한 길은 명시적 approve 뿐이다.
        반환값이 dict 면 턴 종료(실행완료/abandon), None 이면 수집 루프로 복귀
        (파라미터 정정). ★ 판정은 action_agent(LLM)가 한다.
        """
        verdict = classify_confirm(
            decision, action=sc.get("action"), params=sc.get("params"),
            config=config, model_name=model_name)
        print(f"[ACTION confirm_verdict] decision={decision!r} -> {verdict}", flush=True)
        sc["confirm"] = verdict

        # 승인 — 유일하게 execute 로 가는 길
        if verdict == "approve":
            sc["phase"] = "executing"
            return self._execute_and_finalize(sc, config)

        # 명시적 거절 — 종료
        if verdict == "reject":
            sc["phase"] = "abandoned"
            sc["abandon_reason"] = "사용자가 실행을 거절해 명령을 종료합니다."
            return self._abandon(sc)

        # 판정 불가 — 승인/거절이 아니다. 무엇을 하려는 답인지 더 본다.
        # ① 발화 의도부터 분류한다: 취소냐 / 딴 주제의 새 질문(switch)이냐.
        #    ID 가 실려 있어도 "6PDMQ283 위치 찾아줘" 같은 새 질문일 수 있다 —
        #    ID 유무로 정정/의도를 가르면(구버전) 그런 발화를 파라미터 정정으로
        #    오인해 캐리어를 갈아끼운다. 그래서 의도 분류가 정정 시도보다 먼저다.
        # ② 새 질문이 아니면 ID 후보로 '파라미터 정정' 을 시도한다 (값 보존 우선)
        # ③ 둘 다 아니면 바로 접지 말고 승인 질문을 다시 던진다 (상한 있음)
        verdict2 = classify_collect_answer(
            "승인 여부", decision, sc.get("action"),
            question="이 명령을 정말 실행할까요? (승인/거절)",
            config=config, model_name=model_name)
        kind = verdict2.get("kind")
        print(f"[ACTION confirm_verdict] 판정 불가 -> 의도 분류: {kind}", flush=True)

        if kind == "cancel":
            sc["phase"] = "abandoned"
            sc["abandon_reason"] = "사용자가 실행을 취소해 명령을 종료합니다."
            return self._abandon(sc)

        if kind == "switch":
            # 딴 주제의 새 발화. 대기 중이던 명령은 접고(실행 전이라 안전)
            # 그 발화를 새 질문으로 처리한다 — 안 그러면 사용자의 새 질문이
            # '승인 확인 실패' 안내에 먹혀 답을 못 받는다.
            return self._restart(str(decision), config)

        # 새 질문이 아니다 -> 파라미터 정정 시도
        ids = _tool.params_extract_tool.invoke({"text": str(decision)}, config=config)

        if ids["carrier_ids"] or ids["eqp_ids"]:
            # 조회되는 ID 를 줬다 -> 해당 파라미터만 교체하고 다시 검증·승인
            if ids["carrier_ids"]:
                sc["params"]["carrier_id"] = ids["carrier_ids"][0]
            if ids["eqp_ids"]:
                sc["params"]["eqp_id"] = ids["eqp_ids"][0]
            print(f"[ACTION confirm_verdict] 파라미터 정정 -> params={sc['params']}", flush=True)
            sc["phase"] = "param_check"
            return None

        if ids["unknown"]:
            # 고치려 한 건 분명한데 조회가 안 되는 ID -> 그 자리만 비우고 다시 묻는다.
            # 어느 파라미터를 고치려는지 모를 땐 마지막 필수 파라미터로 본다
            # (transport 면 목적지 eqp_id — '바꿔줘'는 대개 목적지를 가리킨다).
            target = _prompt.action_catalog()[sc["action"]]["required_params"][-1]
            sc["params"].pop(target, None)
            sc["last_parse_error"] = (
                f"'{', '.join(ids['unknown'])}' 은(는) 조회되지 않는 ID 입니다.")
            print(f"[ACTION confirm_verdict] 정정 실패 -> {target} 비우고 재수집 (unknown={ids['unknown']})", flush=True)
            sc["phase"] = "param_check"
            return None

        # 잡담/불명 — 승인 질문을 다시 던진다. 반복되면 그때 접는다.
        sc["confirm_retries"] = sc.get("confirm_retries", 0) + 1
        if sc["confirm_retries"] >= cfg.MAX_VALIDATE:
            sc["phase"] = "abandoned"
            sc["abandon_reason"] = "승인 여부를 확인하지 못해 명령을 종료합니다."
            print(f"[ACTION confirm_verdict] 재질문 상한 초과 -> abandon", flush=True)
            return self._abandon(sc)

        print(f"[ACTION confirm_verdict] 재질문 ({sc['confirm_retries']}/{cfg.MAX_VALIDATE})",
              flush=True)
        return self._ask_confirm(sc)


    # ─────────────────────────────────────────────────────────────────────────
    # 본체 — ActionAgent 전 과정을 하나의 노드에서 수행한다
    # ─────────────────────────────────────────────────────────────────────────

    async def action_node(self, state: AgentState, config) -> dict:
        """턴 기반 단일 노드.

        구성: [진입 판정] -> [수집/검증 루프] -> [턴 종료 dict 반환]
        interrupt 를 쓰지 않으므로 재실행(replay) 함정이 없다 — 위에서 아래로
        읽히는 그대로가 실행 순서다.
        """
        sc = dict(state.get("action") or {})
        msgs = state.get("messages") or []
        awaiting = sc.get("awaiting")
        model_name = state.get("model_name")   # 프론트 선택 모델 — 판단 에이전트가 쓴다

        # ── 진입 판정 (구 action_entry — 턴 기반의 관제탑) ──────────────────
        #
        # 우선순위:
        #   1) needs 메일박스 복귀      -> 수집 루프에서 회수
        #   2) awaiting + 새 사용자 발화 -> 그 발화를 '질문에 대한 답'으로 소비
        #   3) 진행 중 스크래치         -> 수집 루프부터 재개 (질문 재조립)
        #   4) 그 외                    -> 의도 추론 (신규 액션)

        if sc.get("needs"):
            print(f"[ACTION entry] needs 복귀 -> param_check", flush=True)
            emit(config, "agent_status", {"agent": "ActionAgent", "detail": "헬퍼 결과 회수"})

        elif awaiting and msgs and isinstance(msgs[-1], HumanMessage):
            # 질문을 던져놓고 기다리던 중 + 새 턴으로 들어온 HITL 답변
            # (Router/Supervisor 를 거쳐 왔다)
            answer = last_user_text(msgs)
            print(f"[ACTION entry] HITL 답변 수신({awaiting['type']}): {answer!r}", flush=True)
            emit(config, "agent_status",
                 {"agent": "ActionAgent", "detail": f"사용자 응답 수신({awaiting['type']})"})
            sc.pop("awaiting", None)

            if awaiting["type"] == "confirm":
                # 승인 TTL — 질문을 던진 지 CONFIRM_TTL_SEC 를 넘긴 답변은
                # 내용과 무관하게 만료다. 방치된 승인 질문을 뒤늦게 눌러
                # 실행되는 사고를 막는다 (판정 LLM 을 태울 것도 없이 시간 초과 —
                # 이것은 판단이 아니라 시계다).
                asked_at = awaiting.get("asked_at")
                if asked_at is not None and time.time() - asked_at > cfg.CONFIRM_TTL_SEC:
                    elapsed = time.time() - asked_at
                    print(f"[ACTION confirm_ttl] 만료 ({elapsed:.0f}s > "
                          f"{cfg.CONFIRM_TTL_SEC}s) -> abandon", flush=True)
                    sc["phase"] = "abandoned"
                    sc["abandon_reason"] = (
                        f"승인 유효시간({cfg.CONFIRM_TTL_SEC}초)이 지나 실행하지 않았습니다. "
                        "필요하면 명령을 다시 요청해 주세요.")
                    return self._abandon(sc)

                out = self._consume_confirm_answer(sc, answer, config, model_name=model_name)
            else:
                out = await self._consume_param_answer(sc, answer, config, model_name=model_name,
                                                question=awaiting.get("prompt"))

            if out is not None:
                return out           # 실행완료 / abandon / restart — 턴 종료

        elif sc.get("phase") in ACTIVE_PHASES:
            # 진행 중 스크래치 (awaiting 인데 새 발화가 없으면 질문을 다시 조립한다)
            print(f"[ACTION entry] 재진입 (phase={sc.get('phase')})", flush=True)
            emit(config, "agent_status", {"agent": "ActionAgent", "detail": "재진입(수집 재개)"})
            sc.pop("awaiting", None)

        else:
            # 신규 진입 -> 의도/파라미터 추출
            text = last_user_text(msgs)
            print(f"[ACTION entry] 신규 진입", flush=True)
            print(f"[ACTION infer_intent] enter text='{text}'", flush=True)
            emit(config, "agent_status", {"agent": "ActionAgent", "detail": "신규 진입"})
            r = await run_action_agent(text, config=config, model_name=model_name)
            sc = {
                "action": r["action"],
                "phase": "param_check",
                # 에이전트가 쓴 검증 요약 — 이번 턴이 승인 질문으로 끝나면 재활용
                "agent_summary": r["final_text"],
                "params": {k: v.upper() for k, v in r["params"].items()},
                "missing": [],
                # 필수값이 비면 원문을 들고 무조건 Supervisor 상담부터 간다.
                # 발화에 간접 표현("~있는 위치로", "로그 분석해서")이 실렸는지,
                # 누가 풀 수 있는지는 전부 Supervisor(needs_dispatch)가 판단한다.
                # 단서가 없으면 NONE 으로 반송되고 사용자에게 직접 묻는다.
                "consult_text": text,
                "collect_retries": 0,
                "validate_retries": 0,
                "hops": 0,
            }
            print(f"[ACTION infer_intent] -> action={r['action']} params={sc['params']}",
                  flush=True)

        # ── 수집/검증 루프 (구 param_check <-> validate) ────────────────────
        for _ in range(MAX_LOOP_TURNS):

            # 0) 헬퍼가 채워준 메일박스 회수 (needs-핸드오프 복귀 경로)
            #
            # 메일박스에는 헬퍼의 '자연어 답변'이 실려 온다. 구조화된 값을 기대하지
            # 않는다 — 어떤 에이전트가 어떤 형태로 답하는지 ActionAgent 는 모르기
            # 때문이다. 사용자의 답변을 읽던 것과 똑같이 ID 판독기로 읽는다.
            if sc.get("needs"):
                res = sc.pop("needs_result", None) or {}
                needs = sc.pop("needs")
                sc.pop("consult_text", None)
                sc["hops"] = sc.get("hops", 0) + 1

                value = None
                if res.get("text"):
                    ids = _tool.params_extract_tool.invoke(
                        {"text": res["text"]}, config=config)
                    pool = (ids["carrier_ids"] if needs["fill"] == "carrier_id"
                            else ids["eqp_ids"])
                    # 분석형 답변은 결론(권장값)이 마지막에 오는 경향이 있어
                    # 같은 종류가 여럿이면 마지막 것을 취한다.
                    value = pool[-1] if pool else None

                if value:
                    sc["params"][needs["fill"]] = value.upper()
                    print(f"[ACTION param_check] 헬퍼({res.get('by')}) 답변에서 판독: "
                          f"{needs['fill']}={value}", flush=True)
                else:
                    # 헬퍼가 없거나, 답변에서 값을 못 읽음 -> 사용자에게 직접(HITL 강등).
                    # 상담도 재질문 한 번으로 세어 MAX_COLLECT 안에서 수렴하게 한다.
                    sc["collect_retries"] = sc.get("collect_retries", 0) + 1
                    sc["last_parse_error"] = res.get("note") or \
                        f"{res.get('by', '동료 에이전트')} 답변에서 값을 찾지 못했습니다. 직접 입력해 주세요."
                    print(f"[ACTION param_check] 헬퍼 실패 -> HITL 강등: {sc['last_parse_error']}", flush=True)

            # 1) 필수 파라미터 충족 검사
            check = param_check_tool.invoke(
                {"action": sc.get("action") or "", "params": sc.get("params", {})},
                config=config)
            sc["params"] = check["normalized"] or sc.get("params", {})
            sc["missing"] = check["missing"]
            # 2) 상담 요청 → Supervisor 에게 (needs-핸드오프)
            #
            # 사용자의 답변(또는 최초 발화)에 필요한 값이 간접적으로 실려 있는데
            # 판독기로 직접 읽히지 않는 경우다. 풀 동료가 없으면 Supervisor 가 빈
            # 결과를 돌려보내고, 위 0) 회수 분기가 사용자에게 직접 묻는 쪽으로
            # 강등한다.
            consult = sc.get("consult_text")
            if consult and sc["missing"]:
                if sc["missing"][0] == "action":
                    # 명령 종류는 동료 워커가 조회해 줄 수 있는 값이 아니다 —
                    # 사용자의 의도다. 상담 없이 바로 사용자에게 고르게 한다.
                    # (안 막으면 헬퍼의 자연어 답에서 판독기가 꺼낸 엉뚱한 ID 가
                    #  params["action"] 에 들어간다 — 실측)
                    print(f"[ACTION param_check] action 미확정은 상담 대상 아님 -> 직접 질문", flush=True)
                    sc.pop("consult_text", None)
                elif sc.get("hops", 0) >= cfg.MAX_HOPS:
                    print(f"[ACTION param_check] MAX_HOPS({cfg.MAX_HOPS}) 초과 -> 상담 포기, 직접 질문", flush=True)
                    sc.pop("consult_text", None)
                else:
                    fill = sc["missing"][0]
                    question = _prompt.action_catalog()[sc["action"]]["param_prompts"].get(fill, "")
                    sc["needs"] = {"fill": fill, "question": question,
                                   "answer": sc.pop("consult_text"),
                                   "params": dict(sc.get("params") or {})}
                    sc["phase"] = "awaiting_helper"
                    print(f"[ACTION param_check] Supervisor 상담 요청: fill={fill} "
                          f"answer='{sc['needs']['answer']}'", flush=True)
                    return self._needs_exit(sc, config)

            # 3) 미충족 → 수집 (한 번에 한 파라미터씩 질문하고 턴 종료)
            if sc["missing"]:
                if sc.get("collect_retries", 0) >= cfg.MAX_COLLECT:
                    sc["abandon_reason"] = "필수 파라미터를 수집하지 못해 요청을 종료합니다."
                    print(f"[ACTION param_check] MAX_COLLECT 초과 -> abandon", flush=True)
                    return self._abandon(sc)
                sc["pending_field"] = sc["missing"][0]
                sc["phase"] = "collecting"
                print(f"[ACTION param_check] 미충족 -> ask '{sc['pending_field']}'", flush=True)
                return self._ask_param(sc)

            # 4) 충족 → 검증
            sc["phase"] = "validating"
            validate = getattr(_tool, f"{sc['action']}_validate_tool")
            print(f"[ACTION validate] enter action={sc['action']} params={sc['params']}", flush=True)
            v = validate.invoke({"params": sc["params"]}, config=config)
            sc["validation"] = v
            if v["ok"]:
                # 검증 통과 → 승인 질문 남기고 턴 종료
                sc["phase"] = "confirming"
                print(f"[ACTION validate] PASS -> ask_confirm", flush=True)
                return self._ask_confirm(sc)

            sc["validate_retries"] = sc.get("validate_retries", 0) + 1
            if sc["validate_retries"] >= cfg.MAX_VALIDATE:
                sc["abandon_reason"] = f"유효성 검증에 반복 실패해 요청을 종료합니다. (사유: {v['reason']})"
                print(f"[ACTION validate] MAX_VALIDATE 초과 -> abandon ({v['reason']})", flush=True)
                return self._abandon(sc)

            # 문제가 된 파라미터만 비우고 재수집 (수렴 보장 지점)
            cleared = {bad: sc["params"].pop(bad, None)
                       for bad in v.get("bad_fields", [])}
            # 검증에서 튕긴 값은 대개 발화에서 '직접 읽어서' 넣었던 값이라,
            # 그 발화를 들고 다시 Supervisor 상담을 가 봐야 캘 것이 없다.
            # 상담 후보를 비우고 사유와 함께 사용자에게 직접 묻는다.
            # 단 예외 하나 — 튕긴 값이 실은 '존재하는 캐리어 ID' 면
            # ("A 를 B 있는 위치로" 의 B 를 에이전트가 eqp_id 에 잘못 꽂은 것)
            # 간접 표현이 남아 있다는 신호이므로 상담을 유지한다.
            # 이 구분은 판단이 아니라 판독기 툴의 DB 조회다.
            mis_slotted = False
            for bad, val in cleared.items():
                if not val:
                    continue
                ids = _tool.params_extract_tool.invoke({"text": str(val)},
                                                       config=config)
                if bad != "carrier_id" and ids["carrier_ids"]:
                    mis_slotted = True
            if not mis_slotted:
                sc.pop("consult_text", None)
            sc["last_parse_error"] = f"검증 실패: {v['reason']}"
            sc["phase"] = "param_check"
            print(f"[ACTION validate] FAIL({v['code']}) -> {v.get('bad_fields')} 비우고 재수집", flush=True)
            # continue -> 루프 맨 위 param_check 부터

        # 루프 상한 — 정상 경로에서는 도달하지 않는다
        sc["abandon_reason"] = "내부 루프 한도를 초과해 요청을 종료합니다."
        print(f"[ACTION loop] MAX_LOOP_TURNS({MAX_LOOP_TURNS}) 초과 -> abandon", flush=True)
        return self._abandon(sc)


action_service = ActionService()
# *************  [app 전용 끝]  *************
