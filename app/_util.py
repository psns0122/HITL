"""공용 헬퍼.

LLM 응답 파싱, 메시지 훑기, SSE 트레이스 이벤트 발행처럼
여러 모듈이 함께 쓰는 잡동사니를 모아둔다.
"""
import json
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

# *************  [app 전용 import — ActionService 가 쓴다]  *************
import app.config as cfg
from app import _prompt, _tool
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


# 사내 코드에서 쓰던 이름 (동일 동작)
last_human_text = last_user_text


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
# ActionAgent 서비스 — 턴 기반 HITL 상태기계 본체.
# _node.action_node 가 여기 위임한다. 원 설계 설명은 클래스 도크스트링 참고.

# 진행 중으로 취급하는 phase (재진입 판정 기준)
ACTIVE_PHASES = {"param_check", "collecting", "awaiting_helper", "validating", "confirming"}

# 수집/검증 내부 루프 폭주 방지 (정상적으로는 MAX_COLLECT/MAX_VALIDATE 에서 먼저 걸린다)
MAX_LOOP_TURNS = 40


def _action_agent():
    """판단 에이전트 지연 로더 — _agent 가 _util 을 import 하므로 순환 회피."""
    from app._agent import action_agent
    return action_agent


class ActionService:
    """ActionAgent — 턴 기반 HITL, 단일 노드 버전.

판단과 흐름의 분리 (사내 규칙)
-----------------------------
이 노드는 판단하지 않는다. 의도 추출·답변 분류·승인 판정 같은 결정은 전부
action_node 가 가진 에이전트(_agent.action_agent)가 LLM 으로 한다. 노드는
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
        """⏸ HITL #2 — 실행 직전 최종 승인 질문을 남기고 턴을 끝낸다."""
        # 툴 바인딩은 네이밍 규칙 — _tool.py 의 {action}_confirm_tool
        guidance = getattr(_tool, f"{sc['action']}_confirm_tool")(sc["params"])

        sc["awaiting"] = {
            "type": "confirm",
            "agent": "ActionAgent",          # 질문 주체 — needs_input 프레임에 실린다
            "action": sc["action"],
            "prompt": guidance,
            "params": sc["params"],
            "options": ["승인", "거절"],
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
        label = _prompt.action_catalog()[sc["action"]]["label"]
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

    def _consume_param_answer(self, sc: dict, answer, config, model_name=None, question=None):
        """수집 질문(collect_param)에 대한 답변 처리.

        분기: 취소 / 맥락이탈(재시작) / 상담 / 액션선택 / 값 / 재질문
        ★ 판단은 노드가 하지 않는다 — action_agent(LLM)가 분류하고,
          노드는 그 결과에 따라 흐름만 잡는다.
        """
        fieldname = sc.get("pending_field")
        print(f"[ACTION merge_param] enter field={fieldname}", flush=True)

        r = _action_agent().classify_collect_answer(
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

        # 액션 선택 (action 을 묻던 중)
        if r["kind"] == "action":
            sc["action"] = r["value"]
            print(f"[ACTION merge_param] 액션 선택 -> {r['value']}", flush=True)
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
            # 이 판단도 에이전트 몫 — 실모드에선 이미 LLM 이 value(정보성)로
            # 분류한 답이므로 그 판단을 신뢰한다.
            if _action_agent().answer_seems_informative(r["text"], config=config,
                                                     model_name=model_name):
                sc["consult_text"] = r["text"]
                print(f"[ACTION merge_param] 판독 불가·정보성 답변 -> Supervisor 상담: '{r['text']}'", flush=True)
                return None

            # 진짜 노이즈 ("음...", "ㅋㅋ") -> 그냥 재질문
            sc["collect_retries"] = sc.get("collect_retries", 0) + 1
            sc["last_parse_error"] = f"입력하신 값에서 {fieldname} 를 찾지 못했습니다."
            print(f"[ACTION merge_param] 값 인식 실패(재시도 {sc['collect_retries']})", flush=True)
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
        verdict = _action_agent().classify_confirm(
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
        # ① ID 후보가 실려 있으면 '파라미터 정정' 시도 (수집한 값 보존이 최우선)
        # ② ID 가 없으면 발화 의도를 분류한다: 취소냐 / 딴 주제의 새 질문이냐
        # ③ 둘 다 아니면 바로 접지 말고 승인 질문을 다시 던진다 (상한 있음)
        ids = _tool.params_extract_tool.invoke({"text": str(decision)}, config=config)
        if not (ids["carrier_ids"] or ids["eqp_ids"] or ids["unknown"]):
            verdict2 = _action_agent().classify_collect_answer(
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

        if ids["carrier_ids"] or ids["eqp_ids"]:
            # 조회되는 ID 를 줬다 -> 해당 파라미터만 교체하고 다시 검증·승인
            if ids["carrier_ids"]:
                sc["params"]["carrier_id"] = ids["carrier_ids"][0]
            if ids["eqp_ids"]:
                sc["params"]["eqp_id"] = ids["eqp_ids"][0]
            print(f"[ACTION confirm_verdict] 파라미터 정정 -> params={sc['params']}", flush=True)
        else:
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


    # ─────────────────────────────────────────────────────────────────────────
    # 본체 — ActionAgent 전 과정을 하나의 노드에서 수행한다


action_service = ActionService()
# *************  [app 전용 끝]  *************
