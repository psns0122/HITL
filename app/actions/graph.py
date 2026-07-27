"""ActionAgent — 단일 노드 + 내부 루프 버전 (서브그래프 버전의 대안).

★ 이 파일이 `claude/hitl-langgraph-chatbot-e67urt` 브랜치와 다른 유일한 파일이다.
  나머지 모듈은 동일하므로, 두 브랜치의 diff 가 곧 두 구현 방식의 차이다.

서브그래프 버전과의 차이
-----------------------
- 노드/엣지로 흐름을 나누지 않고, 하나의 노드 함수 안에서 while 루프로 전 과정을 돈다.
- interrupt() 가 **한 노드 안에 여러 개** 존재한다.

그래서 반드시 알아야 하는 것 (LangGraph 재개 규칙)
------------------------------------------------
1. 노드가 interrupt 되면 그 노드의 상태 변경은 **커밋되지 않는다.**
   resume 하면 노드는 **입력 상태 그대로 맨 위부터 다시 실행**된다.
2. 이미 답변된 interrupt 는 **호출 순서(positional)** 로 저장된 값을 즉시 돌려주고,
   아직 답 안 된 첫 interrupt 에서 다시 멈춘다.
3. 따라서 이 루프는 **재실행마다 완전히 동일한 순서로 interrupt 를 호출해야 한다.**
   (같은 입력 상태 + 같은 replay 값 → 같은 경로가 보장되므로 성립한다.)

이 방식의 대가 (서브그래프 버전이 피하는 것)
-------------------------------------------
- 루프 위쪽의 LLM 호출(infer_intent)이 **resume 마다 다시 실행**된다 → 토큰 이중 과금.
- 진입 로그가 resume 마다 중복 출력된다 (아래 로그에 [replay] 표시를 달아 구분한다).
- 흐름이 엣지가 아니라 코드에 있어, 순서를 알려면 함수를 읽어야 한다.

반대로 얻는 것
-------------
- 파일/노드 수가 적고, 흐름을 한 함수에서 위에서 아래로 읽을 수 있다.
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

import app.config as cfg
from app._agent import extract_intent
from app._state import AgentState
from app._util import emit, last_human_text
from app.actions import resolvers
from app.actions.registry import ACTION_REGISTRY, ACTION_SELECT_PROMPT, REFERENCE_AGENT
from app.actions.tools import param_check_tool

# 진행 중으로 취급하는 phase (재진입 판정 기준)
ACTIVE_PHASES = {"param_check", "collecting", "awaiting_helper", "validating", "confirming"}

# 루프 폭주 방지 (정상적으로는 MAX_COLLECT/MAX_VALIDATE 에서 먼저 걸린다)
MAX_LOOP_TURNS = 40


def _log(step: str, msg: str):
    print(f"[ACTION {step}] {msg}", flush=True)


def _finalize(sc: dict, config) -> dict:
    """성공 종료: 결과 메시지 + facts 기록 + 스크래치 리셋(필수)."""
    spec = ACTION_REGISTRY[sc["action"]]
    res = sc.get("result", {})
    payload = res.get("payload", {})
    lines = [f"✅ {spec.label} 실행 완료",
             f"- Job ID: {res.get('job_id')}",
             f"- 상태: {res.get('status')}"]
    lines += [f"- {k}: {v}" for k, v in payload.items()]
    _log("finalize", f"job={res.get('job_id')}")
    return {
        "messages": [AIMessage(content="\n".join(lines), name="ActionAgent")],
        "facts": {"last_action": {"action": sc["action"], "params": sc.get("params"),
                                  "result": res}},
        "action": {},          # 스크래치 리셋 — 다음 요청 오염 방지
        "next": "Supervisor",
    }


def _abandon(sc: dict, config) -> dict:
    """취소/거절/한도초과 종료: 안내 메시지 + 스크래치 리셋. 재시도 집착 금지."""
    reason = sc.get("abandon_reason") or "요청을 종료했습니다."
    label = ACTION_REGISTRY[sc["action"]].label if sc.get("action") in ACTION_REGISTRY else "명령"
    _log("abandon", reason)
    return {
        "messages": [AIMessage(content=f"🚫 {label} 을(를) 실행하지 않았습니다.\n- 사유: {reason}",
                               name="ActionAgent")],
        "facts": {"last_action": {"action": sc.get("action"), "aborted": True,
                                  "reason": reason}},
        "action": {},          # 스크래치 리셋
        "next": "Supervisor",
    }


def _needs_exit(sc: dict, config) -> dict:
    """동료 에이전트에게 양보하며 노드를 '정상 종료'(interrupt 아님).

    부모 엣지(ActionAgent -> Supervisor)를 타고 Supervisor 가 needs 를 읽어
    헬퍼로 라우팅한다. 여기서 상태가 커밋되므로, 재진입 때는 수집해 둔
    파라미터를 다시 묻지 않는다.
    """
    _log("needs_exit", f"Supervisor 에 양보 -> needs={sc.get('needs')}")
    emit(config, "agent_status",
         {"agent": "ActionAgent",
          "detail": f"{sc['needs']['agent']} 에게 {sc['needs']['fill']} 조회 위임"})
    return {"action": sc, "next": "Supervisor"}


def _restart(text: str, config) -> dict:
    """맥락 이탈 — 진행 중 액션을 접고 사용자의 새 발화로 다시 시작한다.

    사람들은 수집 도중에도 맥락을 벗어난 새 질문을 던진다.
    그 발화를 새 HumanMessage 로 넣어 Supervisor 부터(ExtractAgent 선행) 다시 태운다.
    """
    _log("restart", f"새 질문으로 재시작: '{text}'")
    emit(config, "agent_status",
         {"agent": "ActionAgent", "detail": "이전 작업 중단, 새 질문 처리"})
    return {
        "messages": [HumanMessage(content=text)],
        "action": {},          # 스크래치 리셋
        "next": "Supervisor",
    }


def _read_ids(text: str, config) -> dict:
    """ID 판독기(툴). 발화에서 캐리어/장비 ID 를 인식·검증한다.

    ★ 실무 교체 지점: 지금은 정규식(resolvers.extract_ids)이지만, 캐리어 ID 형식이
      항상 정형화돼 있지 않으므로 실제로는 DB 를 보는 툴로 바꿔야 한다.
    트레이스에 입력(text)과 결과(ids)를 남긴다.
    """
    ids = resolvers.extract_ids(text)
    emit(config, "tool_call", {"agent": "ActionAgent", "tool": "params_extract_tool",
                               "args": {"text": text}, "result": ids})
    return ids


def action_node(state: AgentState, config) -> dict:
    """ActionAgent 전 과정을 하나의 노드에서 수행한다.

    resume 될 때마다 이 함수는 **처음부터** 다시 실행된다는 점을 염두에 두고 읽을 것.
    이미 답변된 interrupt 는 즉시 값을 돌려주므로 루프는 같은 경로를 재현한다.
    """
    sc = dict(state.get("action") or {})
    reentry = sc.get("phase") in ACTIVE_PHASES
    _log("enter", f"단일노드 진입 (reentry={reentry}, phase={sc.get('phase')})")
    emit(config, "agent_status",
         {"agent": "ActionAgent", "detail": "재진입(수집 재개)" if reentry else "신규 진입"})

    # ── 1. 의도 추론 (재진입이면 건너뛴다)
    # 주의: 재진입이 아닌 경우, 이 LLM 호출은 resume 마다 반복된다(이중 과금).
    #       서브그래프 버전은 infer_intent 를 별도 노드로 분리해 이를 피한다.
    if not reentry:
        text = last_human_text(state.get("messages", []))
        _log("infer_intent", f"enter text='{text}'")
        r = extract_intent(text, config=config)
        sc = {
            "action": r.action,
            "phase": "param_check",
            "params": {k: v.upper() for k, v in r.params.items()},
            "missing": [],
            "reference": r.reference,
            "collect_retries": 0,
            "validate_retries": 0,
            "hops": 0,
        }
        _log("infer_intent", f"-> action={r.action} params={sc['params']} ref={r.reference}")

    interrupt_no = 0        # 이 노드 실행에서 몇 번째 interrupt 인지 (replay 여부 표시용)

    for turn in range(MAX_LOOP_TURNS):

        # ── 2. 파라미터 충족 검사 + 헬퍼 메일박스 회수
        if sc.get("needs"):
            res = sc.pop("needs_result", None)
            needs = sc.pop("needs")
            sc["hops"] = sc.get("hops", 0) + 1
            if res and res.get("value"):
                sc["params"][needs["fill"]] = str(res["value"]).upper()
                sc["reference"] = None
                _log("param_check", f"헬퍼({res.get('by')}) 결과 흡수: "
                                    f"{needs['fill']}={res['value']}")
            else:
                # 헬퍼 실패 -> 참조 포기, 사용자에게 직접 묻기(HITL 강등)
                sc["reference"] = None
                sc["last_parse_error"] = (res or {}).get("note") or \
                    f"{needs.get('agent')} 가 값을 찾지 못했습니다. 직접 입력해 주세요."
                _log("param_check", f"헬퍼 실패 -> HITL 강등: {sc['last_parse_error']}")

        check = param_check_tool(sc.get("action"), sc.get("params", {}))
        sc["params"] = check["normalized"] or sc.get("params", {})
        sc["missing"] = check["missing"]
        emit(config, "tool_call", {"agent": "ActionAgent", "tool": "param_check_tool",
                                   "args": {"action": sc.get("action"), "params": sc["params"]},
                                   "result": {"missing": sc["missing"]}})

        # ── 2-a. 참조형 파라미터 → 동료 에이전트 위임 (needs-핸드오프)
        ref = sc.get("reference")
        if ref and ref.get("fill") in sc["missing"]:
            if sc.get("hops", 0) >= cfg.MAX_HOPS:
                _log("param_check", f"MAX_HOPS({cfg.MAX_HOPS}) 초과 -> 참조 포기, 직접 질문")
                sc["reference"] = None
            else:
                sc["needs"] = {"agent": REFERENCE_AGENT[ref["kind"]], "fill": ref["fill"],
                               "kind": ref["kind"], "carrier_id": ref.get("carrier_id"),
                               "query": last_human_text(state.get("messages", []))}
                sc["phase"] = "awaiting_helper"
                return _needs_exit(sc, config)

        # ── 3. 미충족 → ⏸ HITL #1: 부족한 파라미터를 한 번에 하나씩 질문
        if sc["missing"]:
            if sc.get("collect_retries", 0) >= cfg.MAX_COLLECT:
                sc["abandon_reason"] = "필수 파라미터를 수집하지 못해 요청을 종료합니다."
                return _abandon(sc, config)

            fieldname = sc["missing"][0]
            sc["pending_field"] = fieldname
            sc["phase"] = "collecting"
            prompt = (ACTION_SELECT_PROMPT if fieldname == "action"
                      else ACTION_REGISTRY[sc["action"]].param_prompts[fieldname])
            note = sc.pop("last_parse_error", None)
            if note:
                prompt = f"{note}\n{prompt}"

            interrupt_no += 1
            _log("collect_param", f"⏸ interrupt #{interrupt_no} field={fieldname}")
            answer = interrupt({
                "type": "collect_param",
                "action": sc.get("action"),
                "field": fieldname,
                "prompt": prompt,
                "params": sc.get("params", {}),
                "missing": sc.get("missing", []),
            })
            # 여기 도달했다는 건 이 interrupt 에 답이 있다는 뜻(신규 or replay)
            _log("collect_param", f"answer #{interrupt_no} = {answer!r}")

            # ── 3-a. 답변 판정: 취소 / 참조(needs) / 맥락이탈(재시작) / 액션 / 값 / 재질문
            r = resolvers.resolve_param_answer(fieldname, answer, sc.get("action"))

            if r["kind"] == "cancel":
                sc["abandon_reason"] = "사용자 요청으로 명령을 취소했습니다."
                _log("merge_param", "취소 -> abandon")
                return _abandon(sc, config)

            # 맥락 이탈 -> 진행 중 액션 접고 새 질문으로 재시작 (항목 12)
            if r["kind"] == "switch":
                _log("merge_param", f"맥락 이탈 -> 재시작: '{r['text']}'")
                return _restart(r["text"], config)

            if r["kind"] == "reference":
                sc["reference"] = r["reference"]
                _log("merge_param", f"참조형 답변 -> {r['reference']}")

            elif r["kind"] == "action":
                sc["action"] = r["value"]
                _log("merge_param", f"액션 선택 -> {r['value']}")

            elif r["kind"] == "value":
                # 값 후보 -> ID 판독기 툴로 실제 인식·검증 (항목 11)
                ids = _read_ids(r["text"], config)
                pool = ids["carrier_ids"] if fieldname == "carrier_id" else ids["eqp_ids"]
                if pool:
                    sc["params"][fieldname] = pool[0].upper()
                    if ids["carrier_ids"] and not sc["params"].get("carrier_id"):
                        sc["params"]["carrier_id"] = ids["carrier_ids"][0]
                    if ids["eqp_ids"] and not sc["params"].get("eqp_id"):
                        sc["params"]["eqp_id"] = ids["eqp_ids"][0]
                    _log("merge_param", f"값 인식 -> params={sc['params']}")
                else:
                    sc["collect_retries"] = sc.get("collect_retries", 0) + 1
                    sc["last_parse_error"] = f"입력하신 값에서 유효한 {fieldname} 를 찾지 못했습니다."
                    _log("merge_param", f"값 인식 실패(재시도 {sc['collect_retries']})")

            else:  # empty
                sc["collect_retries"] = sc.get("collect_retries", 0) + 1
                sc["last_parse_error"] = r.get("note", "")
                _log("merge_param", f"재질문({sc['collect_retries']}): {r.get('note')}")

            continue        # 다시 param_check 부터

        # ── 4. 검증 (실패 시 잘못된 파라미터만 비우고 수집 루프로)
        sc["phase"] = "validating"
        spec = ACTION_REGISTRY[sc["action"]]
        _log("validate", f"enter action={sc['action']} params={sc['params']}")
        v = spec.validate(sc["params"])
        sc["validation"] = v
        emit(config, "tool_call", {"agent": "ActionAgent", "tool": f"{sc['action']}_validate_tool",
                                   "args": sc["params"], "result": {"ok": v["ok"], "code": v["code"]}})

        if not v["ok"]:
            sc["validate_retries"] = sc.get("validate_retries", 0) + 1
            if sc["validate_retries"] >= cfg.MAX_VALIDATE:
                sc["abandon_reason"] = (f"유효성 검증에 반복 실패해 요청을 종료합니다. "
                                        f"(사유: {v['reason']})")
                _log("validate", f"MAX_VALIDATE 초과 -> abandon ({v['reason']})")
                return _abandon(sc, config)
            for bad in v.get("bad_fields", []):
                sc["params"].pop(bad, None)
            sc["last_parse_error"] = f"검증 실패: {v['reason']}"
            sc["phase"] = "param_check"
            _log("validate", f"FAIL({v['code']}) -> {v.get('bad_fields')} 비우고 재수집")
            continue

        # ── 5. ⏸ HITL #2: 실행 직전 최종 승인
        sc["phase"] = "confirming"
        guidance = spec.confirm_text(sc["params"])
        interrupt_no += 1
        _log("confirm", f"⏸ interrupt #{interrupt_no} guidance=\n{guidance}")
        decision = interrupt({
            "type": "confirm",
            "action": sc["action"],
            "prompt": guidance,
            "params": sc["params"],
            "options": ["승인", "거절"],
        })
        verdict = resolvers.detect_confirm(decision)
        _log("confirm", f"decision #{interrupt_no} = {decision!r} -> {verdict}")
        sc["confirm"] = verdict

        if verdict != "approve":
            sc["phase"] = "abandoned"
            sc["abandon_reason"] = "사용자가 실행을 거절해 명령을 종료합니다."
            return _abandon(sc, config)

        # ── 6. 실행 — 승인 이후에만 도달하는 유일한 side-effect 지점
        sc["phase"] = "executing"
        _log("execute", f"enter action={sc['action']} params={sc['params']}")
        sc["result"] = spec.execute(sc["params"])
        emit(config, "tool_call", {"agent": "ActionAgent", "tool": f"{sc['action']}_execute_tool",
                                   "args": sc["params"], "result": sc["result"]})
        return _finalize(sc, config)

    # 루프 상한 — 정상 경로에서는 도달하지 않는다
    sc["abandon_reason"] = "내부 루프 한도를 초과해 요청을 종료합니다."
    _log("loop", f"MAX_LOOP_TURNS({MAX_LOOP_TURNS}) 초과 -> abandon")
    return _abandon(sc, config)


def build_action_graph():
    """부모 그래프에 등록할 ActionAgent 를 돌려준다.

    서브그래프 버전은 compile() 된 그래프를 반환하지만, 이 버전은 노드 함수 하나를
    그대로 반환한다. 부모(_builder.py)는 양쪽 모두 add_node 로 받으므로 수정 불필요.
    """
    return action_node


# 서브그래프 버전과의 호환용 (이 구현에는 내부 노드가 없다)
ACTION_SUBGRAPH_NODES: set = set()
