"""사용자 발화/HITL 답변 해석기.

merge_param 의 4분기(취소 / 리터럴 / 참조·분석형 / 해석불능)와
infer_intent 의 규칙 기반 추출(FAKE_LLM 모드·LLM 폴백 겸용)을 담당한다.
"""
import re
from dataclasses import dataclass, field


def _log(msg: str):
    print(f"[RESOLVER] {msg}", flush=True)


# ID 후보 토큰: 영숫자 3~20자 중 숫자를 하나라도 포함한 것.
#
# 이건 '유효한 ID 인지' 판정하는 정규식이 아니다. 형식은 여기서 따지지 않는다.
# 캐리어/장비 ID 형식이 항상 정형화돼 있지 않으므로, ID 스러운 건 일단 전부
# 후보로 올려서 판독기(tools.id_lookup_tool)에 넣고, 종류와 유효성은 판독기가
# 조회해서 정한다.
#
# 주의: \b 는 한글도 \w 로 취급해 "STK102로"의 조사 앞에서 경계가 안 잡힌다
#       -> ASCII 영숫자만 배제하는 lookaround 를 쓴다.
ID_CANDIDATE_RE = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{3,20}(?![A-Z0-9])")

CANCEL_RE = re.compile(r"취소|그만|중단|됐어|됐다|안\s*할|안할|말자|하지\s*마|cancel|abort|stop", re.I)
APPROVE_RE = re.compile(r"승인|실행해|진행|허가|좋아|응\b|넵|네\b|예\b|yes|approve|ok|확인|고고|ㄱㄱ|y\b", re.I)
REJECT_RE = re.compile(r"거절|거부|아니|안\s*해|no\b|reject|n\b", re.I)

TRANSPORT_RE = re.compile(r"반송|이송|옮겨|옮기|이동|transport", re.I)
DEST_REQ_RE = re.compile(r"목적지|dest", re.I)

# 수집 도중 사용자가 "새 요청"을 시작한 걸로 볼 신호.
#  - 다른 명령(반송/목적지/명령/실행/요청) 이거나
#  - 다른 에이전트 영역(위치/상태/로그/추출) 질의
# 이게 잡히면 진행 중이던 액션을 접고 새 질문으로 다시 시작한다(맥락 이탈 대응).
CONTEXT_SWITCH_RE = re.compile(
    r"반송|이송|옮겨|옮기|이동|목적지|명령|실행\s*(해|시켜|해줘|하라)|요청\s*(해|생성)"
    r"|위치|어디|상태|서버|큐|로그|이력|에러|원인|추출|알려\s*줘|조회",
    re.I,
)

# "…있는 위치로", "…자리로" 등 위치 참조
LOCATION_REF_RE = re.compile(r"(있는\s*)?(위치|자리|곳)(로|으로|에)?")
# "로그 분석해서 원인 장비로…" 류 분석 참조
LOG_REF_RE = re.compile(r"(로그|이력|에러|장애).{0,12}(분석|원인|해결)|원인\s*(장비|해결)", re.I)


@dataclass
class IntentResult:
    action: str | None = None            # transport | dest_req | None
    params: dict = field(default_factory=dict)
    reference: dict | None = None        # {kind, carrier_id, fill}
    cancel: bool = False


def id_candidates(text: str) -> list:
    """발화에서 ID 일 '수도 있는' 토큰을 느슨하게 전부 뽑는다.

    형식으로 걸러내지 않는다 — 판정은 판독기(tools.id_lookup_tool)가 한다.
    여기서 하는 일은 "숫자가 섞인 영숫자 덩어리"를 순서대로 모아주는 것뿐이다.
    """
    up = (text or "").upper()
    out = []
    for tok in ID_CANDIDATE_RE.findall(up):
        # 숫자가 하나도 없으면 ID 후보로 보지 않는다("STK", "OK" 같은 말 배제)
        if not any(ch.isdigit() for ch in tok):
            continue
        if tok not in out:
            out.append(tok)
    return out


def extract_ids(text: str) -> dict:
    """발화 -> {carrier_ids, eqp_ids, unknown}.

    후보 추출(느슨) + 판독기 조회(권한) 두 단계로 나뉜다.
    종류 판정은 정규식이 아니라 판독기가 한다.
    """
    from app.actions.tools import id_lookup_tool   # 순환 import 회피

    return id_lookup_tool(id_candidates(text))


def detect_intent(text: str) -> str | None:
    if TRANSPORT_RE.search(text or ""):
        return "transport"
    if DEST_REQ_RE.search(text or ""):
        return "dest_req"
    return None


def detect_cancel(answer) -> bool:
    """interrupt resume 답변의 취소 의도. /chat/stop 은 {"aborted": True} 로 들어온다."""
    if isinstance(answer, dict):
        return bool(answer.get("aborted"))
    return bool(CANCEL_RE.search(str(answer or "")))


def detect_confirm_verdict(answer) -> str:
    """confirm interrupt 답변 -> approve | reject | unclear.

    detect_confirm 과 달리 '명시적 거절'과 '판정 불가'를 구분한다.
    구분이 필요한 이유: "STK103으로 바꿔줘" 처럼 정정하려는 답변을 거절로
    처리하면 그때까지 수집한 파라미터가 통째로 버려진다. 호출부가 그런
    답변을 수집 루프로 되돌릴 수 있게 unclear 를 따로 돌려준다.

    실행 여부 판단은 여전히 보수적이다 — approve 는 명시적 승인일 때만.
    """
    if isinstance(answer, dict):
        if answer.get("aborted"):
            return "reject"
        if "approved" in answer:
            return "approve" if answer["approved"] else "reject"
        answer = str(answer)
    text = str(answer or "")
    # 거절/취소 신호가 승인 신호보다 우선 ("아니 실행하지마" 같은 문장 보호)
    if REJECT_RE.search(text) or CANCEL_RE.search(text):
        return "reject"
    if APPROVE_RE.search(text):
        return "approve"
    return "unclear"


def detect_confirm(answer) -> str:
    """confirm interrupt 답변 -> approve | reject.

    판정 불가는 reject 로 떨어뜨린다(실행하지 않는 쪽이 안전).
    파라미터 정정까지 구분해야 하면 detect_confirm_verdict 를 쓴다.
    """
    verdict = detect_confirm_verdict(answer)
    if verdict == "unclear":
        _log(f"detect_confirm: 판정 불가 답변 '{answer}' -> reject 처리")
        return "reject"
    return verdict


def detect_reference(text: str) -> dict | None:
    """참조형 표현 감지. 반환: {kind, carrier_id(optional), fill:'eqp_id'}

    참조 '대상'을 집을 때는 판독기를 태우지 않고 후보 토큰을 그대로 쓴다.
    DB 에 없는 캐리어를 참조했더라도 그 사실은 헬퍼(LocationAgent 등)가
    조회에 실패하면서 알려줘야 하기 때문이다. 여기서 미리 지워버리면
    엉뚱한 캐리어를 참조 대상으로 잡는다.
    """
    t = text or ""
    if LOG_REF_RE.search(t):
        cands = id_candidates(t)
        ref = {"kind": "log_analysis", "fill": "eqp_id",
               "carrier_id": cands[0] if cands else None}
        _log(f"reference detected: {ref}")
        return ref
    if LOCATION_REF_RE.search(t):
        cands = id_candidates(t)
        if cands:
            # "X(캐리어) 있는 위치로" — 위치 참조 대상은 위치 표현에 가장 가까운 것
            m = LOCATION_REF_RE.search(t.upper())
            target = None
            for c in cands:
                pos = t.upper().rfind(c, 0, m.start() + 1)
                if pos != -1:
                    target = c   # 위치 표현 앞에 나온 마지막 후보
            target = target or cands[-1]
            ref = {"kind": "carrier_location", "fill": "eqp_id", "carrier_id": target}
            _log(f"reference detected: {ref}")
            return ref
    return None


def parse_intent(text: str) -> IntentResult:
    """규칙 기반 의도+파라미터 추출 (FAKE_LLM 모드 본선 / LLM 실패 폴백)."""
    _log(f"parse_intent: '{text}'")
    r = IntentResult()
    r.cancel = detect_cancel(text)
    r.action = detect_intent(text)
    r.reference = detect_reference(text)

    ids = extract_ids(text)
    carriers = list(ids["carrier_ids"])
    # carrier_location 참조의 대상 캐리어는 명령 대상(carrier_id) 후보에서 제외.
    # (log_analysis 는 언급된 캐리어가 분석 대상이자 명령 대상인 경우가 보통이라 유지)
    if (r.reference and r.reference["kind"] == "carrier_location"
            and r.reference.get("carrier_id") in carriers):
        carriers.remove(r.reference["carrier_id"])
    if carriers:
        r.params["carrier_id"] = carriers[0]
    if ids["eqp_ids"]:
        r.params["eqp_id"] = ids["eqp_ids"][0]
        r.reference = None   # 리터럴 eqp 가 있으면 참조 불필요
    _log(f"parse_intent -> action={r.action} params={r.params} "
         f"ref={r.reference} cancel={r.cancel}")
    return r


def resolve_param_answer(fieldname: str, answer, current_action: str | None) -> dict:
    """수집(collect_param) 답변이 무엇인지 해석한다.

    HITL 파라미터 질문에 사용자가 답한 내용을 아래 종류 중 하나로 분류해,
    ActionAgent 그래프가 다음 행동을 정하게 한다.

    판정 우선순위
      1) cancel   : 그만/취소  -> 액션 종료
      2) reference : "X 있는 위치로" 처럼 동료 조회가 필요한 답변 -> needs 핸드오프
      3) switch   : 새 명령/다른 에이전트 질의를 시작함 -> 액션 접고 새 질문으로 재시작
      4) action   : (action 을 묻는 중일 때) 반송/목적지 중 선택
      5) value    : 파라미터 값 후보. 실제 ID 인식·검증은 ID 판독기(툴)가 한다.
      6) empty    : 아무것도 못 알아들음 -> 재질문

    반환: {"kind": ..., ...}
      value 는 raw text 를 그대로 넘긴다. 무엇이 유효한 ID 인지는
      호출부(merge_param)가 ID 판독기 툴을 태워서 정한다(항목 11).
    """
    text = str(answer or "")
    _log(f"classify_collect_answer field={fieldname} answer='{text}'")

    # 1) 취소
    if detect_cancel(answer):
        return {"kind": "cancel"}

    # 2) 참조형 ("9ZXCV456 있는 위치로") — switch 보다 먼저 본다.
    #    "위치" 키워드가 switch 로 오인되지 않게 하기 위함.
    ref = detect_reference(text)
    if ref and ref["fill"] == fieldname:
        return {"kind": "reference", "reference": ref}

    # 3) 액션 자체를 묻는 중 — "반송" 은 여기선 정상 답이지 switch 가 아니다.
    if fieldname == "action":
        new_intent = detect_intent(text)
        if new_intent:
            return {"kind": "action", "value": new_intent}
        if CONTEXT_SWITCH_RE.search(text):
            return {"kind": "switch", "text": text}
        return {"kind": "empty", "note": "반송 / 목적지 중 하나로 답해주세요."}

    # 4) 맥락 이탈: 새 명령이나 다른 에이전트 질의를 시작함.
    #    단, 지금 묻는 파라미터에 답한 걸로 볼 만한 짧은 답은 값으로 본다.
    #
    #    '값이냐 새 질문이냐'도 형식이 아니라 판독기 조회로 가른다.
    #      - 묻는 필드 타입으로 조회되면        -> 값
    #      - 반대쪽 타입으로 조회되면           -> 그 ID 에 대한 새 명령일 가능성이
    #                                             크므로 맥락 이탈 판정에 맡긴다
    #        (eqp 를 묻는데 캐리어를 주는 경우: "9ZXCV456 목적지 요청해줘")
    #      - 아무 것도 조회 안 되면             -> 값으로 받아 판독기가 되묻게 한다
    ids = extract_ids(text)
    field_pool = ids["carrier_ids"] if fieldname == "carrier_id" else ids["eqp_ids"]
    other_pool = ids["eqp_ids"] if fieldname == "carrier_id" else ids["carrier_ids"]
    looks_like_bare_value = len(text.strip()) <= 20 and (
        bool(field_pool) or (not other_pool and bool(id_candidates(text))))
    if not looks_like_bare_value and CONTEXT_SWITCH_RE.search(text):
        _log(f"context switch 감지 -> 새 질문으로 재시작: '{text}'")
        return {"kind": "switch", "text": text}

    # 5) 파라미터 값 후보 (raw text 를 넘긴다 — 검증은 호출부가 툴로)
    return {"kind": "value", "text": text}


# 호환 별칭 (혹시 새 이름을 참조하는 코드가 있어도 안 깨지게)
classify_collect_answer = resolve_param_answer
