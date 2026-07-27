"""사용자 발화/HITL 답변 해석기.

merge_param 의 4분기(취소 / 리터럴 / 참조·분석형 / 해석불능)와
infer_intent 의 규칙 기반 추출(FAKE_LLM 모드·LLM 폴백 겸용)을 담당한다.
"""
import re
from dataclasses import dataclass, field


def _log(msg: str):
    print(f"[RESOLVER] {msg}", flush=True)


# carrier: 8자 영숫자(숫자·영문 각 1자 이상), eqp: 영문3 + 숫자3
# 주의: \b 는 한글도 \w 로 취급해 "STK102로"의 조사 앞에서 경계가 안 잡힌다
#       -> ASCII 영숫자만 배제하는 lookaround 사용
CARRIER_RE = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9]{8}(?![A-Z0-9]))(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{8}")
EQP_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{3}\d{3}(?![A-Z0-9])")

CANCEL_RE = re.compile(r"취소|그만|중단|됐어|됐다|안\s*할|안할|말자|하지\s*마|cancel|abort|stop", re.I)
APPROVE_RE = re.compile(r"승인|실행해|진행|허가|좋아|응\b|넵|네\b|예\b|yes|approve|ok|확인|고고|ㄱㄱ|y\b", re.I)
REJECT_RE = re.compile(r"거절|거부|아니|안\s*해|no\b|reject|n\b", re.I)

TRANSPORT_RE = re.compile(r"반송|이송|옮겨|옮기|이동|transport", re.I)
DEST_REQ_RE = re.compile(r"목적지|dest", re.I)

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


def extract_ids(text: str) -> dict:
    up = (text or "").upper()
    eqp_ids = EQP_RE.findall(up)
    carrier_ids = [c for c in CARRIER_RE.findall(up) if c not in eqp_ids]
    return {"carrier_ids": carrier_ids, "eqp_ids": eqp_ids}


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


def detect_confirm(answer) -> str:
    """confirm interrupt 답변 -> approve | reject."""
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
    _log(f"detect_confirm: 판정 불가 답변 '{text}' -> reject 처리")
    return "reject"


def detect_reference(text: str) -> dict | None:
    """참조형 표현 감지. 반환: {kind, carrier_id(optional), fill:'eqp_id'}"""
    t = text or ""
    if LOG_REF_RE.search(t):
        ids = extract_ids(t)
        ref = {"kind": "log_analysis", "fill": "eqp_id",
               "carrier_id": ids["carrier_ids"][0] if ids["carrier_ids"] else None}
        _log(f"reference detected: {ref}")
        return ref
    if LOCATION_REF_RE.search(t):
        ids = extract_ids(t)
        if ids["carrier_ids"]:
            # "X(캐리어) 있는 위치로" — 위치 참조 대상 캐리어는 위치 표현에 가장 가까운 것
            m = LOCATION_REF_RE.search(t.upper())
            target = None
            for c in ids["carrier_ids"]:
                pos = t.upper().rfind(c, 0, m.start() + 1)
                if pos != -1:
                    target = c   # 위치 표현 앞에 나온 마지막 캐리어
            target = target or ids["carrier_ids"][-1]
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
    """collect_param interrupt 답변 해석 4분기.

    반환: {"kind": "cancel" | "filled" | "reference" | "flip" | "unparsed", ...}
    """
    text = str(answer or "")
    _log(f"resolve_param_answer field={fieldname} answer='{text}'")

    if detect_cancel(answer):
        return {"kind": "cancel"}

    # 의도 전환 감지 (수집 중 사용자가 다른 액션으로 피벗)
    new_intent = detect_intent(text)
    if fieldname != "action" and new_intent and current_action and new_intent != current_action:
        _log(f"intent flip: {current_action} -> {new_intent}")
        return {"kind": "flip", "action": new_intent, "ids": extract_ids(text)}

    if fieldname == "action":
        if new_intent:
            return {"kind": "filled", "field": "action", "value": new_intent,
                    "ids": extract_ids(text)}
        return {"kind": "unparsed", "note": "반송/목적지 중 하나로 답해주세요."}

    ref = detect_reference(text)
    if ref and ref["fill"] == fieldname:
        return {"kind": "reference", "reference": ref}

    ids = extract_ids(text)
    pool = ids["carrier_ids"] if fieldname == "carrier_id" else ids["eqp_ids"]
    if pool:
        return {"kind": "filled", "field": fieldname, "value": pool[0], "ids": ids}

    return {"kind": "unparsed",
            "note": f"답변에서 {fieldname} 값을 찾지 못했습니다. 형식을 확인해 주세요."}
