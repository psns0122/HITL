"""ID 판독기 — ExtractAgent 소유의 툴 (params_extract_tool 자리).

소유권과 예외
-------------
이 판독기는 ExtractAgent 의 툴이다. 원칙대로라면 다른 에이전트가 판독이
필요할 때도 Supervisor 를 거쳐 ExtractAgent 에게 부탁해야 한다.

단 하나 예외를 둔다: **ActionAgent 는 이 툴을 공용으로 직접 쓴다.**
HITL 수집 루프는 사용자 답변이 올 때마다 판독이 필요한데, 그때마다
Supervisor-ExtractAgent 왕복을 태우면 답변 하나에 그래프가 한 바퀴씩
돌아 HITL 이 감당 못 하게 무거워진다. 그래서 '에이전트'가 아니라
'툴' 수준에서만 공유한다 — ActionAgent 가 ExtractAgent 를 아는 게 아니라,
둘 다 이 모듈을 아는 것뿐이다.

동작 원리 (형식이 아니라 조회가 권한)
-------------------------------------
캐리어/장비 ID 형식은 항상 정형화돼 있지 않다. 그래서 두 단계로 나눈다.
  1) id_candidates : ID '스러운' 토큰을 형식 안 따지고 전부 후보로 (느슨)
  2) id_lookup_tool: 후보가 캐리어인지 장비인지 아무것도 아닌지 조회 (권한)

★★ 사내 반입 시 id_lookup_tool 본문만 사내 조회 코드로 갈아끼우면 된다.
   (여기서는 mock_db 로 땡 처리. 인터페이스는 그대로 유지)
   이 한 곳을 바꾸면 ExtractAgent·ActionAgent(infer/merge/confirm)가
   모두 따라온다.
"""
import re

from app.actions import mock_db


# ID 후보 토큰: 영숫자 3~20자 중 숫자를 하나라도 포함한 것.
#
# 이건 '유효한 ID 인지' 판정하는 정규식이 아니다. 형식은 여기서 따지지 않는다.
# ID 스러운 건 일단 전부 후보로 올려서 판독기에 넣고, 종류와 유효성은
# 판독기가 조회해서 정한다.
#
# 주의: \b 는 한글도 \w 로 취급해 "STK102로"의 조사 앞에서 경계가 안 잡힌다
#       -> ASCII 영숫자만 배제하는 lookaround 를 쓴다.
ID_CANDIDATE_RE = re.compile(r"(?<![A-Z0-9])[A-Z0-9]{3,20}(?![A-Z0-9])")


def id_candidates(text: str) -> list:
    """발화에서 ID 일 '수도 있는' 토큰을 느슨하게 전부 뽑는다.

    형식으로 걸러내지 않는다 — 판정은 id_lookup_tool 이 한다.
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


def id_lookup_tool(candidates: list) -> dict:
    """★ 후보 토큰들이 실제로 무엇인지 조회해서 종류를 정한다 (사내 교체 지점).

    Returns:
        {"carrier_ids": [...], "eqp_ids": [...], "unknown": [...]}
        unknown = 후보로는 올라왔지만 조회에 걸리지 않은 것들.
                  (사용자에게 되물을 때 근거로 쓴다)
    """
    carrier_ids, eqp_ids, unknown = [], [], []

    for cand in candidates or []:
        if mock_db.get_carrier(cand):
            carrier_ids.append(cand)
        elif mock_db.get_equipment(cand):
            eqp_ids.append(cand)
        else:
            unknown.append(cand)

    print(f"[ID_READER] 후보={list(candidates or [])} -> carrier={carrier_ids} "
          f"eqp={eqp_ids} unknown={unknown}", flush=True)
    return {"carrier_ids": carrier_ids, "eqp_ids": eqp_ids, "unknown": unknown}


def extract_ids(text: str) -> dict:
    """발화 -> {carrier_ids, eqp_ids, unknown}. (후보 추출 + 조회 판정)"""
    return id_lookup_tool(id_candidates(text))
