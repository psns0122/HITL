"""하드코딩 목업 DB.

실 DB 연결 전 HITL 동작 검증용. 형식은 사내 Logpresso createTransportJob
이력(psns0122/log)과 맞춰둠:
- carrier_id : 8자 영숫자 (예: 6PDMQ283)
- 완료 상태  : COMPLETED / CANCELED, 단 DESTTYPE=PORT + DESTMACHINETYPE2=STOCKER 조합은 부당 완료
- 에러       : (reason, description) 콤보, 1분 내 연속이면 xN

LocationAgent / LogAgent / ActionAgent 검증 툴이 모두 이 모듈의 조회 함수를
공유한다(동일 로직 중복 구현 금지).
"""
from datetime import datetime, timedelta


MOCK_DB = {
    # carrier_id -> 현재 상태
    "carriers": {
        "6PDMQ283": {"status": "IDLE",         "current_eqp": "STK101", "lot": "LOT-A123"},
        "3KWQ7712": {"status": "TRANSFERRING", "current_eqp": "PHT201", "lot": "LOT-B456"},
        "9ZXCV456": {"status": "IDLE",         "current_eqp": "STK102", "lot": "LOT-C789"},
        "7HITL001": {"status": "IDLE",         "current_eqp": "PHT201", "lot": "LOT-D012"},
        # 형식이 다른 캐리어. 구 정규식(8자 영숫자)으로는 절대 안 잡히던 형태다.
        # ID 판독기가 형식이 아니라 조회로 종류를 정한다는 걸 확인하는 회귀 방지용.
        "TESTCAR0001": {"status": "IDLE",      "current_eqp": "STK101", "lot": "LOT-E345"},
    },
    # eqp_id -> 장비 정보
    "equipment": {
        "STK101": {"type": "STOCKER", "online": True},
        "STK102": {"type": "STOCKER", "online": True},
        "PHT201": {"type": "PHOTO",   "online": True},
        "ETC301": {"type": "ETCH",    "online": True},
        "CLN501": {"type": "CLEAN",   "online": True},
        "DFF401": {"type": "DIFF",    "online": False},   # 오프라인 → 반송 목적지로 부적합
        # 형식이 다른 장비(구 정규식은 영문3+숫자3만 인정했다). 위와 같은 목적.
        "STOCKER9": {"type": "STOCKER", "online": True},
    },
    # from_eqp -> 도달 가능한 목적지 목록
    "reachable": {
        "STK101": ["PHT201", "ETC301", "STK102", "CLN501", "STOCKER9"],
        "STK102": ["STK101", "PHT201", "CLN501"],
        "PHT201": ["STK101", "STK102", "ETC301"],
        "ETC301": ["STK101", "PHT201"],
        "CLN501": ["STK101", "STK102"],
        "DFF401": [],
    },
    # dest_req(목적지요청) 시 캐리어별 허용 목적지 정책
    "dest_policy": {
        "6PDMQ283": ["STK102", "PHT201"],
        "9ZXCV456": ["STK101"],
        "7HITL001": ["STK101", "CLN501"],
        "3KWQ7712": [],   # 이송 중 → 목적지요청 불가 (validation fail 데모)
    },
    # LogAgent 목업 분석용 반송 이력 (log 레포의 콤보 규칙 재현용)
    "transport_logs": [
        {"_time": "2026-07-27 08:00:11.000", "carrier": "6PDMQ283", "state": "CREATED",
         "eqp": "ETC301", "reason": "",        "description": ""},
        {"_time": "2026-07-27 08:01:02.000", "carrier": "6PDMQ283", "state": "",
         "eqp": "ETC301", "reason": "E-STALL", "description": "OHT stall detected"},
        {"_time": "2026-07-27 08:01:31.000", "carrier": "6PDMQ283", "state": "",
         "eqp": "ETC301", "reason": "E-STALL", "description": "OHT stall detected"},
        {"_time": "2026-07-27 08:01:55.000", "carrier": "6PDMQ283", "state": "",
         "eqp": "ETC301", "reason": "E-STALL", "description": "OHT stall detected"},
        {"_time": "2026-07-27 08:03:40.000", "carrier": "6PDMQ283", "state": "",
         "eqp": "ETC301", "reason": "E-PORT",  "description": "port not ready"},
        {"_time": "2026-07-27 08:04:05.000", "carrier": "6PDMQ283", "state": "",
         "eqp": "ETC301", "reason": "E-PORT",  "description": "port not ready"},
        {"_time": "2026-07-27 08:05:00.000", "carrier": "6PDMQ283", "state": "CANCELED",
         "eqp": "ETC301", "reason": "",        "description": ""},
        {"_time": "2026-07-27 09:10:00.000", "carrier": "9ZXCV456", "state": "",
         "eqp": "PHT201", "reason": "E-ID",    "description": "carrier id read fail"},
        {"_time": "2026-07-27 09:12:00.000", "carrier": "9ZXCV456", "state": "COMPLETED",
         "eqp": "STK102", "reason": "",        "description": "success"},
    ],
}


# ── 공유 조회 함수 (LocationAgent / ActionAgent 등이 함께 사용) ──────────────

def get_carrier(carrier_id: str) -> dict | None:
    return MOCK_DB["carriers"].get((carrier_id or "").upper())


def get_carrier_location(carrier_id: str) -> str | None:
    """캐리어의 현재 장비 위치. LocationAgent 와 ActionAgent 참조 해석이 공유."""
    c = get_carrier(carrier_id)
    loc = c["current_eqp"] if c else None
    print(f"[MOCK_DB] get_carrier_location({carrier_id}) -> {loc}", flush=True)
    return loc


def get_equipment(eqp_id: str) -> dict | None:
    return MOCK_DB["equipment"].get((eqp_id or "").upper())


def is_reachable(from_eqp: str, to_eqp: str) -> bool:
    ok = (to_eqp or "").upper() in MOCK_DB["reachable"].get((from_eqp or "").upper(), [])
    print(f"[MOCK_DB] is_reachable({from_eqp} -> {to_eqp}) -> {ok}", flush=True)
    return ok


def get_dest_policy(carrier_id: str) -> list:
    return MOCK_DB["dest_policy"].get((carrier_id or "").upper(), [])


def _parse_t(s: str) -> datetime:
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def analyze_transport_logs(carrier_id: str | None = None) -> dict:
    """LogAgent 목업 분석: (reason, description) 이 1분 내 연속이면 콤보 xN.
    에러가 가장 많은 장비 = 원인 장비, 대체 목적지 = 온라인 STOCKER 중 원인 장비가 아닌 곳.
    (psns0122/log 노트북 summarize_children 로직의 축소판)
    """
    cid = (carrier_id or "").upper() or None
    rows = [r for r in MOCK_DB["transport_logs"]
            if (cid is None or r["carrier"] == cid)]
    print(f"[MOCK_DB] analyze_transport_logs(carrier={cid}) rows={len(rows)}", flush=True)

    combos: list[dict] = []
    err_count_by_eqp: dict[str, int] = {}
    for r in rows:
        reason, desc = r["reason"], r["description"]
        if not reason and not desc:
            continue
        if desc.lower() == "success":     # 성공은 생략
            continue
        t = _parse_t(r["_time"])
        key = (reason, desc)
        if combos and combos[-1]["key"] == key and t - combos[-1]["last_t"] <= timedelta(minutes=1):
            combos[-1]["count"] += 1
            combos[-1]["last_t"] = t
        else:
            combos.append({"key": key, "count": 1, "first_t": t, "last_t": t, "eqp": r["eqp"]})
        err_count_by_eqp[r["eqp"]] = err_count_by_eqp.get(r["eqp"], 0) + 1

    cause_eqp = max(err_count_by_eqp, key=err_count_by_eqp.get) if err_count_by_eqp else None
    # 권장 대체 목적지: 온라인 STOCKER 중 원인 장비/캐리어 현재 위치 제외
    current = (get_carrier(cid) or {}).get("current_eqp") if cid else None
    recommended = None
    for eqp_id, info in MOCK_DB["equipment"].items():
        if (info["type"] == "STOCKER" and info["online"]
                and eqp_id != cause_eqp and eqp_id != current):
            recommended = eqp_id
            break

    result = {
        "carrier_id": cid,
        "combos": [
            {"reason": c["key"][0], "description": c["key"][1],
             "count": c["count"], "eqp": c["eqp"],
             "first_t": c["first_t"].strftime("%H:%M:%S")}
            for c in combos
        ],
        "cause_eqp": cause_eqp,
        "recommended_dest": recommended,
    }
    print(f"[MOCK_DB] analyze -> cause_eqp={cause_eqp}, recommended_dest={recommended}, combos={len(combos)}", flush=True)
    return result


_JOB_SEQ = {"n": 0}


def next_job_id(prefix: str) -> str:
    _JOB_SEQ["n"] += 1
    return f"{prefix}-{datetime.now().strftime('%Y%m%d')}-{_JOB_SEQ['n']:04d}"
