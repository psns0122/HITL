"""유량 제어 / 동시성 제어.

LLM 그래프 실행은 무겁고 게이트웨이 동시 요청 한도도 있어서, 아무 제한 없이
받으면 몰릴 때 게이트웨이가 429/타임아웃을 낸다. 하지만 전역으로 하나만 막으면
"내 긴 작업 때문에 남의 가벼운 질문이 기다리는" 불공정이 생긴다.

그래서 두 층으로 나눈다.

  1) 사용자별 동시성 (공정성)  : thread_id(세션) 하나가 동시에 도는 스트림 수를
       MAX_CONCURRENT_PER_USER 로 제한. 한 사람이 여러 칸을 독점하지 못하므로,
       내 작업이 길어져도 다른 사람 요청은 자기 슬롯으로 바로 들어간다.
       (FastAPI 는 async 라, LLM 응답을 await 하는 동안 다른 요청이 자유롭게 돈다.
        서로를 막는 건 이 세마포어뿐이니, 이걸 사용자별로 두면 상호 민폐가 사라진다.)

  2) 전역 동시성 (게이트웨이 보호): 그래도 전체 동시 LLM 호출 수에 넉넉한 상한
       MAX_CONCURRENT 를 둔다. 평소엔 안 걸리고, 극단적 부하일 때만 게이트웨이를 지킨다.

  3) 유량(레이트리밋)           : 클라이언트(IP)당 1분 요청 수를 RATE_LIMIT_PER_MIN 으로
       제한. 넘으면 429 즉시 거절(대기 아님) — 도배/연타를 막는다.

전부 in-memory 라 단일 프로세스 기준이다. 멀티 워커면 Redis 등으로 옮겨야 한다.
"""
import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import app.config as cfg

# ── 전역 동시성 (게이트웨이 보호) ───────────────────────────────────────────
_global_sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)

# ── 사용자별 동시성 (공정성) ────────────────────────────────────────────────
# user_id -> Semaphore. 처음 보는 사용자면 그때 만든다.
_user_sems: dict[str, asyncio.Semaphore] = {}


def _user_sem(user_id: str) -> asyncio.Semaphore:
    sem = _user_sems.get(user_id)
    if sem is None:
        sem = asyncio.Semaphore(cfg.MAX_CONCURRENT_PER_USER)
        _user_sems[user_id] = sem
    return sem


@asynccontextmanager
async def concurrency_slot(user_id: str):
    """동시 실행 슬롯을 잡는다.

    사용자별 슬롯을 먼저 잡고(공정성), 그 다음 전역 슬롯을 잡는다(게이트웨이 보호).
    한 사용자가 자기 상한을 넘겨 요청하면 '자기 자신'만 기다린다.
    """
    async with _user_sem(user_id):
        async with _global_sem:
            yield


def concurrency_state() -> dict:
    """헬스체크용 현재 동시성 상태."""
    return {
        "max_concurrent": cfg.MAX_CONCURRENT,
        "max_concurrent_per_user": cfg.MAX_CONCURRENT_PER_USER,
        # Semaphore._value 는 남은 슬롯 수 (비공식이지만 관측용으로만 쓴다)
        "global_available": getattr(_global_sem, "_value", None),
        "active_users": len(_user_sems),
    }


# ── 유량(레이트리밋) ────────────────────────────────────────────────────────
# client_id -> 최근 요청 시각(deque). 창(60초) 밖은 버린다.
_hits: dict[str, deque] = defaultdict(deque)
_WINDOW_SEC = 60.0


def rate_limit_ok(client_id: str) -> tuple[bool, int]:
    """이 클라이언트가 지금 요청해도 되는지.

    반환: (허용?, 재시도까지 남은 초)
    """
    limit = cfg.RATE_LIMIT_PER_MIN
    if limit <= 0:
        return True, 0   # 0 이하면 유량 제어 끔

    now = time.time()
    dq = _hits[client_id]

    # 창 밖(오래된) 기록 제거
    while dq and now - dq[0] > _WINDOW_SEC:
        dq.popleft()

    if len(dq) >= limit:
        retry_after = int(_WINDOW_SEC - (now - dq[0])) + 1
        return False, retry_after

    dq.append(now)
    return True, 0
