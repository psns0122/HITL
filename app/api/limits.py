"""유량 제어 / 동시성 제어.

LLM 그래프 실행은 무겁고 게이트웨이 동시 요청 한도도 있어서, 아무 제한 없이
받으면 한꺼번에 몰릴 때 게이트웨이가 429/타임아웃을 낸다.

두 가지를 둔다.
  1) 동시성 : asyncio.Semaphore 로 '동시에 도는 스트림 수'를 MAX_CONCURRENT 로 제한.
              넘치면 순서대로 대기(거절 아님).
  2) 유량   : 클라이언트(IP)당 1분 요청 수를 RATE_LIMIT_PER_MIN 으로 제한.
              넘으면 429 를 즉시 반환(대기 아님).

둘 다 in-memory 라 단일 프로세스 기준이다. 멀티 워커로 뜨면 Redis 등으로 옮겨야 한다.
"""
import asyncio
import time
from collections import defaultdict, deque

import app.config as cfg

# ── 동시성 ────────────────────────────────────────────────────────────────
# 동시에 도는 그래프 스트림 수 상한. _generate 가 이 세마포어를 잡고 돈다.
_semaphore = asyncio.Semaphore(cfg.MAX_CONCURRENT)


def concurrency_slot():
    """`async with concurrency_slot():` 로 동시 실행 슬롯을 하나 잡는다."""
    return _semaphore


def concurrency_state() -> dict:
    """헬스체크용 현재 동시성 상태."""
    return {
        "max_concurrent": cfg.MAX_CONCURRENT,
        # Semaphore._value 는 남은 슬롯 수 (비공식이지만 관측용으로만 쓴다)
        "available": getattr(_semaphore, "_value", None),
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
