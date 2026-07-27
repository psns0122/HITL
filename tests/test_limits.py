"""유량/동시성 제어 테스트.

실행: python3 tests/test_limits.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.config as cfg


def test_rate_limit():
    """분당 상한을 넘기면 거절되는지."""
    from app.api import limits

    cfg.RATE_LIMIT_PER_MIN = 3          # 테스트용으로 낮춘다
    limits._hits.clear()

    results = [limits.rate_limit_ok("1.2.3.4")[0] for _ in range(5)]
    # 앞 3개 허용, 뒤 2개 거절
    assert results == [True, True, True, False, False], results
    # 다른 클라이언트는 독립적으로 허용
    assert limits.rate_limit_ok("9.9.9.9")[0] is True
    print("rate_limit PASS")


async def test_global_concurrency():
    """전역 세마포어가 전체 동시 실행 수를 상한 이하로 유지하는지."""
    from app.api import limits

    # 여러 사용자가 동시에 몰려도 전역 상한을 넘지 않아야 한다
    order = []

    async def worker(user, i):
        async with limits.concurrency_slot(f"user-{user}"):
            order.append(("in", user, i))
            await asyncio.sleep(0.05)
            order.append(("out", user, i))

    # 사용자 여러 명이 각자 여러 번 (전역 상한보다 훨씬 많게)
    tasks = [worker(u, i) for u in range(cfg.MAX_CONCURRENT + 4) for i in range(2)]
    await asyncio.gather(*tasks)

    live, peak = 0, 0
    for kind, *_ in order:
        live += 1 if kind == "in" else -1
        peak = max(peak, live)
    assert peak <= cfg.MAX_CONCURRENT, f"peak={peak} > {cfg.MAX_CONCURRENT}"
    print(f"global concurrency PASS (peak={peak}, max={cfg.MAX_CONCURRENT})")


async def test_fairness():
    """한 사용자가 자기 상한을 넘게 요청해도, 다른 사용자는 안 기다리는지(공정성)."""
    from app.api import limits

    started_at = {}

    async def heavy(user, i, dur):
        async with limits.concurrency_slot(user):
            started_at.setdefault(user, []).append(asyncio.get_event_loop().time())
            await asyncio.sleep(dur)

    # A: 자기 상한(+2)을 넘겨 긴 작업을 잔뜩 던진다 -> A 는 자기들끼리 대기
    # B: 나중에 가벼운 요청 하나 -> A 뒤에서 안 기다리고 바로 시작돼야 한다
    t0 = asyncio.get_event_loop().time()
    a_tasks = [heavy("A", i, 0.2) for i in range(cfg.MAX_CONCURRENT_PER_USER + 3)]
    await asyncio.sleep(0.02)                 # A 가 먼저 자리 잡게
    b_task = heavy("B", 0, 0.01)
    await asyncio.gather(*a_tasks, b_task)

    b_start = started_at["B"][0] - t0
    # B 는 A 의 긴 작업들이 끝나길 기다리지 않고 0.1s 안에 시작돼야 한다
    assert b_start < 0.1, f"B 가 {b_start:.2f}s 나 기다림 (공정성 실패)"
    print(f"fairness PASS (B 시작 {b_start*1000:.0f}ms — A 뒤에서 안 기다림)")


if __name__ == "__main__":
    test_rate_limit()
    asyncio.run(test_global_concurrency())
    asyncio.run(test_fairness())
    print("\nALL LIMIT TESTS PASS")
