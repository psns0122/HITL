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


async def test_concurrency():
    """동시성 세마포어가 슬롯 수를 제한하는지."""
    from app.api import limits

    order = []

    async def worker(i):
        async with limits.concurrency_slot():
            order.append(("in", i))
            await asyncio.sleep(0.05)
            order.append(("out", i))

    # MAX_CONCURRENT 개까지만 동시에 in 상태여야 한다
    await asyncio.gather(*[worker(i) for i in range(cfg.MAX_CONCURRENT + 3)])

    # 임의 시점의 동시 in 개수가 상한을 넘지 않았는지 재구성해 확인
    live, peak = 0, 0
    for kind, _ in order:
        live += 1 if kind == "in" else -1
        peak = max(peak, live)
    assert peak <= cfg.MAX_CONCURRENT, f"peak={peak} > {cfg.MAX_CONCURRENT}"
    print(f"concurrency PASS (peak={peak}, max={cfg.MAX_CONCURRENT})")


if __name__ == "__main__":
    test_rate_limit()
    asyncio.run(test_concurrency())
    print("\nALL LIMIT TESTS PASS")
