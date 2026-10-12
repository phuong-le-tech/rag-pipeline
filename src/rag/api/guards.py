"""In-process abuse guards. shortcut: per-process memory, move to Redis if we run >1 replica."""

import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime


class RateLimiter:
    """Sliding one-minute window per key. ``check`` returns 0 if allowed, else seconds to wait."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self.clock = clock
        self.calls: dict[str, deque[float]] = {}

    def check(self, key: str) -> int:
        now = self.clock()
        # Forget clients idle for a minute, so the map can't grow without bound.
        for k in [k for k, w in self.calls.items() if now - w[-1] >= 60]:
            del self.calls[k]
        window = self.calls.setdefault(key, deque())
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return int(60 - (now - window[0])) + 1
        window.append(now)
        return 0


class DailyBudget:
    """Caps LLM calls per UTC day, so a free tier or a credit balance can't be drained."""

    def __init__(self, limit: int, today: Callable[[], str] | None = None) -> None:
        self.limit = limit
        self.today = today or (lambda: datetime.now(UTC).date().isoformat())
        self.day = self.today()
        self.used = 0

    def take(self) -> bool:
        if self.today() != self.day:
            self.day, self.used = self.today(), 0
        if self.used >= self.limit:
            return False
        self.used += 1
        return True
