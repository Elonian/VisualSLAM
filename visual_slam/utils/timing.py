from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    start: float = 0.0
    end: float = 0.0

    def tic(self) -> None:
        self.start = time.perf_counter()

    def toc(self) -> float:
        self.end = time.perf_counter()
        return self.end - self.start


@contextmanager
def timed_block(stats: dict, key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        stats[key] = stats.get(key, 0.0) + dt
