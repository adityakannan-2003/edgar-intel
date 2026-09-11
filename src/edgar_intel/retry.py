"""Retry with exponential backoff and jitter.

Twenty lines instead of a dependency. Written out rather than imported because
the policy is something you should be able to explain: which exceptions are
retryable, how long the backoff grows, and why there is jitter.

Jitter is the part people leave out and then regret. Without it, every client
that failed at the same moment retries at the same moment, and the backoff
turns a brief blip into a synchronised stampede against a service that is
already struggling.
"""

from __future__ import annotations

import functools
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    attempts: int = 4,
    initial: float = 1.0,
    maximum: float = 20.0,
    jitter: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable:
    """Retry on `exceptions`, doubling the wait each time, capped at `maximum`.

    The last attempt re-raises rather than swallowing: a caller needs to know
    the operation failed, and a retry wrapper that returns None on exhaustion
    turns a loud failure into a confusing one.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = initial
            last: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last = exc
                    if attempt == attempts:
                        break
                    sleep(min(maximum, delay) * (1 + random.random() * jitter))
                    delay = min(maximum, delay * 2)
            assert last is not None
            raise last

        return wrapper

    return decorator
