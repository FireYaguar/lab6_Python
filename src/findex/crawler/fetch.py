"""One polite fetch: timeout, bounded retries with exponential backoff + jitter."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
type Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int  # 0 = no HTTP response (timeout, connection error, redirect loop)
    bytes: int
    elapsed: float  # seconds, all attempts included
    attempts: int
    content_type: str = ""
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def backoff_delay(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    """Exponential backoff with jitter: min(cap, base * 2**attempt) + U(0, base)."""
    return min(cap, base * 2**attempt) + random.uniform(0, base)


def retry_after(resp: httpx.Response) -> float | None:
    """Retry-After as seconds (both forms: '120' or an HTTP date)."""
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    if value.strip().isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    sleep: Sleep = asyncio.sleep,
) -> FetchResult:
    """GET url. Retries 429, 5xx and timeouts/connection errors (at most max_retries times),
    honoring Retry-After; never retries other 4xx. Never raises for HTTP/network errors —
    the result records status/error so the crawl goes on (CancelledError still propagates)."""
    t0 = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        resp: httpx.Response | None = None
        error = ""
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)
        except TimeoutError:
            error = f"timeout after {timeout:g}s"
        except httpx.TooManyRedirects:
            return FetchResult(
                url,
                url,
                0,
                0,
                time.perf_counter() - t0,
                attempt,
                error="too many redirects (loop?)",
            )
        except httpx.TransportError as e:
            error = f"{type(e).__name__}: {e}"

        if resp is not None and resp.status_code not in RETRY_STATUSES:
            return FetchResult(
                url,
                str(resp.url),
                resp.status_code,
                len(resp.content),
                time.perf_counter() - t0,
                attempt,
                resp.headers.get("Content-Type", ""),
                resp.text if resp.is_success else "",
            )
        if attempt > max_retries:
            status = resp.status_code if resp is not None else 0
            return FetchResult(
                url,
                url,
                status,
                0,
                time.perf_counter() - t0,
                attempt,
                error=error or f"HTTP {status} after {attempt} attempts",
            )
        ra = retry_after(resp) if resp is not None else None
        delay = ra if ra is not None else backoff_delay(attempt - 1)
        log.warning(
            "retry %d/%d %s after %s, sleeping %.2fs",
            attempt,
            max_retries,
            url,
            error or f"HTTP {resp.status_code if resp else 0}",
            delay,
        )
        await sleep(delay)
