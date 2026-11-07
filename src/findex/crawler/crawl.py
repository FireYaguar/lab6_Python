"""crawl(): an async generator of pages.

frontier (asyncio.Queue) -> N worker tasks in a TaskGroup -> output queue -> `async for`.
Politeness: robots.txt, a global semaphore (all requests), a semaphore per host and a
minimum delay between two requests to the same host; domain allowlist; max_pages.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
from selectolax.lexbor import LexborHTMLParser

from findex.crawler import USER_AGENT
from findex.crawler.fetch import FetchResult, fetch
from findex.crawler.robots import RobotsCache
from findex.crawler.urls import canonicalize, host_of

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str

    def to_json(self) -> dict[str, str]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "fetched_at": self.fetched_at,
        }


@dataclass
class CrawlStats:
    """Live counters for the progress display (read by the CLI, written by workers)."""

    pages: int = 0
    errors: int = 0
    retries: int = 0
    skipped_robots: int = 0
    in_flight: int = 0
    queued: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_s(self) -> float:
        return self.pages / max(time.perf_counter() - self.started, 1e-9)


def parse_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """(title, visible text, absolute canonical links). selectolax: C parser, ~1 ms/page."""
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = title_node.text(strip=True) if title_node else ""
    links: list[str] = []
    for a in tree.css("a[href]"):
        href = a.attributes.get("href")
        if href and (url := canonicalize(href, base_url)):
            links.append(url)
    for node in tree.css("script, style, nav"):
        node.decompose()
    body = tree.body
    text = body.text(separator="\n") if body else ""
    return title, text, links


class HostPoliteness:
    """Per-host semaphore + minimum delay between request starts to the same host."""

    def __init__(self, per_host: int, delay: float) -> None:
        self.per_host, self.delay = per_host, delay
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._next: dict[str, float] = {}

    def semaphore(self, host: str) -> asyncio.Semaphore:
        return self._sems.setdefault(host, asyncio.Semaphore(self.per_host))

    async def wait_turn(self, host: str) -> None:
        if self.delay <= 0:
            return
        async with self._locks.setdefault(host, asyncio.Lock()):
            loop = asyncio.get_running_loop()
            wait = self._next.get(host, 0.0) - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)  # the sleep that does NOT block the loop
            self._next[host] = loop.time() + self.delay


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    delay: float = 0.0,
    allowed_domains: Iterable[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    on_result: Callable[[FetchResult], None] | None = None,
    timeout: float = 10.0,
) -> AsyncIterator[Page]:
    """Yield pages as they arrive, until max_pages or the frontier is exhausted."""
    stats = stats if stats is not None else CrawlStats()
    seed_list = [u for s in seeds if (u := canonicalize(s))]
    domains = set(allowed_domains or {host_of(u) for u in seed_list})
    frontier: asyncio.Queue[str] = asyncio.Queue()
    out: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    for u in seed_list:
        if u not in seen:
            seen.add(u)
            frontier.put_nowait(u)
    global_sem = asyncio.Semaphore(concurrency)
    hosts = HostPoliteness(per_host, delay)
    done = asyncio.Event()

    async with httpx.AsyncClient(  # ONE client = one connection pool for the whole crawl
        transport=transport,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        max_redirects=5,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:
        robots = RobotsCache(client)

        async def process(url: str) -> None:
            if not await robots.allowed(url):
                stats.skipped_robots += 1
                log.info("robots.txt disallows %s", url)
                return
            host = host_of(url)
            async with global_sem, hosts.semaphore(host):
                await hosts.wait_turn(host)
                stats.in_flight += 1
                try:
                    res = await fetch(client, url, timeout=timeout)
                finally:
                    stats.in_flight -= 1
            stats.retries += res.attempts - 1
            log.info(
                "%s %d %dB %.0fms try=%d %s",
                url,
                res.status,
                res.bytes,
                res.elapsed * 1000,
                res.attempts,
                res.error,
            )
            if on_result:
                on_result(res)
            if not res.ok:
                stats.errors += 1
                return
            if "html" not in res.content_type or done.is_set():
                return
            title, text, links = parse_html(res.text, res.final_url)
            for link in links:
                if link not in seen and host_of(link) in domains:
                    seen.add(link)
                    frontier.put_nowait(link)
            stats.pages += 1
            await out.put(Page(res.final_url, title, text, datetime.now(UTC).isoformat()))
            if stats.pages >= max_pages:
                done.set()

        async def worker() -> None:
            while True:
                url = await frontier.get()
                try:
                    if not done.is_set():
                        await process(url)
                finally:
                    stats.queued = frontier.qsize()
                    frontier.task_done()

        async def run() -> None:
            async with asyncio.TaskGroup() as tg:  # structured: workers live inside this block
                workers = [tg.create_task(worker()) for _ in range(concurrency)]
                drained = tg.create_task(frontier.join())
                limit = tg.create_task(done.wait())
                await asyncio.wait([drained, limit], return_when=asyncio.FIRST_COMPLETED)
                for t in [*workers, drained, limit]:
                    t.cancel()  # max_pages or empty frontier: stop everyone
            await out.put(None)

        runner = asyncio.create_task(run())  # keep a reference: tasks are weakly held
        try:
            while (page := await out.get()) is not None:
                yield page
            await runner  # re-raise a worker's unexpected exception, if any
        finally:
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass
