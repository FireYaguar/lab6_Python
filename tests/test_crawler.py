"""Crawler tests: httpx.MockTransport only — no network."""

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from findex.corpus import iter_documents
from findex.crawler.crawl import CrawlStats, crawl, parse_html
from findex.crawler.fetch import FetchResult, backoff_delay, fetch, retry_after
from findex.crawler.robots import RobotsCache
from findex.crawler.runner import crawl_to_jsonl
from findex.crawler.urls import canonicalize

type Handler = Callable[[httpx.Request], httpx.Response]


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def run_fetch(handler: Handler) -> tuple[FetchResult, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go() -> FetchResult:
        async with client_for(handler) as c:
            return await fetch(c, "http://x.test/a", sleep=fake_sleep)

    return asyncio.run(go()), sleeps


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://Docs.Example.COM/a/b/#frag", "http://docs.example.com/a/b"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com:443/x?b=2&a=1", "https://example.com/x?a=1&b=2"),
        ("http://example.com:8080/x/", "http://example.com:8080/x"),
        ("mailto:me@example.com", None),
        ("javascript:void(0)", None),
    ],
)
def test_canonicalize(raw: str, expected: str | None) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_relative() -> None:
    assert canonicalize("../c.html#x", "http://h.test/a/b/page.html") == "http://h.test/a/c.html"


def test_fetch_ok_records_status_bytes_time() -> None:
    res, sleeps = run_fetch(
        lambda r: httpx.Response(200, text="hello", headers={"Content-Type": "text/html"})
    )
    assert (res.status, res.bytes, res.attempts, res.text, sleeps) == (200, 5, 1, "hello", [])
    assert res.elapsed >= 0 and res.ok


def test_fetch_retries_5xx_with_backoff() -> None:
    calls = Counter[str]()

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] < 3 else 200, text="ok")

    res, sleeps = run_fetch(handler)
    assert res.status == 200 and res.attempts == 3 and len(sleeps) == 2
    assert 0.5 <= sleeps[0] <= 1.0 and 1.0 <= sleeps[1] <= 1.5  # base*2**n + jitter


def test_fetch_honors_retry_after_on_429() -> None:
    calls = Counter[str]()

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200)

    res, sleeps = run_fetch(handler)
    assert res.status == 200 and sleeps == [3.0]


def test_fetch_gives_up_after_three_retries() -> None:
    res, sleeps = run_fetch(lambda r: httpx.Response(500))
    assert res.status == 500 and res.attempts == 4 and len(sleeps) == 3 and not res.ok


@pytest.mark.parametrize("status", [400, 403, 404, 410])
def test_fetch_does_not_retry_other_4xx(status: int) -> None:
    res, sleeps = run_fetch(lambda r: httpx.Response(status))
    assert res.status == status and res.attempts == 1 and sleeps == []


def test_fetch_timeout_is_retried() -> None:
    async def slow(r: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200)

    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    async def go() -> FetchResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as c:
            return await fetch(c, "http://x.test/", timeout=0.01, sleep=fake_sleep)

    res = asyncio.run(go())
    assert res.status == 0 and "timeout" in res.error and res.attempts == 4 and len(sleeps) == 3


def test_fetch_redirect_loop_is_an_error_not_a_hang() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        nxt = "/b" if r.url.path == "/a" else "/a"
        return httpx.Response(302, headers={"Location": nxt})

    res, sleeps = run_fetch(handler)
    assert res.status == 0 and "redirect" in res.error and sleeps == []


def test_retry_after_http_date_and_backoff_cap() -> None:
    resp = httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert retry_after(resp) == 0.0  # date in the past
    assert retry_after(httpx.Response(200)) is None
    assert backoff_delay(20, base=0.5, cap=10) <= 10.5


def test_robots_cache_fetches_once_per_host() -> None:
    hits = Counter[str]()

    def handler(r: httpx.Request) -> httpx.Response:
        hits[str(r.url)] += 1
        if r.url.host == "a.test":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(404)

    async def go() -> list[bool]:
        async with client_for(handler) as c:
            robots = RobotsCache(c)
            return list(
                await asyncio.gather(
                    robots.allowed("http://a.test/public"),
                    robots.allowed("http://a.test/private/x"),
                    robots.allowed("http://b.test/anything"),
                )
            )

    assert asyncio.run(go()) == [True, False, True]
    assert hits["http://a.test/robots.txt"] == 1


def site(pages: dict[str, str], robots: str = "") -> Handler:
    """A fake website: path -> list of hrefs (as HTML)."""

    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/robots.txt":
            return httpx.Response(200, text=robots)
        if r.url.path not in pages:
            return httpx.Response(404)
        return httpx.Response(200, html=pages[r.url.path])

    return handler


def html(title: str, *links: str) -> str:
    anchors = "".join(f'<a href="{h}">x</a>' for h in links)
    return f"<html><head><title>{title}</title><script>var x;</script></head><body><p>{title} text</p>{anchors}</body></html>"


SITE = {
    "/": html(
        "Home", "/a", "/a#top", "/A/../a", "/b/", "/private/x", "https://other.test/", "mailto:x@y"
    ),
    "/a": html("A", "/", "/c?y=2&x=1"),
    "/b": html("B", "/c?x=1&y=2", "/missing"),
    "/c": html("C"),
    "/private/x": html("Secret"),
}


def collect(
    handler: Handler, max_pages: int = 200, concurrency: int = 10
) -> tuple[list[str], CrawlStats, Counter[str]]:
    requests = Counter[str]()

    def counting(r: httpx.Request) -> httpx.Response:
        requests[str(r.url)] += 1
        return handler(r)

    async def go() -> tuple[list[str], CrawlStats]:
        stats = CrawlStats()
        urls = [
            p.url
            async for p in crawl(
                ["http://site.test/"],
                transport=httpx.MockTransport(counting),
                stats=stats,
                max_pages=max_pages,
                concurrency=concurrency,
            )
        ]
        return urls, stats

    urls, stats = asyncio.run(go())
    return urls, stats, requests


def test_parse_html_title_text_links() -> None:
    title, text, links = parse_html(SITE["/"], "http://site.test/")
    assert title == "Home" and "Home text" in text and "var x" not in text
    assert "http://site.test/a" in links and "https://other.test/" in links


def test_crawl_visits_each_page_once_and_obeys_robots() -> None:
    urls, stats, requests = collect(site(SITE, robots="User-agent: *\nDisallow: /private/\n"))
    assert sorted(urls) == [
        "http://site.test/",
        "http://site.test/a",
        "http://site.test/b",
        "http://site.test/c?x=1&y=2",
    ]
    assert max(requests.values()) == 1  # canonical URLs + seen set: no duplicate fetches
    assert "http://site.test/private/x" not in requests and stats.skipped_robots == 1
    assert not any("other.test" in u for u in requests)  # domain allowlist
    assert stats.errors == 1  # /missing -> 404


def test_crawl_stops_at_max_pages() -> None:
    urls, stats, _ = collect(site(SITE), max_pages=2, concurrency=4)
    assert len(urls) == 2 and stats.pages == 2


def test_per_host_limit_is_respected() -> None:
    in_flight = Counter[str]()
    peak = Counter[str]()
    many = {"/": html("Home", *[f"/p{i}" for i in range(20)])} | {
        f"/p{i}": html(f"P{i}") for i in range(20)
    }

    async def handler(r: httpx.Request) -> httpx.Response:
        in_flight["n"] += 1
        peak["n"] = max(peak["n"], in_flight["n"])
        await asyncio.sleep(0.01)
        in_flight["n"] -= 1
        return site(many)(r)

    async def go() -> int:
        return len(
            [
                p
                async for p in crawl(
                    ["http://site.test/"],
                    transport=httpx.MockTransport(handler),
                    concurrency=10,
                    per_host=3,
                )
            ]
        )

    assert asyncio.run(go()) == 21
    assert peak["n"] <= 3


def test_crawl_to_jsonl_is_readable_by_iter_documents(tmp_path: Path) -> None:
    out = tmp_path / "crawl.jsonl"
    asyncio.run(
        crawl_to_jsonl(
            ["http://site.test/"],
            out,
            max_pages=10,
            concurrency=2,
            per_host=2,
            delay=0.0,
            stats=CrawlStats(),
            transport=httpx.MockTransport(site(SITE)),
        )
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    docs = list(iter_documents(out))
    assert len(rows) == len(docs) == 5
    assert {d.title for d in docs} == {"Home", "A", "B", "C", "Secret"}
    assert docs[0].path.startswith("http://site.test/")
