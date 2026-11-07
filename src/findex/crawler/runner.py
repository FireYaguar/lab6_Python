"""Glue for `findex crawl`: stream pages into a JSON-lines file as they arrive."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import FetchResult


async def crawl_to_jsonl(
    seeds: Iterable[str],
    out_path: Path,
    *,
    max_pages: int,
    concurrency: int,
    per_host: int,
    delay: float,
    stats: CrawlStats,
    transport: httpx.AsyncBaseTransport | None = None,
    on_result: Callable[[FetchResult], None] | None = None,
) -> CrawlStats:
    """One JSON line per page, written (and flushed) the moment the page arrives — nothing
    is buffered, so memory stays flat and `iter_documents` can read the file (Lab 1).
    Indexing is NOT done here: `findex index` runs afterwards, outside the event loop."""
    with out_path.open("w", encoding="utf-8") as f:
        async for page in crawl(
            seeds,
            max_pages=max_pages,
            concurrency=concurrency,
            per_host=per_host,
            delay=delay,
            stats=stats,
            transport=transport,
            on_result=on_result,
        ):
            f.write(json.dumps(page.to_json(), ensure_ascii=False) + "\n")
            f.flush()
    return stats
