"""ONE crawl in a fresh process -> JSON line (wall, pages, errors, retries, peak RSS).

python scripts/lab06_run.py <seed> <max_pages> <concurrency> <per_host> <delay> <log>
"""

from __future__ import annotations

import asyncio
import json
import logging
import resource
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from findex.crawler.crawl import CrawlStats
from findex.crawler.runner import crawl_to_jsonl


def main() -> None:
    seed, max_pages, conc, per_host, delay, log_path = sys.argv[1:7]
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    lg = logging.getLogger("findex.crawler")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    stats = CrawlStats()
    with TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        asyncio.run(
            crawl_to_jsonl(
                [seed],
                Path(tmp) / "c.jsonl",
                max_pages=int(max_pages),
                concurrency=int(conc),
                per_host=int(per_host),
                delay=float(delay),
                stats=stats,
            )
        )
        wall = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
        json.dumps(
            {
                "wall": wall,
                "pages": stats.pages,
                "errors": stats.errors,
                "retries": stats.retries,
                "rss": rss,
            }
        )
    )


if __name__ == "__main__":
    main()
