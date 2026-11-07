"""What NOT to do: index inside the event loop. Run with asyncio debug to see the stall.

uv run python scripts/lab06_blocking_demo.py http://127.0.0.1:8765/contents.html
"""

from __future__ import annotations

import asyncio
import logging
import sys

from findex.corpus import Document
from findex.crawler.crawl import crawl
from findex.index import build_index


async def main(seed: str) -> None:
    docs: list[Document] = []
    async for page in crawl([seed], max_pages=200, concurrency=20, per_host=20):
        docs.append(Document(len(docs), page.url, page.title, page.text))
        if len(docs) % 50 == 0:
            ix = build_index(docs)  # CPU work in a coroutine: every other task is frozen
            print(f"{len(docs)} pages -> {ix!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main(sys.argv[1]), debug=True)
