"""robots.txt per host, fetched once and cached."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler import USER_AGENT

log = logging.getLogger(__name__)
DISALLOW_ALL = ["User-agent: *", "Disallow: /"]


class RobotsCache:
    """allowed(url): obey the host's robots.txt.

    404 (no robots.txt) -> everything allowed; 401/403, 5xx or network error -> nothing
    allowed for that host (the conservative reading of the robots spec).
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str = USER_AGENT) -> None:
        self.client = client
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _parser(self, origin: str) -> RobotFileParser:
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:  # many workers, one robots.txt request per host
            if origin in self._parsers:
                return self._parsers[origin]
            parser = RobotFileParser(origin + "/robots.txt")
            try:
                async with asyncio.timeout(10):
                    resp = await self.client.get(origin + "/robots.txt")
                if resp.status_code == 200:
                    parser.parse(resp.text.splitlines())
                elif resp.status_code in (401, 403) or resp.status_code >= 500:
                    parser.parse(DISALLOW_ALL)
                else:
                    parser.parse([])  # no robots.txt: everything allowed
            except (httpx.HTTPError, TimeoutError) as e:
                log.warning("robots.txt for %s failed (%s): disallowing host", origin, e)
                parser.parse(DISALLOW_ALL)
            log.info("robots.txt %s loaded", origin)
            self._parsers[origin] = parser
            return parser

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        parser = await self._parser(f"{parts.scheme}://{parts.netloc}")
        return parser.can_fetch(self.user_agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        parts = urlsplit(url)
        parser = await self._parser(f"{parts.scheme}://{parts.netloc}")
        delay = parser.crawl_delay(self.user_agent)
        return float(delay) if delay is not None else None
