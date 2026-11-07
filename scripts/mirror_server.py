"""Local mirror HTTP server that behaves like a real remote site (Lab 6 crawl target).

    uv run python scripts/mirror_server.py mirror/ [--port 8765] [--latency 0.1] [--rate 100]

- network latency: every response is delayed by latency * U(0.5, 1.5) seconds
- robots.txt: disallows /whatsnew/ (the crawler must skip it)
- rate limit: more than --rate requests within one second -> 429 with Retry-After: 1
- /library/ -> 301 to /library/index.html (redirects)
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from collections import deque
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROBOTS = b"User-agent: *\nDisallow: /whatsnew/\n"


class Limiter:
    def __init__(self, rate: int) -> None:
        self.rate, self.times, self.lock = rate, deque[float](), threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        with self.lock:
            while self.times and now - self.times[0] > 1.0:
                self.times.popleft()
            if len(self.times) >= self.rate:
                return False
            self.times.append(now)
            return True


class Handler(SimpleHTTPRequestHandler):
    latency = 0.1
    limiter = Limiter(100)

    def log_message(self, format: str, *args: object) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        time.sleep(self.latency * random.uniform(0.5, 1.5))
        if self.path == "/robots.txt":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(ROBOTS)))
            self.end_headers()
            self.wfile.write(ROBOTS)
            return
        if not self.limiter.allow():
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        super().do_GET()


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: object, client_address: object) -> None:
        pass  # the crawler closes connections it cancels at max_pages: not an error here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--latency", type=float, default=0.1)
    ap.add_argument("--rate", type=int, default=100)
    a = ap.parse_args()
    Handler.latency, Handler.limiter = a.latency, Limiter(a.rate)
    server = QuietServer(("127.0.0.1", a.port), partial(Handler, directory=a.root))
    print(f"serving {a.root} on http://127.0.0.1:{a.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
