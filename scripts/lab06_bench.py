"""Lab 6 benchmark against the local mirror (fresh server per run).

uv run python scripts/lab06_bench.py mirror/ [--pages 300] [--runs 3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8799
SEED = f"http://127.0.0.1:{PORT}/contents.html"
CONFIGS = [  # (concurrency, per_host, delay)
    (1, 1, 0.0),
    (5, 5, 0.0),
    (20, 20, 0.0),
    (20, 2, 0.0),
    (20, 4, 0.05),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mirror")
    ap.add_argument("--pages", type=int, default=300)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    Path("/tmp/lab06").mkdir(exist_ok=True)
    print(
        "| Concurrency | Per host | Delay (s) | Pages | Wall (s, median) | Pages/s | Errors | "
        "Retries | Peak RSS (MiB) |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for conc, per_host, delay in CONFIGS:
        rows = []
        for i in range(a.runs):
            server = subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "mirror_server.py"),
                    a.mirror,
                    "--port",
                    str(PORT),
                    "--latency",
                    "0.1",
                    "--rate",
                    "100",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            assert server.stdout is not None
            server.stdout.readline()  # "serving ..."
            try:
                log = f"/tmp/lab06/c{conc}_h{per_host}_d{delay}_{i}.log"
                out = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "lab06_run.py"),
                        SEED,
                        str(a.pages),
                        str(conc),
                        str(per_host),
                        str(delay),
                        log,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                rows.append(json.loads(out.strip().splitlines()[-1]))
            finally:
                server.terminate()
                server.wait()
                time.sleep(0.2)
        med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
        print(
            f"| {conc} | {per_host} | {delay} | {med['pages']:.0f} | {med['wall']:.2f} "
            f"| {med['pages'] / med['wall']:.1f} | {med['errors']:.0f} | {med['retries']:.0f} "
            f"| {med['rss']:.0f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
