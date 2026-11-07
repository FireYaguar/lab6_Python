"""Build a LOCAL MIRROR of the Python docs as HTML (the crawl target for Lab 6).

    uv run python scripts/make_mirror.py data/python-docs mirror/

Each .rst becomes <path>.html with: <title>, an <h1>, the source text in <pre>, and links
from `.. toctree::` entries, :doc:`...` and :mod:`...` roles (so the pages form a graph like
docs.python.org). mirror/ is git-ignored, like data/.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from findex.corpus import _title_from  # pyright: ignore[reportPrivateUsage]  # same titles as the index

DOC_RE = re.compile(r":doc:`(?:[^`<]*<)?([^`>]+)>?`")
MOD_RE = re.compile(r":mod:`!?~?([\w.]+)`")


def toctree_entries(text: str) -> list[str]:
    out: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != ".. toctree::":
            continue
        for entry in lines[i + 1 :]:
            if entry and not entry.startswith((" ", "\t")):
                break
            e = entry.strip()
            if not e or e.startswith(":"):
                continue
            m = re.search(r"<([^>]+)>$", e)
            out.append(m.group(1) if m else e)
    return out


def resolve(src: Path, target: str, root: Path) -> str | None:
    target = target.removesuffix(".rst")
    if "*" in target:
        return None
    p = Path(target.lstrip("/")) if target.startswith("/") else src.parent / target
    p = Path(*[part for part in p.parts if part != "."])
    return f"/{p.as_posix()}.html" if (root / f"{p.as_posix()}.rst").is_file() else None


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = 0
    for rst in sorted(root.rglob("*.rst")):
        rel = rst.relative_to(root)
        text = rst.read_text(encoding="utf-8", errors="replace")
        title = _title_from(text, rel.stem)
        links = {resolve(rel, t, root) for t in toctree_entries(text)}
        links |= {resolve(rel, "/" + t, root) for t in DOC_RE.findall(text)}
        links |= {f"/library/{m}.html" for m in MOD_RE.findall(text)}  # some 404: realistic
        anchors = "\n".join(
            f'<li><a href="{h}">{h}</a></li>' for h in sorted(x for x in links if x)
        )
        page = (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}"
            f"</title></head><body><nav><a href='/contents.html'>Home</a></nav>"
            f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
            f"<ul class='links'>{anchors}</ul></body></html>"
        )
        dest = out / rel.with_suffix(".html")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page, encoding="utf-8")
        n += 1
    print(f"{n} pages -> {out}")


if __name__ == "__main__":
    main()
