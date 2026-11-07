"""URL canonicalization: one spelling per page, so `seen` really prevents refetching."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Absolute, normalized URL, or None for non-http(s) links (mailto:, javascript:, ...).

    - scheme and host lowercased, default port dropped
    - fragment (#...) removed
    - empty path -> "/"; trailing slash removed from non-root paths
    - query parameters sorted
    """
    if base is not None:
        url = urljoin(base, url)
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if parts.port and parts.port != DEFAULT_PORTS[scheme]:
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc
