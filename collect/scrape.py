"""HTTP scraping for live metrics endpoints (vLLM /metrics, dcgm-exporter).

Pure stdlib ``urllib`` — the live path is I/O-bound, so there is no reason to add
a dependency. This is the thin acquisition layer; parsing belongs to ``parsers/``.
A failed scrape raises ``ScrapeError`` so the caller (a ``Source``) can decide a
single tick degrades rather than the whole loop dying.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from urllib.parse import urlparse

_ALLOWED_SCHEMES = ("http", "https")


class ScrapeError(RuntimeError):
    """A metrics endpoint could not be scraped (network error, timeout, bad status)."""


def http_get(url: str, timeout: float = 10.0) -> str:
    """GET a text endpoint and return its decoded body.

    Raises ``ScrapeError`` on any failure (connection refused, timeout, non-200,
    decode error) so the live loop can survive a transient server hiccup. The URL
    scheme is restricted to http/https so a watch target can never reach
    ``file://`` / ``ftp://`` and friends.
    """
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ScrapeError(
            f"unsupported URL scheme {scheme or '(none)'!r} for {url} "
            f"(only {'/'.join(_ALLOWED_SCHEMES)})"
        )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            # urlopen already raises HTTPError for 4xx/5xx; this guards the rare
            # non-2xx success that still returns a body we should not trust.
            if not (200 <= status < 300):
                raise ScrapeError(f"{url} returned HTTP {status}")
            return resp.read().decode("utf-8")
    except ScrapeError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ScrapeError(f"failed to scrape {url}: {type(exc).__name__}: {exc}") from exc


__all__ = ["ScrapeError", "http_get"]
