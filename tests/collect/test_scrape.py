"""Tests for collect/scrape.py — the thin HTTP acquisition layer.

A watch target is a user-supplied URL handed to urllib; the scheme is restricted
to http/https so it can never resolve a local ``file://`` (or ``ftp://`` etc.).
"""

from __future__ import annotations

import pytest

from collect.scrape import ScrapeError, http_get


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://host/x", "/etc/passwd"])
def test_non_http_scheme_is_rejected(url: str) -> None:
    with pytest.raises(ScrapeError) as exc:
        http_get(url)
    assert "scheme" in str(exc.value)
