"""Tests for collect/sources.py — replay ordering and HTTP-source robustness.

A source must never raise out of poll(): a scrape failure records last_error and
returns None so the loop survives. Replay sources hand out captured files in
order and flip `exhausted` when dry.
"""

from __future__ import annotations

from pathlib import Path

import collect.sources as sources_mod
from collect.scrape import ScrapeError
from collect.sources import ReplaySource, VllmSource
from parsers.vllm import parse_vllm_metrics_file

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "collect" / "replay"
_VLLM_FILES = sorted((_REPLAY / "vllm").glob("*.prom"))


def test_replay_yields_files_in_order_then_none() -> None:
    src = ReplaySource(_VLLM_FILES, parse_vllm_metrics_file)
    seen = [src.poll() for _ in _VLLM_FILES]
    assert all(dx is not None for dx in seen)
    assert src.poll() is None
    assert src.exhausted


def test_replay_provenance_tracks_each_file() -> None:
    src = ReplaySource(_VLLM_FILES, parse_vllm_metrics_file)
    first = src.poll()
    assert str(_VLLM_FILES[0]) in first.source_files


def test_empty_replay_is_exhausted() -> None:
    src = ReplaySource([], parse_vllm_metrics_file)
    assert src.exhausted
    assert src.poll() is None


def test_http_source_scrape_failure_returns_none(monkeypatch) -> None:
    def boom(url, timeout):
        raise ScrapeError("connection refused")

    monkeypatch.setattr(sources_mod, "http_get", boom)
    src = VllmSource("http://localhost:9/metrics")
    assert src.poll() is None
    assert src.last_error is not None
    assert "connection refused" in src.last_error


def test_http_source_parses_on_success(monkeypatch) -> None:
    text = (_REPLAY / "vllm" / "00.prom").read_text()
    monkeypatch.setattr(sources_mod, "http_get", lambda url, timeout: text)
    src = VllmSource("http://x/metrics", "m", "H100")
    dx = src.poll()
    assert dx is not None
    assert dx.vllm_serving is not None
    assert src.last_error is None
