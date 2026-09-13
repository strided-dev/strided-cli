"""Shared Prometheus text-exposition-format line parsing.

Used by the DCGM exporter parser (dcgm-exporter serves Prometheus text, not the
JSON the dmon path reads). Kept deliberately small: it extracts
``(metric name, labels, value)`` per line and nothing more — no histogram
accumulation, no type awareness. The vLLM parser predates this and keeps its own
copy; it may adopt this helper later (non-blocking).
"""

from __future__ import annotations

import math
import re
from typing import Iterator, Optional

# A Prometheus metric line: name{labels} value [timestamp]
METRIC_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)'
)

_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_labels(label_str: str) -> dict[str, str]:
    """Parse ``{k="v", k2="v2"}`` → ``{'k': 'v', 'k2': 'v2'}``."""
    if not label_str:
        return {}
    inner = label_str.strip("{}")
    return {m.group(1): m.group(2) for m in _LABEL_RE.finditer(inner)}


def safe_float(s: str) -> Optional[float]:
    """``float(s)`` but ``None`` for NaN/inf/garbage."""
    try:
        v = float(s)
    except (ValueError, TypeError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def iter_samples(text: str) -> Iterator[tuple[str, dict[str, str], float]]:
    """Yield ``(name, labels, value)`` for each metric line.

    Comment lines (``# HELP`` / ``# TYPE``), blank lines, and lines whose value
    is non-finite or unparseable are skipped.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_RE.match(line)
        if not m:
            continue
        value = safe_float(m.group("value"))
        if value is None:
            continue
        yield m.group("name"), parse_labels(m.group("labels") or ""), value


__all__ = ["METRIC_RE", "parse_labels", "safe_float", "iter_samples"]
