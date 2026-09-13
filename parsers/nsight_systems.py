"""
Nsight Systems (`nsys`) timeline parser.

Distinct from `parsers/nsight.py`, which parses Nsight **Compute** (`ncu`) — a deep
per-kernel profiler whose output is a set of aggregate kernel metrics with no
wall-clock placement. Nsight **Systems** is the system-wide *timeline* profiler:
it records *when* things ran. That ordering is the whole point for r08
(prefill↔decode interference), which needs to see long prefills stalling the
decode cadence — invisible to any single aggregate snapshot.

Two entry points, mirroring the Compute parser's CSV-first / binary-stub split:

1. parse_nsys_timeline_csv(text)
   Parses the NVTX push/pop range export, i.e. what
   `nsys stats --report nvtx_pushpop_trace --format csv report.nsys-rep` emits.
   NVTX ranges are how a framework (or a user instrumenting for this analysis)
   marks scheduler steps; their *names* carry the phase, so the timeline is
   phase-tagged without guessing from kernel names (which is unreliable — both
   phases run the same kernels; see the aggregation note in parsers/nsight.py).

2. parse_nsys_rep(path)            [STUB — not yet implemented]
   Parses the binary `.nsys-rep` / exported `.sqlite` directly via the nsys
   Python/SQLite interface. Needs the full Nsight Systems install; stubbed so the
   interface is locked, matching parse_nsight_ncu_rep().

Step-range naming convention (documented contract with the trace producer):
  - The NVTX range name contains one of `prefill`, `decode`, or `mixed`
    (case-insensitive). `mixed` is checked first so "mixed_prefill_decode" reads
    as mixed. Ranges matching none of the three are ignored (not step ranges).
  - Optional token/seq counts may ride in the name as `key=value` / `key:value`,
    e.g. `prefill tokens=4096` or `mixed prefill_tokens=512 decode_seqs=24`.
    Absent → left as None; r08 treats the counts as optional corroborators.

Everything produced here is RAW (start/end/phase/counts parsed verbatim). Overlap,
stall, and the recommended chunk budget are r08's to derive, never the parser's.
"""

from __future__ import annotations

import csv
import io
import math
import re
from typing import Optional

from schema import DiagnosisInput, EngineStep, EngineStepPhase, NsysTimeline


# ---------------------------------------------------------------------------
# Header / unit handling — nsys column names drift across versions, so match
# loosely (substring, case-insensitive) rather than pinning exact strings.
# ---------------------------------------------------------------------------

def _norm(h: str) -> str:
    return h.strip().strip('"').lstrip("﻿").lower()


def _unit_factor_to_ms(header: str) -> float:
    """Multiplier converting a column's native time unit to milliseconds.

    nsys defaults to nanoseconds; the unit is usually in the header, e.g.
    "Start (ns)". Fall back to ns when no unit hint is present.
    """
    h = header.lower()
    if "(ms)" in h or "msec" in h:
        return 1.0
    if "(us)" in h or "(µs)" in h or "usec" in h or "microsec" in h:
        return 1e-3
    if "(ns)" in h or "nsec" in h or "nanosec" in h:
        return 1e-6
    # A bare "(s)"/"sec"/"seconds" — but guard against matching the "s" in other
    # words by requiring an explicit token.
    if "(s)" in h or "seconds" in h:
        return 1000.0
    return 1e-6  # nsys default is nanoseconds


def _find(headers: list[str], *needles: str) -> Optional[str]:
    """First normalised header containing every needle (all lower-case)."""
    for h in headers:
        n = _norm(h)
        if all(needle in n for needle in needles):
            return h
    return None


# ---------------------------------------------------------------------------
# Phase + payload extraction from an NVTX range name
# ---------------------------------------------------------------------------

_PREFILL_TOKENS_RE = re.compile(r"(?:prefill[_ ]?tokens|tokens|prompt[_ ]?tokens)\s*[=:]\s*(\d+)", re.I)
_DECODE_SEQS_RE = re.compile(r"(?:decode[_ ]?seqs|num[_ ]?decode[_ ]?seqs|seqs)\s*[=:]\s*(\d+)", re.I)


def classify_phase(name: str) -> Optional[EngineStepPhase]:
    """Map an NVTX range name to a step phase, or None if it is not a step range.

    "mixed" is tested first so a name carrying both words ("mixed_prefill") is not
    misread as a pure prefill step.
    """
    n = name.lower()
    if "mixed" in n:
        return "mixed"
    if "prefill" in n:
        return "prefill"
    if "decode" in n:
        return "decode"
    return None


def _extract_counts(name: str) -> tuple[Optional[int], Optional[int]]:
    """Pull optional (num_prefill_tokens, num_decode_seqs) from a range name."""
    p = _PREFILL_TOKENS_RE.search(name)
    d = _DECODE_SEQS_RE.search(name)
    return (int(p.group(1)) if p else None, int(d.group(1)) if d else None)


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_steps(text: str) -> tuple[list[EngineStep], list[str]]:
    """Parse NVTX-range CSV rows into ordered EngineSteps. Returns (steps, warnings)."""
    warnings: list[str] = []
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return [], ["nsys CSV parser: empty or invalid CSV."]

    headers = list(reader.fieldnames)
    name_col = _find(headers, "name") or _find(headers, "message") or _find(headers, "text") or _find(headers, "range")
    start_col = _find(headers, "start")
    end_col = _find(headers, "end")
    dur_col = _find(headers, "duration") or _find(headers, "dur (")

    if name_col is None:
        warnings.append(
            "nsys CSV parser: no range-name column ('Name'/'Message'/'Text') found. "
            "Export with `nsys stats --report nvtx_pushpop_trace --format csv`."
        )
        return [], warnings
    if start_col is None or (end_col is None and dur_col is None):
        warnings.append(
            "nsys CSV parser: need a 'Start' column and either 'End' or 'Duration'. "
            "Export the nvtx_pushpop_trace report."
        )
        return [], warnings

    start_f = _unit_factor_to_ms(start_col)
    end_f = _unit_factor_to_ms(end_col) if end_col else None
    dur_f = _unit_factor_to_ms(dur_col) if dur_col else None

    steps: list[EngineStep] = []
    skipped_non_step = 0
    skipped_bad = 0

    for row in reader:
        row = {k: (v or "") for k, v in row.items() if k}
        name = row.get(name_col, "").strip().strip('"')
        phase = classify_phase(name)
        if phase is None:
            skipped_non_step += 1
            continue

        try:
            start_ms = float(row[start_col].strip().strip('"').replace(",", "")) * start_f
            if end_col and row.get(end_col, "").strip():
                end_ms = float(row[end_col].strip().strip('"').replace(",", "")) * end_f
            else:
                end_ms = start_ms + float(row[dur_col].strip().strip('"').replace(",", "")) * dur_f
        except (ValueError, KeyError, TypeError):
            skipped_bad += 1
            continue

        # Prometheus-style exports can carry literal NaN/Inf and float() accepts
        # them. NaN would crash EngineStep's ge=0 validator (uncaught pydantic
        # ValidationError); Inf slips past the end<start check (inf < inf is
        # False) and poisons every downstream duration stat. Non-finite bounds
        # are unparseable rows, same as text — skip and count them.
        if not (math.isfinite(start_ms) and math.isfinite(end_ms)):
            skipped_bad += 1
            continue

        if end_ms < start_ms:
            skipped_bad += 1
            continue

        prefill_tokens, decode_seqs = _extract_counts(name)
        steps.append(
            EngineStep(
                start_ms=start_ms,
                end_ms=end_ms,
                phase=phase,
                num_prefill_tokens=prefill_tokens,
                num_decode_seqs=decode_seqs,
            )
        )

    # Order by start time: the timeline is meaningless unordered, and exports are
    # not guaranteed sorted (thread interleaving, summary merges).
    steps.sort(key=lambda s: (s.start_ms, s.end_ms))

    if not steps:
        warnings.append(
            "nsys CSV parser: no prefill/decode/mixed step ranges recognised "
            f"({skipped_non_step} non-step ranges, {skipped_bad} unparseable rows). "
            "Step ranges must be NVTX-named with 'prefill', 'decode', or 'mixed'."
        )
    elif skipped_bad:
        warnings.append(f"nsys CSV parser: skipped {skipped_bad} unparseable step rows.")

    return steps, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_nsys_timeline_csv(
    text: str,
    model_name: str = "unknown",
    gpu_type: str = "unknown",
    source_file: str = "<nsys_csv>",
) -> DiagnosisInput:
    """Parse an `nsys` NVTX push/pop CSV export into a DiagnosisInput.

    Produce it with:
        nsys stats --report nvtx_pushpop_trace --format csv \
            --output . report.nsys-rep

    Args:
        text:        Raw CSV text of the nvtx_pushpop_trace report.
        model_name:  Supplied by the caller (not in the trace).
        gpu_type:    Supplied by the caller (not in the trace).
        source_file: Provenance label.

    Returns:
        DiagnosisInput with `nsys_timeline` populated (empty steps + a warning
        when nothing recognisable was found — never raises on a benign export).
    """
    steps, warnings = _parse_steps(text)

    trace_duration_ms: Optional[float] = None
    if steps:
        trace_duration_ms = max(s.end_ms for s in steps) - min(s.start_ms for s in steps)

    return DiagnosisInput(
        model_name=model_name,
        gpu_type=gpu_type,
        inference_engine="unknown",
        nsys_timeline=NsysTimeline(steps=steps, trace_duration_ms=trace_duration_ms),
        source_files=[source_file],
        parse_warnings=warnings,
    )


def parse_nsys_timeline_csv_file(path: str, **kwargs) -> DiagnosisInput:
    """Convenience wrapper: read an nsys CSV export file and parse it."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    return parse_nsys_timeline_csv(text, source_file=path, **kwargs)


def parse_nsys_rep(path: str, **kwargs) -> DiagnosisInput:
    """
    [STUB] Parse a binary `.nsys-rep` (or exported `.sqlite`) directly.

    NOT YET IMPLEMENTED. Requires:
      - Nsight Systems installed (the `nsys` CLI / SQLite export), and
      - the NVTX step ranges already present in the capture.

    When implemented this will export to SQLite (`nsys export --type sqlite`),
    read the NVTX_EVENTS table, and build the same EngineStep list as the CSV
    path. For now, raise a clear error rather than returning garbage.
    """
    raise NotImplementedError(
        "parse_nsys_rep() is not yet implemented. Export the timeline to CSV with "
        "`nsys stats --report nvtx_pushpop_trace --format csv <report.nsys-rep>` and "
        "use parse_nsys_timeline_csv_file() instead."
    )


__all__ = [
    "classify_phase",
    "parse_nsys_timeline_csv",
    "parse_nsys_timeline_csv_file",
    "parse_nsys_rep",
]
