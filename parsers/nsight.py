"""
Nsight Compute parser.

Two entry points:

1. parse_nsight_csv(text)
   Parses the CSV export from Nsight Compute UI or ncu --csv.
   This is the format publicly available papers (TritonForge, Mind the Memory Gap,
   etc.) actually publish, and what you can generate from academic reproductions.
   CSV is the primary target for the 90-day sprint.

2. parse_nsight_ncu_rep(path)      [STUB — not yet implemented]
   Parses the binary .ncu-rep format via the nvidia-nsight-compute Python
   bindings. This requires the full Nsight Compute installation and a CUDA GPU.
   Stub is here so the interface is locked; implementation follows once we
   have real .ncu-rep files from customer interviews.

Key Nsight Compute metrics we extract:
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum  → global load bytes
  lts__t_bytes.sum                               → L2 traffic bytes
  dram__bytes.sum                                → HBM bytes read/written
  sm__warps_active.avg.pct_of_peak_sustained_active → SM occupancy
  gpu__time_duration.sum                         → kernel duration (ns)
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared → shared memory conflicts
  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum (+ fmul, ffma) → FLOPs

CSV format from `ncu --csv --page raw`:
  "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",
  "Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"

CSV format from `ncu --csv --page details`:
  Simpler: kernel rows, one metric per row.


Phase attribution (schema 1.8.0) — what is achievable and what is not
---------------------------------------------------------------------
`LayerMetrics.phase` exists so a rule can scope to prefill or decode. Filling it
means answering: which inference phase was each kernel launch part of? The three
candidate joins, and the honest verdict on each:

1. **ncu's own NVTX column — WORKS, and is the only exact path.**
   `ncu --nvtx --csv ...` adds a column holding the NVTX push/pop stack that was
   active at the launch, e.g. ``<default domain>/decode seqs=24``. That is not a
   correlation at all: the profiler attributes the launch to the range itself,
   inside the same process, with no clock to reconcile. When the export carries
   such a column, `phase` is exact. It requires the capture to have been taken
   with `--nvtx` AND the engine to emit the step ranges
   `parsers/nsight_systems.py` already documents. Neither is true of the standard
   capture path today (a bare `ncu --csv --page raw`), so this path is implemented and, on current captures, unused.

2. **Timestamp correlation against `NsysTimeline` — NOT ACHIEVABLE from a stock
   ncu CSV.** This is the join schema 1.8.0 was designed around, and the data
   does not support it, for three independent reasons, any one of which is
   fatal:
     a. *There is no timestamp to join on.* `_METRIC_MAP` below is the complete
        set of metrics this parser reads, and every one is a counter or a
        duration. `gpu__time_duration.sum` is how LONG a kernel ran, never WHEN.
        The raw page's "Kernel Time" column is a human-readable wall-clock
        stamp at ~second resolution; engine steps are 5-50 ms, so even parsed it
        cannot place a launch within a step. Nothing else in the export is an
        instant.
     b. *The clocks are not the same clock.* An `nsys` timeline is on the trace
        origin; an ncu export has no trace origin at all. There is no offset to
        apply because there is no second instant to apply it to.
     c. *The runs are not the same run.* ncu serialises kernels and REPLAYS them
        to collect counter sets, so the wall-clock ordering under ncu is not the
        ordering the engine actually executed. ncu and nsys also do not
        generally profile one process concurrently. So even a perfect timestamp
        would be a timestamp from a different, distorted execution.
   The narrow case that IS achievable: an export carrying an explicit NUMERIC
   launch-timestamp column already expressed on the timeline's clock — which
   strided's own capture tooling could emit, and which no stock ncu invocation
   does. `correlate_phase_by_timestamp` implements exactly that case and refuses
   everything else.

3. **Kernel-name heuristics — REFUSED.** Both phases run attention and both run
   GEMMs. A name-based tag is not a weak signal, it is a wrong one, and it would
   be indistinguishable in the schema from a measured one. This is the r01 scope-dilution
   lesson applied to a new field: an honest None is worth more than a guessed
   tag.

So `phase` is None on every kernel of every capture the repo currently takes,
and the parser says so in `parse_warnings` rather than leaving the reader to
infer it from an absence.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Optional

from parsers.nsight_systems import classify_phase
from schema import (
    DiagnosisInput,
    EngineStepPhase,
    LayerMetrics,
    NsysTimeline,
    PhaseMetrics,
    RooflinePosition,
)


# ---------------------------------------------------------------------------
# Metric name → semantic mapping
# ---------------------------------------------------------------------------

# Metric patterns → (semantic_field, needs_normalisation, peak_value_hint)
# peak_value_hint is only used for roofline estimation when absolute values
# are available but percentages are not.
_METRIC_MAP: dict[str, str] = {
    # SM occupancy (reported as % of peak)
    "sm__warps_active.avg.pct_of_peak_sustained_active": "sm_occupancy_pct",
    "achieved_occupancy": "sm_occupancy_pct",  # older Nsight naming
    # HBM / DRAM bandwidth
    "dram__bytes.sum": "dram_bytes",
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "l2_global_load_bytes": "dram_bytes",            # older naming
    # L2 / memory traffic
    "lts__t_bytes.sum": "l2_bytes",
    # Kernel duration (canonical Nsight metric, nanoseconds)
    "gpu__time_duration.sum": "duration_ns",
    # FLOPs (we sum across add/mul/fma)
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": "flop_fadd",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": "flop_fmul",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "flop_ffma",
    # Memory bound indicators
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": "shared_mem_conflicts",
    # Warp stalls — key for memory-bound diagnosis
    "smsp__warp_issue_stall_long_scoreboard_per_warp_active.pct": "stall_mem_pct",
    "smsp__warp_issue_stall_mio_throttle_per_warp_active.pct": "stall_mio_pct",
}

# How to aggregate a semantic metric across repeated launches of the same kernel.
# Counters (bytes, time, FLOPs) sum; percentages are duration-weighted means (a
# 10 ms launch's occupancy should outweigh a 10 µs one), falling back to an
# unweighted mean when durations are absent.
_ADDITIVE_METRICS = frozenset({
    "duration_ns", "dram_bytes", "dram_bytes_read", "dram_bytes_write",
    "l2_bytes", "flop_fadd", "flop_fmul", "flop_ffma", "shared_mem_conflicts",
})
_PCT_METRICS = frozenset({"sm_occupancy_pct", "stall_mem_pct", "stall_mio_pct"})

# Peak HBM bandwidth in bytes/sec for known GPU types. Used to convert observed
# byte counts into a [0, 1] bandwidth utilisation. Add entries as needed.
_PEAK_HBM_BPS: dict[str, float] = {
    "H100-SXM":  3.35e12,
    "H100-PCIe": 2.00e12,
    "A100-80G":  2.039e12,
    "A100-40G":  1.555e12,
    # Aliases for the names real tooling reports. `nvidia-smi --query-gpu=name`
    # returns "NVIDIA A100-PCIE-40GB", and a field capture passed exactly that
    # through --gpu-type: it missed every key here, fell back to the H100-SXM
    # default, and scored an A100 against 3.35 TB/s instead of 1.555 — reading
    # every kernel 2.15x under-utilised, which is what pushed r01's batch-1
    # decode to 0.276 when the corrected figure is 0.595.
    "A100-PCIE-40GB":  1.555e12,
    "A100-PCIE-80GB":  1.935e12,   # PCIe 80GB is HBM2e @ 1935 GB/s
    "A100-SXM4-40GB":  1.555e12,
    "A100-SXM4-80GB":  2.039e12,
    "H100-80GB-HBM3":  3.35e12,
    "L40S":      0.864e12,
    # Local-validation cards (GDDR6, not HBM — same roofline arithmetic). Without
    # these entries a 3050 trace is scored against the H100 default and every
    # kernel reads ~15x under-utilised.
    "RTX-3050":        0.224e12,   # desktop, 128-bit GDDR6 @ 14 Gbps
    "RTX-3050-Laptop": 0.192e12,   # laptop, 128-bit GDDR6 @ 12 Gbps
}
_DEFAULT_PEAK_HBM_BPS = 3.35e12  # assume H100-SXM if gpu_type unknown


def _peak_hbm_bps_checked(gpu_type: str) -> tuple[float, bool]:
    """Return (peak bytes/s, whether the GPU was recognised).

    Every HBM utilisation is `bytes / (duration x peak)`, so an unrecognised
    gpu_type does not fail — it silently rescales every kernel in the trace. The
    default is the FASTEST card in the table, so an unknown GPU always reads
    *under*-utilised, biasing the roofline toward "compute-bound" and rules
    toward abstaining. That is a silent false negative, which is why the caller
    surfaces the miss instead of quietly proceeding.
    """
    if gpu_type in _PEAK_HBM_BPS:
        return _PEAK_HBM_BPS[gpu_type], True
    # Tolerate the vendor prefix and casing that real tools emit — `nvidia-smi
    # --query-gpu=name` gives "NVIDIA A100-PCIE-40GB".
    key = gpu_type.strip()
    if key.upper().startswith("NVIDIA "):
        key = key[len("NVIDIA "):].strip()
    for known, bps in _PEAK_HBM_BPS.items():
        if known.upper() == key.upper():
            return bps, True
    return _DEFAULT_PEAK_HBM_BPS, False


def _peak_hbm_bps(gpu_type: str) -> float:
    return _peak_hbm_bps_checked(gpu_type)[0]

# Kernel name patterns → layer type classification.
# The attention alternation covers the fused families seen in real traces:
# flash-attn's kernels are named flash_fwd_kernel / flash_fwd_splitkv_kernel
# (no "attn" substring!), PyTorch SDPA's mem-efficient path is fmha_cutlass*,
# and Triton attention kernels under a flash:: C++ namespace. Deliberately NOT
# matched: bare "_fwd_kernel" (Triton's generic suffix — it would swallow
# unrelated kernels); that naming miss is recorded in the r07 spec doc.
_LAYER_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"flash_attn|flash_fwd|flash::|fmha|sdpa|attention|attn", re.I), "attention"),
    (re.compile(r"mlp|ffn|feed_forward|gemm|matmul|mm_", re.I), "mlp"),
    (re.compile(r"layernorm|rmsnorm|norm", re.I), "norm"),
    (re.compile(r"allreduce|all_reduce|nccl", re.I), "allreduce"),
    (re.compile(r"embedding|embed", re.I), "embedding"),
    (re.compile(r"softmax", re.I), "softmax"),
]


def _classify_layer(kernel_name: str) -> str:
    for pattern, layer_type in _LAYER_TYPE_PATTERNS:
        if pattern.search(kernel_name):
            return layer_type
    return "other"


# ---------------------------------------------------------------------------
# Phase attribution (schema 1.8.0)
# ---------------------------------------------------------------------------
# Read the module docstring's "Phase attribution" section before touching any of
# this. The short version: there are exactly two evidence sources the parser will
# accept, and kernel names are not one of them.

# Column names, normalised, that carry ncu's NVTX push/pop stack for a launch.
# ncu's own header has drifted ("NVTX", "NVTX Range", "NVTX Push-Pop Range"), so
# match on the substring rather than pinning a version's exact spelling — the
# same loose-header policy parsers/nsight_systems.py already uses.
_NVTX_COLUMN_NEEDLE = "nvtx"

# Column names, normalised, that could carry a NUMERIC launch instant on the
# timeline's clock. Deliberately does NOT include ncu's "Kernel Time": that
# column is a human-readable wall-clock stamp at ~second resolution from a
# replayed execution, and treating it as a trace-relative instant is the
# fabricated correlation this module refuses to make.
_TIMESTAMP_COLUMN_NEEDLES = ("start (", "start_ns", "start_ms", "launch time", "timestamp")


def _normalise_header(header: str) -> str:
    """Lower-cased, quote- and BOM-stripped header, for loose matching."""
    return header.strip().strip('"').lstrip("﻿").lower()


def _find_nvtx_column(headers: list[str]) -> Optional[str]:
    """The header carrying ncu's NVTX range stack, if the export has one.

    Present only when the capture was taken with `ncu --nvtx`. The shipped
    capture scripts do not pass it, so this returns None on every trace the repo
    currently produces — which is the point: the caller then reports an honest
    None rather than reaching for a heuristic.
    """
    for h in headers:
        if _NVTX_COLUMN_NEEDLE in _normalise_header(h):
            return h
    return None


def _find_timestamp_column(headers: list[str]) -> Optional[str]:
    """The header carrying a numeric launch instant, if the export has one.

    See `_TIMESTAMP_COLUMN_NEEDLES` for why "Kernel Time" is excluded. A hit
    here is necessary but not sufficient — `correlate_phase_by_timestamp` still
    requires the values to parse as finite numbers and a timeline to join
    against, and abstains per-launch when they do not.
    """
    for h in headers:
        n = _normalise_header(h)
        if any(needle in n for needle in _TIMESTAMP_COLUMN_NEEDLES):
            return h
    return None


def phase_from_nvtx_range(nvtx_stack: str) -> Optional[EngineStepPhase]:
    """Phase of a launch from the NVTX push/pop stack active at its launch.

    Exact, not correlated: ncu recorded which range was open *inside the same
    process at the moment of the launch*, so there is no clock to reconcile and
    no interval to guess at. Delegates the name->phase mapping to
    `parsers/nsight_systems.classify_phase` rather than re-implementing it, so
    the ncu path and the nsys path cannot disagree about what "mixed_prefill"
    means. A stack naming no phase returns None (it was not a step range).

    The stack may be slash-separated and deeply nested, e.g.
    ``<default domain>/engine_loop/decode seqs=24``. `classify_phase` scans the
    whole string and tests "mixed" first, which is the behaviour we want here
    too: an outer `mixed` range containing an inner `prefill` range is a mixed
    step, and reading it as prefill would misattribute the decode work sharing
    that step.
    """
    if not nvtx_stack or not nvtx_stack.strip():
        return None
    return classify_phase(nvtx_stack)


def correlate_phase_by_timestamp(
    launch_instant_ms: float, timeline: NsysTimeline
) -> Optional[EngineStepPhase]:
    """Phase of a launch by interval containment in a timeline's steps.

    **This is the join that a stock ncu CSV cannot feed.** It is implemented for
    the narrow case that is genuinely achievable — an export carrying an explicit
    numeric launch instant already expressed on the timeline's clock — and it is
    the caller's job to have established that, which `_parse_ncu_csv` only does
    when `_find_timestamp_column` finds such a column. See the module docstring
    for why "Kernel Time" does not qualify and why replay makes ncu's wall clock
    the wrong clock in the first place.

    Returns None, never a nearest-neighbour guess, when the instant falls in no
    step. A launch in an idle gap between steps is unattributed; snapping it to
    the closest step would manufacture exactly the plausible-but-wrong tag this
    field exists to avoid. Steps are half-open [start, end) so a launch landing
    exactly on a boundary belongs to the step that was starting, not the one that
    had ended.
    """
    for step in timeline.steps:
        if step.start_ms <= launch_instant_ms < step.end_ms:
            return step.phase
    # A zero-length step is degenerate but legal (start == end); catch an exact
    # hit on one so it is not silently unattributable.
    for step in timeline.steps:
        if step.start_ms == step.end_ms == launch_instant_ms:
            return step.phase
    return None


def _roofline_from_metrics(
    sm_occupancy: Optional[float],
    hbm_util: Optional[float],
    stall_mem_pct: Optional[float],
) -> Optional[RooflinePosition]:
    """
    Heuristic roofline classification from Nsight metrics.
    Rules derived from "Mind the Memory Gap" (arxiv 2503.08311) and
    NVIDIA Nsight documentation.
    """
    if sm_occupancy is None or hbm_util is None:
        return "unknown"
    if stall_mem_pct is not None and stall_mem_pct > 40.0:
        return "memory_bound"
    if hbm_util > 0.85 and sm_occupancy < 0.40:
        return "memory_bound"
    if hbm_util < 0.40 and sm_occupancy > 0.70:
        return "compute_bound"
    return "balanced"


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _reduce_instances(instances: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate repeated launches of one kernel name into a single record.

    Additive metrics (time, bytes, FLOPs) sum across launches; percentage
    metrics take the duration-weighted mean (unweighted when any instance lacks
    a duration). A single instance passes through untouched.
    """
    if len(instances) == 1:
        return dict(instances[0])

    out: dict[str, float] = {}
    for metric in _ADDITIVE_METRICS:
        vals = [inst[metric] for inst in instances if metric in inst]
        if vals:
            out[metric] = sum(vals)
    for metric in _PCT_METRICS:
        pairs = [
            (inst[metric], inst.get("duration_ns")) for inst in instances if metric in inst
        ]
        if not pairs:
            continue
        if all(d is not None and d > 0 for _, d in pairs):
            out[metric] = sum(v * d for v, d in pairs) / sum(d for _, d in pairs)
        else:
            out[metric] = sum(v for v, _ in pairs) / len(pairs)
    return out


def _strip_csv_preamble(text: str) -> tuple[str, int]:
    """Drop any non-CSV lines preceding the Nsight header row.

    Returns (text_from_header_onward, lines_skipped). If no header row is found
    the text is returned unchanged so the caller's own diagnostics still fire.

    A header is identified structurally — a parsed row containing a "Kernel Name"
    field — rather than by matching text, so it holds for both the raw and
    details pages and does not depend on column order.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "Kernel Name" not in line:
            continue                      # cheap reject before paying for csv
        try:
            fields = next(csv.reader([line]))
        except (csv.Error, StopIteration):
            continue
        if any(f.strip().strip('"').lstrip("﻿") == "Kernel Name" for f in fields):
            return "\n".join(lines[i:]), i
    return text, 0


# Grouping key for a reduced kernel record: the kernel's symbol plus the phase
# its launches ran under. The phase is PART OF THE KEY, not a label bolted on
# afterwards, and that is load-bearing: one kernel name legitimately runs in both
# phases (a decode GEMM and a prefill GEMM are the same symbol), so reducing by
# name alone and then asking "what phase was that?" has no answer. Keying by the
# pair means a symbol appearing in both phases produces two records with two
# duration sums, which is the only shape in which a phase-scoped rule can read a
# per-phase number. `None` is a normal member of the key space and is the whole
# key space on a capture with no phase evidence — in which case this collapses
# to exactly the pre-1.8.0 by-name grouping, entry for entry, in order.
_KernelKey = tuple[str, Optional[EngineStepPhase]]


def _parse_ncu_csv(
    text: str, timeline: Optional[NsysTimeline] = None
) -> tuple[dict[_KernelKey, dict[str, float]], list[str]]:
    """
    Parse ncu --csv output.

    ncu profiles each kernel *launch* separately, so the same kernel name can
    appear once per launch (raw page: distinguished by the "ID" column; details
    page: one row per launch). Launches are grouped into instances and reduced
    per (kernel name, phase) — durations SUM, percentages take the
    duration-weighted mean — so time-share signals (r05's nccl_time_pct, r07's
    attention/softmax shares, the aggregate phase) reflect every launch rather
    than whichever instance the CSV listed last.

    Phase (schema 1.8.0) comes from one of two evidence sources, in order, and
    from nothing else — read the module docstring before adding a third:

      1. an ncu `--nvtx` range column on the row, which is exact; or
      2. a numeric launch-instant column joined against `timeline`, which is
         only offered when the export genuinely carries such a column on the
         timeline's clock.

    With neither, every key's phase is None and the grouping is byte-identical
    to the pre-1.8.0 by-name grouping.

    Args:
        text:     Raw CSV text.
        timeline: Optional `NsysTimeline` to join launch instants against. Used
                  only when the CSV carries a numeric instant column; a timeline
                  alone cannot phase-tag an export that has no instants, and
                  passing one does not make the parser guess.

    Returns:
        per_kernel: dict mapping (kernel_name, phase) → {metric_semantic: value}
        warnings:   list of parse warnings
    """
    per_kernel: dict[_KernelKey, dict[str, float]] = {}
    warnings: list[str] = []

    # `ncu --csv` writes its table to STDOUT, and so does the profiled
    # application — so the documented capture command (`ncu --csv --page raw
    # <workload> > kernels.csv`) yields a file whose real header is preceded by
    # however many lines the workload logged, plus ncu's own "==PROF==" notices.
    # csv.DictReader treats line 1 as the header, so without this the whole file
    # parses as one garbage column and every metric goes missing. Skip forward to
    # the first line that actually looks like the Nsight header.
    text, skipped = _strip_csv_preamble(text)
    if skipped:
        warnings.append(
            f"Nsight CSV parser: skipped {skipped} line(s) of non-CSV preamble "
            "before the header (application log output captured alongside the "
            "ncu table)."
        )

    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames is None:
        return per_kernel, ["Nsight CSV parser: empty or invalid CSV."]

    # Normalise header names (strip BOM, quotes, whitespace)
    headers = [h.strip().strip('"').lstrip('\ufeff') for h in (reader.fieldnames or [])]

    # Detect format: "Kernel Name" + "Metric Name" + "Metric Value" → raw page
    # or simpler formats
    has_metric_name_col = "Metric Name" in headers
    has_kernel_col = "Kernel Name" in headers
    has_id_col = "ID" in headers

    if not has_kernel_col:
        warnings.append(
            "Nsight CSV parser: 'Kernel Name' column not found. "
            "Export with --page raw or --page details."
        )
        return per_kernel, warnings

    # Which phase evidence, if any, this export actually carries. Resolved once
    # from the header rather than re-sniffed per row, so a malformed row cannot
    # silently switch the parser between evidence sources mid-file.
    nvtx_col = _find_nvtx_column(headers)
    ts_col = _find_timestamp_column(headers) if timeline is not None else None
    # Counted so the caller can report honestly on a PARTIAL tagging rather than
    # implying the whole trace was attributed. A launch that had evidence
    # available but fell outside every step (an idle gap) is unattributed, and
    # that is a different fact from "the export had no evidence at all".
    phase_attributed = 0
    phase_unattributed = 0

    # (kernel name, phase) → list of launch instances (each a {semantic: value}).
    instances: dict[_KernelKey, list[dict[str, float]]] = {}
    # Raw page with an ID column: (kernel, phase, launch id) → its instance dict.
    by_launch_id: dict[tuple[str, Optional[str], str], dict[str, float]] = {}

    for row in reader:
        # Normalise keys
        row = {k.strip().strip('"'): v.strip().strip('"') for k, v in row.items() if k}

        kernel = row.get("Kernel Name", "unknown_kernel")
        if not kernel:
            continue

        phase = _row_phase(row, nvtx_col, ts_col, timeline)
        if nvtx_col is not None or ts_col is not None:
            if phase is None:
                phase_unattributed += 1
            else:
                phase_attributed += 1
        key: _KernelKey = (kernel, phase)

        if has_metric_name_col:
            # raw-page format: one metric per row
            metric_name = row.get("Metric Name", "")
            metric_value_str = row.get("Metric Value", "")
            semantic = _METRIC_MAP.get(metric_name)
            if semantic is None:
                continue
            try:
                value = float(metric_value_str.replace(",", ""))
            except ValueError:
                continue
            if has_id_col:
                launch_key = (kernel, phase, row.get("ID", ""))
                inst = by_launch_id.get(launch_key)
                if inst is None:
                    inst = {}
                    by_launch_id[launch_key] = inst
                    instances.setdefault(key, []).append(inst)
            else:
                # No launch IDs: one launch's rows arrive together, so a
                # repeated semantic key means a new launch started.
                kernel_insts = instances.setdefault(key, [])
                if not kernel_insts or semantic in kernel_insts[-1]:
                    kernel_insts.append({})
                inst = kernel_insts[-1]
            inst[semantic] = value
        else:
            # details-page format: metric columns are the headers; one row = one launch
            inst = {}
            for col_name, col_value in row.items():
                semantic = _METRIC_MAP.get(col_name)
                if semantic is None:
                    continue
                try:
                    inst[semantic] = float(col_value.replace(",", ""))
                except ValueError:
                    continue
            if inst:
                instances.setdefault(key, []).append(inst)

    total_launches = sum(len(v) for v in instances.values())
    per_kernel = {k: _reduce_instances(insts) for k, insts in instances.items()}
    if total_launches > len(per_kernel):
        warnings.append(
            f"Nsight CSV: {total_launches} kernel launches aggregated into "
            f"{len(per_kernel)} kernel names (durations summed, percentages "
            f"duration-weighted)."
        )

    # Report the attribution as a rate, not a boolean. "Phases were tagged" is
    # true of a trace where one launch in a thousand landed inside a step, and a
    # reader who sees only the happy sentence will trust the resulting per-phase
    # aggregate far past what it can carry.
    if nvtx_col is not None:
        warnings.append(
            f"Nsight CSV: phase attributed from the NVTX range column "
            f"{nvtx_col!r} for {phase_attributed} of "
            f"{phase_attributed + phase_unattributed} launch rows "
            f"({phase_unattributed} carried no recognised prefill/decode/mixed "
            "range and are left phase=None)."
        )
    elif ts_col is not None:
        warnings.append(
            f"Nsight CSV: phase correlated from launch instants in {ts_col!r} "
            f"against {len(timeline.steps) if timeline else 0} timeline steps "
            f"for {phase_attributed} of {phase_attributed + phase_unattributed} "
            f"launch rows ({phase_unattributed} fell outside every step — an "
            "idle gap, or a clock the two captures do not share — and are left "
            "phase=None rather than snapped to the nearest step)."
        )

    return per_kernel, warnings


def _row_phase(
    row: dict[str, str],
    nvtx_col: Optional[str],
    ts_col: Optional[str],
    timeline: Optional[NsysTimeline],
) -> Optional[EngineStepPhase]:
    """Phase for one CSV row, from whichever evidence the export carries.

    NVTX first because it is exact (the profiler attributed the launch itself);
    the timestamp join is the fallback and is only reachable when the caller
    already established that the instant column is numeric and on the timeline's
    clock. No evidence means None, and None is returned rather than raising:
    a phase-less export is the normal case, not an error.
    """
    if nvtx_col is not None:
        phase = phase_from_nvtx_range(row.get(nvtx_col, ""))
        if phase is not None:
            return phase
        # Fall through: an NVTX column can be present but empty on some rows
        # (a launch outside any range), and a timestamp column may still place
        # it. Not an error either way.
    if ts_col is not None and timeline is not None:
        raw = row.get(ts_col, "").strip().replace(",", "")
        if raw:
            try:
                instant_ms = float(raw)
            except ValueError:
                return None
            if instant_ms != instant_ms or instant_ms in (float("inf"), float("-inf")):
                # Non-finite instants are unparseable rows, not distant ones.
                return None
            return correlate_phase_by_timestamp(instant_ms, timeline)
    return None


def _kernel_metrics_to_layer(
    kernel_name: str,
    metrics: dict[str, float],
    peak_hbm_bps: float,
    phase: Optional[EngineStepPhase] = None,
) -> LayerMetrics:
    """Convert a dict of semantic metrics for one kernel into a LayerMetrics.

    ``phase`` is passed through verbatim — it was decided at grouping time from
    the row's evidence, and nothing here may second-guess it or fill a None.
    """
    sm_pct = metrics.get("sm_occupancy_pct")
    sm_occ = sm_pct / 100.0 if sm_pct is not None else None

    duration_ns = metrics.get("duration_ns")
    duration_ms = duration_ns / 1e6 if duration_ns is not None else None

    # HBM utilisation = (observed bytes / duration) / peak bandwidth.
    # Requires both duration and a byte counter; otherwise None.
    hbm_util: Optional[float] = None
    dram_bytes = metrics.get("dram_bytes") or (
        (metrics.get("dram_bytes_read") or 0) + (metrics.get("dram_bytes_write") or 0)
    )
    if dram_bytes and duration_ns and duration_ns > 0:
        bw_bps = dram_bytes / (duration_ns * 1e-9)
        hbm_util = min(1.0, bw_bps / peak_hbm_bps)

    stall_mem_pct = metrics.get("stall_mem_pct")
    roofline = _roofline_from_metrics(sm_occ, hbm_util, stall_mem_pct)

    # FLOPs estimate: fadd + fmul + 2*ffma (each ffma = 2 FLOPs)
    flops_raw = (
        metrics.get("flop_fadd", 0)
        + metrics.get("flop_fmul", 0)
        + 2 * metrics.get("flop_ffma", 0)
    )
    achieved_tflops = None
    if flops_raw and duration_ns and duration_ns > 0:
        achieved_tflops = flops_raw / (duration_ns * 1e-9) / 1e12

    return LayerMetrics(
        layer_name=kernel_name,
        layer_type=_classify_layer(kernel_name),
        phase=phase,
        duration_ms=duration_ms,
        sm_occupancy=sm_occ,
        hbm_bandwidth_util=hbm_util,
        achieved_flops=achieved_tflops,
        roofline_position=roofline,
    )


def _aggregate_phase(layers: list[LayerMetrics]) -> Optional[PhaseMetrics]:
    """
    Aggregate per-kernel LayerMetrics into a single PhaseMetrics.
    Means are duration-weighted so a 10 ms GEMM outweighs a 10 µs softmax.

    **This function is not the r01 scope-dilution defect and changing its
    weighting would not fix it.** The weighting here is already duration-weighted;
    the field finding is that the *population* is
    unscoped. 17.4% of the window in near-zero-bandwidth kernels drags a 0.720
    subset reading down to 0.595 at ANY weighting, because weighting decides how
    loudly a kernel votes and not whether it is in the denominator. The fix is
    kernel scoping in r01, not this function's job.

    Passing a phase-filtered `layers` here (which `parse_nsight_csv` now does
    when the trace carries phase evidence) narrows the population along the
    phase axis only. It does not address the scope dilution, which is a separate
    axis and a separate contaminant.
    """
    if not layers:
        return None

    total_ms = sum(l.duration_ms for l in layers if l.duration_ms is not None)

    def weighted(attr: str) -> Optional[float]:
        num = den = 0.0
        for l in layers:
            v = getattr(l, attr)
            d = l.duration_ms
            if v is not None and d is not None and d > 0:
                num += v * d
                den += d
        return num / den if den > 0 else None

    sm_occ = weighted("sm_occupancy")
    hbm_util = weighted("hbm_bandwidth_util")
    roofline = _roofline_from_metrics(sm_occ, hbm_util, None)

    return PhaseMetrics(
        duration_ms=total_ms or None,
        sm_occupancy=sm_occ,
        hbm_bandwidth_util=hbm_util,
        roofline_position=roofline,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_nsight_csv(
    text: str,
    model_name: str = "unknown",
    gpu_type: str = "unknown",
    source_file: str = "<nsight_csv>",
    timeline: Optional[NsysTimeline] = None,
) -> DiagnosisInput:
    """
    Parse Nsight Compute CSV export into a DiagnosisInput.

    Export from Nsight UI: File → Export → CSV (raw or details page)
    Or from CLI: ncu --csv --page raw <workload>

    Args:
        text:        Raw CSV text.
        model_name:  Must be supplied by the caller (not in the CSV).
        gpu_type:    Must be supplied by the caller (or extracted from Nsight header).
        source_file: Provenance label.
        timeline:    Optional `NsysTimeline` (schema 1.8.0) to attribute launches
                     to a phase by interval containment. Only usable when the CSV
                     carries a numeric launch-instant column on the timeline's
                     clock, which a stock `ncu` export does not — see the module
                     docstring. Passing a timeline never causes a guess; with no
                     instant column it is simply unused.

    Returns:
        DiagnosisInput with per-layer breakdown and aggregate phase metrics.
        `layers[].phase` is populated only from real evidence, and `prefill` /
        `decode` are split only when that evidence exists.
    """
    per_kernel, warnings = _parse_ncu_csv(text, timeline=timeline)

    if not per_kernel:
        return DiagnosisInput(
            model_name=model_name,
            gpu_type=gpu_type,
            inference_engine="unknown",
            source_files=[source_file],
            parse_warnings=warnings or ["Nsight CSV parser: no kernels extracted."],
        )

    peak_hbm_bps, _gpu_known = _peak_hbm_bps_checked(gpu_type)
    if not _gpu_known:
        warnings.append(
            f"Nsight CSV parser: unrecognised gpu_type {gpu_type!r}; HBM "
            f"utilisation scored against the default "
            f"{_DEFAULT_PEAK_HBM_BPS / 1e12:.2f} TB/s peak. That default is the "
            "fastest card known, so utilisation reads LOW and the roofline is "
            "biased toward compute-bound. Pass a supported gpu_type "
            f"({', '.join(sorted(_PEAK_HBM_BPS))}) for a trustworthy verdict."
        )
    layers = [
        _kernel_metrics_to_layer(kname, kmetrics, peak_hbm_bps, phase=kphase)
        for (kname, kphase), kmetrics in per_kernel.items()
    ]

    prefill_metrics, decode_metrics = _phase_aggregates(layers, warnings)

    # NCCL dominance: time fraction of allreduce kernels. Computed over EVERY
    # layer regardless of phase — collectives run in both, and r05's question is
    # what share of the job's device time went to the fabric, not what share of
    # one phase's did. Splitting it by phase here would silently change a
    # shipped rule's input the moment a trace gained NVTX ranges.
    allreduce_layers = [l for l in layers if l.layer_type == "allreduce"]
    nccl_time_pct: Optional[float] = None
    if allreduce_layers:
        total_dur = sum(l.duration_ms for l in layers if l.duration_ms is not None)
        nccl_dur = sum(l.duration_ms for l in allreduce_layers if l.duration_ms is not None)
        if total_dur > 0:
            nccl_time_pct = nccl_dur / total_dur

    return DiagnosisInput(
        model_name=model_name,
        gpu_type=gpu_type,
        inference_engine="unknown",
        prefill=prefill_metrics,
        decode=decode_metrics,
        layers=layers,
        nccl_time_pct=nccl_time_pct,
        source_files=[source_file],
        parse_warnings=warnings,
    )


def _phase_aggregates(
    layers: list[LayerMetrics], warnings: list[str]
) -> tuple[Optional[PhaseMetrics], Optional[PhaseMetrics]]:
    """Split the layer list into the `prefill` and `decode` PhaseMetrics slots.

    Two branches, and the difference between them is the whole content of schema
    1.8.0.

    **No phase evidence (the case for every capture the repo takes today).**
    Behaviour is byte-identical to 1.7.0: the flat aggregate over every kernel
    goes to `decode` and `prefill` stays None. That is a MISATTRIBUTION and it is
    kept deliberately. Prefill kernels are in that number; the parser cannot say
    which, and moving the aggregate somewhere else (to `prefill`, to both, to
    neither) would be a different guess, not a smaller one. What 1.8.0 changes is
    that the misattribution is now *stated* — in the warning below, in
    `layers[].phase is None`, and in the changelog — instead of being visible
    only to someone who reads the constructor call. r01 reads `decode`, so r01's
    number carries prefill contamination on every A-tier capture we have taken,
    independently of the r01 scope dilution.

    **Phase evidence present.** Each slot aggregates only the layers tagged with
    it. Three things deliberately do NOT happen:
      - `mixed` layers are not folded into either slot. A chunked-prefill step
        ran prompt tokens and in-flight decodes in one launch batch; assigning
        its kernels to either pure phase is the misattribution again, one level
        down. They stay in `layers` with `phase="mixed"` where a rule can decide
        for itself.
      - Untagged layers in a partially-attributed trace are not swept into
        `decode`. They are reported by their share of device time, so a reader
        can see how much of the trace the per-phase numbers do not cover.
      - An empty slot stays None, never a zero-valued PhaseMetrics. "No prefill
        kernels were attributed" and "prefill did no work" are different claims
        and only one of them is supported.
    """
    tagged = [l for l in layers if l.phase is not None]

    if not tagged:
        warnings.append(
            "Nsight CSV: trace does not tag prefill/decode phases, so every "
            "kernel — prefill kernels included — is aggregated into 'decode' "
            "and 'prefill' is left None. Read 'decode' as 'all device work', "
            "not as decode work: this is a misattribution, not a label. "
            "Per-kernel data is in 'layers' with phase=None. To get a real "
            "split, capture with `ncu --nvtx` against an engine emitting "
            "prefill/decode NVTX step ranges (see parsers/nsight_systems.py)."
        )
        return None, _aggregate_phase(layers)

    def _slot(phase: EngineStepPhase) -> Optional[PhaseMetrics]:
        subset = [l for l in layers if l.phase == phase]
        return _aggregate_phase(subset) if subset else None

    total_ms = sum(l.duration_ms for l in layers if l.duration_ms is not None)
    untagged_ms = sum(
        l.duration_ms for l in layers if l.phase is None and l.duration_ms is not None
    )
    mixed_ms = sum(
        l.duration_ms for l in layers if l.phase == "mixed" and l.duration_ms is not None
    )
    if total_ms > 0 and (untagged_ms or mixed_ms):
        warnings.append(
            f"Nsight CSV: per-phase aggregates cover "
            f"{100.0 * (total_ms - untagged_ms - mixed_ms) / total_ms:.1f}% of "
            f"device time; {100.0 * mixed_ms / total_ms:.1f}% ran in 'mixed' "
            f"(chunked-prefill) steps and {100.0 * untagged_ms / total_ms:.1f}% "
            "could not be attributed. Neither is folded into 'prefill' or "
            "'decode' — both are in 'layers' with their own phase tag."
        )

    return _slot("prefill"), _slot("decode")


def parse_nsight_csv_file(path: str, **kwargs) -> DiagnosisInput:
    """Convenience wrapper: read a .csv file and parse it."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    return parse_nsight_csv(text, source_file=path, **kwargs)


def parse_nsight_ncu_rep(path: str, **kwargs) -> DiagnosisInput:
    """
    [STUB] Parse a binary .ncu-rep file via nvidia-nsight-compute Python bindings.

    NOT YET IMPLEMENTED. Requires:
      - Nsight Compute 2024.x+ installed on the machine
      - ncu Python bindings (ncu.py / pynvml)
      - An actual NVIDIA GPU

    When implemented, this will:
      1. Open the .ncu-rep via ncu.open_file()
      2. Iterate over ranges/kernels
      3. Extract the same metric set as the CSV parser
      4. Call _kernel_metrics_to_layer() for each kernel
      5. Return a DiagnosisInput

    For now, raise a clear error rather than silently returning garbage.
    """
    raise NotImplementedError(
        "parse_nsight_ncu_rep() is not yet implemented. "
        "Export from Nsight Compute UI to CSV and use parse_nsight_csv_file() instead."
    )