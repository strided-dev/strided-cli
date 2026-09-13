"""
DCGM JSON parser.

Read up on DCGM here: https://developer.nvidia.com/dcgm
Parses DCGM (Data Center GPU Manager) JSON output into DiagnosisInput.
DCGM is the standard cluster-level GPU telemetry tool — it gives you
per-GPU utilisation, memory bandwidth, NVLink stats, and per-rank timing
(for tensor-parallel setups).

This parser targets the output of:
  dcgmi dmon -e <field_ids> --json
  dcgm-exporter (Prometheus-format JSON)
  DCGM Python bindings export

Input format (one of):
  A) dcgmi dmon --json output:  top-level "DCGM_FI_*" keyed dict
  B) dcgm-exporter snapshot:    list of {"name": ..., "labels": {...}, "value": ...}

Output: DiagnosisInput with cluster-level fields populated.
        KV cache and per-layer fields are left as None (those come from vLLM / Nsight).

DCGM field IDs we care about:
  DCGM_FI_DEV_GPU_UTIL          (203)  - SM utilisation [0,100]
  DCGM_FI_DEV_MEM_COPY_UTIL     (204)  - memory copy engine util [0,100]
  DCGM_FI_DEV_FB_USED           (250)  - framebuffer used (MiB)
  DCGM_FI_DEV_FB_FREE           (251)  - framebuffer free (MiB)
  DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL (409) - NVLink bandwidth (MB/s)
  DCGM_FI_PROF_DRAM_ACTIVE       (1005) - HBM bandwidth utilisation [0,1]
  DCGM_FI_PROF_SM_OCCUPANCY      (1007) - SM occupancy [0,1]
  DCGM_FI_PROF_PIPE_TENSOR_ACTIVE (1004) - tensor core active [0,1]
"""

from __future__ import annotations

import json
from typing import Any, Optional

from parsers._prom import iter_samples
from schema import DiagnosisInput, PhaseMetrics


# ---------------------------------------------------------------------------
# Field ID → semantic name mapping
# ---------------------------------------------------------------------------
_FIELD_ID_MAP: dict[str | int, str] = {
    # by numeric ID
    100:  "sm_clock",
    150:  "gpu_temp",
    203:  "gpu_util",
    204:  "mem_copy_util",
    250:  "fb_used_mib",
    251:  "fb_free_mib",
    409:  "nvlink_bw_mbps",
    1004: "tensor_active",
    1005: "dram_active",
    1007: "sm_occupancy",
    # by string ID
    "DCGM_FI_DEV_SM_CLOCK":               "sm_clock",
    "DCGM_FI_DEV_GPU_TEMP":               "gpu_temp",
    "DCGM_FI_DEV_GPU_UTIL":               "gpu_util",
    "DCGM_FI_DEV_MEM_COPY_UTIL":          "mem_copy_util",
    "DCGM_FI_DEV_FB_USED":                "fb_used_mib",
    "DCGM_FI_DEV_FB_FREE":                "fb_free_mib",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL": "nvlink_bw_mbps",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE":    "tensor_active",
    "DCGM_FI_PROF_DRAM_ACTIVE":           "dram_active",
    "DCGM_FI_PROF_SM_OCCUPANCY":          "sm_occupancy",
}


def _norm(v: float, divisor: float = 100.0) -> float:
    """Normalise a percentage (0-100) to [0, 1]."""
    return max(0.0, min(1.0, v / divisor))


# ---------------------------------------------------------------------------
# Format A: dcgmi dmon --json
# ---------------------------------------------------------------------------
def _parse_dmon_format(data: dict) -> tuple[list[dict[str, Any]], list[str]]:
    """
    dcgmi dmon --json produces something like:
    {
      "DCGM_FI_DEV_GPU_UTIL": {"0": 87, "1": 82, "2": 91, "3": 85},
      "DCGM_FI_PROF_DRAM_ACTIVE": {"0": 0.91, "1": 0.89, ...},
      ...
    }
    Returns a list of per-rank dicts: [{"gpu_util": 0.87, "dram_active": 0.91, ...}, ...]
    """
    warnings: list[str] = []
    per_rank: dict[int, dict[str, Any]] = {}

    for field_key, rank_values in data.items():
        semantic = _FIELD_ID_MAP.get(field_key)
        if semantic is None and isinstance(field_key, str) and field_key.isdigit():
            semantic = _FIELD_ID_MAP.get(int(field_key))
        if semantic is None:
            continue
        if not isinstance(rank_values, dict):
            continue
        for rank_str, value in rank_values.items():
            try:
                rank = int(rank_str)
            except ValueError:
                continue
            per_rank.setdefault(rank, {})[semantic] = float(value)

    if not per_rank:
        warnings.append("DCGM dmon parser: no recognised fields found in dmon format.")

    return [per_rank[k] for k in sorted(per_rank)], warnings


# ---------------------------------------------------------------------------
# Format B: dcgm-exporter snapshot (list of metric objects)
# ---------------------------------------------------------------------------
def _parse_exporter_format(data: list) -> tuple[list[dict[str, Any]], list[str]]:
    """
    dcgm-exporter produces a list like:
    [
      {"name": "DCGM_FI_PROF_DRAM_ACTIVE", "labels": {"gpu": "0", ...}, "value": 0.91},
      ...
    ]
    """
    warnings: list[str] = []
    per_rank: dict[int, dict[str, Any]] = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        field_key = item.get("name", "")
        semantic = _FIELD_ID_MAP.get(field_key)
        if semantic is None:
            continue
        labels = item.get("labels", {})
        rank_str = labels.get("gpu") or labels.get("rank") or labels.get("device") or "0"
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        value = item.get("value")
        if value is None:
            continue
        per_rank.setdefault(rank, {})[semantic] = float(value)

    if not per_rank:
        warnings.append("DCGM exporter parser: no recognised fields found in exporter format.")

    return [per_rank[k] for k in sorted(per_rank)], warnings


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _rank_aggregate(per_rank: list[dict[str, Any]], key: str) -> Optional[float]:
    vals = [r[key] for r in per_rank if key in r]
    return _mean(vals) if vals else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_dcgm_json(
    text: str,
    model_name: str = "unknown",
    gpu_type: str = "unknown",
    source_file: str = "<dcgm>",
) -> DiagnosisInput:
    """
    Parse DCGM JSON into a DiagnosisInput.

    Accepts either:
      - dcgmi dmon --json format (top-level dict of field → rank → value)
      - dcgm-exporter format (list of metric objects with labels)

    Args:
        text:        Raw JSON string from DCGM.
        model_name:  Passed through; DCGM doesn't track this.
        gpu_type:    Passed through; or extracted from DCGM labels if present.
        source_file: Provenance label.

    Returns:
        DiagnosisInput with cluster-level and phase-level fields populated.
    """
    warnings: list[str] = []

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return DiagnosisInput(
            model_name=model_name,
            gpu_type=gpu_type,
            inference_engine="unknown",
            source_files=[source_file],
            parse_warnings=[f"DCGM parser: JSON decode failed — {e}"],
        )

    # Detect format
    if isinstance(data, list):
        per_rank, fmt_warnings = _parse_exporter_format(data)
    elif isinstance(data, dict):
        per_rank, fmt_warnings = _parse_dmon_format(data)
    else:
        return DiagnosisInput(
            model_name=model_name,
            gpu_type=gpu_type,
            inference_engine="unknown",
            source_files=[source_file],
            parse_warnings=["DCGM parser: unrecognised JSON structure (not list or dict)."],
        )

    warnings.extend(fmt_warnings)
    return _build_dcgm_input(per_rank, model_name, gpu_type, source_file, warnings)


def _build_dcgm_input(
    per_rank: list[dict[str, Any]],
    model_name: str,
    gpu_type: str,
    source_file: str,
    warnings: list[str],
) -> DiagnosisInput:
    """Assemble a DiagnosisInput from per-rank DCGM fields.

    Shared by the JSON (``parse_dcgm_json``) and Prometheus-text
    (``parse_dcgm_prometheus``) front-ends: both reduce their input to the same
    per-rank list, and the derived-field logic (HBM/SM aggregation, roofline
    heuristic, TP-rank timing proxy, PROF→decode placement) lives here once.
    """
    if not per_rank:
        return DiagnosisInput(
            model_name=model_name,
            gpu_type=gpu_type,
            inference_engine="unknown",
            source_files=[source_file],
            parse_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------
    num_gpus = len(per_rank)

    # HBM bandwidth utilisation: prefer DRAM_ACTIVE (profiling counter), fall back
    # to MEM_COPY_UTIL (coarser, percentage-based)
    hbm_util: Optional[float] = _rank_aggregate(per_rank, "dram_active")
    if hbm_util is None:
        raw_mcu = _rank_aggregate(per_rank, "mem_copy_util")
        hbm_util = _norm(raw_mcu) if raw_mcu is not None else None

    # SM occupancy: already in [0,1] from PROF counter, or normalise from util
    sm_occ: Optional[float] = _rank_aggregate(per_rank, "sm_occupancy")
    if sm_occ is None:
        raw_util = _rank_aggregate(per_rank, "gpu_util")
        sm_occ = _norm(raw_util) if raw_util is not None else None

    # Tensor core activity (optional extra signal for roofline position)
    tensor_active: Optional[float] = _rank_aggregate(per_rank, "tensor_active")

    # Roofline heuristic from aggregate cluster metrics
    roofline: Optional[str] = None
    if hbm_util is not None and sm_occ is not None:
        if hbm_util > 0.85 and sm_occ < 0.4:
            roofline = "memory_bound"
        elif hbm_util < 0.5 and sm_occ > 0.7:
            roofline = "compute_bound"
        elif hbm_util > 0.6 and sm_occ > 0.6:
            roofline = "balanced"

    # Per-rank SM clock + temperature for TP imbalance detection (r04). These are
    # RAW telemetry — the parser writes them as-is; r04 derives the comparable
    # slowness ratio (max_clock / clock) itself, because a derived ratio is a rule
    # concept, not schema state. gpu_util is deliberately NOT captured as a slowness
    # signal: the old open question — does a straggler show the highest util
    # (busy-waiting) or the lowest (stalled)? — is resolved by declining to guess.
    # Under the all-reduce barrier the fast ranks spin-wait in NCCL at ~100% util,
    # so the direction is ambiguous; with no clocks r04 simply has no signal and
    # abstains, rather than risk an inverted straggler call (r04 review, 2026-06).
    # Each list is surfaced only when present for *every* rank, so rank indices stay
    # aligned.
    clock_vals = [r["sm_clock"] for r in per_rank if "sm_clock" in r]
    temp_vals = [r["gpu_temp"] for r in per_rank if "gpu_temp" in r]

    tp_rank_sm_clocks = clock_vals if len(clock_vals) == num_gpus and num_gpus >= 2 else None
    tp_rank_temps = temp_vals if len(temp_vals) == num_gpus and num_gpus >= 2 else None

    # Build a single PhaseMetrics representing the cluster aggregate.
    # DCGM cannot separate prefill from decode, so the aggregate is placed in
    # `decode` (the longer-running phase). Rules that assume `decode` is truly
    # decode-only should cross-check the source via parse_warnings.
    aggregate_phase = PhaseMetrics(
        sm_occupancy=sm_occ,
        hbm_bandwidth_util=hbm_util,
        roofline_position=roofline,
    )
    warnings.append(
        "DCGM parser: cluster snapshot does not distinguish prefill from decode. "
        "Aggregate metrics placed in 'decode'; use Nsight for phase-resolved data."
    )

    return DiagnosisInput(
        model_name=model_name,
        gpu_type=gpu_type,
        inference_engine="unknown",
        num_gpus=num_gpus,
        tensor_parallel_size=num_gpus,   # assume full TP; override if known
        decode=aggregate_phase,
        tp_rank_sm_clocks=tp_rank_sm_clocks,
        tp_rank_temps=tp_rank_temps,
        source_files=[source_file],
        parse_warnings=warnings,
    )


def parse_dcgm_json_file(path: str, **kwargs) -> DiagnosisInput:
    """Convenience wrapper: read a .json file and parse it."""
    with open(path, "r") as f:
        text = f.read()
    return parse_dcgm_json(text, source_file=path, **kwargs)


def parse_dcgm_prometheus(
    text: str,
    model_name: str = "unknown",
    gpu_type: str = "unknown",
    source_file: str = "<dcgm>",
) -> DiagnosisInput:
    """Parse dcgm-exporter Prometheus *text* into a DiagnosisInput.

    dcgm-exporter serves the Prometheus text exposition format (one
    ``DCGM_FI_*{gpu="N",...} value`` line per GPU per field), not the JSON the
    dmon path reads. This is the live-pollable form ``watch`` scrapes. We reduce
    each recognised line to the same per-rank list ``_parse_exporter_format``
    produces, keyed by the ``gpu``/``rank``/``device`` label, then hand off to the
    shared ``_build_dcgm_input``.
    """
    warnings: list[str] = []
    per_rank: dict[int, dict[str, Any]] = {}

    for name, labels, value in iter_samples(text):
        semantic = _FIELD_ID_MAP.get(name)
        if semantic is None:
            continue
        rank_str = labels.get("gpu") or labels.get("rank") or labels.get("device") or "0"
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        per_rank.setdefault(rank, {})[semantic] = float(value)

    ranks = [per_rank[k] for k in sorted(per_rank)]
    if not ranks:
        warnings.append(
            "DCGM exporter parser: no recognised DCGM_FI_* fields found in "
            "Prometheus text."
        )
    return _build_dcgm_input(ranks, model_name, gpu_type, source_file, warnings)


def parse_dcgm_prometheus_file(path: str, **kwargs) -> DiagnosisInput:
    """Convenience wrapper: read a dcgm-exporter text dump and parse it."""
    with open(path, "r") as f:
        text = f.read()
    return parse_dcgm_prometheus(text, source_file=path, **kwargs)