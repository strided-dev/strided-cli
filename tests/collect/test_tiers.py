"""Tests for collect/tiers.py — pin the live/deep tiering to the rules' reality.

The tier table is data *about* the rules, kept out of the sacred contract. The
risk is drift: the table claiming a rule is live-capable when its real
INSUFFICIENT_DATA guards say otherwise. These tests pin the table to what the
rules actually do on a representative live snapshot, and pin the field→source map
to the fields r01 genuinely reports missing.
"""

from __future__ import annotations

from pathlib import Path

from collect import tiers
from engine import run_diagnosis
from engine.registry import rule_ids
from parsers.dcgm import parse_dcgm_prometheus_file
from parsers.vllm import parse_vllm_metrics_file
from rules.base import InsufficientData
from rules.r01_decode_memory_bound import DecodeMemoryBoundRule
from schema import DiagnosisInput, ThroughputSample
from schema.merge import merge_inputs

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "collect" / "replay"


def _full_live_snapshot() -> DiagnosisInput:
    """A merged vLLM + DCGM snapshot — every live source connected.

    Includes the sustained throughput_history the watch loop accumulates across
    ticks (a healthy, flat run), since r06 reads a *window* not a single scrape:
    a fully-connected live watch provides it, so a 'full' snapshot must too. The
    flat series keeps r06 BELOW_THRESHOLD (not INSUFFICIENT_DATA) — the rule had
    its data, the workload just wasn't decaying.
    """
    v = parse_vllm_metrics_file(str(_REPLAY / "vllm" / "00.prom"))
    d = parse_dcgm_prometheus_file(str(_REPLAY / "dcgm" / "00.prom"))
    merged = merge_inputs([v, d], model_name="m", gpu_type="H100")
    history = [
        ThroughputSample(t=float(i * 15), token_throughput_gen=100.0,
                         num_requests_waiting=5.0)
        for i in range(8)
    ]
    return merged.model_copy(update={"throughput_history": history})


def test_live_capable_rules_are_not_insufficient_on_a_full_snapshot() -> None:
    report = run_diagnosis(_full_live_snapshot())
    insufficient = {n.rule_id for n in report.insufficient_data}
    for rid in rule_ids():
        t = tiers.tier(rid)
        if t is not None and not t.dump_only:
            assert rid not in insufficient, (
                f"{rid} is marked live-capable but landed in INSUFFICIENT_DATA on a "
                "fully-connected live snapshot — the tier table has drifted."
            )


def test_dump_only_rule_is_suppressed_live() -> None:
    assert tiers.is_surfaced("r05") is False  # NCCL — dump only
    assert tiers.is_surfaced("r07") is False  # attention bottleneck — Nsight layers only
    assert tiers.is_surfaced("r01") is True
    assert tiers.is_surfaced("r02") is True


def test_field_source_map_covers_r01_missing_fields() -> None:
    # r01 abstains naming the fields it lacks; each must map to the source to enable.
    result = DecodeMemoryBoundRule().evaluate(DiagnosisInput(model_name="m", gpu_type="g"))
    assert isinstance(result, InsufficientData)
    assert result.missing  # it named something
    for field in result.missing:
        assert tiers.source_for_field(field) == tiers.DCGM


def test_banner_flags_dcgm_connection_state() -> None:
    connected = tiers.startup_banner(have_dcgm=True, registered_ids=rule_ids())
    disconnected = tiers.startup_banner(have_dcgm=False, registered_ids=rule_ids())
    assert "connected" in connected
    assert "--dcgm" in disconnected
    # Only registered rules appear; r05 isn't registered yet, so no Nsight line.
    assert "r02" in connected and "r03" in connected
