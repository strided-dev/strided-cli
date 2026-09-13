"""Tests for collect/session.py — emit-on-change and graceful degradation.

The session is pure of I/O (it prints nothing), so we drive it with fake sources
and a synthetic clock. Change detection keys on the *set of fired rules*: a
re-run with identical state must not report `changed`; a cleared rule must.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from collect.session import WatchSession
from collect.sources import ReplaySource
from parsers.dcgm import parse_dcgm_prometheus_file
from parsers.vllm import parse_vllm_metrics_file
from schema import DiagnosisInput, VllmServingMetrics

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "collect" / "replay"

# Parsed fixture snapshots: tick 00 fires r02+r03, tick 02 fires r02 only (r03
# cleared as kv_cache dropped below the bar).
_V_BOTH = parse_vllm_metrics_file(str(_REPLAY / "vllm" / "00.prom"))
_V_R02_ONLY = parse_vllm_metrics_file(str(_REPLAY / "vllm" / "02.prom"))


class _FakeSource:
    """Returns a preset sequence of DiagnosisInputs, then None (exhausted)."""

    name = "fake"

    def __init__(self, seq: list[Optional[DiagnosisInput]]) -> None:
        self._seq = list(seq)
        self._i = 0
        self.last_error: Optional[str] = None
        self.exhausted = len(self._seq) == 0

    def poll(self) -> Optional[DiagnosisInput]:
        if self._i >= len(self._seq):
            self.exhausted = True
            return None
        v = self._seq[self._i]
        self._i += 1
        if self._i >= len(self._seq):
            self.exhausted = True
        return v


class _FailingSource:
    """Always fails to scrape (records an error, returns None)."""

    name = "dcgm"
    exhausted = False

    def __init__(self) -> None:
        self.last_error: Optional[str] = None

    def poll(self) -> Optional[DiagnosisInput]:
        self.last_error = "scrape failed: connection refused"
        return None


def _fired(report) -> set[str]:
    return {r.diagnosis.rule_id for r in report.diagnoses}


def test_first_tick_changes_then_identical_state_is_quiet() -> None:
    session = WatchSession([_FakeSource([_V_BOTH, _V_BOTH])])
    r1 = session.tick(0.0)
    r2 = session.tick(5.0)
    assert r1.changed is True
    assert r2.changed is False
    assert _fired(r1.report) == _fired(r2.report) == {"r02", "r03"}


def test_cleared_rule_re_emits() -> None:
    session = WatchSession([_FakeSource([_V_BOTH, _V_R02_ONLY])])
    r1 = session.tick(0.0)
    r2 = session.tick(5.0)
    assert r1.changed and r2.changed
    assert _fired(r1.report) == {"r02", "r03"}
    assert _fired(r2.report) == {"r02"}  # r03 cleared


def test_partial_source_failure_still_diagnoses() -> None:
    session = WatchSession([_FakeSource([_V_BOTH]), _FailingSource()])
    result = session.tick(0.0)
    assert result.merged is not None
    assert _fired(result.report) == {"r02", "r03"}
    assert any("scrape failed" in w for w in result.warnings)


def test_no_inputs_is_not_a_state_change() -> None:
    session = WatchSession([_FailingSource()])
    result = session.tick(0.0)
    assert result.merged is None
    assert result.report is None
    assert result.changed is False


def test_paired_replay_fires_all_three_offline() -> None:
    vfiles = sorted((_REPLAY / "vllm").glob("*.prom"))
    dfiles = sorted((_REPLAY / "dcgm").glob("*.prom"))
    session = WatchSession([
        ReplaySource(vfiles, parse_vllm_metrics_file),
        ReplaySource(dfiles, parse_dcgm_prometheus_file),
    ])
    result = session.tick(0.0)
    assert {"r01", "r02", "r03"} <= _fired(result.report)


def test_windowed_throughput_populates_on_second_tick() -> None:
    # Two vLLM ticks with growing counters → positive gen throughput on tick 2.
    session = WatchSession([_FakeSource([
        parse_vllm_metrics_file(str(_REPLAY / "vllm" / "00.prom")),
        parse_vllm_metrics_file(str(_REPLAY / "vllm" / "01.prom")),
    ])])
    r1 = session.tick(0.0)
    r2 = session.tick(5.0)
    assert r1.merged.token_throughput_gen is None     # first tick: no window
    assert r2.merged.token_throughput_gen is not None  # 01.prom gen > 00.prom gen
    assert r2.merged.token_throughput_gen > 0


def test_throughput_history_accumulates_across_ticks() -> None:
    # Growing generation counter → a windowed rate from tick 2 on; the watch loop
    # appends one ThroughputSample per rate-bearing tick to throughput_history,
    # carrying the backlog co-sample r06 reads. (Feeds r06; one-shot path has none.)
    def _vllm(gen: int) -> DiagnosisInput:
        return DiagnosisInput(
            model_name="m", gpu_type="g", inference_engine="vllm",
            vllm_serving=VllmServingMetrics(
                generation_tokens_total=gen, request_success_total=10,
                num_requests_waiting=5),
        )

    session = WatchSession([_FakeSource([_vllm(0), _vllm(1000), _vllm(1800)])])
    r1 = session.tick(0.0)
    r2 = session.tick(30.0)
    r3 = session.tick(60.0)
    assert r1.merged.throughput_history is None          # no rate on tick 1
    assert len(r2.merged.throughput_history) == 1
    assert len(r3.merged.throughput_history) == 2
    last = r3.merged.throughput_history[-1]
    assert last.token_throughput_gen > 0
    assert last.num_requests_waiting == 5


def test_history_cosamples_are_windowed_deltas_not_lifetime() -> None:
    # THE semantics pin for schema 1.4.0: the pressure co-samples must be
    # per-window deltas, not lifetime ratios/means. Three ticks with cumulative
    # counters chosen so lifetime and window values diverge:
    #   tick2 window: Δpre=5,  Δsucc=10 → 0.5;  Δqsum=1000, Δqcount=10 → 100 ms
    #   tick3 window: Δpre=30, Δsucc=20 → 1.5;  Δqsum=8000, Δqcount=20 → 400 ms
    # (lifetime at tick3 would read pre 35/30 ≈ 1.17 and queue 9100/31 ≈ 294 ms).
    def _vllm(gen, pre, succ, qsum, qcount) -> DiagnosisInput:
        return DiagnosisInput(
            model_name="m", gpu_type="g", inference_engine="vllm",
            vllm_serving=VllmServingMetrics(
                generation_tokens_total=gen,
                num_preemptions_total=pre,
                request_success_total=succ,
                request_queue_time_ms_sum=qsum,
                request_queue_time_count=qcount,
            ),
        )

    session = WatchSession([_FakeSource([
        _vllm(0, 0, 0, 100.0, 1.0),
        _vllm(1000, 5, 10, 1100.0, 11.0),
        _vllm(1800, 35, 30, 9100.0, 31.0),
    ])])
    session.tick(0.0)
    r2 = session.tick(30.0)
    r3 = session.tick(60.0)
    s2 = r2.merged.throughput_history[-1]
    s3 = r3.merged.throughput_history[-1]
    assert s2.preemption_rate == pytest.approx(0.5)      # 5 / 10, this window only
    assert s2.queue_time_ms == pytest.approx(100.0)      # 1000 / 10
    assert s3.preemption_rate == pytest.approx(1.5)      # 30 / 20 — rises with pressure
    assert s3.queue_time_ms == pytest.approx(400.0)      # 8000 / 20


def test_history_queue_cosample_none_when_no_completions_in_window() -> None:
    # Δcount == 0 → "no requests completed this window" reads as no signal (None),
    # not a zero that would drag the r06 trend down.
    def _vllm(gen, qsum, qcount) -> DiagnosisInput:
        return DiagnosisInput(
            model_name="m", gpu_type="g", inference_engine="vllm",
            vllm_serving=VllmServingMetrics(
                generation_tokens_total=gen,
                request_queue_time_ms_sum=qsum,
                request_queue_time_count=qcount,
            ),
        )

    session = WatchSession([_FakeSource([
        _vllm(0, 500.0, 5.0),
        _vllm(1000, 500.0, 5.0),   # nothing completed in this window
    ])])
    session.tick(0.0)
    r2 = session.tick(30.0)
    s2 = r2.merged.throughput_history[-1]
    assert s2.queue_time_ms is None
    assert s2.preemption_rate is None   # preemption counter absent entirely


def test_all_sources_exhausted_reports_exhausted() -> None:
    session = WatchSession([_FakeSource([_V_BOTH])])
    session.tick(0.0)          # serves the one input
    result = session.tick(5.0)  # nothing left
    assert result.exhausted is True
    assert result.merged is None
