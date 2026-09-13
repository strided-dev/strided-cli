"""Tests for collect/window.py — cumulative counters → per-second rates.

The window is the live path's distinguishing capability (the one-shot path cannot
compute a rate from a single scrape). These pin the rate maths and the two
defensive cases: a non-positive dt and a counter that went backwards (restart).
"""

from __future__ import annotations

import pytest

from collect.window import ThroughputHistory, ThroughputWindow


def test_first_tick_has_no_rates_or_deltas() -> None:
    w = ThroughputWindow()
    wu = w.update({"generation_tokens_total": 1000.0}, 0.0)
    assert wu.rates == {}
    assert wu.deltas == {}


def test_rate_is_delta_over_dt() -> None:
    w = ThroughputWindow()
    w.update(
        {"generation_tokens_total": 1000.0, "request_success_total": 100.0,
         "prompt_tokens_total": 5000.0}, 0.0)
    rates = w.update(
        {"generation_tokens_total": 1500.0, "request_success_total": 110.0,
         "prompt_tokens_total": 5200.0}, 5.0).rates
    assert rates["token_throughput_gen"] == pytest.approx(100.0)    # 500 / 5
    assert rates["request_throughput"] == pytest.approx(2.0)        # 10 / 5
    assert rates["token_throughput_prompt"] == pytest.approx(40.0)  # 200 / 5


def test_counter_reset_skips_window_then_recovers() -> None:
    w = ThroughputWindow()
    w.update({"generation_tokens_total": 1000.0}, 0.0)
    # Server restarted: counter went backwards → no rate this window.
    assert "token_throughput_gen" not in w.update({"generation_tokens_total": 5.0}, 5.0).rates
    # Baseline advanced, so the next window is clean.
    rates = w.update({"generation_tokens_total": 55.0}, 10.0).rates
    assert rates["token_throughput_gen"] == pytest.approx(10.0)  # (55 - 5) / 5


def test_nonpositive_dt_skips() -> None:
    w = ThroughputWindow()
    w.update({"generation_tokens_total": 1000.0}, 5.0)
    wu = w.update({"generation_tokens_total": 2000.0}, 5.0)  # dt == 0
    assert wu.rates == {}
    assert wu.deltas == {}


def test_only_present_counters_contribute() -> None:
    w = ThroughputWindow()
    w.update({"generation_tokens_total": 1000.0}, 0.0)
    rates = w.update({"generation_tokens_total": 1100.0}, 1.0).rates
    assert set(rates) == {"token_throughput_gen"}


# --------------------------------------------------------------------------- #
# Per-window counter deltas — the windowed r06 co-samples (schema 1.4.0).
# --------------------------------------------------------------------------- #

def test_deltas_are_per_window_increments() -> None:
    w = ThroughputWindow()
    w.update(
        {"num_preemptions_total": 3.0, "request_success_total": 100.0,
         "request_queue_time_ms_sum": 1000.0, "request_queue_time_count": 100.0}, 0.0)
    deltas = w.update(
        {"num_preemptions_total": 8.0, "request_success_total": 120.0,
         "request_queue_time_ms_sum": 2500.0, "request_queue_time_count": 110.0}, 5.0).deltas
    assert deltas["num_preemptions_total"] == pytest.approx(5.0)
    assert deltas["request_success_total"] == pytest.approx(20.0)
    assert deltas["request_queue_time_ms_sum"] == pytest.approx(1500.0)
    assert deltas["request_queue_time_count"] == pytest.approx(10.0)


def test_delta_reset_skips_only_that_counter() -> None:
    w = ThroughputWindow()
    w.update({"num_preemptions_total": 50.0, "request_success_total": 100.0}, 0.0)
    # Preemption counter went backwards (restart); successes kept climbing.
    deltas = w.update({"num_preemptions_total": 2.0, "request_success_total": 130.0}, 5.0).deltas
    assert "num_preemptions_total" not in deltas
    assert deltas["request_success_total"] == pytest.approx(30.0)


def test_deltas_only_for_counters_in_both_snapshots() -> None:
    w = ThroughputWindow()
    w.update({"num_preemptions_total": 3.0}, 0.0)
    deltas = w.update(
        {"num_preemptions_total": 4.0, "request_queue_time_ms_sum": 500.0}, 5.0).deltas
    assert set(deltas) == {"num_preemptions_total"}


# --------------------------------------------------------------------------- #
# ThroughputHistory — the rolling series r06 reads.
# --------------------------------------------------------------------------- #

def test_history_skips_tick_without_rate() -> None:
    # The first tick has no window rate (None) → nothing accumulates.
    h = ThroughputHistory()
    assert h.update(0.0, None) == []


def test_history_accumulates_in_order_with_corroborators() -> None:
    h = ThroughputHistory()
    h.update(0.0, 100.0, preemption_rate=0.0, num_requests_waiting=8.0)
    out = h.update(10.0, 90.0, preemption_rate=0.01, queue_time_ms=20.0)
    assert [s.token_throughput_gen for s in out] == [100.0, 90.0]
    assert out[-1].preemption_rate == pytest.approx(0.01)
    assert out[-1].queue_time_ms == pytest.approx(20.0)


def test_history_evicts_by_count() -> None:
    h = ThroughputHistory(max_samples=3)
    out: list = []
    for i in range(5):
        out = h.update(float(i), 100.0 - i)
    assert [s.t for s in out] == [2.0, 3.0, 4.0]


def test_history_evicts_by_age() -> None:
    h = ThroughputHistory(max_window_s=100.0)
    h.update(0.0, 100.0)
    h.update(50.0, 95.0)
    out = h.update(160.0, 80.0)   # cutoff = 60 → only t=160 survives
    assert [s.t for s in out] == [160.0]
