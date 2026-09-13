"""Unit tests for parsers/vllm.py.

The vLLM /metrics endpoint is the cheapest real data source, so the parser must
be robust to partial, malformed, and label-decorated Prometheus text. These
tests pin: histogram → Distribution maths, field extraction, the batch-size
semantics, and — load-bearing for the merge — the rule that TPOT is a per-token
latency and must NOT be written into decode.duration_ms.
"""

from __future__ import annotations

import math

import pytest

from parsers.vllm import (
    _HistogramAccumulator,
    _parse_labels,
    _safe_float,
    parse_vllm_metrics,
    parse_vllm_metrics_file,
)
from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# A small but complete /metrics dump: KV cache, 4 running requests, TTFT
# (sum=6.0, count=100 → mean 60ms) and TPOT (sum=2.0, count=100 → mean 20ms).
_METRICS = """\
# HELP vllm:gpu_cache_usage_perc KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="m"} 0.5
vllm:num_requests_running{model_name="m"} 4
vllm:time_to_first_token_seconds_bucket{le="0.05"} 50
vllm:time_to_first_token_seconds_bucket{le="0.10"} 100
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 100
vllm:time_to_first_token_seconds_count 100
vllm:time_to_first_token_seconds_sum 6.0
vllm:time_per_output_token_seconds_bucket{le="0.05"} 100
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 100
vllm:time_per_output_token_seconds_count 100
vllm:time_per_output_token_seconds_sum 2.0
"""


@pytest.fixture
def parsed() -> DiagnosisInput:
    return parse_vllm_metrics(_METRICS, model_name="Llama", gpu_type="A100-80G")


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_parse_labels_basic(self) -> None:
        assert _parse_labels('{a="1", b="two"}') == {"a": "1", "b": "two"}

    def test_parse_labels_empty(self) -> None:
        assert _parse_labels("") == {}

    def test_safe_float_rejects_nan_inf(self) -> None:
        assert _safe_float("nan") is None
        assert _safe_float("inf") is None
        assert _safe_float("-inf") is None

    def test_safe_float_rejects_garbage(self) -> None:
        assert _safe_float("not-a-number") is None

    def test_safe_float_parses_number(self) -> None:
        assert _safe_float("3.5") == 3.5


class TestHistogramAccumulator:
    def test_mean_from_sum_over_count_with_scale(self) -> None:
        acc = _HistogramAccumulator()
        acc.count, acc.sum = 100.0, 6.0
        dist = acc.to_distribution(scale=1000.0)
        assert dist is not None
        assert dist.mean == pytest.approx(60.0)  # 6.0/100 * 1000

    def test_none_when_count_missing_or_zero(self) -> None:
        acc = _HistogramAccumulator()
        assert acc.to_distribution() is None
        acc.count = 0.0
        assert acc.to_distribution() is None

    def test_inf_bucket_is_ignored(self) -> None:
        acc = _HistogramAccumulator()
        acc.add_bucket("+Inf", 100.0)
        assert acc.buckets == []

    def test_percentile_interpolated_between_bounds(self) -> None:
        acc = _HistogramAccumulator()
        acc.count = 100.0
        acc.add_bucket("0.05", 50.0)
        acc.add_bucket("0.10", 100.0)
        dist = acc.to_distribution(scale=1.0)
        # p50 target=50 lands exactly on the first bound edge.
        assert dist.p50 == pytest.approx(0.05)
        # p95 target=95 sits 45/50 of the way from 0.05 to 0.10.
        assert dist.p95 == pytest.approx(0.05 + (45 / 50) * 0.05)


# --------------------------------------------------------------------------- #
# Field extraction
# --------------------------------------------------------------------------- #

class TestFieldExtraction:
    def test_engine_is_vllm(self, parsed: DiagnosisInput) -> None:
        assert parsed.inference_engine == "vllm"

    def test_passthrough_model_and_gpu(self, parsed: DiagnosisInput) -> None:
        assert parsed.model_name == "Llama"
        assert parsed.gpu_type == "A100-80G"

    def test_kv_cache_util(self, parsed: DiagnosisInput) -> None:
        assert parsed.kv_cache_util == pytest.approx(0.5)

    def test_latency_distributions_scaled_to_ms(self, parsed: DiagnosisInput) -> None:
        assert parsed.ttft_ms.mean == pytest.approx(60.0)
        assert parsed.tpot_ms.mean == pytest.approx(20.0)

    def test_source_file_recorded(self, parsed: DiagnosisInput) -> None:
        assert parsed.source_files == ["<vllm_metrics>"]

    def test_comments_and_blank_lines_ignored(self) -> None:
        text = "# a comment\n\nvllm:gpu_cache_usage_perc 0.3\n"
        out = parse_vllm_metrics(text)
        assert out.kv_cache_util == pytest.approx(0.3)

    def test_nan_value_line_skipped(self) -> None:
        # A poisoned gauge must not reach the schema.
        out = parse_vllm_metrics("vllm:gpu_cache_usage_perc NaN\n")
        assert out.kv_cache_util is None


class TestBatchSize:
    def test_running_requests_become_batch_size(self, parsed: DiagnosisInput) -> None:
        assert parsed.batch_size == 4

    def test_zero_running_leaves_batch_none(self) -> None:
        out = parse_vllm_metrics("vllm:num_requests_running 0\n")
        assert out.batch_size is None  # not silently rounded up to 1

    def test_absent_running_leaves_batch_none(self) -> None:
        out = parse_vllm_metrics("vllm:gpu_cache_usage_perc 0.3\n")
        assert out.batch_size is None


# --------------------------------------------------------------------------- #
# Phase derivation — the merge-critical scoping contract
# --------------------------------------------------------------------------- #

class TestPhaseDerivation:
    def test_prefill_duration_from_ttft(self, parsed: DiagnosisInput) -> None:
        # TTFT is a whole-phase wall-clock, so it is a legitimate prefill duration.
        assert parsed.prefill.duration_ms == pytest.approx(60.0)
        assert parsed.prefill.latency_ms == pytest.approx(60.0)

    def test_decode_duration_is_none(self, parsed: DiagnosisInput) -> None:
        # TPOT is per-output-token, NOT a phase wall-clock. Writing it into
        # decode.duration_ms would mislabel it and, on merge, clobber Nsight's
        # kernel-measured duration. It must live only in latency_ms.
        assert parsed.decode.duration_ms is None
        assert parsed.decode.latency_ms == pytest.approx(20.0)

    def test_decode_carries_no_kernel_fields(self, parsed: DiagnosisInput) -> None:
        # vLLM cannot supply SM/HBM; those stay None so Nsight can enrich them.
        assert parsed.decode.sm_occupancy is None
        assert parsed.decode.hbm_bandwidth_util is None


# --------------------------------------------------------------------------- #
# Degenerate input
# --------------------------------------------------------------------------- #

class TestDegenerate:
    def test_unrecognised_text_warns(self) -> None:
        out = parse_vllm_metrics("totally unrelated content\n")
        assert any("no recognised metrics" in w for w in out.parse_warnings)

    def test_empty_text_warns_and_does_not_crash(self) -> None:
        out = parse_vllm_metrics("")
        assert isinstance(out, DiagnosisInput)
        assert out.parse_warnings  # at least one warning surfaced

    def test_file_wrapper_reads_and_labels_source(self, tmp_path) -> None:
        p = tmp_path / "metrics.prom"
        p.write_text(_METRICS)
        out = parse_vllm_metrics_file(str(p), model_name="M", gpu_type="A100-80G")
        assert out.source_files == [str(p)]
        assert out.kv_cache_util == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# KV-cache fragmentation estimate (feeds rule r03)
# --------------------------------------------------------------------------- #

# block_size=16 (from cache_config_info) + prompt-token mean 20 (sum 2000 / count
# 100). ceil(20/16)=2 blocks → allocated 32, waste 12 → fragmentation 12/32 = 0.375.
_FRAG_METRICS = """\
vllm:kv_cache_usage_perc{model_name="m"} 0.86
vllm:cache_config_info{block_size="16",cache_dtype="auto"} 1.0
vllm:request_prompt_tokens_bucket{le="10"} 0
vllm:request_prompt_tokens_bucket{le="50"} 100
vllm:request_prompt_tokens_bucket{le="+Inf"} 100
vllm:request_prompt_tokens_count 100
vllm:request_prompt_tokens_sum 2000
"""


class TestKvFragmentation:
    def test_block_size_from_cache_config_info(self) -> None:
        out = parse_vllm_metrics(_FRAG_METRICS)
        assert out.kv_block_size == 16

    def test_new_usage_metric_name_read(self) -> None:
        # vllm:kv_cache_usage_perc is the newer name for the gauge.
        out = parse_vllm_metrics(_FRAG_METRICS)
        assert out.kv_cache_util == pytest.approx(0.86)

    def test_fragmentation_estimate_computed(self) -> None:
        out = parse_vllm_metrics(_FRAG_METRICS)
        assert out.kv_cache_fragmentation == pytest.approx(0.375, abs=1e-3)
        assert 0.0 < out.kv_cache_fragmentation < 1.0

    def test_fragmentation_none_without_block_size(self) -> None:
        # The plain dump has no cache_config_info and no prompt histogram.
        out = parse_vllm_metrics(_METRICS)
        assert out.kv_cache_fragmentation is None
        assert out.kv_block_size is None

    def test_fragmentation_none_without_seq_len(self) -> None:
        # block_size present but no prompt histogram → no sequence-length mean.
        text = 'vllm:cache_config_info{block_size="16"} 1.0\nvllm:kv_cache_usage_perc 0.9\n'
        out = parse_vllm_metrics(text)
        assert out.kv_block_size == 16
        assert out.kv_cache_fragmentation is None

    def test_bad_block_size_label_ignored(self) -> None:
        text = 'vllm:cache_config_info{block_size="not-a-number"} 1.0\n'
        out = parse_vllm_metrics(text)
        assert out.kv_block_size is None


# A dump carrying the serving-layer counters r02 consumes. request_success_total
# is split across two finished_reason labels (210 + 90 = 300) to pin the
# counter-summation path; running/waiting are single-series gauges.
_SERVING = """\
vllm:num_requests_running 4
vllm:num_requests_waiting 6
vllm:num_preemptions_total 9
vllm:request_success_total{finished_reason="stop"} 210
vllm:request_success_total{finished_reason="length"} 90
vllm:prompt_tokens_total 420000
vllm:generation_tokens_total 980000
"""


class TestServingMetrics:
    def test_counters_summed_across_labels(self) -> None:
        out = parse_vllm_metrics(_SERVING)
        assert out.vllm_serving is not None
        # 210 + 90 across the two finished_reason label sets.
        assert out.vllm_serving.request_success_total == 300

    def test_counter_and_gauge_fields_extracted(self) -> None:
        s = parse_vllm_metrics(_SERVING).vllm_serving
        assert s.num_preemptions_total == 9
        assert s.prompt_tokens_total == 420000
        assert s.generation_tokens_total == 980000
        assert s.num_requests_running == 4
        assert s.num_requests_waiting == 6

    def test_values_coerced_to_int(self) -> None:
        # Prometheus values parse as floats; the block stores ints.
        s = parse_vllm_metrics(_SERVING).vllm_serving
        assert isinstance(s.num_preemptions_total, int)
        assert isinstance(s.request_success_total, int)

    def test_connector_not_set_from_metrics(self) -> None:
        # kv_transfer_connector is launch-config, never present in /metrics.
        s = parse_vllm_metrics(_SERVING).vllm_serving
        assert s.kv_transfer_connector is None

    def test_absent_block_is_none(self) -> None:
        # No serving counters/gauges at all -> no block (r02 then abstains).
        out = parse_vllm_metrics("vllm:gpu_cache_usage_perc 0.3\n")
        assert out.vllm_serving is None

    def test_v1_counter_names_without_total_suffix(self) -> None:
        # vLLM V1 drops the `_total` suffix from counters; the same dump with
        # V1 names must populate the same schema fields (which keep the
        # canonical *_total names — schema is stable, wire names alias).
        v1 = _SERVING.replace("_total", "")
        s = parse_vllm_metrics(v1).vllm_serving
        assert s is not None
        assert s.num_preemptions_total == 9
        assert s.request_success_total == 300
        assert s.prompt_tokens_total == 420000
        assert s.generation_tokens_total == 980000

    def test_v0_name_wins_are_not_double_counted(self) -> None:
        # A dump carrying both names (deprecation overlap window) must read one,
        # not sum the pair; the V1 name is preferred.
        both = "vllm:num_preemptions 9\nvllm:num_preemptions_total 7\n"
        s = parse_vllm_metrics(both).vllm_serving
        assert s.num_preemptions_total == 9

    def test_partial_block_still_built(self) -> None:
        out = parse_vllm_metrics("vllm:num_preemptions_total 5\n")
        assert out.vllm_serving is not None
        assert out.vllm_serving.num_preemptions_total == 5
        assert out.vllm_serving.request_success_total is None

    def test_queue_time_raw_totals_extracted(self) -> None:
        # Schema 1.4.0: the raw histogram totals ride along (seconds → ms for the
        # sum) so the live loop can difference consecutive scrapes into a
        # current-window mean queue time.
        text = (
            "vllm:request_queue_time_seconds_sum 1.5\n"
            "vllm:request_queue_time_seconds_count 30\n"
        )
        s = parse_vllm_metrics(text).vllm_serving
        assert s is not None
        assert s.request_queue_time_ms_sum == 1500.0
        assert s.request_queue_time_count == 30.0


# --------------------------------------------------------------------------- #
# Two-scrape window correction (feeds rule r03's current-pressure signal)
# --------------------------------------------------------------------------- #

# Two scrapes of one server. Cumulative metrics grew between them; the WINDOW
# (current - baseline) describes current load, not the diluted lifetime average.
#   prompt tokens : Δsum 400 / Δcount 20  → window mean 20 → frag(block16) 0.375
#   queue time    : Δsum 0.42 / Δcount 20 → window mean 21 ms
#   preemptions   : 13 - 10 = 3 ;  successes : 1200 - 1000 = 200
_WIN_BASELINE = """\
vllm:cache_config_info{block_size="16"} 1.0
vllm:kv_cache_usage_perc 0.50
vllm:request_prompt_tokens_count 100
vllm:request_prompt_tokens_sum 3000
vllm:request_queue_time_seconds_count 100
vllm:request_queue_time_seconds_sum 1.0
vllm:num_preemptions_total 10
vllm:request_success_total 1000
"""

_WIN_CURRENT = """\
vllm:cache_config_info{block_size="16"} 1.0
vllm:kv_cache_usage_perc 0.62
vllm:request_prompt_tokens_count 120
vllm:request_prompt_tokens_sum 3400
vllm:request_queue_time_seconds_count 120
vllm:request_queue_time_seconds_sum 1.42
vllm:num_preemptions_total 13
vllm:request_success_total 1200
"""


class TestWindowDelta:
    def test_single_scrape_is_lifetime(self) -> None:
        out = parse_vllm_metrics(_WIN_CURRENT)
        assert out.pressure_window == "lifetime"
        # Lifetime prompt-token mean 3400/120 ≈ 28.3 → frag ≈ 0.115.
        assert out.kv_cache_fragmentation == pytest.approx(0.1146, abs=1e-3)
        # Lifetime queue mean 1.42/120 s ≈ 11.8 ms.
        assert out.queue_time_ms.mean == pytest.approx(11.83, abs=0.1)
        assert out.vllm_serving.num_preemptions_total == 13
        assert out.vllm_serving.request_success_total == 1200

    def test_two_scrapes_difference_to_window(self) -> None:
        out = parse_vllm_metrics(_WIN_CURRENT, baseline_text=_WIN_BASELINE)
        assert out.pressure_window == "delta"
        # Window prompt-token mean 400/20 = 20 → frag 12/32 = 0.375.
        assert out.kv_cache_fragmentation == pytest.approx(0.375, abs=1e-3)
        # Window queue mean 0.42/20 s = 21 ms (the falsified-sweep value).
        assert out.queue_time_ms.mean == pytest.approx(21.0, abs=0.1)
        # Window counters, not lifetime totals.
        assert out.vllm_serving.num_preemptions_total == 3
        assert out.vllm_serving.request_success_total == 200

    def test_gauges_use_current_scrape_not_delta(self) -> None:
        # kv_cache_usage is a point-in-time gauge: keep the current value (0.62),
        # never difference it.
        out = parse_vllm_metrics(_WIN_CURRENT, baseline_text=_WIN_BASELINE)
        assert out.kv_cache_util == pytest.approx(0.62)

    def test_counter_reset_clamps_to_zero(self) -> None:
        # If "current" is the smaller scrape (server restarted between captures),
        # deltas clamp to 0 rather than going negative; a zero window count drops
        # the distribution entirely.
        out = parse_vllm_metrics(_WIN_BASELINE, baseline_text=_WIN_CURRENT)
        assert out.pressure_window == "delta"
        assert out.vllm_serving.num_preemptions_total == 0
        assert out.queue_time_ms is None
        assert out.kv_cache_fragmentation is None
