"""
vLLM /metrics parser.

Parses the Prometheus-format text endpoint that vLLM exposes at /metrics.
This is the easiest real data source: scrape it live, or paste the output
from a benchmark run.

Input:  raw text from vLLM's /metrics endpoint (or a saved .prom file)
Output: DiagnosisInput (partially filled — no Nsight fields)

vLLM metric reference (as of vLLM 0.4+):
  vllm:num_requests_running          - requests in-flight
  vllm:num_requests_waiting          - requests in queue
  vllm:kv_cache_usage_perc           - KV cache utilisation [0,1] (newer name)
  vllm:gpu_cache_usage_perc          - KV cache utilisation [0,1] (legacy name)
  vllm:cache_config_info             - info gauge; block_size exposed as a label
  vllm:cpu_cache_usage_perc          - CPU KV cache utilisation [0,1]
  vllm:request_success_total         - counter (V1 drops _total: vllm:request_success)
  vllm:prompt_tokens_total           - counter (V1: vllm:prompt_tokens)
  vllm:generation_tokens_total       - counter (V1: vllm:generation_tokens)
  vllm:num_preemptions_total         - counter (V1: vllm:num_preemptions)
  vllm:time_to_first_token_seconds   - histogram
  vllm:time_per_output_token_seconds - histogram
  vllm:e2e_request_latency_seconds   - histogram
  vllm:request_prompt_tokens         - histogram (per-request prompt length)
  vllm:request_generation_tokens     - histogram (per-request gen length)

Histogram suffixes: _bucket, _count, _sum
Summary statistics are computed from _sum / _count for mean,
and from bucket boundaries for p50/p95/p99.
"""

from __future__ import annotations

import re
import math
from typing import Optional

from schema import DiagnosisInput, PhaseMetrics, Distribution, VllmServingMetrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches a Prometheus metric line: name{labels} value [timestamp]
_METRIC_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[^\s]+)'
)

def _parse_labels(label_str: str) -> dict[str, str]:
    """Parse '{key="val", key2="val2"}' → {'key': 'val', 'key2': 'val2'}"""
    result: dict[str, str] = {}
    if not label_str:
        return result
    inner = label_str.strip("{}")
    for part in re.findall(r'(\w+)="([^"]*)"', inner):
        result[part[0]] = part[1]
    return result


def _safe_float(s: str) -> Optional[float]:
    try:
        v = float(s)
        return None if math.isnan(v) or math.isinf(v) else v
    except (ValueError, TypeError):
        return None


def _estimate_internal_fragmentation(
    block_size: Optional[int], seq_len_mean: Optional[float]
) -> Optional[float]:
    """Estimate internal KV-cache fragmentation from paged-block granularity.

    A sequence of ``seq_len_mean`` tokens occupies ``ceil(L / B)`` blocks of ``B``
    slots; its final block wastes ``ceil(L/B)*B - L`` slots. The wasted *fraction*
    is the bounded internal-fragmentation residual that remains after
    PagedAttention removes external/reservation waste (Kwon et al., SOSP 2023).

    This is an *estimate*, not a measurement: vLLM /metrics exports neither
    external fragmentation (PagedAttention drives it to ~0) nor per-block
    occupancy. It uses the prompt-token mean as a stand-in for total context and
    computes from the mean rather than integrating over the distribution. Returns
    None unless both inputs are usable.
    """
    if block_size is None or block_size < 1:
        return None
    if seq_len_mean is None or seq_len_mean <= 0:
        return None
    allocated = math.ceil(seq_len_mean / block_size) * block_size
    if allocated <= 0:
        return None
    return max(0.0, min(1.0, (allocated - seq_len_mean) / allocated))


class _HistogramAccumulator:
    """
    Collects _bucket / _count / _sum lines for one histogram metric,
    then computes mean + percentile estimates via linear interpolation.
    """

    def __init__(self):
        self.buckets: list[tuple[float, float]] = []  # (le_bound, cumulative_count)
        self.count: Optional[float] = None
        self.sum: Optional[float] = None

    def add_bucket(self, le: str, value: float) -> None:
        if le == "+Inf":
            return
        bound = _safe_float(le)
        if bound is not None:
            self.buckets.append((bound, value))

    def to_distribution(self, scale: float = 1.0) -> Optional[Distribution]:
        """
        scale: multiply all values by this factor (e.g. 1000 to convert s → ms)
        Returns None if not enough data.
        """
        if self.count is None or self.count == 0:
            return None

        mean = (self.sum / self.count * scale) if self.sum is not None else None

        sorted_buckets = sorted(self.buckets, key=lambda x: x[0])
        total = self.count

        def interpolate_pct(pct: float) -> Optional[float]:
            target = pct * total
            prev_bound, prev_count = 0.0, 0.0
            for bound, count in sorted_buckets:
                if count >= target:
                    if count == prev_count:
                        return bound * scale
                    frac = (target - prev_count) / (count - prev_count)
                    return (prev_bound + frac * (bound - prev_bound)) * scale
                prev_bound, prev_count = bound, count
            return None

        return Distribution(
            mean=mean if mean is not None else 0.0,
            p50=interpolate_pct(0.50),
            p95=interpolate_pct(0.95),
            p99=interpolate_pct(0.99),
        )

    def subtract(self, baseline: "_HistogramAccumulator") -> "_HistogramAccumulator":
        """Return a new accumulator holding this scrape minus an earlier one.

        Differences sum, count, and every bucket so the result describes only the
        requests that completed in the window between the two scrapes — turning a
        lifetime cumulative histogram into a current-window one. Negative results
        (a counter reset / server restart between scrapes) clamp to 0; a zero
        window count then makes ``to_distribution`` return None, which correctly
        reads as "no traffic in the window."
        """
        result = _HistogramAccumulator()
        if self.sum is not None:
            result.sum = max(0.0, self.sum - (baseline.sum or 0.0))
        if self.count is not None:
            result.count = max(0.0, self.count - (baseline.count or 0.0))
        base_counts = {bound: c for bound, c in baseline.buckets}
        for bound, count in self.buckets:
            result.buckets.append((bound, max(0.0, count - base_counts.get(bound, 0.0))))
        return result


def _build_serving_metrics(
    scalars: dict[str, float],
    counters: dict[str, float],
    histograms: dict[str, "_HistogramAccumulator"],
) -> Optional[VllmServingMetrics]:
    """Assemble the vLLM serving-metrics block from parsed scalars/counters.

    Counters (cumulative) are read from the summed `counters` table; gauges
    (point-in-time) from `scalars`; the raw queue-time histogram totals
    (`_sum`/`_count`, cumulative) from `histograms` — the live watch loop
    differences those totals between scrapes into a current-window mean queue
    time (schema 1.4.0). `kv_transfer_connector` is intentionally not
    set here: it is launch-config, not a /metrics value, so topology is inferred
    downstream from dump shape. Returns None when none of the fields are present,
    so r02 abstains with INSUFFICIENT_DATA rather than seeing an empty block.
    """

    def counter(*names: str) -> Optional[int]:
        # First match wins. vLLM V1 dropped the `_total` suffix from counter
        # names (vllm:num_preemptions_total → vllm:num_preemptions); accept
        # both, newest first — the same rename-tolerance pattern as the
        # kv_cache_usage gauge.
        for name in names:
            v = counters.get(name)
            if v is not None:
                return int(v)
        return None

    def gauge(name: str) -> Optional[int]:
        v = scalars.get(name)
        return int(v) if v is not None else None

    num_preemptions = counter("vllm:num_preemptions", "vllm:num_preemptions_total")
    request_success = counter("vllm:request_success", "vllm:request_success_total")
    prompt_tokens = counter("vllm:prompt_tokens", "vllm:prompt_tokens_total")
    generation_tokens = counter("vllm:generation_tokens", "vllm:generation_tokens_total")
    running = gauge("vllm:num_requests_running")
    waiting = gauge("vllm:num_requests_waiting")

    queue_hist = histograms.get("vllm:request_queue_time_seconds")
    queue_ms_sum = (
        queue_hist.sum * 1000.0 if queue_hist is not None and queue_hist.sum is not None else None
    )
    queue_count = queue_hist.count if queue_hist is not None else None

    if all(
        v is None
        for v in (num_preemptions, request_success, prompt_tokens,
                  generation_tokens, running, waiting, queue_ms_sum, queue_count)
    ):
        return None

    return VllmServingMetrics(
        num_preemptions_total=num_preemptions,
        request_success_total=request_success,
        prompt_tokens_total=prompt_tokens,
        generation_tokens_total=generation_tokens,
        num_requests_running=running,
        num_requests_waiting=waiting,
        request_queue_time_ms_sum=queue_ms_sum,
        request_queue_time_count=queue_count,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _accumulate(
    text: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, _HistogramAccumulator], dict[str, str]]:
    """Parse Prometheus text into (scalars, counters, histograms, config_labels).

    scalars      — last-wins, correct for single-sample gauges.
    counters     — summed across label partitions, correct for counters split by
                   label (e.g. request_success_total per finished_reason).
    histograms   — one accumulator per histogram base name.
    config_labels — merged labels from vllm:cache_config_info (block_size etc.).
    """
    scalars: dict[str, float] = {}
    # Counters summed across label sets: vLLM splits some counters by label
    # (e.g. request_success_total{finished_reason=...}), so the same metric name
    # appears on several lines that must be added, not overwritten. Gauges stay
    # in `scalars` (last-wins) since they are single-series snapshots.
    counters: dict[str, float] = {}
    histograms: dict[str, _HistogramAccumulator] = {}
    config_labels: dict[str, str] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        m = _METRIC_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        labels = _parse_labels(m.group("labels") or "")
        value = _safe_float(m.group("value"))
        if value is None:
            continue

        # Info gauge: value is a constant 1.0; the payload lives in the labels.
        if name == "vllm:cache_config_info":
            config_labels.update(labels)
            continue

        # Histogram buckets
        if name.endswith("_bucket"):
            base = name[: -len("_bucket")]
            histograms.setdefault(base, _HistogramAccumulator()).add_bucket(
                labels.get("le", "+Inf"), value
            )
        elif name.endswith("_count"):
            base = name[: -len("_count")]
            histograms.setdefault(base, _HistogramAccumulator()).count = value
        elif name.endswith("_sum"):
            base = name[: -len("_sum")]
            histograms.setdefault(base, _HistogramAccumulator()).sum = value
        else:
            scalars[name] = value
            counters[name] = counters.get(name, 0.0) + value

    return scalars, counters, histograms, config_labels


def _delta_counters(
    current: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    """Difference cumulative counters to the window between two scrapes.

    Counters only grow, so current - baseline is the window's contribution; a
    negative result means the server restarted between scrapes — clamp to 0.
    """
    return {
        name: max(0.0, value - baseline.get(name, 0.0))
        for name, value in current.items()
    }


def _delta_histograms(
    current: dict[str, _HistogramAccumulator],
    baseline: dict[str, _HistogramAccumulator],
) -> dict[str, _HistogramAccumulator]:
    """Difference each cumulative histogram to the window between two scrapes."""
    out: dict[str, _HistogramAccumulator] = {}
    for name, acc in current.items():
        base = baseline.get(name)
        out[name] = acc.subtract(base) if base is not None else acc
    return out


def parse_vllm_metrics(
    text: str,
    model_name: str = "unknown",
    gpu_type: str = "unknown",
    source_file: str = "<vllm_metrics>",
    baseline_text: Optional[str] = None,
) -> DiagnosisInput:
    """
    Parse raw vLLM /metrics text into a DiagnosisInput.

    Args:
        text:         Raw Prometheus text (the current / later scrape).
        model_name:   Passed through; vLLM doesn't expose this in /metrics.
        gpu_type:     Passed through; not in /metrics.
        source_file:  Label for provenance tracking.
        baseline_text: An optional earlier scrape. When given, every CUMULATIVE
            metric (histograms, counters) is differenced against it so the
            cache/queue/preemption signals describe the *current window* rather
            than the server's lifetime, and `pressure_window` is set to "delta".
            Point-in-time gauges (utilisation, queue depth) always use the
            current scrape.

    Returns:
        DiagnosisInput with KV cache, throughput, and latency fields populated.
        Nsight-specific fields (sm_occupancy, hbm_bandwidth_util, etc.) are None.
    """
    scalars, counters, histograms, config_labels = _accumulate(text)
    warnings: list[str] = []

    # Window correction: difference cumulative metrics against an earlier scrape so
    # they reflect current pressure, not a diluted lifetime average. Gauges stay on
    # the current scrape (they are point-in-time already).
    pressure_window = "lifetime"
    if baseline_text is not None:
        _, b_counters, b_histograms, _ = _accumulate(baseline_text)
        counters = _delta_counters(counters, b_counters)
        histograms = _delta_histograms(histograms, b_histograms)
        pressure_window = "delta"

    # ------------------------------------------------------------------
    # Extract fields
    # ------------------------------------------------------------------

    # KV cache utilisation. vLLM renamed the gauge from gpu_cache_usage_perc to
    # kv_cache_usage_perc; accept both.
    # Throughput counters are cumulative; rates require two snapshots, so we
    # also leave request/token throughput unset from a single /metrics dump.
    kv_cache_util = scalars.get("vllm:kv_cache_usage_perc")
    if kv_cache_util is None:
        kv_cache_util = scalars.get("vllm:gpu_cache_usage_perc")

    # Block size is exposed only as a label on cache_config_info, never as a
    # standalone metric. Capture it for the fragmentation estimate and fix text.
    kv_block_size: Optional[int] = None
    bs_raw = config_labels.get("block_size")
    if bs_raw is not None:
        try:
            bs = int(float(bs_raw))
            kv_block_size = bs if bs >= 1 else None
        except (ValueError, TypeError):
            kv_block_size = None

    # vLLM serving-layer counters/gauges (preemptions, request/token totals,
    # queue depth) that r02 and future serving-diagnosis rules consume. Counters
    # were summed across label partitions in _accumulate and, with a baseline
    # scrape, differenced to the current window above; gauges stay point-in-time.
    # r03 leans on these cumulative signals because the kv_cache_usage gauge
    # sawtooths and reads low between bursts, while a windowed counter reflects
    # *current* pressure. The raw queue-time histogram totals ride along so the
    # live loop can difference them between scrapes (schema 1.4.0).
    vllm_serving = _build_serving_metrics(scalars, counters, histograms)

    # Latency distributions (convert seconds → ms)
    ttft_dist = histograms.get("vllm:time_to_first_token_seconds")
    tpot_dist = histograms.get("vllm:time_per_output_token_seconds")
    e2e_dist = histograms.get("vllm:e2e_request_latency_seconds")

    ttft_ms = ttft_dist.to_distribution(scale=1000.0) if ttft_dist else None
    tpot_ms = tpot_dist.to_distribution(scale=1000.0) if tpot_dist else None
    e2e_latency_ms = e2e_dist.to_distribution(scale=1000.0) if e2e_dist else None

    # Queue time: a cumulative histogram. With a baseline scrape it was differenced
    # to the current window above; from a single scrape its mean is a lifetime
    # average (see pressure_window). It is the capacity-pressure signal r03 relies
    # on, and the one that moves on vLLM V1, which queues rather than preempting.
    queue_dist = histograms.get("vllm:request_queue_time_seconds")
    queue_time_ms = queue_dist.to_distribution(scale=1000.0) if queue_dist else None

    # Derive PhaseMetrics for prefill / decode from TTFT / TPOT.
    #
    # TTFT is a whole-phase wall-clock (queue + prefill compute), so it is a
    # reasonable proxy for prefill.duration_ms. TPOT is *per output token*, not
    # a phase duration: storing it in decode.duration_ms would mislabel a
    # per-token latency as the decode-phase wall-clock and, on merge, clobber
    # Nsight's kernel-measured duration. Keep TPOT in latency_ms only (where the
    # schema documents it belongs); let Nsight own decode.duration_ms.
    prefill_metrics = None
    if ttft_ms is not None:
        prefill_metrics = PhaseMetrics(
            latency_ms=ttft_ms.mean,
            duration_ms=ttft_ms.mean,
        )

    decode_metrics = None
    if tpot_ms is not None:
        decode_metrics = PhaseMetrics(
            latency_ms=tpot_ms.mean,
        )

    # Sequence length from vLLM prompt/gen token histograms
    seq_len_dist = None
    prompt_hist = histograms.get("vllm:request_prompt_tokens")
    if prompt_hist:
        seq_len_dist = prompt_hist.to_distribution(scale=1.0)

    # Internal-fragmentation estimate from block granularity x mean sequence
    # length. None unless both block size and a sequence-length mean are present
    # (e.g. the bare /metrics example has neither).
    kv_cache_fragmentation = _estimate_internal_fragmentation(
        kv_block_size, seq_len_dist.mean if seq_len_dist is not None else None
    )

    # Batch size: number of in-flight requests. 0 means no batch active;
    # leave batch_size unset in that case rather than silently rounding to 1.
    running = scalars.get("vllm:num_requests_running")
    batch_size = int(running) if running is not None and running >= 1 else None

    # Validate we got something useful
    has_any = any([
        kv_cache_util is not None,
        ttft_ms is not None,
        tpot_ms is not None,
        e2e_latency_ms is not None,
        vllm_serving is not None,
    ])
    if not has_any:
        warnings.append(
            "vLLM parser: no recognised metrics found. "
            "Check that the input is from vLLM's /metrics endpoint."
        )

    return DiagnosisInput(
        model_name=model_name,
        gpu_type=gpu_type,
        inference_engine="vllm",
        batch_size=batch_size,
        seq_len_distribution=seq_len_dist,
        prefill=prefill_metrics,
        decode=decode_metrics,
        kv_cache_util=kv_cache_util,
        vllm_serving=vllm_serving,
        kv_cache_fragmentation=kv_cache_fragmentation,
        kv_block_size=kv_block_size,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_latency_ms=e2e_latency_ms,
        queue_time_ms=queue_time_ms,
        pressure_window=pressure_window,
        source_files=[source_file],
        parse_warnings=warnings,
    )


def parse_vllm_metrics_file(
    path: str, baseline_path: Optional[str] = None, **kwargs
) -> DiagnosisInput:
    """Convenience wrapper: read a .prom file (and optional earlier baseline) and parse it.

    When ``baseline_path`` is given it is read as the earlier scrape so cumulative
    metrics are differenced to the current window (see ``parse_vllm_metrics``).
    """
    with open(path, "r") as f:
        text = f.read()
    baseline_text = None
    if baseline_path is not None:
        with open(baseline_path, "r") as f:
            baseline_text = f.read()
    return parse_vllm_metrics(
        text, source_file=path, baseline_text=baseline_text, **kwargs
    )