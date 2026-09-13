"""
strided canonical schema.

THIS IS THE CONTRACT.
Do not change field names, types, or semantics without maintainer review.
Parsers write to this. Rules read from it. Nothing else crosses the boundary.

Version: 1.8.0

Changelog:
  1.8.0 — Per-layer phase attribution (`LayerMetrics.phase`, additive, Optional).
          Added:
            - `LayerMetrics.phase` — `prefill | decode | mixed | None`, the
              inference phase the kernel's launches ran under. Typed against the
              existing `EngineStepPhase` alias, which is the same vocabulary
              `EngineStep.phase` already uses: a layer's phase and the step it
              ran inside must be nameable in one language or the join between
              them is not checkable.
          What this fixes, stated precisely, because an earlier draft of this
          entry got it wrong. `parsers/nsight.py` does NOT assign its aggregate
          to both phases. It constructs `DiagnosisInput(decode=aggregate, ...)`
          and leaves `prefill` unset — so *every* kernel, prefill kernels
          included, lands in the decode slot. `decode` today is therefore a
          MISATTRIBUTION, not a shared value carrying a vague label: prefill
          work is being counted as decode work and no field can say so. That is
          a second contaminant of r01's reading, independent of the scope
          dilution measured in field testing (the pathological
          kernel reads 0.795 HBM utilisation, the flat aggregate 0.595). Scope
          dilution mixes the wrong *kernels* into the population; phase
          misattribution mixes the wrong *phase* in. Fixing either alone leaves
          the other, and neither was previously expressible.
          Why a field rather than a parser change alone: the phase is a property
          OF THE KERNEL, parsed from the trace, so it belongs beside the
          kernel's other parsed properties. Storing it lets a rule scope to a
          phase instead of trusting that whatever is in `decode` is
          decode.
          None is a first-class value and the common one. A stock `ncu --csv
          --page raw` export carries NO phase annotation and NO per-kernel
          timestamp on the trace clock, so there is nothing to correlate against
          the NVTX step ranges in `nsys_timeline` — see the correlation notes in
          `parsers/nsight.py`. The parser populates `phase` only when the export
          actually carries the evidence (an ncu `--nvtx` range column, or
          explicit numeric launch timestamps plus a timeline to join against)
          and leaves it None otherwise, with a warning naming the limitation. An
          honest None is worth more than a guessed tag; guessing by kernel name
          (attention -> prefill, MLP -> decode) is simply wrong, because both
          phases run both kernel families.
          Purely additive: `phase` is Optional with a None default, so every
          1.7.0 input validates unchanged, no shipped rule reads it, and the
          phase-less parse path is byte-identical to 1.7.0's (aggregate still
          goes to `decode`, `prefill` still stays None — the misattribution is
          now DESCRIBED rather than silently repaired, because repairing it
          without evidence would be the same guess in the other direction).
  1.7.0 — Nsight Systems timeline telemetry (`NsysTimeline`, surfaced as
          `DiagnosisInput.nsys_timeline`). Every prior block is a single
          aggregate *snapshot* — phase means, cumulative counters, latency
          distributions — with no wall-clock placement, so nothing in the schema
          can express *when* prefill and decode actually run relative to each
          other. r08 (prefill↔decode interference) needs exactly that: a time-
          resolved view of the engine's scheduler steps to see long prefills
          stalling the decode cadence and spiking TTFT. `NsysTimeline.steps` is a
          list of `EngineStep` intervals (start/end in ms, a `prefill|decode|mixed`
          phase tag, and optional token/seq counts) parsed verbatim from the NVTX
          step ranges in an `nsys` trace — RAW telemetry, never a derived overlap
          or stall figure (r08 computes those itself; see
          schema-keeps-only-parsed-telemetry). Purely additive: `nsys_timeline`
          is Optional, so every 1.6.0 input validates unchanged. `EngineStep` and
          `NsysTimeline` adopt the same `allow_inf_nan=False` hardening 1.5.0 gave
          every other model, so a non-finite step bound is rejected at
          construction rather than leaking a NaN duration into a diagnosis.
          Block shape: NVTX step intervals over raw kernel spans, not a
          pre-summarised overlap block.
  1.6.0 — Per-rank clock history for persistence-gated straggler detection
          (additive, Optional).
          Added:
            - `TpRankSample` — one tick of per-rank SM clocks (+ optional
              temps), the unit of `tp_rank_history`.
            - `tp_rank_history` — the rolling per-tick series the live `watch`
              loop appends; r04 reads it to require that the SAME rank is the
              outlier across consecutive ticks.
          Motivation (field-measured): a single SM-clock
          snapshot cannot distinguish a thermally-throttled rank from one that
          merely DVFS'd to the idle floor between load bursts. Both read as
          "slow", and r04 fired on 16/100 ticks against a HEALTHY 3-replica
          pack, blaming ranks near-uniformly (6/6/4) — the signature of noise
          rather than detection. Replaying the same captures, a persistence
          gate of 2 consecutive ticks reduces that to 0/100.
          Raw samples only: the schema stores no outlier or persistence verdict.
          Purely additive — 1.5.0 inputs validate unchanged, and the one-shot
          path leaves it None (a static dump has no transient to confuse).
  1.5.0 — Contract hardening: non-finite floats are rejected at construction
          (`allow_inf_nan=False` on every model). No field added or renamed; the
          change is validation semantics only — inputs carrying NaN/Inf, which
          previously validated, now fail loudly at the boundary. Why: comparison
          -based validators are blind to NaN (every comparison returns False),
          so `Distribution(mean=nan)` passed the `v < 0` check, and adversarial
          testing demonstrated a fired r02 diagnosis carrying `tpot_tail_ratio: nan`
          into the final report, and an Inf queue-time mean reading as queue
          pressure in r03. The parser boundary already drops non-finite
          (`parsers/_prom.py`), so no real scrape path changes behavior; this
          closes the programmatic/future-parser hole. Also corrected:
          `PhaseMetrics.clamp_utilisation` clamps finite overshoot only — NaN
          previously clamped to 1.0 (Python min/max ordering), silently reading
          garbage as full utilisation. Direction pre-recorded in
          `engine/runner.py`'s `_scan_non_finite` docstring as the intended fix.
  1.4.0 — Sustained-run throughput history + windowed co-samples (additive, all
          Optional).
          Added:
            - `ThroughputSample` — one tick of a sustained run: the window
              generation throughput (tok/s) plus the co-sampled corroborators r06
              needs (preemption rate, queue time, waiting backlog, SM clock, temp).
            - `throughput_history` — the rolling per-tick series the live `watch`
              loop appends to (oldest→newest); r06 (throughput decay) derives a
              decay slope and its significance from it. Raw samples only — the rule
              derives the trend, the schema stores no slope (same parsers-write-raw,
              rules-derive split as `tp_rank_sm_clocks`/`slowness_from_sm_clocks`).
              Only the live path populates it; the one-shot scrape has no time base,
              so r06 abstains there.
          Added to `VllmServingMetrics`:
            - `request_queue_time_ms_sum` — cumulative `_sum` of vLLM's
              `request_queue_time_seconds` histogram, converted to ms.
            - `request_queue_time_count` — cumulative `_count` of the same
              histogram.
          Both are raw Prometheus telemetry (the exporter emits `_sum`/`_count`
          directly); storing them lets the live watch loop difference two scrapes
          into a *current-window* mean queue time, exactly as r03's baseline path
          differences the full histogram. Motivation: fed lifetime ratios/means, the
          ThroughputSample co-samples would be diluted by a long-running server's
          past bursts — the r06 mechanism leg could miss genuine current pressure.
          So the watch loop feeds `ThroughputSample.preemption_rate` and
          `.queue_time_ms` with per-window deltas (see those field comments).
          Semantics-safe: r06 tests only the *direction* of the trend, and the
          first tick never yields a sample, so every appended sample is windowed.
          Purely additive: 1.3.0 inputs validate unchanged.
  1.3.0 — Tensor-parallel + queue telemetry, and a contract correction.
          Added (raw telemetry, all Optional — additive):
            - `tp_rank_sm_clocks` — per-rank SM clock (MHz) from DCGM
              `DCGM_FI_DEV_SM_CLOCK`; the straggler signal r04 reads.
            - `tp_rank_temps` — per-rank temperature (°C) from
              `DCGM_FI_DEV_GPU_TEMP`; r04's thermal-throttle corroborator.
            - `queue_time_ms` — request queue-time distribution from vLLM's
              `request_queue_time_seconds` histogram (see field comment).
            - `pressure_window` — provenance ("lifetime" vs "delta") of the
              cumulative-derived signals (queue time, preemption rate, the
              fragmentation estimate); r03 reads it to cap lifetime-only firings.
          REMOVED (breaking): `tp_rank_timings`. It was a 1.0.0/1.1.0 placeholder
            typed "ms" but never populated with real per-rank step times; r04's
            DCGM path had begun reinterpreting it as a derived `max_clock / clock`
            *ratio*. A derived ratio is not raw telemetry and so does not belong in
            the schema (parsers write raw, rules derive) — the computation now lives
            in r04 (`rules/_stats.slowness_from_sm_clocks`), reading the raw
            `tp_rank_sm_clocks` above. No parser or shipped consumer depended on the
            old ms field.
  1.1.0 — Completed the vLLM serving-metrics block (`VllmServingMetrics`,
          surfaced as `DiagnosisInput.vllm_serving`). The schema previously
          modelled vLLM with only KV-cache state, which under-represented what
          /metrics exposes; serving counters (preemptions, request/token totals,
          queue depth) are bread-and-butter signals for serving-diagnosis rules.
          Purely additive: every new field is Optional, so 1.0.0 inputs validate
          unchanged. Customer-policy values (latency SLOs) are deliberately NOT
          modelled here — they are not parsed from any dump.
  1.0.0 — Initial contract.
"""

from __future__ import annotations

import math
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Contract hardening (see 1.5.0 changelog entry): no non-finite float crosses
# the boundary. Comparison-based validators are blind to NaN (every comparison
# is False), so a NaN could pass a `v < 0` check and poison downstream
# arithmetic — adversarial testing demonstrated a fired diagnosis carrying
# `tpot_tail_ratio: nan` in the final report. Rejecting at construction makes
# the parser guards (`parsers/_prom.py`) the second line of defense, not the
# only one. Every model in this module opts in.
_FINITE_FLOATS = ConfigDict(allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Primitive enums / type aliases
# ---------------------------------------------------------------------------
RooflinePosition = Literal["compute_bound", "memory_bound", "balanced", "unknown"]
InferenceEngine = Literal["vllm", "sglang", "trt-llm", "unknown"]
# Provenance of cumulative-derived pressure signals: differenced between two
# scrapes ("delta", reflects the current window) or read from one ("lifetime").
PressureWindow = Literal["delta", "lifetime"]
# Phase of one inference-engine scheduler step on the timeline. "mixed" is a step
# that batched prefill tokens together with in-flight decodes (what chunked prefill
# produces); it is a PARSED property of the step (the scheduler's own decision,
# recorded in the NVTX range), not a derived label.
EngineStepPhase = Literal["prefill", "decode", "mixed"]

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------
class Distribution(BaseModel):
    """Describes a distribution of values (e.g. sequence lengths in a batch)."""
    model_config = _FINITE_FLOATS

    mean: float
    p50: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    @field_validator("mean")
    @classmethod
    def mean_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("mean must be non-negative")
        return v


class PhaseMetrics(BaseModel):
    """
    Metrics for a single inference phase (prefill or decode).
    All utilisation fields are in [0, 1]. duration_ms is wall-clock time.
    Fields are Optional because not all data sources provide all metrics.
    """
    model_config = _FINITE_FLOATS

    duration_ms: Optional[float] = Field(None, ge=0)

    # From Nsight Compute
    sm_occupancy: Optional[float] = Field(None, ge=0, le=1)
    hbm_bandwidth_util: Optional[float] = Field(None, ge=0, le=1)
    achieved_flops: Optional[float] = Field(None, ge=0)   # in TFLOP/s
    roofline_position: Optional[RooflinePosition] = None

    # From vLLM metrics
    tokens_per_second: Optional[float] = Field(None, ge=0)
    latency_ms: Optional[float] = Field(None, ge=0)       # TTFT for prefill, TPOT for decode

    @field_validator("sm_occupancy", "hbm_bandwidth_util", mode="before")
    @classmethod
    def clamp_utilisation(cls, v):
        """Guard against parsers emitting e.g. 101% from rounding.

        Clamps FINITE overshoot only. A non-finite value passes through so the
        model's ``allow_inf_nan=False`` validation rejects it — Python's min/max
        ordering would otherwise silently clamp NaN to 1.0, turning garbage into
        a full-utilisation reading (adversarial testing caught r01 firing at 0.90
        confidence on a NaN input this way).
        """
        if v is not None:
            v = float(v)
            if not math.isfinite(v):
                return v
            return max(0.0, min(1.0, v))
        return v


class LayerMetrics(BaseModel):
    """Per-layer breakdown. Only populated when Nsight Compute data is present."""
    model_config = _FINITE_FLOATS

    layer_name: str
    layer_type: Optional[str] = None          # "attention", "mlp", "norm", etc.
    # Inference phase this kernel's launches ran under (schema 1.8.0).
    #
    # Deliberately the SAME alias `EngineStep.phase` uses. A layer and the step
    # containing it have to be nameable in one vocabulary or the join between
    # them cannot be checked; two parallel phase enums would drift the first
    # time either grew a member.
    #
    # None means "the capture did not say", and it is the common case, not an
    # oversight. A stock `ncu --csv --page raw` export carries no phase
    # annotation and no per-kernel timestamp on the trace clock, so there is
    # nothing to correlate against `nsys_timeline.steps`. The parser fills this
    # only from real evidence (an ncu `--nvtx` range column, or numeric launch
    # timestamps joined against a timeline) and leaves it None otherwise.
    # Never infer it from `layer_type`: both phases run attention AND MLP
    # kernels, so a name-based tag is not a weak signal, it is a wrong one.
    #
    # A rule reading a None phase must abstain or widen its scope. It must not
    # read None as "decode" — that is precisely the misattribution the field
    # exists to expose (see the 1.8.0 changelog entry).
    phase: Optional[EngineStepPhase] = None
    duration_ms: Optional[float] = Field(None, ge=0)
    sm_occupancy: Optional[float] = Field(None, ge=0, le=1)
    hbm_bandwidth_util: Optional[float] = Field(None, ge=0, le=1)
    achieved_flops: Optional[float] = Field(None, ge=0)
    roofline_position: Optional[RooflinePosition] = None
    # MoE fields
    expert_utilisation: Optional[list[float]] = None  # one entry per expert


class VllmServingMetrics(BaseModel):
    """
    vLLM serving-layer metrics from a single /metrics scrape.

    Completes the vLLM ingestion. The schema previously modelled vLLM with only
    KV-cache state (`kv_cache_*`); these are the serving counters and gauges that
    /metrics also exposes and that serving-diagnosis rules need (preemptions,
    request/token counters, queue depth). High-populability: if the customer runs
    vLLM, /metrics is almost always present.

    These are raw cumulative counters and point-in-time gauges, NOT derived rates.
    Rules compute the rates they need (preemption_rate, pd_work_ratio,
    queue_pressure) themselves — the schema stores no derived quantity.

    Latency percentiles are intentionally NOT duplicated here: TPOT/TTFT live in
    the existing top-level `tpot_ms` / `ttft_ms` Distribution fields, which remain
    the single source of truth for latency.
    """
    model_config = _FINITE_FLOATS

    # Cumulative counters (since server start).
    num_preemptions_total: Optional[int] = Field(None, ge=0)
    request_success_total: Optional[int] = Field(None, ge=0)
    prompt_tokens_total: Optional[int] = Field(None, ge=0)
    generation_tokens_total: Optional[int] = Field(None, ge=0)

    # Point-in-time queue gauges. A single scrape can miss bursty pressure, so
    # rules should treat these as low-weight corroborators, not primary signals.
    num_requests_running: Optional[int] = Field(None, ge=0)
    num_requests_waiting: Optional[int] = Field(None, ge=0)

    # Raw cumulative totals of the request_queue_time_seconds histogram (in ms /
    # requests). Prometheus exposes `_sum`/`_count` directly, so these are raw
    # telemetry, not derived rates. The live watch loop differences consecutive
    # scrapes into a current-window mean queue time (Δsum/Δcount) for the
    # ThroughputSample series; a single scrape only supports the lifetime mean
    # already carried by the top-level `queue_time_ms` Distribution.
    request_queue_time_ms_sum: Optional[float] = Field(None, ge=0)
    request_queue_time_count: Optional[float] = Field(None, ge=0)

    # Topology signal: the KV-transfer connector class name (e.g. "NixlConnector")
    # if disaggregated prefilling is configured, else None. This is a *parsed*
    # value (from the launch config / connector signature); rules use it to INFER
    # deployment topology. The topology label itself is never stored — only inferred.
    kv_transfer_connector: Optional[str] = None


class EngineStep(BaseModel):
    """One inference-engine scheduler step, placed on the wall clock.

    The unit of a `nsys` (Nsight Systems) timeline as strided models it: a single
    forward pass the engine ran, with a start/end and a phase tag. Continuous
    batching runs these strictly one-at-a-time on the GPU, so a *long* prefill (or
    an oversized `mixed`) step does not run *concurrently* with decode — it blocks
    the next decode step, stretching the gap between token emissions. That temporal
    blocking, visible only with wall-clock placement, is the interference r08 reads.

    All fields are parsed verbatim from the NVTX step range (`start_ms`/`end_ms`
    from the range bounds; `phase` and the token/seq counts from the range
    name/payload). Nothing here is derived: overlap, stall, and the recommended
    chunk budget are r08's to compute, never stored.
    """

    model_config = _FINITE_FLOATS

    start_ms: float = Field(..., ge=0, description="Step start, ms from trace origin.")
    end_ms: float = Field(..., ge=0, description="Step end, ms from trace origin.")
    phase: EngineStepPhase
    # Tokens of prefill work in this step (a chunk of a prompt, or a whole prompt
    # when chunking is off). None when the trace did not annotate it.
    num_prefill_tokens: Optional[int] = Field(None, ge=0)
    # Decode sequences advanced one token in this step. None when not annotated.
    num_decode_seqs: Optional[int] = Field(None, ge=0)

    @model_validator(mode="after")
    def end_after_start(self) -> "EngineStep":
        if self.end_ms < self.start_ms:
            raise ValueError(
                f"EngineStep.end_ms ({self.end_ms}) precedes start_ms ({self.start_ms})"
            )
        return self

    @property
    def duration_ms(self) -> float:
        """Wall-clock length of the step. Derived on read, never stored."""
        return self.end_ms - self.start_ms


class NsysTimeline(BaseModel):
    """A time-resolved trace of engine scheduler steps from an `nsys` capture.

    Completes the *timeline* ingestion the schema previously lacked: every other
    block is one aggregate snapshot, so no rule could reason about ordering or
    wall-clock placement. `steps` are the raw per-step intervals; rules that need
    the temporal structure (r08) read them and derive their own signals. The list
    may be empty (a trace with no recognised step ranges) — rules guard for it.
    """

    model_config = _FINITE_FLOATS

    steps: list[EngineStep] = Field(default_factory=list)
    # Total wall-clock span of the captured trace (ms). May exceed the summed step
    # durations (idle gaps between steps). Optional: a summary export omits it.
    trace_duration_ms: Optional[float] = Field(None, ge=0)


class ThroughputSample(BaseModel):
    """One tick of a sustained serving run — the unit of `throughput_history`.

    The live `watch` loop appends one per tick (see `collect/window.py`); r06
    (throughput decay) reads the resulting series to detect a sustained downward
    trend in generation throughput and to attribute it. Raw co-samples only: the
    rule derives the slope and its significance, the schema stores no derived
    trend — the same split as `tp_rank_sm_clocks` / `slowness_from_sm_clocks`.

    `t` is monotonic wall-clock seconds (the loop's clock); only `t` and the
    window throughput are required. The corroborators feed r06's demand/pressure
    gate and are Optional because not every source provides them.
    """
    model_config = _FINITE_FLOATS

    t: float = Field(..., description="monotonic wall-clock seconds at this sample")
    token_throughput_gen: float = Field(..., ge=0, description="window generation throughput, tok/s")

    # Co-sampled corroborators (Optional) — the gate that distinguishes a
    # work-limited decay (real) from a demand-limited one (benign load taper).
    # As of schema 1.4.0 the live loop feeds both with per-window deltas
    # (Δpreemptions/Δsuccesses; Δqueue_sum/Δqueue_count between consecutive
    # scrapes) so a long-running server's lifetime totals cannot dilute the
    # current pressure the r06 mechanism leg looks for.
    preemption_rate: Optional[float] = Field(None, ge=0)       # window preemptions / successes at this tick
    queue_time_ms: Optional[float] = Field(None, ge=0)         # window mean request queue time at this tick
    num_requests_waiting: Optional[float] = Field(None, ge=0)  # scheduler backlog gauge
    sm_clock_mhz: Optional[float] = Field(None, ge=0)          # thermal/DVFS throttle corroborator
    gpu_temp_c: Optional[float] = None                         # thermal corroborator (°C)


class TpRankSample(BaseModel):
    """One tick of per-rank SM clocks — the unit of `tp_rank_history`.

    The live `watch` loop appends one per tick; r04 (tensor-parallel rank
    imbalance) reads the series to require that the SAME rank is the outlier
    across consecutive ticks before calling it a straggler.

    Why the history exists: a single clock snapshot cannot distinguish a rank
    that is *thermally throttled* from one that has merely DVFS'd down to the
    idle floor between load bursts — both read as "slow". Field testing
    measured r04 firing on 16 of 100
    ticks against a healthy pack, blaming ranks near-uniformly, which is the
    signature of that confusion. Persistence separates them: a throttled rank
    stays slow, an idle one does not.

    Raw samples only — the schema stores no derived outlier or persistence
    verdict; r04 computes those, the same parsers-write-raw/rules-derive split
    as `tp_rank_sm_clocks` / `slowness_from_sm_clocks`.
    """
    model_config = _FINITE_FLOATS

    t: float = Field(..., description="monotonic wall-clock seconds at this sample")
    sm_clocks: list[float] = Field(..., description="per-rank SM clock (MHz), rank order")
    temps: Optional[list[float]] = None      # per-rank temperature (°C), rank order


# ---------------------------------------------------------------------------
# Top-level DiagnosisInput
# ---------------------------------------------------------------------------
class DiagnosisInput(BaseModel):
    """
    Everything a rule needs to fire a diagnosis.
    All fields outside workload context are Optional — parsers populate what
    they can; rules guard against None before using a field.
    """
    model_config = _FINITE_FLOATS

    # ------------------------------------------------------------------
    # Workload context (required)
    # ------------------------------------------------------------------
    model_name: str = Field(..., description="E.g. 'meta-llama/Llama-3-70B'")
    gpu_type: str = Field(..., description="E.g. 'H100-SXM', 'A100-80G'")
    inference_engine: InferenceEngine = "unknown"

    # ------------------------------------------------------------------
    # Workload shape (optional, from vLLM or config)
    # ------------------------------------------------------------------
    model_params_b: Optional[float] = Field(None, ge=0, description="Model size in billions of parameters")
    batch_size: Optional[int] = Field(None, ge=1)
    seq_len_distribution: Optional[Distribution] = None
    num_gpus: Optional[int] = Field(None, ge=1)
    tensor_parallel_size: Optional[int] = Field(None, ge=1)
    dtype: Optional[str] = None  # "bfloat16", "float16", "int8", "fp8"

    # ------------------------------------------------------------------
    # Phase-level timing (optional, from Nsight / vLLM)
    # ------------------------------------------------------------------
    prefill: Optional[PhaseMetrics] = None
    decode: Optional[PhaseMetrics] = None

    # ------------------------------------------------------------------
    # Per-layer breakdown (optional, from Nsight)
    # ------------------------------------------------------------------
    layers: Optional[list[LayerMetrics]] = None

    # ------------------------------------------------------------------
    # Timeline trace (optional, from Nsight Systems / nsys)
    # ------------------------------------------------------------------
    # Time-resolved scheduler steps. The one block carrying wall-clock placement;
    # r08 reads it to see long prefills stalling the decode cadence. Unlike the
    # `layers`/`prefill`/`decode` aggregates, steps are ordered and timestamped.
    nsys_timeline: Optional[NsysTimeline] = None

    # ------------------------------------------------------------------
    # Cluster-level (optional, from DCGM / NCCL)
    # ------------------------------------------------------------------
    # Per-rank SM clock (MHz) from DCGM; a throttling/straggler GPU down-clocks, so
    # lower clock = slower rank. This is RAW telemetry. r04 derives the comparable
    # slowness ratio (max_clock / clock) from it inside the rule — the schema does
    # not store derived ratios. gpu_util is intentionally not an alternative here:
    # its straggler direction is ambiguous under the NCCL barrier (fast ranks
    # busy-wait at ~100%), so with no clocks r04 has no signal and abstains.
    tp_rank_sm_clocks: Optional[list[float]] = None  # per-rank SM clock (MHz); lower = slower rank
    tp_rank_temps: Optional[list[float]] = None       # per-rank temperature (°C); throttling corroborator
    nccl_time_pct: Optional[float] = Field(None, ge=0, le=1)

    # ------------------------------------------------------------------
    # KV cache state (optional, from vLLM)
    # ------------------------------------------------------------------
    kv_cache_util: Optional[float] = Field(None, ge=0, le=1)
    kv_cache_fragmentation: Optional[float] = Field(None, ge=0, le=1)
    kv_block_size: Optional[int] = Field(None, ge=1)
    kv_num_blocks_total: Optional[int] = Field(None, ge=0)
    kv_num_blocks_used: Optional[int] = Field(None, ge=0)

    # ------------------------------------------------------------------
    # vLLM serving-layer metrics (optional, from vLLM /metrics + launch config)
    # ------------------------------------------------------------------
    vllm_serving: Optional[VllmServingMetrics] = None

    # ------------------------------------------------------------------
    # Throughput & latency summary (from vLLM /metrics or benchmark logs)
    # ------------------------------------------------------------------
    request_throughput: Optional[float] = Field(None, ge=0)    # req/s
    token_throughput_prompt: Optional[float] = Field(None, ge=0)  # tok/s
    token_throughput_gen: Optional[float] = Field(None, ge=0)     # tok/s
    ttft_ms: Optional[Distribution] = None        # time-to-first-token
    tpot_ms: Optional[Distribution] = None        # time-per-output-token
    e2e_latency_ms: Optional[Distribution] = None
    # Time requests spent queued before scheduling. Derived from vLLM's
    # request_queue_time_seconds histogram — a CUMULATIVE histogram, so one scrape
    # *survives* (unlike the num_requests_waiting gauge), but its mean is then a
    # LIFETIME average: on a long-running server it dilutes below a fresh burst
    # (false negative) or stays high after a burst passes (false positive). Pass a
    # second, earlier scrape so the parser differences it into the current window
    # (sets `pressure_window="delta"`). It is the pressure signal that moves on
    # vLLM V1, which queues rather than preempting (num_preemptions can stay 0
    # under genuine pressure — observed on a real GPU 2026-06-13).
    queue_time_ms: Optional[Distribution] = None
    # Provenance of the cumulative-derived signals above (queue_time_ms, the
    # vllm_serving counters, and the kv_cache_fragmentation estimate): "delta" when
    # parsed from two scrapes and differenced to the current window, "lifetime"
    # when from a single scrape (a lifetime aggregate). None when not applicable.
    # r03 keeps lifetime-only firings capped because a lifetime mean is a dilutable
    # proxy for *current* pressure.
    pressure_window: Optional[PressureWindow] = None

    # ------------------------------------------------------------------
    # Sustained-run throughput history (from the live `watch` loop only)
    # ------------------------------------------------------------------
    # Rolling per-tick throughput series, oldest→newest, appended by the watch
    # loop's ThroughputHistory accumulator. r06 (throughput decay) reads it to
    # detect a sustained downward trend; it abstains (INSUFFICIENT_DATA) when the
    # series is absent or too short. The one-shot `diagnose` path leaves this None
    # (a single scrape has no time base), so r06 is a live-only rule — exactly like
    # the per-second throughput fields above. Raw samples only; the slope is
    # derived in the rule, never stored here.
    throughput_history: Optional[list[ThroughputSample]] = None

    # Per-tick per-rank clock series, appended by the live watch loop. r04 uses
    # it to require a PERSISTENT outlier rank; the one-shot path leaves it None
    # and r04 falls back to single-snapshot behaviour (a dump has no DVFS
    # transient to confuse it).
    tp_rank_history: Optional[list[TpRankSample]] = None

    # ------------------------------------------------------------------
    # Data provenance (informational, not used by rules)
    # ------------------------------------------------------------------
    source_files: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_least_one_metric(self) -> "DiagnosisInput":
        """
        Warn (don't crash) if we have basically nothing to diagnose.
        A diagnosis with only model_name and gpu_type is valid structurally,
        but rules will likely all return None.
        """
        has_data = any([
            self.prefill, self.decode, self.layers,
            self.kv_cache_util, self.tp_rank_sm_clocks,
            self.request_throughput, self.vllm_serving,
            self.nsys_timeline is not None and bool(self.nsys_timeline.steps),
        ])
        if not has_data:
            self.parse_warnings.append(
                "DiagnosisInput has no metric fields set; rules will produce no diagnoses."
            )
        return self
