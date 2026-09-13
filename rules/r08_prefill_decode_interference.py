"""
r08 — Prefill↔decode interference (timeline), adaptive chunk-budget fix.

The first *dynamic* rule. Every other rule reads one aggregate snapshot — phase
means, cumulative counters, a latency distribution — and asks "is this number
bad?" r08 reads the time-resolved `nsys` timeline (`dx.nsys_timeline.steps`) and
asks a question only ordering can answer: *are long prefills stalling the decode
cadence?* Continuous batching runs scheduler steps one-at-a-time on the GPU, so a
long prefill step (or an oversized `mixed` step) does not overlap decode — it
*blocks* it, inserting a gap in the token stream that spikes TTFT for the landing
request and TPOT for everyone in flight. That blocking is invisible to r02's
static /metrics fingerprint; it is plainly visible on the timeline.

Two things make r08 "dynamic, as opposed to the static rules":
  1. Dynamic INPUT — it measures temporal structure (decode-step baseline, the
     prefill-step duration tail, the share of busy time lost to excess prefill),
     not a single snapshot value.
  2. Dynamic OUTPUT — the fix is a *computed* chunked-prefill budget
     (`--max-num-batched-tokens`) derived from the trace's own prefill token rate
     and decode-step time budget, not a fixed "512–2048" copied from a blog. This
     is the "policy vs raw size" distinction: size the chunk to the decode-step
     slack, not to an arbitrary token count.

Relation to r02: same root pathology (prefill contends with decode), different
evidence. r02 infers it correlationally from serving counters; r08 measures it
directly from the schedule and prescribes the knob value. They CORROBORATE
(engine/relations.py); neither suppresses the other.

What r08 does NOT do: it does not claim prefill and decode run *concurrently*
(they do not, on one stream) — the interference is temporal blocking. It does not
fire on a workload that is already well-chunked (bounded step durations): that is
the dominant false positive, guarded by the step-tail floor. Confidence is capped
while THRESHOLDS_UNCALIBRATED is True — the bands are calibration seeds, there is
no published "tail ratio that means mis-tuned chunking."
"""

from __future__ import annotations

from typing import Optional

from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput, EngineStep

# --------------------------------------------------------------------------- #
# Thresholds — CALIBRATION SEEDS, NOT literature values.
# Sarathi-Serve (arXiv:2403.02310) establishes that an unbounded prefill batched
# with decodes is what inflates TPOT/TTFT, and that a token budget bounds it —
# but the *numeric* tail ratio / busy-time share that should trigger a retune is
# workload-specific and unpublished. Validate on real colocated nsys traces
# before trusting the bands; until then THRESHOLDS_UNCALIBRATED caps confidence.
# --------------------------------------------------------------------------- #
MIN_STEPS = 20             # below this the timeline is too short to trust
MIN_DECODE_STEPS = 5       # decode-only steps needed to baseline the cadence
MIN_PREFILL_STEPS = 2      # prefill-bearing steps needed to see interference

TAIL_MIN = 2.0             # p95 prefill step must be >=2x a decode step to be "spiky"
TAIL_STRONG = 8.0          # >=8x = severe stalls
INTERFERENCE_MIN = 0.15    # >=15% of busy time in excess-prefill = material
INTERFERENCE_STRONG = 0.50 # >=50% = decode is badly starved

BUDGET_TARGET_MULT = 1.5   # aim: keep a step within ~1.5x the decode baseline
BUDGET_ROUND_TO = 128      # round the recommended token budget to a sane multiple
BUDGET_FLOOR = 256         # never recommend a budget below this (too-small thrashes)

# Confidence model (mirrors r05's linear-in-signal skeleton).
_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.40
_CONFIDENCE_CEILING = 0.90
THRESHOLDS_UNCALIBRATED = True
_UNCALIBRATED_CEILING = 0.65


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile of `values` (q in [0, 1]). Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    frac = idx - lo
    if lo + 1 >= len(s):
        return s[-1]
    return s[lo] + frac * (s[lo + 1] - s[lo])


def _median(values: list[float]) -> float:
    return _percentile(values, 0.5)


# --------------------------------------------------------------------------- #
# Timeline analysis (pure)
# --------------------------------------------------------------------------- #

class _TimelineStats:
    """Derived interference statistics for one timeline. Computed, never stored."""

    def __init__(self, steps: list[EngineStep]) -> None:
        decode = [s for s in steps if s.phase == "decode"]
        prefill_bearing = [s for s in steps if s.phase in ("prefill", "mixed")]
        mixed = [s for s in steps if s.phase == "mixed"]

        self.num_steps = len(steps)
        self.num_decode = len(decode)
        self.num_prefill_bearing = len(prefill_bearing)
        self.num_mixed = len(mixed)

        self.decode_baseline_ms = _median([s.duration_ms for s in decode])
        pb_durs = [s.duration_ms for s in prefill_bearing]
        self.prefill_p95_ms = _percentile(pb_durs, 0.95)

        # "Spike" magnitude: worst prefill step vs a normal decode step.
        self.step_tail_ratio = (
            self.prefill_p95_ms / self.decode_baseline_ms
            if self.decode_baseline_ms > 0 else 0.0
        )

        # Interference: share of GPU-busy time spent on prefill work *in excess of*
        # a decode step's worth — the time each long prefill step displaced from the
        # decode stream. Excess-over-baseline (not full duration) so a healthy short
        # prefill does not register.
        total_step_ms = sum(s.duration_ms for s in steps)
        excess_ms = sum(max(0.0, s.duration_ms - self.decode_baseline_ms) for s in prefill_bearing)
        self.interference_fraction = excess_ms / total_step_ms if total_step_ms > 0 else 0.0

        # Chunking already engaged? `mixed` steps mean prefill chunks were fused
        # into decode batches (vLLM chunked prefill). No mixed steps + huge prefill
        # steps => chunking is likely OFF.
        self.chunking_active = self.num_mixed > 0

        # Prefill token throughput (tokens/ms) from steps that annotated counts —
        # the basis for an *absolute* budget. None when the trace omitted counts.
        # The denominator is the *prefill-attributable* time, not the raw step
        # duration: a `mixed` step's wall-clock is decode work + a prefill chunk, so
        # charging its whole duration to prefill under-reads the true prefill rate
        # (measuring 75 tok/ms for a prefill actually running at 100). Net out the
        # decode baseline for mixed steps; a pure prefill step is charged in full. A
        # step with no positive prefill-attributable time (e.g. a zero-duration
        # mixed step) contributes neither term, so it can never divide in.
        prefill_tokens = 0
        prefill_time_ms = 0.0
        for s in prefill_bearing:
            if s.num_prefill_tokens is None:
                continue
            attributable_ms = (
                s.duration_ms - self.decode_baseline_ms
                if s.phase == "mixed" else s.duration_ms
            )
            if attributable_ms <= 0:
                continue
            prefill_tokens += s.num_prefill_tokens
            prefill_time_ms += attributable_ms
        self.prefill_tokens_per_ms: Optional[float] = (
            prefill_tokens / prefill_time_ms
            if prefill_time_ms > 0 and prefill_tokens > 0 else None
        )

    def recommended_budget(self) -> tuple[Optional[int], float]:
        """(recommended --max-num-batched-tokens, target_step_ms).

        Budget is sized so that, once chunked prefill is on, a fused step lands
        within `BUDGET_TARGET_MULT` decode steps — policy tied to the decode-step
        budget, not a raw constant. The subtlety the first cut got wrong: a mixed
        step still has to run its decode work, so only the *slack* beyond one decode
        step (`target_step_ms - decode_baseline_ms`) is available for the prefill
        chunk. Sizing the chunk to the whole target double-counts the decode time
        and recommends a budget ~3× too large — the operator applies it, re-runs,
        and r08 fires again (non-convergent). Sizing to the slack makes the
        recommendation a one-shot fixed point. None when the trace carried no token
        counts (then r08 falls back to a ratio target in the fix text).
        """
        target_step_ms = BUDGET_TARGET_MULT * self.decode_baseline_ms
        prefill_slack_ms = target_step_ms - self.decode_baseline_ms
        if self.prefill_tokens_per_ms is None or prefill_slack_ms <= 0:
            return None, target_step_ms
        raw = self.prefill_tokens_per_ms * prefill_slack_ms
        rounded = int(round(raw / BUDGET_ROUND_TO) * BUDGET_ROUND_TO)
        return max(BUDGET_FLOOR, rounded), target_step_ms


# --------------------------------------------------------------------------- #
# Fix text
# --------------------------------------------------------------------------- #

def _fix(stats: _TimelineStats) -> str:
    budget, target_step_ms = stats.recommended_budget()

    if stats.chunking_active:
        opening = (
            "Chunked prefill is already on (the trace has mixed prefill+decode "
            "steps) but its budget is too large — chunks are still long enough to "
            "stall decode. Lower the budget."
        )
    else:
        opening = (
            "Chunked prefill appears to be off (no mixed steps — prefills run as "
            "their own long steps). Enable it (`--enable-chunked-prefill`, the "
            "default on vLLM V1) and set a budget."
        )

    if budget is not None:
        prefill_slack_ms = target_step_ms - stats.decode_baseline_ms
        knob = (
            f"Set `--max-num-batched-tokens` ≈ {budget}: at this trace's measured "
            f"~{stats.prefill_tokens_per_ms:.0f} prefill tok/ms, that many tokens "
            f"fit in the ~{prefill_slack_ms:.0f}ms of slack beyond one "
            f"{stats.decode_baseline_ms:.0f}ms decode step, so a fused prefill+decode "
            f"step stays within ~{target_step_ms:.0f}ms (~{BUDGET_TARGET_MULT:g}× a "
            f"decode step) instead of the {stats.step_tail_ratio:.0f}× it is now. "
        )
    else:
        knob = (
            "The trace carried no per-step token counts, so an absolute budget "
            "cannot be computed here. Lower `--max-num-batched-tokens` until the "
            f"longest prefill step is within ~{BUDGET_TARGET_MULT:g}× your median "
            f"decode step (it is {stats.step_tail_ratio:.0f}× now). "
        )

    policy = (
        "Treat the budget as a policy tied to your decode-step time, not a fixed "
        "raw token count: re-derive it whenever the model, GPU, or decode batch "
        "size changes. If TTFT then rises too far (chunks too small lengthen "
        "prefill), raise it back toward the knee. `--long-prefill-token-threshold` "
        "can additionally cap which prompts get chunked. (Sarathi-Serve, "
        "arXiv:2403.02310.)"
    )
    return f"{opening} {knob}{policy}"


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

class PrefillDecodeInterferenceRule(Rule):
    """Long prefills stall the decode cadence on the timeline; retune the chunk budget."""

    rule_id = "r08"
    title = "Prefill↔decode interference (timeline)"
    references = (
        "Agrawal et al., 'Taming Throughput-Latency Tradeoff in LLM Inference with "
        "Sarathi-Serve,' OSDI 2024. arXiv:2403.02310. (Chunked prefill + token "
        "budget; an unbounded prefill batched with decodes inflates TPOT/TTFT.)",
        "Zhong et al., 'DistServe: Disaggregating Prefill and Decoding,' OSDI 2024. "
        "arXiv:2401.09670. (Prefill↔decode interference, characterised.)",
        "vLLM optimization & chunked-prefill docs. "
        "https://docs.vllm.ai/en/latest/performance/optimization.html",
        "NVIDIA Nsight Systems user guide (NVTX timeline). "
        "https://docs.nvidia.com/nsight-systems/",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required: a timeline with steps -------------------------------
        tl = dx.nsys_timeline
        if tl is None or not tl.steps:
            return InsufficientData(
                missing=("nsys_timeline.steps",),
                reason="no nsys timeline steps present; r08 needs a time-resolved trace",
            )
        steps = list(tl.steps)
        if len(steps) < MIN_STEPS:
            return InsufficientData(
                reason=f"only {len(steps)} timeline steps; need >={MIN_STEPS} for a "
                "trustworthy cadence baseline"
            )

        stats = _TimelineStats(steps)

        # ---- Need a decode cadence AND prefill events to interfere with it --
        if stats.num_prefill_bearing < MIN_PREFILL_STEPS:
            # Data present, but no prefill activity to stall decode — rule N/A.
            return Abstention.BELOW_THRESHOLD
        if stats.num_decode == 0:
            # No decode steps: nothing for prefill to stall (prefill-only trace).
            return Abstention.BELOW_THRESHOLD
        if stats.num_decode < MIN_DECODE_STEPS:
            return InsufficientData(
                reason=f"only {stats.num_decode} decode-only steps; need "
                f">={MIN_DECODE_STEPS} to baseline the decode cadence"
            )
        if stats.decode_baseline_ms <= 0:
            return InsufficientData(
                reason="decode steps have zero duration; cannot baseline the cadence"
            )

        # ---- Firing condition: spiky AND materially stalling ---------------
        # Below the tail floor the prefill steps are bounded (already well-chunked,
        # or naturally short) — chunking will not help. This is the dominant FP guard.
        if stats.step_tail_ratio < TAIL_MIN:
            return Abstention.BELOW_THRESHOLD
        if stats.interference_fraction < INTERFERENCE_MIN:
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength + confidence ----------------------------------
        tail_signal = _clamp01((stats.step_tail_ratio - TAIL_MIN) / (TAIL_STRONG - TAIL_MIN))
        interf_signal = _clamp01(
            (stats.interference_fraction - INTERFERENCE_MIN)
            / (INTERFERENCE_STRONG - INTERFERENCE_MIN)
        )
        signal_strength = 0.5 * tail_signal + 0.5 * interf_signal

        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )
        if THRESHOLDS_UNCALIBRATED:
            confidence = min(confidence, _UNCALIBRATED_CEILING)
        if confidence < _CONFIDENCE_FLOOR:
            return Abstention.BELOW_THRESHOLD

        # ---- Build the diagnosis -------------------------------------------
        budget, target_step_ms = stats.recommended_budget()

        cause = (
            "On the captured timeline, long prefill steps are stalling the decode "
            f"cadence. The worst prefill steps ran {stats.step_tail_ratio:.1f}× a "
            f"normal decode step ({stats.prefill_p95_ms:.0f}ms p95 vs a "
            f"{stats.decode_baseline_ms:.0f}ms median decode step), and "
            f"{stats.interference_fraction:.0%} of GPU-busy time went to prefill work "
            "beyond one decode step's worth. Because the engine runs steps one at a "
            "time, each long prefill blocks every in-flight decode for its full "
            "duration — which is what spikes TTFT for the arriving request and TPOT "
            "for the rest."
        )

        notes = (
            f"timeline rule: {stats.num_steps} steps "
            f"({stats.num_decode} decode / {stats.num_prefill_bearing} prefill-bearing"
            f", {stats.num_mixed} mixed); chunking "
            f"{'on' if stats.chunking_active else 'off'}; "
            "signal = mean(tail, interference)"
        )
        if budget is None:
            notes += "; no token counts in trace — budget given as a ratio target"
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=_fix(stats),
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=self._data_completeness(dx, stats),
                notes=notes,
            ),
            evidence={
                "num_steps": stats.num_steps,
                "num_decode_steps": stats.num_decode,
                "num_prefill_bearing_steps": stats.num_prefill_bearing,
                "num_mixed_steps": stats.num_mixed,
                "decode_baseline_ms": round(stats.decode_baseline_ms, 2),
                "prefill_p95_ms": round(stats.prefill_p95_ms, 2),
                "step_tail_ratio": round(stats.step_tail_ratio, 2),
                "interference_fraction": round(stats.interference_fraction, 3),
                "chunking_active": stats.chunking_active,
                "target_step_ms": round(target_step_ms, 2),
                "recommended_max_num_batched_tokens": budget,
                "signal_strength": round(signal_strength, 2),
            },
        )

    @staticmethod
    def _data_completeness(dx: DiagnosisInput, stats: _TimelineStats) -> float:
        """Fraction of optional corroborators that sharpen the diagnosis/fix."""
        optional = [
            stats.prefill_tokens_per_ms is not None,   # enables an absolute budget
            dx.nsys_timeline.trace_duration_ms is not None,
            dx.ttft_ms is not None,                    # external corroboration of the spike
            stats.num_mixed > 0,                       # tells us the chunking state
        ]
        return sum(optional) / len(optional)
