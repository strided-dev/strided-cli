"""
r06 — Throughput decay over a sustained run.

Fires when generation throughput (tokens/sec) trends DOWN over a sustained
serving run AND the decline is *work-limited* (the engine does less per unit time
despite continued demand) rather than *demand-limited* (fewer requests arrived —
benign). Throughput naturally falls when offered load tapers, so a downtrend
alone is never enough: the rule also requires that a mechanism is visibly
building —

  * memory pressure: the KV cache fills and the scheduler falls behind, so the
    preemption rate and/or request queue time climb (Kwon et al., PagedAttention,
    SOSP 2023; vLLM preemption/swap path); or
  * a thermal / DVFS throttle: the SM clock falls while temperature climbs, so
    each step is slower (arXiv:2603.23640, "thermal management, not peak compute,
    is the binding constraint for sustained inference"; arXiv:2010.06291).

A rising preemption rate / queue time is itself evidence that demand is sustained
(an idle server does not preempt or queue), and a falling clock under a *rising*
temperature distinguishes a throttle from a benign idle down-clock (idle cools).
When a waiting-queue series is available and the backlog has drained to ~0, the
drop is demand-driven and the rule abstains — the abstentions are the point.

As of schema 1.4.0 the live loop feeds the preemption/queue co-samples as
per-window deltas (Δ between consecutive scrapes), not lifetime ratios, so a
long-running server's past bursts cannot dilute the current pressure this rule's
mechanism leg looks for. The rule itself is unchanged — it only ever tested the
trend's direction — but the trend it sees is now truthful on old servers too.

This is a LIVE-ONLY rule: it reads ``dx.throughput_history``, the per-tick series
the watch loop appends. The one-shot ``diagnose`` path has no time base, so r06
abstains there (INSUFFICIENT_DATA) — exactly like the per-second throughput
fields. The trend is detected with the Theil–Sen slope (robust magnitude) gated
by the Mann–Kendall monotonic-trend test (significance), so the decode sawtooth
does not masquerade as decay. While THRESHOLDS_UNCALIBRATED is True the firing
bands are calibration seeds and confidence is capped.
"""

from __future__ import annotations

import math
import statistics
from typing import Optional

from rules._stats import TrendTest, mann_kendall, theil_sen_slope
from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput
from schema.diagnosis_input import ThroughputSample

# --------------------------------------------------------------------------- #
# Firing thresholds — CALIBRATION SEEDS (THRESHOLDS_UNCALIBRATED caps confidence).
# A "sustained run" needs both a minimum sample count and a minimum wall-clock
# span — two ticks 3s apart are not a sustained run, and the Mann–Kendall normal
# approximation is only asymptotically valid (so below the sample floor it is a
# conservative seed, and confidence stays capped). DECAY_MIN is a fractional
# decline per minute relative to the run's median throughput; 5%/min sustained
# is a real, actionable slide. P_MAX is the two-sided Mann–Kendall significance a
# downtrend must clear to count as monotonic rather than sawtooth noise.
# --------------------------------------------------------------------------- #
MIN_SAMPLES = 6
MIN_WINDOW_S = 60.0
TRUSTED_WINDOW_S = 180.0       # below this the window is short → keep confidence capped
DECAY_MIN = 0.05               # frac throughput decline / minute to fire
DECAY_SEVERE = 0.20            # frac/min that counts as a full-strength decay
P_MAX = 0.10                   # Mann–Kendall two-sided p for a significant trend
WAITING_MIN = 1.0              # recent backlog below this = queue drained (benign)
TEMP_HOT_C = 80.0              # temperature band that corroborates a throttle
Z_STRONG = 3.0                 # |MK z| that counts as a fully-clear mechanism trend

# Confidence model constants (see r01 for the geometric-mean rationale).
_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.80
_CONFIDENCE_CEILING = 0.90

THRESHOLDS_UNCALIBRATED = True  # flip to False only after a real-GPU sweep
_UNCALIBRATED_CEILING = 0.65    # confidence cap while uncalibrated / short window

_FIX_MEMORY_VLLM = (
    "Throughput is decaying as the KV cache fills and the scheduler falls behind. "
    "Relieve memory pressure: raise `--gpu-memory-utilization` or add GPUs to grow "
    "the cache, cap admitted concurrency with `--max-num-seqs` so in-flight requests "
    "are not starved, and track `vllm:num_preemptions_total` and request queue time "
    "to confirm the relief. (Kwon et al., SOSP 2023.)"
)

_FIX_MEMORY_GENERIC = (
    "Throughput is decaying under building memory pressure (rising preemptions / "
    "queue time). Grow the KV cache (more GPU-memory headroom or more GPUs) and cap "
    "admitted concurrency so the engine stops thrashing eviction/recompute under "
    "sustained load."
)

_FIX_THERMAL = (
    "Throughput is decaying as the GPU thermally throttles (SM clock falling while "
    "temperature climbs). Improve cooling/airflow, lower the ambient temperature, or "
    "set a sustained power/clock floor (`nvidia-smi -pl` / `-lgc`) so the clock holds "
    "steady instead of stepping down. Thermal headroom — not peak compute — is the "
    "binding constraint for a sustained run (arXiv:2603.23640)."
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _series(hist: list[ThroughputSample], attr: str) -> tuple[list[float], list[float]]:
    """Parallel (t, value) lists for samples where ``attr`` is populated."""
    ts: list[float] = []
    ys: list[float] = []
    for s in hist:
        v = getattr(s, attr)
        if v is not None:
            ts.append(s.t)
            ys.append(float(v))
    return ts, ys


def _trend(ys: list[float]) -> Optional[TrendTest]:
    """Mann–Kendall result, or None when there are too few points."""
    return mann_kendall(ys)


def _rising(ys: list[float]) -> Optional[TrendTest]:
    t = _trend(ys)
    if t is not None and t.s > 0 and t.p_value < P_MAX:
        return t
    return None


def _falling(ys: list[float]) -> Optional[TrendTest]:
    t = _trend(ys)
    if t is not None and t.s < 0 and t.p_value < P_MAX:
        return t
    return None


def _recent_mean(ys: list[float]) -> float:
    """Mean of the last third (>=1) of a series — the 'where it ended up' level."""
    k = max(1, len(ys) // 3)
    return statistics.mean(ys[-k:])


def _early_mean(ys: list[float]) -> float:
    """Mean of the first third (>=1) of a series — the 'where it started' level."""
    k = max(1, len(ys) // 3)
    return statistics.mean(ys[:k])


class ThroughputDecayRule(Rule):
    """Generation throughput trends down over a sustained run, with a cause."""

    rule_id = "r06"
    title = "Throughput decay"
    references = (
        "Kwon et al., 'Efficient Memory Management for Large Language Model "
        "Serving with PagedAttention,' SOSP 2023. arXiv:2309.06180.",
        "'LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency "
        "Trade-offs Under Sustained Load,' arXiv:2603.23640 — sustained-load "
        "thermal throttling settles throughput well below peak (reported 33–44%).",
        "'Impact of Thermal Throttling on Long-Term Visual Inference in a "
        "CPU-Based Edge Device,' Electronics 9(12):2106, 2020. arXiv:2010.06291.",
        "Sen, 'Estimates of the Regression Coefficient Based on Kendall's Tau,' "
        "JASA 1968; Mann, Econometrica 1945; Kendall, 'Rank Correlation Methods,' "
        "1975 — Theil–Sen slope and the Mann–Kendall monotonic-trend test.",
        "vLLM Metrics design. https://docs.vllm.ai/en/latest/design/metrics/",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required: a sustained per-tick throughput series -------------
        hist = dx.throughput_history
        if not hist:
            # None on the one-shot path (no time base); empty is the same gap.
            return InsufficientData(missing=("throughput_history",))
        if len(hist) < MIN_SAMPLES:
            return InsufficientData(
                reason=(
                    f"throughput_history has {len(hist)} sample(s); need "
                    f">= {MIN_SAMPLES} over a sustained run to judge a trend."
                )
            )
        ts, tput = _series(hist, "token_throughput_gen")
        window_s = ts[-1] - ts[0]
        if window_s < MIN_WINDOW_S:
            return InsufficientData(
                reason=(
                    f"throughput_history spans {window_s:.0f}s; need "
                    f">= {MIN_WINDOW_S:.0f}s to judge a sustained trend."
                )
            )

        scale = statistics.median(tput)
        if scale <= 0:
            # No generation throughput at all — nothing to call a decay.
            return Abstention.BELOW_THRESHOLD

        # ---- Leg 1: a significant, monotonic downward trend ---------------
        slope_per_s = theil_sen_slope(ts, tput)        # tok/s per second
        tput_trend = _trend(tput)
        if slope_per_s is None or tput_trend is None:
            return Abstention.BELOW_THRESHOLD
        # Fractional decline per minute, robust scale. Positive = decaying.
        decay_frac_per_min = -slope_per_s * 60.0 / scale
        significant_down = tput_trend.s < 0 and tput_trend.p_value < P_MAX
        if not (significant_down and decay_frac_per_min >= DECAY_MIN):
            return Abstention.BELOW_THRESHOLD

        # ---- Benign-drain override: if we can see the queue and it emptied,
        # the drop is demand-driven, not work-limited → abstain. ------------
        _, waiting = _series(hist, "num_requests_waiting")
        backlog_drained = bool(waiting) and _recent_mean(waiting) < WAITING_MIN
        if backlog_drained:
            return Abstention.BELOW_THRESHOLD

        # ---- Leg 2: an attributable, *building* mechanism -----------------
        _, preempt = _series(hist, "preemption_rate")
        _, queue = _series(hist, "queue_time_ms")
        _, sm_clock = _series(hist, "sm_clock_mhz")
        _, temp = _series(hist, "gpu_temp_c")

        preempt_rising = _rising(preempt)
        queue_rising = _rising(queue)
        memory_pressure = preempt_rising or queue_rising

        clock_falling = _falling(sm_clock)
        temp_rising = _rising(temp)
        temp_hot = bool(temp) and _recent_mean(temp) >= TEMP_HOT_C
        thermal_throttle = clock_falling is not None and (temp_rising is not None or temp_hot)

        if not (memory_pressure or thermal_throttle):
            # Throughput fell but nothing explains it as work-limited — most
            # likely the load tapered. Do not invent a cause.
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength ----------------------------------------------
        decay_strength = _clamp01((decay_frac_per_min - DECAY_MIN) / (DECAY_SEVERE - DECAY_MIN))
        # Only legs that opened may contribute strength: a trend that exists but
        # failed its leg's gate (e.g. a falling clock with no thermal
        # corroboration) is not an attributed mechanism and must not inflate
        # the score.
        mech_zs: list[float] = []
        if memory_pressure:
            for t in (preempt_rising, queue_rising):
                if t is not None:
                    mech_zs.append(abs(t.z))
        if thermal_throttle:
            for t in (clock_falling, temp_rising):
                if t is not None:
                    mech_zs.append(abs(t.z))
        mechanism_strength = _clamp01((max(mech_zs) if mech_zs else 0.0) / Z_STRONG)
        # Geometric mean: both the decay and its mechanism must be clear (r01).
        signal_strength = math.sqrt(decay_strength * mechanism_strength)

        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )
        # A short sustained window is a weaker basis (analogous to r03's lifetime
        # cap): keep it capped even after THRESHOLDS_UNCALIBRATED is flipped off.
        short_window = window_s < TRUSTED_WINDOW_S
        if THRESHOLDS_UNCALIBRATED or short_window:
            confidence = min(confidence, _UNCALIBRATED_CEILING)

        # ---- Mechanism labelling, cause, fix ------------------------------
        mechanisms = []
        if memory_pressure:
            mechanisms.append("memory_pressure")
        if thermal_throttle:
            mechanisms.append("thermal_throttle")
        mechanism = "+".join(mechanisms)

        early, recent = _early_mean(tput), _recent_mean(tput)
        total_drop_frac = _clamp01((early - recent) / early) if early > 0 else 0.0

        clause_parts = []
        if memory_pressure:
            if queue_rising is not None and queue:
                clause_parts.append(f"request queue time climbed to {queue[-1]:.0f}ms")
            if preempt_rising is not None and preempt:
                clause_parts.append(f"the preemption rate climbed to {preempt[-1]:.0%}")
        if thermal_throttle and sm_clock:
            temp_clause = f" as temperature reached {temp[-1]:.0f}°C" if temp else ""
            clause_parts.append(
                f"the SM clock fell from {sm_clock[0]:.0f} to {sm_clock[-1]:.0f} MHz{temp_clause}"
            )
        mechanism_clause = "; ".join(clause_parts) or "a building bottleneck"
        mechanism_noun = (
            "a building memory bottleneck" if memory_pressure and not thermal_throttle
            else "a thermal throttle" if thermal_throttle and not memory_pressure
            else "memory pressure and thermal throttling"
        )

        cause = (
            f"Generation throughput decayed ~{decay_frac_per_min:.0%}/min over a "
            f"{window_s:.0f}s run ({early:.0f}→{recent:.0f} tok/s, ~{total_drop_frac:.0%} "
            f"below the early-run level) while {mechanism_clause} — the decline tracks "
            f"{mechanism_noun}, not reduced load."
        )

        if thermal_throttle and not memory_pressure:
            fix = _FIX_THERMAL
        elif dx.inference_engine in ("vllm", "sglang"):
            fix = _FIX_MEMORY_VLLM
        else:
            fix = _FIX_MEMORY_GENERIC

        notes = (
            f"geometric mean of decay and mechanism strength (mechanism={mechanism}, "
            f"window={window_s:.0f}s, n={len(hist)}, MK p={tput_trend.p_value:.3f})"
        )
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"
        elif short_window:
            notes += f"; short window (<{TRUSTED_WINDOW_S:.0f}s), confidence capped at {_UNCALIBRATED_CEILING}"

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=fix,
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=self._data_completeness(hist),
                notes=notes,
            ),
            evidence={
                "samples": len(hist),
                "window_seconds": round(window_s, 1),
                "throughput_first": round(early, 1),
                "throughput_last": round(recent, 1),
                "decay_frac_per_min": round(decay_frac_per_min, 4),
                "throughput_drop_frac": round(total_drop_frac, 4),
                "mann_kendall_p": round(tput_trend.p_value, 4),
                "mechanism": mechanism,
                "preemption_rate_last": round(preempt[-1], 4) if preempt else None,
                "queue_time_ms_last": round(queue[-1], 1) if queue else None,
                "sm_clock_first": round(sm_clock[0], 0) if sm_clock else None,
                "sm_clock_last": round(sm_clock[-1], 0) if sm_clock else None,
                "gpu_temp_last": round(temp[-1], 0) if temp else None,
                "inference_engine": dx.inference_engine,
                "signal_strength": round(signal_strength, 2),
            },
        )

    @staticmethod
    def _data_completeness(hist: list[ThroughputSample]) -> float:
        """Fraction of corroborator series that had enough points to test."""
        attrs = ("preemption_rate", "queue_time_ms", "num_requests_waiting",
                 "sm_clock_mhz", "gpu_temp_c")
        present = 0
        for attr in attrs:
            n = sum(1 for s in hist if getattr(s, attr) is not None)
            if n >= 3:
                present += 1
        return present / len(attrs)
