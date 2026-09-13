"""
r12 — Queue growth (super-linear waiting-queue trend).

Fires when the scheduler's waiting backlog (`num_requests_waiting`) trends UP
over a sustained run AND the growth is *super-linear* (accelerating) rather than
a steady, absorbed overload. A backlog that grows linearly is a server running
hot but keeping pace; a backlog whose growth rate is itself increasing is a
server past its saturation point, where by Little's Law and the M/M/1 result
L = rho/(1 - rho) the queue diverges as utilisation -> 1. That acceleration is
the signature of a queue *death spiral*: every unit of time the server falls
further behind than the last, and TTFT/latency follow the backlog up.

The fix is to SHED or ROUTE load — reduce offered load, add replicas, or move to
least-outstanding-requests routing. This is deliberately the OPPOSITE remedy to
its sibling R-SCHED-CAP (scheduler cap, not yet built): both present the same
symptom surface (a persistent waiting queue), but R-SCHED-CAP fires when the
ceiling is *artificial* — `num_requests_running` pinned at `max-num-seqs` while
KV util has headroom — and its fix is to *raise* the ceiling. r12 fires on
genuine saturation (the GPU is already full); raising the ceiling there only
deepens the spiral. The two are disambiguated by KV headroom, and MUST be
declared a symmetric CONFLICT_SET in engine/relations.py the moment R-SCHED-CAP
lands (see the reservation note at the bottom of this module). They are not
registered as conflicting yet because engine/relations._validate rejects
relations that name a rule id no registry knows.

This is a LIVE-ONLY rule: it reads `dx.throughput_history`, the per-tick series
the watch loop appends. The one-shot `diagnose` path has no time base, so r12
abstains there (INSUFFICIENT_DATA) — a single snapshot cannot distinguish a
server momentarily at its ceiling from one accelerating into a spiral.

The trend is detected with the Mann-Kendall monotonic-trend test (significance)
and the acceleration with a robust first-half vs second-half Theil-Sen slope
comparison (convexity), so a bursty-but-mean-reverting backlog does not
masquerade as a spiral. While THRESHOLDS_UNCALIBRATED is True the firing bands
are calibration seeds and confidence is capped.

Substrate note: the >=2-strictly-increasing-points gate (`_monotone_series`)
follows the watch loop's windowed-history validation. A sample whose
timestamp does not advance is dropped, and a series only trends at >= 2 points.
"""

from __future__ import annotations

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
# A spiral needs a sustained run to be visible: a two-tick blip is not a trend,
# and the Mann-Kendall normal approximation is only asymptotically valid, so
# below the sample floor these bands are conservative seeds and confidence stays
# capped. ACCEL_MIN is how much steeper the *second half's* backlog slope must be
# than the first half's to count as accelerating rather than linear overload.
# WAITING_FLOOR keeps a backlog oscillating around 0-1 from ever looking like a
# spiral — there must be a real, growing queue.
# --------------------------------------------------------------------------- #
MIN_SAMPLES = 6
MIN_WINDOW_S = 60.0
TRUSTED_WINDOW_S = 180.0        # below this the window is short -> keep confidence capped
P_MAX = 0.10                    # Mann-Kendall two-sided p for a significant uptrend
ACCEL_MIN = 0.50                # 2nd-half slope must exceed 1st-half by >= this fraction
ACCEL_SEVERE = 2.0              # 2nd-half slope this many x the 1st = full-strength spiral
WAITING_FLOOR = 1.0            # recent backlog must exceed this (else no real queue)
Z_STRONG = 3.0                  # |MK z| that counts as a fully-clear backlog trend
KV_HEADROOM_MAX = 0.75          # KV util below this = headroom -> the ceiling is likely
                                # artificial (R-SCHED-CAP territory), not genuine
                                # saturation; r12 must not prescribe shedding there

THRESHOLDS_UNCALIBRATED = True  # flip to False only after a real-GPU / field sweep

_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.80
_CONFIDENCE_CEILING = 0.90
_UNCALIBRATED_CEILING = 0.65    # confidence cap while uncalibrated / short window

_FIX_SHED_VLLM = (
    "The GPU is at genuine saturation and the waiting queue is accelerating — "
    "shed or redistribute load, do not raise the scheduler ceiling. If this is "
    "one replica, scale horizontally and route with least-outstanding-requests; "
    "if offered load is elastic, apply admission control / backpressure upstream. "
    "Raising --max-num-seqs here deepens the spiral (contrast r-sched-cap, which "
    "fires only when the ceiling is artificial and KV util has headroom)."
)
_FIX_SHED_GENERIC = (
    "The serving tier is past its saturation point and the request backlog is "
    "growing super-linearly — reduce offered load, add serving replicas, or add "
    "admission control upstream. Adding capacity or shedding load is the remedy; "
    "a larger in-flight batch will not help a GPU that is already full."
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _monotone_series(
    hist: list[ThroughputSample], attr: str
) -> tuple[list[float], list[float]]:
    """Parallel (t, value) lists for samples where ``attr`` is populated and the
    timestamp strictly advances, matching the windowed-history validation:
    a sample whose ``t`` does not advance past the last kept sample is dropped
    (a stalled/duplicate scrape must not fake a trend point). The caller trends
    only when the result has >= 2 points.
    """
    ts: list[float] = []
    ys: list[float] = []
    for s in hist:
        v = getattr(s, attr)
        if v is None:
            continue
        if ts and s.t <= ts[-1]:
            continue  # keep timestamps strictly increasing
        ts.append(s.t)
        ys.append(float(v))
    return ts, ys


def _trend(ys: list[float]) -> Optional[TrendTest]:
    """Mann-Kendall result, or None when there are too few points."""
    return mann_kendall(ys)


def _half_slopes(ts: list[float], ys: list[float]) -> Optional[tuple[float, float]]:
    """Theil-Sen slope of the first and second half of the series. None if either
    half has < 2 points (can't slope a single point)."""
    n = len(ys)
    if n < 4:
        return None
    mid = n // 2
    first = theil_sen_slope(ts[:mid], ys[:mid])
    second = theil_sen_slope(ts[mid:], ys[mid:])
    if first is None or second is None:
        return None
    return first, second


def _recent_mean(ys: list[float]) -> float:
    k = max(1, len(ys) // 3)
    return statistics.mean(ys[-k:])


class QueueGrowthRule(Rule):
    """The scheduler's waiting backlog accelerates upward — genuine saturation."""

    rule_id = "r12"
    title = "Queue growth"
    references = (
        "Little, 'A Proof for the Queuing Formula L = lambda W,' Operations "
        "Research 9(3), 1961 — with the M/M/1 result L = rho/(1 - rho), the "
        "expected queue length diverges super-linearly as utilisation rho -> 1, "
        "so an accelerating backlog is the signature of approaching saturation.",
        "Yu et al., 'Orca: A Distributed Serving System for Transformer-Based "
        "Generative Models,' OSDI 2022 — continuous batching admits until the "
        "batch ceiling; past it the waiting queue is the saturation signal.",
        "Kwon et al., 'Efficient Memory Management for Large Language Model "
        "Serving with PagedAttention,' SOSP 2023. arXiv:2309.06180 — vLLM "
        "scheduler and waiting-queue semantics (num_requests_waiting).",
        "vLLM Metrics design. https://docs.vllm.ai/en/latest/design/metrics/",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required: a sustained per-tick backlog series ----------------
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

        ts, waiting = _monotone_series(hist, "num_requests_waiting")
        if len(waiting) < MIN_SAMPLES:
            return InsufficientData(missing=("throughput_history.num_requests_waiting",))
        window_s = ts[-1] - ts[0]
        if window_s < MIN_WINDOW_S:
            return InsufficientData(
                reason=(
                    f"waiting-queue series spans {window_s:.0f}s; need "
                    f">= {MIN_WINDOW_S:.0f}s to judge a sustained trend."
                )
            )

        # ---- A real, growing queue (not oscillation around zero) ----------
        if _recent_mean(waiting) < WAITING_FLOOR:
            return Abstention.BELOW_THRESHOLD

        # ---- Self-guard: genuine saturation, not an artificial ceiling ----
        # An accelerating backlog with ample KV headroom is R-SCHED-CAP's
        # signature (the admission ceiling is set too low), and its fix —
        # raise the ceiling — is the OPPOSITE of ours. Shedding load there is
        # the wrong-remedy failure this rule must never prescribe. Absent the
        # gauge, fire (completeness reflects the missing corroborator).
        if dx.kv_cache_util is not None and dx.kv_cache_util < KV_HEADROOM_MAX:
            return Abstention.BELOW_THRESHOLD

        # ---- Leg 1: a significant, monotonic UPWARD backlog trend ---------
        trend = _trend(waiting)
        if trend is None or not (trend.s > 0 and trend.p_value < P_MAX):
            # Flat, draining, or bursty-but-mean-reverting -> not a spiral.
            return Abstention.BELOW_THRESHOLD

        # ---- Leg 2: super-linear growth (convex: 2nd-half slope > 1st) ----
        halves = _half_slopes(ts, waiting)
        if halves is None:
            return Abstention.BELOW_THRESHOLD
        first_slope, second_slope = halves
        if first_slope <= 0 or second_slope <= 0:
            return Abstention.BELOW_THRESHOLD
        accel_ratio = second_slope / first_slope
        if accel_ratio < (1.0 + ACCEL_MIN):
            # Rising but linear (steady, absorbed overload) — not accelerating.
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength ----------------------------------------------
        trend_strength = _clamp01(abs(trend.z) / Z_STRONG)
        accel_strength = _clamp01(
            (accel_ratio - (1.0 + ACCEL_MIN)) / (ACCEL_SEVERE - (1.0 + ACCEL_MIN))
        )
        # Geometric mean: both the uptrend and its acceleration must be clear.
        signal_strength = (trend_strength * accel_strength) ** 0.5

        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )
        short_window = window_s < TRUSTED_WINDOW_S
        if THRESHOLDS_UNCALIBRATED or short_window:
            confidence = min(confidence, _UNCALIBRATED_CEILING)

        # ---- Corroborator: is the experienced wait growing too? -----------
        _, queue_ms = _monotone_series(hist, "queue_time_ms")
        queue_trend = _trend(queue_ms) if len(queue_ms) >= 2 else None
        queue_rising = queue_trend is not None and queue_trend.s > 0 and queue_trend.p_value < P_MAX

        early, recent = waiting[0], waiting[-1]
        wait_clause = (
            f"; mean request queue time climbed to {queue_ms[-1]:.0f}ms"
            if queue_rising and queue_ms
            else ""
        )
        cause = (
            f"The scheduler backlog grew super-linearly over a {window_s:.0f}s run "
            f"(num_requests_waiting {early:.0f}->{recent:.0f}, second-half growth "
            f"{accel_ratio:.1f}x the first) — an accelerating queue is the signature "
            f"of a server past its saturation point, not steady overload{wait_clause}."
        )
        fix = _FIX_SHED_VLLM if dx.inference_engine in ("vllm", "sglang") else _FIX_SHED_GENERIC

        notes = (
            f"geometric mean of backlog-trend and acceleration strength "
            f"(MK p={trend.p_value:.3f}, accel={accel_ratio:.2f}x, window={window_s:.0f}s, "
            f"n={len(waiting)})"
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
                data_completeness=self._data_completeness(dx, hist),
                notes=notes,
            ),
            evidence={
                "samples": len(waiting),
                "kv_cache_util": dx.kv_cache_util,
                "window_seconds": round(window_s, 1),
                "waiting_first": round(early, 1),
                "waiting_last": round(recent, 1),
                "first_half_slope_per_s": round(first_slope, 4),
                "second_half_slope_per_s": round(second_slope, 4),
                "acceleration_ratio": round(accel_ratio, 3),
                "mann_kendall_p": round(trend.p_value, 4),
                "queue_time_ms_last": round(queue_ms[-1], 1) if queue_ms else None,
                "queue_time_rising": queue_rising,
            },
        )

    @staticmethod
    def _data_completeness(dx: DiagnosisInput, hist: list[ThroughputSample]) -> float:
        """Fraction of the corroborators r12 can use that are actually present.
        The backlog gauge is required to fire; the queue-time series and the KV
        utilisation gauge (the artificial-ceiling self-guard) are bonuses.
        Completeness reflects how much of the ideal evidence we had."""
        _, queue_ms = _monotone_series(hist, "queue_time_ms")
        score = 0.5
        if len(queue_ms) >= 2:
            score += 0.25
        if dx.kv_cache_util is not None:
            score += 0.25
        return score


# --------------------------------------------------------------------------- #
# RESERVATION — do NOT delete. When R-SCHED-CAP (scheduler cap) lands and takes
# its ledger id, register the symmetric conflict in engine/relations.py:
#     CONFLICT_SETS += (frozenset({"r12", "<r-sched-cap id>"}),)
# and resolve toward the rule whose KV-headroom gate matched (R-SCHED-CAP has
# headroom; r12 does not). engine/relations._validate rejects a set that names
# an unknown id, which is why the conflict is not declared here today.
# --------------------------------------------------------------------------- #
