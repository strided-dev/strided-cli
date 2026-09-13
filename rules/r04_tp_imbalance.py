"""
r04 — Tensor-parallel rank imbalance.

Fires when one tensor-parallel rank is a consistent outlier — slower than its
peers. Tensor-parallel all-reduce is a synchronisation barrier, so the slowest rank
gates every step: all GPUs advance at the straggler's pace. The fix is a
differential-diagnosis investigation (hardware fault vs noisy neighbour vs sharding
imbalance), NOT a generic "rebalance."

The signal is **per-rank SM clock** (`tp_rank_sm_clocks`, raw DCGM telemetry). A
throttling GPU down-clocks, so the rule derives a comparable slowness ratio in-rule
— `slowness_from_sm_clocks` returns `max_clock / clock`, so the slowest rank gets
the largest value — and runs the robust outlier stat on that. The *ratio* is a rule
concept and is computed here, never stored in the schema. gpu_util is not used as a
fallback: its straggler direction under the NCCL barrier is ambiguous (fast ranks
spin-wait high), so a dump with no SM clocks yields no signal and this rule abstains
rather than risk an inverted straggler call. This is weak from static dumps — true
step timing needs a multi-rank profiler trace most customers lack — and a single
snapshot cannot confirm "consistently," so confidence is capped while
THRESHOLDS_UNCALIBRATED is True, and the rule abstains below 3 ranks (you cannot
call one rank an outlier against a single peer).
"""

from __future__ import annotations

import statistics
from typing import Optional

from rules._stats import (
    MIN_RANKS,
    PERSISTENCE_TICKS,
    ImbalanceStats,
    compute_imbalance,
    outlier_persists,
    slowness_from_sm_clocks,
)
from rules.base import Abstention, ConfidenceBreakdown, Diagnosis, Rule, RuleResult
from schema import DiagnosisInput

# The pure stat helpers live in rules/_stats.py (shared infrastructure) so r05 can
# reuse them without importing r04 — rules stay blind to one another. Re-exported
# here for backward compatibility.
__all__ = [
    "TpRankImbalanceRule",
    "compute_imbalance",
    "ImbalanceStats",
    "MIN_RANKS",
    "PERSISTENCE_TICKS",
    "outlier_persists",
    "slowness_from_sm_clocks",
]

# --------------------------------------------------------------------------- #
# Thresholds — CALIBRATION SEEDS, not literature values.
# Well-balanced TP shows a few-percent per-rank jitter; a rank >=10% slower than
# the median peer is beyond normal jitter (Megatron-LM / Narayanan et al. SC'21
# establish the barrier mechanism, not a numeric threshold). MODIFIED_Z_MIN is the
# Iglewicz-Hoaglin robust-outlier cutoff. Validate on real multi-rank dumps before
# trusting the bands; until then THRESHOLDS_UNCALIBRATED caps confidence.
# --------------------------------------------------------------------------- #
LAG_MIN = 0.10           # slowest rank >= 10% above the median peer
MODIFIED_Z_MIN = 3.5     # robust-outlier threshold (Iglewicz & Hoaglin 1993)
_Z_DISPLAY_CAP = 99.9    # cap the (possibly infinite) z-score for display/evidence

# Confidence model (see r01 for the geometric-mean rationale).
_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.80
_CONFIDENCE_CEILING = 0.90
THRESHOLDS_UNCALIBRATED = True
_UNCALIBRATED_CEILING = 0.65

# Temperature corroboration (booster, never a gate): a genuine thermal straggler runs
# hottest. When the slowest rank is also the hottest by this margin, nudge confidence.
_TEMP_CORROBORATION_DELTA_C = 5.0
_TEMP_CORROBORATION_BONUS = 0.05

# The persistence gate itself (`PERSISTENCE_TICKS`, `outlier_persists`) lives in
# rules/_stats.py so r05's self-guard can apply the same bar without importing
# r04. Re-exported above; see that module for the rationale and field evidence.


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _fix_for_rank(idx: int) -> str:
    return (
        f"Investigate rank {idx} specifically — do not blindly re-shard. In order of "
        "likelihood: (1) Hardware fault — confirm the GPU is not thermal-throttling or "
        "down-clocking (check DCGM temperature and SM clocks) and that its NVLink/PCIe "
        "links are healthy; a single hot or degraded GPU produces exactly this "
        "barrier-straggler signature. (2) Noisy neighbour — on a shared node, another "
        "tenant on that GPU can steal SM cycles or memory bandwidth. (3) Sharding "
        "imbalance — only after the hardware is ruled out, check for uneven layer/head "
        "assignment across ranks and rebalance the partition. Cross-check per-GPU "
        "temperature, power and clocks before re-sharding. (Shoeybi et al. 2019; "
        "Narayanan et al. SC'21.)"
    )


class TpRankImbalanceRule(Rule):
    """One tensor-parallel rank lags its peers and gates the all-reduce barrier."""

    rule_id = "r04"
    title = "Tensor-parallel rank imbalance"
    references = (
        "Shoeybi et al., 'Megatron-LM: Training Multi-Billion Parameter Language "
        "Models Using Model Parallelism,' arXiv:1909.08053.",
        "Narayanan et al., 'Efficient Large-Scale Language Model Training on GPU "
        "Clusters Using Megatron-LM,' SC'21. arXiv:2104.04473.",
        "Pope et al., 'Efficiently Scaling Transformer Inference,' MLSys 2023. "
        "arXiv:2211.05102. (Tensor-parallel partitioning cost model.)",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required field: raw per-rank SM clocks ------------------------
        # The schema carries only the raw clocks; r04 derives its own comparable
        # slowness ratio (max_clock / clock, higher = slower) from them. gpu_util is
        # not a fallback — see the module docstring — so absent clocks mean no signal.
        clocks = dx.tp_rank_sm_clocks
        if clocks is None or len(clocks) < MIN_RANKS:
            # None, single-GPU, or TP=2: cannot identify an outlier rank.
            return Abstention.INSUFFICIENT_DATA

        timings = slowness_from_sm_clocks(clocks)
        if timings is None:
            # A non-positive clock (corrupt scrape) leaves no usable slowness signal.
            return Abstention.INSUFFICIENT_DATA

        stats = compute_imbalance(timings)
        if stats is None:
            return Abstention.INSUFFICIENT_DATA

        # ---- Firing condition ---------------------------------------------
        if not (stats.relative_lag > LAG_MIN and stats.modified_z > MODIFIED_Z_MIN):
            return Abstention.BELOW_THRESHOLD

        # ---- Persistence gate (live path only) -----------------------------
        # With a clock history available, require the SAME rank to be the
        # outlier across consecutive samples. A throttled rank stays slow; an
        # idle-downclocked one does not, and a single snapshot cannot tell them
        # apart. The one-shot `diagnose` path has no history and is
        # unaffected — a static dump carries no DVFS transient to confuse.
        if not outlier_persists(
            dx.tp_rank_history,
            stats.slowest_index,
            lag_min=LAG_MIN,
            z_min=MODIFIED_Z_MIN,
        ):
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength ----------------------------------------------
        lag_strength = _clamp01((stats.relative_lag - LAG_MIN) / 0.40)  # 10%→0, 50%→1
        z_strength = _clamp01(
            (stats.modified_z - MODIFIED_Z_MIN) / (10.0 - MODIFIED_Z_MIN)
        )  # 3.5→0, 10→1
        signal_strength = (lag_strength * z_strength) ** 0.5  # geometric mean

        # ---- Temperature corroboration (booster, not a gate) --------------
        # A genuine thermal straggler runs hottest. If the slowest rank is also the
        # hottest by a clear margin, that independently corroborates throttling and
        # nudges confidence; firing stays owned by lag + z-score alone.
        temps = dx.tp_rank_temps
        slowest_temp: Optional[float] = None
        temp_margin: Optional[float] = None
        temp_corroborated = False
        if temps is not None and len(temps) == len(timings):
            slowest_temp = temps[stats.slowest_index]
            peer_temps = temps[: stats.slowest_index] + temps[stats.slowest_index + 1 :]
            temp_margin = slowest_temp - statistics.median(peer_temps)
            temp_corroborated = (
                slowest_temp == max(temps) and temp_margin >= _TEMP_CORROBORATION_DELTA_C
            )

        confidence = _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        if temp_corroborated:
            confidence += _TEMP_CORROBORATION_BONUS
        confidence = min(_CONFIDENCE_CEILING, confidence)
        if THRESHOLDS_UNCALIBRATED:
            confidence = min(confidence, _UNCALIBRATED_CEILING)

        # ---- Build the diagnosis ------------------------------------------
        z_display = (
            f">{_Z_DISPLAY_CAP:.0f}"
            if stats.modified_z == float("inf")
            else f"{stats.modified_z:.1f}"
        )
        z_value = min(stats.modified_z, _Z_DISPLAY_CAP)  # finite, JSON/format-safe

        cause = (
            f"Tensor-parallel rank {stats.slowest_index} was the slowest of "
            f"{stats.num_ranks} ranks: its per-step time sat {stats.relative_lag:.0%} "
            f"above the median rank and stood apart as an outlier (modified z-score "
            f"{z_display}). Because the tensor-parallel all-reduce is a synchronisation "
            f"barrier, the slowest rank gates every step — all {stats.num_ranks} GPUs "
            f"advance at rank {stats.slowest_index}'s pace."
        )
        if temp_corroborated:
            cause += (
                f" Rank {stats.slowest_index} also ran hottest at {slowest_temp:.0f}°C "
                f"({temp_margin:.0f}°C above its peers), consistent with thermal throttling."
            )

        notes = (
            "signal from per-rank SM clock; robust outlier (median + MAD); "
            "geometric mean of lag and z-score strength"
        )
        if temp_corroborated:
            notes += "; temperature corroborates throttling"
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"

        evidence: dict[str, object] = {
            # Derived in-rule from the raw clocks (max_clock / clock); not a schema field.
            "tp_rank_slowness": [round(t, 3) for t in timings],
            "tp_rank_sm_clocks": [round(c, 1) for c in clocks],
            "slowest_rank": stats.slowest_index,
            "relative_lag": round(stats.relative_lag, 3),
            "modified_z_score": round(z_value, 1),
            "num_ranks": stats.num_ranks,
            "signal_strength": round(signal_strength, 2),
        }
        if slowest_temp is not None:
            evidence["slowest_temp_c"] = round(slowest_temp, 1)
            evidence["temp_corroborated"] = temp_corroborated

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=_fix_for_rank(stats.slowest_index),
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=self._data_completeness(dx, stats.num_ranks),
                notes=notes,
            ),
            evidence=evidence,
        )

    @staticmethod
    def _data_completeness(dx: DiagnosisInput, num_ranks: int) -> float:
        """Fraction of optional corroborating context the rule had available.

        The raw SM clocks are required to fire, so they are not counted here; this
        measures the *extra* context that strengthens the call.
        """
        optional = [
            num_ranks >= 4,                          # more ranks → more reliable outlier
            dx.tensor_parallel_size is not None,     # confirms these are TP (not PP) ranks
            dx.num_gpus is not None,
            dx.tp_rank_temps is not None,            # thermal corroborator available
        ]
        return sum(optional) / len(optional)
