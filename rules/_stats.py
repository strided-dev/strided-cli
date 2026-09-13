"""
Shared pure statistics for rules — no rule-specific or cross-rule logic.

`compute_imbalance` is the robust median + MAD outlier helper used by r04
(tensor-parallel rank imbalance) and reused by r05's r04 self-guard. It lives
here, not inside r04, so r05 can use it without importing another rule: rules
are, by contract (see rules/base.py and engine/relations.py's docstring,
"a rule is, by contract, blind to every other rule"), blind to one another.
This module is shared infrastructure, so depending on it keeps that invariant
literally true.

`outlier_persists` is here for the same reason. r04 gates firing on it, and
r05's self-guard consults it so it only defers to a straggler r04 would
actually claim — without either rule importing the other. It reads
`TpRankSample` (schema 1.6.0), the one schema type this module touches: the
persistence question is inherently about a *series*, not a single vector.

`theil_sen_slope` and `mann_kendall` are the robust monotonic-trend toolkit r06
(throughput decay) uses on the per-tick throughput series. Both are
non-parametric and resistant to the throughput sawtooth — the right tools for
"did tokens/sec genuinely trend down, or is this noise?" (Theil 1950; Sen 1968;
Mann 1945; Kendall 1975), the same way r04 grounds its outlier cutoff in
Iglewicz & Hoaglin. The slope gives magnitude; Mann–Kendall gives significance.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional

from schema.diagnosis_input import TpRankSample

# Below this you cannot distinguish an outlier from a pair.
MIN_RANKS = 3
# MAD → std-dev consistency constant (Iglewicz & Hoaglin 1993).
_MAD_SCALE = 0.6745

# How many consecutive samples must agree on the outlier rank before it counts as
# persistent. A thermally-throttled rank stays slow tick after tick; a rank that
# merely DVFS'd to the idle floor between load bursts does not. Field testing
# measured r04 firing on 16/100 ticks against a HEALTHY pack from single
# snapshots, blaming ranks near-uniformly (6/6/4) — noise, not detection;
# requiring 2 consecutive samples reduced that to 0/100 on the same captures
# Lives here rather than in r04
# so r05's self-guard can apply the same bar without importing r04.
PERSISTENCE_TICKS = 2


def slowness_from_sm_clocks(clocks: Optional[list[float]]) -> Optional[list[float]]:
    """Per-rank *slowness ratio* from raw SM clocks: ``max_clock / clock``.

    A throttling GPU down-clocks, so the slowest rank has the lowest clock and the
    largest ratio (the pack baselines at ~1.0). This derivation is the rule's job,
    not the schema's — the schema carries only the raw ``tp_rank_sm_clocks`` (raw
    telemetry); r04 and r05's self-guard call this to turn them into a comparable
    slowness signal. gpu_util is deliberately not a fallback: its straggler
    direction is ambiguous under the NCCL barrier, so absent clocks yield no signal.

    Returns None when there is nothing usable (no clocks, or a non-positive clock
    that would divide by zero).
    """
    if not clocks or min(clocks) <= 0:
        return None
    max_clock = max(clocks)
    return [max_clock / c for c in clocks]


@dataclass(frozen=True)
class ImbalanceStats:
    """Robust outlier statistics for the slowest TP rank."""

    slowest_index: int
    median_peers: float
    relative_lag: float   # (slowest - median_peers) / median_peers
    modified_z: float     # robust z-score of the slowest rank (may be inf)
    num_ranks: int


def compute_imbalance(timings: list[float]) -> Optional[ImbalanceStats]:
    """Locate the slowest rank and quantify how far it stands from the pack.

    Returns None when there are too few ranks or the peer baseline is degenerate.
    Uses median + MAD (robust to the very outlier we are hunting) rather than
    mean + std-dev, which the outlier would itself inflate.
    """
    n = len(timings)
    if n < MIN_RANKS:
        return None

    slowest = max(timings)
    slowest_index = timings.index(slowest)
    peers = timings[:slowest_index] + timings[slowest_index + 1 :]
    median_peers = statistics.median(peers)
    if median_peers <= 0:
        return None

    relative_lag = (slowest - median_peers) / median_peers

    median_all = statistics.median(timings)
    mad = statistics.median([abs(t - median_all) for t in timings])
    if mad > 0:
        modified_z = _MAD_SCALE * (slowest - median_all) / mad
    else:
        # Peers are identical; the slowest above them is a perfect outlier.
        modified_z = float("inf") if slowest > median_all else 0.0

    return ImbalanceStats(slowest_index, median_peers, relative_lag, modified_z, n)


def outlier_persists(
    history: Optional[list[TpRankSample]],
    slowest_index: int,
    *,
    lag_min: float,
    z_min: Optional[float] = None,
    ticks: int = PERSISTENCE_TICKS,
) -> bool:
    """True when `slowest_index` is the outlier across the last `ticks` samples.

    A single SM-clock snapshot cannot separate a *thermally throttled* rank from
    one that has merely DVFS'd to the idle floor between load bursts — both read
    as "slow". A throttled rank stays slow tick after tick; an idle one does not,
    so agreement across consecutive samples separates them on physics rather than
    on a tuned threshold.

    Callers pass their own bands, because they ask different questions of the
    same series: r04 requires its full firing condition (lag *and* z-score) to
    hold in every sample, while r05's self-guard passes only `lag_min` —
    deliberately broader, since it steps aside for any *plausible* straggler, not
    only one that would clear r04's stricter isolation bar.

    Returns True when there is no history, or when the series is shorter than
    `ticks`: the one-shot path keeps its single-snapshot behaviour (a static dump
    carries no DVFS transient to confuse), and a live watch must not go blind on
    its first tick — the transients this targets die at the second sample anyway.
    """
    if not history or len(history) < ticks:
        return True
    for sample in history[-ticks:]:
        timings = slowness_from_sm_clocks(sample.sm_clocks)
        if timings is None:
            return False
        st = compute_imbalance(timings)
        # Every recent sample must independently flag the SAME rank, at the
        # bands the caller fires on.
        if st is None or st.slowest_index != slowest_index:
            return False
        if not st.relative_lag > lag_min:
            return False
        if z_min is not None and not st.modified_z > z_min:
            return False
    return True


# --------------------------------------------------------------------------- #
# Robust monotonic-trend toolkit (r06: throughput decay)
# --------------------------------------------------------------------------- #

def theil_sen_slope(xs: list[float], ys: list[float]) -> Optional[float]:
    """Median of all pairwise slopes — the robust Theil–Sen trend estimate.

    Returns the slope of ``ys`` against ``xs`` in y-units per x-unit. Unlike
    ordinary least squares it has a ~29% breakdown point and assumes no error
    distribution, so a few throughput spikes (the decode sawtooth, a GC pause)
    cannot drag the estimate the way they drag OLS. Returns None when fewer than
    two points have distinct x (no slope is defined). (Theil 1950; Sen 1968.)
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[j] - xs[i]
            if dx == 0:
                continue
            slopes.append((ys[j] - ys[i]) / dx)
    if not slopes:
        return None
    return statistics.median(slopes)


@dataclass(frozen=True)
class TrendTest:
    """Mann–Kendall monotonic-trend result. ``s`` sign gives direction."""

    s: float          # S statistic = sum of pairwise signs (>0 up, <0 down)
    z: float          # continuity-corrected normal z-score
    p_value: float    # two-sided significance (normal approximation)


def mann_kendall(ys: list[float]) -> Optional[TrendTest]:
    """Non-parametric Mann–Kendall test for a monotonic trend.

    Detects an increasing or decreasing trend without assuming normality and with
    low sensitivity to outliers (Mann 1945; Kendall 1975). ``s > 0`` is an upward
    trend, ``s < 0`` downward; ``p_value`` is two-sided via the tie-corrected
    normal approximation (asymptotically valid for n ≳ 10; below that it is a
    conservative calibration seed, which is why r06 also requires a minimum sample
    count and caps confidence). Returns None for fewer than three points.
    """
    n = len(ys)
    if n < 3:
        return None

    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = ys[j] - ys[i]
            s += (d > 0) - (d < 0)  # sign(d), as int

    # Variance with the standard tie correction.
    counts: dict[float, int] = {}
    for v in ys:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values())
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p_value = 2.0 * (1.0 - statistics.NormalDist().cdf(abs(z)))
    return TrendTest(s=float(s), z=z, p_value=p_value)
