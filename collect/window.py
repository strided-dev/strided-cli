"""Windowed throughput: cumulative vLLM counters → per-second rates.

The one-shot path leaves request/token throughput unset because a single scrape
of cumulative counters has no time base (see ``parsers/vllm.py``). ``watch`` has
two scrapes, so it can divide deltas by elapsed wall-clock — the live path's
distinguishing capability. A counter going backwards means the server restarted;
we rebaseline and emit no rate for that window rather than a negative spike.

Scope note: these rates populate the *existing* DiagnosisInput throughput fields
(which the one-shot path leaves ``None``). They feed the live status line and any
future rate-based rule; they do **not** change r01–r03 firing, which read
point-in-time gauges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schema.diagnosis_input import TpRankSample
from schema import ThroughputSample

# vLLM cumulative-counter field name → the DiagnosisInput throughput field it feeds.
_RATE_FIELDS: dict[str, str] = {
    "request_success_total": "request_throughput",
    "generation_tokens_total": "token_throughput_gen",
    "prompt_tokens_total": "token_throughput_prompt",
}

# Cumulative counters whose per-window *deltas* (not rates) feed the r06
# co-samples: the session derives a window preemption rate (Δpreemptions /
# Δsuccesses) and a window mean queue time (Δsum / Δcount) from these, so a
# long-running server's lifetime totals cannot dilute current pressure.
_DELTA_FIELDS: tuple[str, ...] = (
    "num_preemptions_total",
    "request_success_total",
    "request_queue_time_ms_sum",
    "request_queue_time_count",
)

# Every counter the window consumes — what the session must snapshot per tick.
COUNTER_FIELDS: tuple[str, ...] = tuple(dict.fromkeys((*_RATE_FIELDS, *_DELTA_FIELDS)))


@dataclass(frozen=True)
class WindowUpdate:
    """One window's derived views of the cumulative counters.

    ``rates`` maps DiagnosisInput throughput fields to per-second rates (as
    before); ``deltas`` maps raw counter names to their per-window increments.
    Both are empty on the first tick / non-positive dt; a counter that went
    backwards (server restart) is skipped individually in both views.
    """

    rates: dict[str, float]
    deltas: dict[str, float]


_EMPTY_UPDATE = WindowUpdate(rates={}, deltas={})


@dataclass
class ThroughputWindow:
    """Holds the previous counter snapshot to turn cumulative counters into rates."""

    _prev: Optional[dict[str, float]] = None
    _prev_t: Optional[float] = None

    def update(self, counters: dict[str, float], t: float) -> WindowUpdate:
        """Return the current window's ``WindowUpdate`` (rates + counter deltas).

        Empty on the first tick, on a non-positive ``dt``, or for any counter that
        went backwards (server restart). Only counters present in *both* the
        previous and current snapshot contribute. The baseline always advances to
        the latest snapshot so the next window is clean.
        """
        prev, prev_t = self._prev, self._prev_t
        self._prev = dict(counters)
        self._prev_t = t

        if prev is None or prev_t is None:
            return _EMPTY_UPDATE
        dt = t - prev_t
        if dt <= 0:
            return _EMPTY_UPDATE

        def _delta(counter_name: str) -> Optional[float]:
            cur = counters.get(counter_name)
            old = prev.get(counter_name)
            if cur is None or old is None:
                return None
            delta = cur - old
            if delta < 0:
                # Counter reset (restart); skip — baseline already advanced.
                return None
            return delta

        rates: dict[str, float] = {}
        for counter_name, field_name in _RATE_FIELDS.items():
            d = _delta(counter_name)
            if d is not None:
                rates[field_name] = d / dt

        deltas: dict[str, float] = {}
        for counter_name in _DELTA_FIELDS:
            d = _delta(counter_name)
            if d is not None:
                deltas[counter_name] = d

        return WindowUpdate(rates=rates, deltas=deltas)


@dataclass
class ThroughputHistory:
    """Bounded rolling per-tick throughput series for the r06 decay rule.

    The watch loop calls ``update`` once per tick with the current-window
    generation throughput (from ``ThroughputWindow``) plus the corroborators r06
    needs to attribute a decay. A sample is appended only when a window rate is
    available — the first tick has none, so it is skipped — and the buffer is
    evicted by both age and count, so a long-running watch holds a bounded, recent
    window rather than growing without limit. The returned list is what
    ``WatchSession`` injects onto the merged input as ``throughput_history``;
    r06 derives the trend, so this stores raw samples only.
    """

    max_samples: int = 240          # ~ the most recent N ticks
    max_window_s: float = 1800.0    # drop samples older than 30 min
    _samples: list[ThroughputSample] = field(default_factory=list)

    def update(
        self,
        t: float,
        token_throughput_gen: Optional[float],
        *,
        preemption_rate: Optional[float] = None,
        queue_time_ms: Optional[float] = None,
        num_requests_waiting: Optional[float] = None,
        sm_clock_mhz: Optional[float] = None,
        gpu_temp_c: Optional[float] = None,
    ) -> list[ThroughputSample]:
        """Append this tick's sample (if a rate exists) and return the series."""
        if token_throughput_gen is None:
            return list(self._samples)
        self._samples.append(
            ThroughputSample(
                t=t,
                token_throughput_gen=token_throughput_gen,
                preemption_rate=preemption_rate,
                queue_time_ms=queue_time_ms,
                num_requests_waiting=num_requests_waiting,
                sm_clock_mhz=sm_clock_mhz,
                gpu_temp_c=gpu_temp_c,
            )
        )
        cutoff = t - self.max_window_s
        self._samples = [s for s in self._samples if s.t >= cutoff]
        if len(self._samples) > self.max_samples:
            self._samples = self._samples[-self.max_samples:]
        return list(self._samples)


__all__ = ["ThroughputWindow", "ThroughputHistory", "WindowUpdate", "COUNTER_FIELDS"]


@dataclass
class TpRankHistory:
    """Bounded rolling per-tick series of per-rank SM clocks, for r04.

    Mirrors ``ThroughputHistory``: the watch loop calls ``update`` once per tick
    with the raw per-rank clocks (and temps when present); the buffer evicts by
    age and count so a long watch stays bounded. ``WatchSession`` injects the
    result as ``tp_rank_history``.

    Why r04 needs it: a single clock snapshot cannot separate a thermally
    throttled rank from one that merely DVFS'd to the idle floor between load
    bursts. Persistence across ticks can.
    A sample is appended only when clocks are present and the rank count is
    stable — a changed rank count means a different pack, not a slower one.
    """

    max_samples: int = 240
    max_window_s: float = 1800.0
    _samples: list[TpRankSample] = field(default_factory=list)

    def update(
        self,
        t: float,
        sm_clocks: Optional[list[float]],
        *,
        temps: Optional[list[float]] = None,
    ) -> list[TpRankSample]:
        """Append this tick's per-rank sample (if clocks exist) and return the series."""
        if not sm_clocks:
            return list(self._samples)
        # A changed rank count means the pack itself changed; comparing across
        # that boundary would be meaningless, so restart the series.
        if self._samples and len(self._samples[-1].sm_clocks) != len(sm_clocks):
            self._samples.clear()
        self._samples.append(TpRankSample(
            t=t,
            sm_clocks=list(sm_clocks),
            temps=list(temps) if temps and len(temps) == len(sm_clocks) else None,
        ))
        cutoff = t - self.max_window_s
        self._samples = [s for s in self._samples if s.t >= cutoff][-self.max_samples:]
        return list(self._samples)
