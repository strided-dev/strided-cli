"""The watch loop's brain: poll → merge → window → diagnose → diff.

``WatchSession.tick`` is pure of I/O beyond what the sources and engine do (it
prints nothing), so it is unit-testable with fake sources and a synthetic clock.
Change detection is keyed on the *set of fired rule ids* plus the set of surfaced
insufficiencies — a CHANGED block means the conclusions changed, not that a
confidence wobbled. Confidence still rides along in the rendered card and the
status line; it just does not, on its own, re-emit a block every tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from collect import tiers
from collect.sources import Source
from collect.window import COUNTER_FIELDS, ThroughputHistory, ThroughputWindow, TpRankHistory
from engine import run_diagnosis
from engine.runner import DiagnosisReport
from schema import DiagnosisInput
from schema.merge import merge_inputs


def _mean(values: Optional[list[float]]) -> Optional[float]:
    """Mean of a per-rank telemetry list (SM clock / temp), or None if empty.

    A single-GPU watch sees a one-element list; r06 reads the device-level clock /
    temperature, so collapsing the per-rank list to its mean is the right summary.
    """
    if not values:
        return None
    return sum(values) / len(values)


@dataclass(frozen=True)
class TickResult:
    """One tick's outcome. ``merged``/``report`` are None when no source produced data."""

    merged: Optional[DiagnosisInput]
    report: Optional[DiagnosisReport]
    changed: bool
    warnings: tuple[str, ...]
    exhausted: bool


class WatchSession:
    def __init__(
        self,
        sources: list[Source],
        *,
        model_name: Optional[str] = None,
        gpu_type: Optional[str] = None,
        window: Optional[ThroughputWindow] = None,
        history: Optional[ThroughputHistory] = None,
        rank_history: Optional[TpRankHistory] = None,
    ) -> None:
        self._sources = list(sources)
        self._model = model_name
        self._gpu = gpu_type
        self._window = window or ThroughputWindow()
        self._history = history or ThroughputHistory()
        self._rank_history = rank_history or TpRankHistory()
        self._last_state: Optional[frozenset] = None

    def tick(self, now: float) -> TickResult:
        """Poll every source once, diagnose the merged snapshot, diff against last."""
        inputs: list[DiagnosisInput] = []
        warnings: list[str] = []
        for src in self._sources:
            dx = src.poll()
            if src.last_error:
                warnings.append(f"[{src.name}] {src.last_error}")
            if dx is not None:
                inputs.append(dx)

        exhausted = bool(self._sources) and all(
            getattr(s, "exhausted", False) for s in self._sources
        )

        if not inputs:
            # Nothing to diagnose this tick (all sources failed or ran dry). Do not
            # touch the change state — a transient gap is not a state change.
            return TickResult(None, None, False, tuple(warnings), exhausted)

        merged = merge_inputs(inputs, model_name=self._model, gpu_type=self._gpu)

        wu = self._window.update(self._counters(merged), now)
        if wu.rates:
            merged = merged.model_copy(update=wu.rates)

        # Accumulate the sustained-run series r06 reads. token_throughput_gen is
        # set by the rates copy above (None on the first tick → no sample); the
        # pressure co-samples are per-window deltas (schema 1.4.0) so a
        # long-running server's lifetime totals cannot dilute current pressure;
        # the point-in-time corroborators ride along from the merged snapshot.
        history = self._history.update(
            now,
            merged.token_throughput_gen,
            preemption_rate=self._window_preemption_rate(wu.deltas),
            queue_time_ms=self._window_queue_time_ms(wu.deltas),
            num_requests_waiting=(
                merged.vllm_serving.num_requests_waiting if merged.vllm_serving else None
            ),
            sm_clock_mhz=_mean(merged.tp_rank_sm_clocks),
            gpu_temp_c=_mean(merged.tp_rank_temps),
        )
        if history:
            merged = merged.model_copy(update={"throughput_history": history})

        # Per-rank clock series for r04's persistence gate (schema 1.6.0). Kept
        # separate from throughput_history because it survives ticks where no
        # window rate exists — clocks are a gauge, not a windowed rate.
        rank_hist = self._rank_history.update(
            now, merged.tp_rank_sm_clocks, temps=merged.tp_rank_temps
        )
        if rank_hist:
            merged = merged.model_copy(update={"tp_rank_history": rank_hist})

        report = run_diagnosis(merged)
        state = self._state_of(report)
        changed = state != self._last_state
        self._last_state = state
        return TickResult(merged, report, changed, tuple(warnings), exhausted)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _counters(merged: DiagnosisInput) -> dict[str, float]:
        s = merged.vllm_serving
        if s is None:
            return {}
        out: dict[str, float] = {}
        for name in COUNTER_FIELDS:
            v = getattr(s, name)
            if v is not None:
                out[name] = float(v)
        return out

    @staticmethod
    def _window_preemption_rate(deltas: dict[str, float]) -> Optional[float]:
        """Window preemptions / successes — the r06 memory-pressure co-sample.

        Same ratio r03's delta path reads, differenced between consecutive
        scrapes: it reflects pressure in *this* window, so a past burst on a
        long-running server cannot dilute it.
        None when the preemption counter is absent from either snapshot; the
        max(Δsuccesses, 1) floor keeps a preemption burst visible even in a
        window where nothing completed.
        """
        d_pre = deltas.get("num_preemptions_total")
        if d_pre is None:
            return None
        return d_pre / max(deltas.get("request_success_total", 0.0), 1.0)

    @staticmethod
    def _window_queue_time_ms(deltas: dict[str, float]) -> Optional[float]:
        """Window mean queue time (Δsum/Δcount) — r06's V1 pressure co-sample.

        vLLM V1 queues rather than preempting, so on V1 this is the co-sample
        that moves under pressure. None when no request completed in the window
        (Δcount == 0) — correctly "no signal", not zero; r06's series helper
        skips None samples.
        """
        d_sum = deltas.get("request_queue_time_ms_sum")
        d_count = deltas.get("request_queue_time_count")
        if d_sum is None or d_count is None or d_count <= 0:
            return None
        return d_sum / d_count

    @staticmethod
    def _state_of(report: DiagnosisReport) -> frozenset:
        """A hashable signature of the diagnosis state for change detection.

        Fired rules contribute ``("fire", rule_id)``; surfaced insufficiencies
        contribute ``("insuf", rule_id)``. Dump-only rules are filtered so their
        permanent live-insufficiency is not part of the state. Confidence is
        deliberately excluded: a CHANGED block tracks which conclusions hold, not
        how a confidence drifts tick to tick.
        """
        fired = {
            ("fire", d.diagnosis.rule_id)
            for d in report.diagnoses
        }
        surfaced = {
            ("insuf", n.rule_id)
            for n in report.insufficient_data
            if tiers.is_surfaced(n.rule_id)
        }
        return frozenset(fired | surfaced)


__all__ = ["WatchSession", "TickResult"]
