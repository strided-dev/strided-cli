"""Unit tests for the CLI report renderer (``cli.render.render_report``).

The renderer returns a finished string and never writes to stdout, so we can
drive it directly with hand-built report objects. Several branches — rule
errors, conflict suppression, corroboration/conflict annotations, the
adjusted-vs-base confidence sub-line — are unreachable from the CLI today
because the boost/suppression toggles ship off and unexposed. They are still
the user-facing contract for when r02–r10 land, so we pin them here.

Unless a test is specifically about colour, we render with ``color=False`` so
assertions read against plain text. A fixed ``width`` keeps wrapping
deterministic regardless of the terminal running the suite.
"""

from __future__ import annotations

import click
import pytest

from cli import brand
from cli.render import brand_header, logo_banner, render_report, render_status_line
from cli.style import Styler
from engine.runner import (
    DiagnosisReport,
    InsufficientDataNote,
    RankedDiagnosis,
    RuleError,
    SuppressedDiagnosis,
)
from rules.base import ConfidenceBreakdown, Diagnosis
from schema import DiagnosisInput, PhaseMetrics

_WIDTH = 100


def _dx(**overrides) -> DiagnosisInput:
    base = dict(
        model_name="Llama-3-70B",
        gpu_type="H100-SXM",
        inference_engine="vllm",
        batch_size=32,
        prefill=PhaseMetrics(duration_ms=42.0, roofline_position="compute_bound"),
        decode=PhaseMetrics(duration_ms=89.0, roofline_position="memory_bound"),
    )
    base.update(overrides)
    return DiagnosisInput(**base)


def _diag(rule_id: str = "r01", *, confidence: float = 0.71, cause: str = "a cause",
          fix: str = "a fix", evidence: dict | None = None) -> Diagnosis:
    return Diagnosis(
        rule_id=rule_id,
        cause=cause,
        fix=fix,
        confidence=confidence,
        confidence_breakdown=ConfidenceBreakdown(signal_strength=0.5, data_completeness=1.0),
        evidence=evidence or {},
    )


def _ranked(d: Diagnosis, *, rank: int = 1, adjusted: float | None = None,
            corroborated=(), conflicts=()) -> RankedDiagnosis:
    return RankedDiagnosis(
        rank=rank,
        diagnosis=d,
        adjusted_confidence=d.confidence if adjusted is None else adjusted,
        corroborated_by=tuple(corroborated),
        conflicts_with=tuple(conflicts),
        adjustment_reason="",
    )


def _report(**kw) -> DiagnosisReport:
    base = dict(
        diagnoses=(), insufficient_data=(), suppressed=(), errors=(),
        warnings=(), rules_run=1,
    )
    base.update(kw)
    return DiagnosisReport(**base)


def _render(dx: DiagnosisInput, report: DiagnosisReport, *, color: bool = False) -> str:
    return render_report(dx, report, color=color, width=_WIDTH, elapsed_s=0.03)


def _line_with(out: str, token: str) -> str:
    """Return the single output line containing ``token`` (fails if 0 or >1)."""
    matches = [ln for ln in out.splitlines() if token in ln]
    assert len(matches) == 1, f"expected exactly one line with {token!r}, got {matches}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Header & phase blocks
# --------------------------------------------------------------------------- #

class TestHeader:
    def test_banner_and_loaded_block(self) -> None:
        out = _render(_dx(), _report())
        assert out.startswith("strided v")
        assert "loaded" in out
        model_line = _line_with(out, "Llama-3-70B")
        assert model_line.lstrip().startswith("model")
        engine_line = _line_with(out, "vllm")
        assert engine_line.lstrip().startswith("engine")
        batch_line = _line_with(out, "batch_size")
        assert batch_line.rstrip().endswith("32")

    def test_phase_block_shows_duration_and_roofline(self) -> None:
        out = _render(_dx(), _report())
        assert "phase breakdown" in out
        prefill = _line_with(out, "prefill")
        assert "42.0 ms" in prefill and "compute-bound" in prefill
        decode = _line_with(out, "decode")
        assert "89.0 ms" in decode and "memory-bound" in decode

    def test_no_phase_block_when_no_durations(self) -> None:
        out = _render(DiagnosisInput(model_name="m", gpu_type="g"), _report())
        assert "phase breakdown" not in out

    def test_batch_row_omitted_when_absent(self) -> None:
        out = _render(DiagnosisInput(model_name="m", gpu_type="g"), _report())
        assert "batch_size" not in out

    def test_gpu_count_shown_when_present(self) -> None:
        out = _render(_dx(num_gpus=8), _report())
        assert "H100-SXM × 8" in out


# --------------------------------------------------------------------------- #
# Diagnosis rendering
# --------------------------------------------------------------------------- #

class TestDiagnoses:
    def test_single_fired_card(self) -> None:
        report = _report(diagnoses=(_ranked(_diag(evidence={"decode.sm_occupancy": 0.14})),))
        out = _render(_dx(), report)
        assert "diagnosis / 1 rule fired, ranked by confidence" in out
        header = _line_with(out, "Decode memory-bound at low batch")
        assert header.startswith("▌")
        assert "r01" in header
        assert header.rstrip().endswith("71%")
        assert "decode.sm_occupancy=0.14" in _line_with(out, "evidence")

    def test_plural_count(self) -> None:
        report = _report(diagnoses=(
            _ranked(_diag("r01"), rank=1),
            _ranked(_diag("r02"), rank=2),
        ))
        out = _render(_dx(), report)
        assert "2 rules fired, ranked by confidence" in out

    def test_no_rules_fired(self) -> None:
        out = _render(_dx(), _report())
        assert "diagnosis" in out
        assert "no rules fired" in out

    def test_adjusted_confidence_shows_base(self) -> None:
        report = _report(diagnoses=(
            _ranked(_diag(confidence=0.70), adjusted=0.80, corroborated=("r03",)),
        ))
        out = _render(_dx(), report)
        # Header carries the adjusted value; the sub-line carries provenance.
        assert _line_with(out, "Decode memory-bound at low batch").rstrip().endswith("80%")
        assert "base 70%" in out
        assert "corroborated by r03" in out

    def test_conflict_annotation_rendered(self) -> None:
        report = _report(diagnoses=(_ranked(_diag("r02"), conflicts=("r01",)),))
        out = _render(_dx(), report)
        assert "conflicts with r01" in out

    def test_long_fix_wraps_with_hanging_indent(self) -> None:
        long = " ".join(["increase"] * 40)
        report = _report(diagnoses=(_ranked(_diag(fix=long)),))
        out = _render(_dx(), report).splitlines()
        fix_idx = next(i for i, ln in enumerate(out) if "fix" in ln and "increase" in ln)
        cont = out[fix_idx + 1]
        # Continuation hangs under the text column: 4 (card indent) + 10 (label).
        assert cont.startswith(" " * 14)
        assert cont.strip().startswith("increase")

    def test_evidence_float_above_one_uses_sig_figs(self) -> None:
        report = _report(diagnoses=(_ranked(_diag(evidence={"achieved_flops": 1234.5})),))
        out = _render(_dx(), report)
        assert "achieved_flops=1.23e+03" in out


# --------------------------------------------------------------------------- #
# Secondary sections + footer
# --------------------------------------------------------------------------- #

class TestSecondarySections:
    def test_input_warnings(self) -> None:
        out = _render(_dx(), _report(warnings=("non-finite value at decode.x",)))
        assert "input warnings" in out
        assert "- non-finite value at decode.x" in out

    def test_parse_warnings_from_input(self) -> None:
        dx = _dx()
        dx.parse_warnings.append("a parse warning")
        out = _render(dx, _report())
        assert "parse warnings" in out
        assert "- a parse warning" in out

    def test_insufficient_data_section(self) -> None:
        report = _report(insufficient_data=(InsufficientDataNote("r01", "Decode memory-bound"),))
        out = _render(_dx(), report)
        assert "could not evaluate / missing data" in out
        assert "- r01 Decode memory-bound" in out

    def test_insufficient_data_names_missing_field(self) -> None:
        report = _report(insufficient_data=(
            InsufficientDataNote("r01", "Decode memory-bound",
                                 missing=("decode.sm_occupancy",)),
        ))
        out = _render(_dx(), report)
        assert "r01 Decode memory-bound, needs decode.sm_occupancy" in out

    def test_insufficient_data_shows_reason(self) -> None:
        report = _report(insufficient_data=(
            InsufficientDataNote("r02", "Colocation contention",
                                 reason="only 40 successful requests; need >=200"),
        ))
        out = _render(_dx(), report)
        assert "need >=200" in out


class TestOneShotTemporalQuiet:
    """One-shot diagnose collapses temporal rules' insufficiency into one dim
    pointer at watch; live rendering keeps the itemized rows (there, "needs
    more history" is a real, changing state)."""

    @pytest.fixture(autouse=True)
    def fake_temporal_tier(self, monkeypatch) -> None:
        from collect import tiers
        monkeypatch.setitem(
            tiers._TIERS, "r98",
            tiers.RuleTier("r98", "Fake trend A", (tiers.VLLM,), temporal=True))
        monkeypatch.setitem(
            tiers._TIERS, "r99",
            tiers.RuleTier("r99", "Fake trend B", (tiers.VLLM,), temporal=True))

    def _temporal_note(self, rid: str = "r98", title: str = "Fake trend A"):
        return InsufficientDataNote(rid, title, missing=("throughput_history",))

    def test_one_shot_collapses_temporal_rows(self) -> None:
        report = _report(insufficient_data=(self._temporal_note(),))
        out = render_report(_dx(), report, width=_WIDTH, one_shot=True)
        # No itemized row, no section header just for it — one quiet line.
        assert "could not evaluate" not in out
        assert "needs throughput_history" not in out
        assert "1 trend rule idle (r98): it observes over a run" in out
        assert "strided watch" in out

    def test_one_shot_counts_and_names_all_idle_temporal_rules(self) -> None:
        report = _report(insufficient_data=(
            self._temporal_note(),
            self._temporal_note("r99", "Fake trend B"),
        ))
        out = render_report(_dx(), report, width=_WIDTH, one_shot=True)
        assert "2 trend rules idle (r98, r99)" in out

    def test_one_shot_keeps_non_temporal_rows_itemized(self) -> None:
        report = _report(insufficient_data=(
            InsufficientDataNote("r01", "Decode memory-bound",
                                 missing=("decode.sm_occupancy",)),
            self._temporal_note(),
        ))
        out = render_report(_dx(), report, width=_WIDTH, one_shot=True)
        assert "could not evaluate / missing data" in out
        assert "r01 Decode memory-bound, needs decode.sm_occupancy" in out
        assert "r98" not in out.split("trend rule idle")[0].split("could not evaluate")[1]
        assert "1 trend rule idle (r98)" in out

    def test_live_rendering_keeps_temporal_rows(self) -> None:
        # Default (one_shot=False) — watch's path — is unchanged.
        report = _report(insufficient_data=(self._temporal_note(),))
        out = _render(_dx(), report)
        assert "r98 Fake trend A, needs throughput_history" in out
        assert "trend rule idle" not in out

    def test_rule_errors_section(self) -> None:
        report = _report(errors=(RuleError("r07", "ValueError"),))
        out = _render(_dx(), report)
        assert "rule errors" in out
        assert "- r07: ValueError" in out

    def test_suppressed_section(self) -> None:
        suppressed = SuppressedDiagnosis(
            diagnosis=_diag("r02"), suppressed_by="r01",
            reason="conflicts with higher-confidence r01",
        )
        out = _render(_dx(), _report(suppressed=(suppressed,)))
        assert "suppressed by conflict resolution" in out
        assert "- r02 (conflicts with higher-confidence r01)" in out

    def test_footer_reports_real_rule_count_and_timing(self) -> None:
        out = _render(_dx(), _report(rules_run=1))
        assert "parsed in 0.03s · 1 rule evaluated" in out

    def test_footer_pluralises_rule_count(self) -> None:
        out = render_report(_dx(), _report(rules_run=10), color=False, width=_WIDTH)
        assert "10 rules evaluated" in out
        assert "parsed in" not in out  # elapsed_s omitted → no timing fragment


# --------------------------------------------------------------------------- #
# Colour behaviour
# --------------------------------------------------------------------------- #

class TestColor:
    def test_plain_output_has_no_ansi(self) -> None:
        out = _render(_dx(), _report(diagnoses=(_ranked(_diag()),)), color=False)
        assert "\x1b[" not in out

    def test_colored_output_has_ansi(self) -> None:
        out = _render(_dx(), _report(diagnoses=(_ranked(_diag()),)), color=True)
        assert "\x1b[" in out

    def test_color_does_not_change_text_content(self) -> None:
        # Scope: this byte-identical invariant is for the *report body*
        # (render_report). The start-up wordmark (logo_banner) intentionally renders
        # block-art only in colour — see TestBanner for its narrower contract.
        import re

        report = _report(diagnoses=(_ranked(_diag()),))
        colored = _render(_dx(), report, color=True)
        stripped = re.sub(r"\x1b\[[0-9;]*m", "", colored)
        assert stripped == _render(_dx(), report, color=False)

    def test_control_chars_in_untrusted_fields_are_stripped(self) -> None:
        # A crafted model name carrying an ANSI escape must not reach the terminal,
        # in either mode — it would inject colour and break the no-ANSI contract.
        evil = "Llama\x1b[31m-3\x1b[0m\x07"
        for color in (False, True):
            out = render_report(_dx(model_name=evil), _report(), color=color, width=_WIDTH)
            assert "\x1b[31m" not in out and "\x07" not in out
            assert "Llama-3" in out  # the printable text survives


# --------------------------------------------------------------------------- #
# Start-up banner / wordmark
# --------------------------------------------------------------------------- #

class TestBanner:
    def test_plain_logo_banner_is_the_text_wordmark(self) -> None:
        # Colour off: the banner collapses to the plain text wordmark, identically
        # to brand_header — no block-art, no ANSI. Keeps pipes/logs clean.
        st = Styler(False)
        assert logo_banner(st, 80) == brand_header(st, 80)

    def test_plain_logo_banner_has_no_ansi_or_blockart(self) -> None:
        out = "\n".join(logo_banner(Styler(False), 80))
        assert "\x1b[" not in out
        assert "▀" not in out  # block-art only appears in colour

    def test_colored_logo_banner_uses_blockart(self) -> None:
        # Documents the intentional divergence: colour adds the block-art lockup.
        out = "\n".join(logo_banner(Styler(True), 80))
        assert "▀" in out and "\x1b[" in out


# --------------------------------------------------------------------------- #
# Brand palette / terminal surface
# --------------------------------------------------------------------------- #

class TestSurface:
    """Which half of the palette we paint with, and how that half is chosen."""

    def test_defaults_to_dark(self) -> None:
        # Most terminals report nothing at all; dark is what most of them are.
        assert brand.resolve_surface({}) == brand.DARK

    def test_colorfgbg_reports_the_background(self) -> None:
        assert brand.resolve_surface({"COLORFGBG": "0;15"}) == brand.LIGHT
        assert brand.resolve_surface({"COLORFGBG": "15;0"}) == brand.DARK
        # Some terminals interpose a third field; the background is still last.
        assert brand.resolve_surface({"COLORFGBG": "0;default;15"}) == brand.LIGHT

    def test_malformed_colorfgbg_falls_back_to_dark(self) -> None:
        for value in ("", ";", "nonsense", "0;default"):
            assert brand.resolve_surface({"COLORFGBG": value}) == brand.DARK

    def test_explicit_env_var_wins(self) -> None:
        env = {"STRIDED_SURFACE": "light", "COLORFGBG": "15;0"}
        assert brand.resolve_surface(env) == brand.LIGHT
        env = {"STRIDED_SURFACE": "DARK", "COLORFGBG": "0;15"}
        assert brand.resolve_surface(env) == brand.DARK

    def test_unrecognised_env_var_defers_to_the_terminal(self) -> None:
        env = {"STRIDED_SURFACE": "porcelain", "COLORFGBG": "0;15"}
        assert brand.resolve_surface(env) == brand.LIGHT

    def test_each_surface_paints_its_documented_purple(self) -> None:
        # Light gets the canonical logo purple; dark gets the lifted tint, because
        # #4a0081 measures 1.6:1 against a black terminal.
        assert Styler(True, surface=brand.LIGHT).accent("x") == click.style(
            "x", fg=brand.rgb("#4a0081"), bold=True
        )
        assert Styler(True, surface=brand.DARK).accent("x") == click.style(
            "x", fg=brand.rgb("#A85AE2"), bold=True
        )

    def test_each_surface_dims_with_its_canonical_grey(self) -> None:
        # ink-soft on paper, ink-faint on a dark terminal where ink-soft would
        # not clear a legible contrast.
        assert Styler(True, surface=brand.LIGHT).dim("x") == click.style(
            "x", fg=brand.rgb(brand.INK_SOFT)
        )
        assert Styler(True, surface=brand.DARK).dim("x") == click.style(
            "x", fg=brand.rgb(brand.INK_FAINT)
        )

    def test_surface_never_reaches_plain_output(self) -> None:
        # The surface only picks a paint. With colour off there is no paint, so
        # both surfaces must render the identical bytes.
        report = _report(diagnoses=(_ranked(_diag()),))
        rendered = [
            "\n".join(logo_banner(Styler(False, surface=s), 80))
            + render_report(_dx(), report, color=False, width=_WIDTH)
            for s in (brand.LIGHT, brand.DARK)
        ]
        assert rendered[0] == rendered[1]
        assert "\x1b[" not in rendered[0]

    def test_mark_leaves_the_ink_bars_to_the_terminal(self) -> None:
        # Ink in a terminal is the surface's own foreground: painting it would
        # make the lower bars vanish on a theme that disagrees with us.
        st = Styler(True, surface=brand.DARK)
        assert st.mark("▀▀", ink=True) == "▀▀"
        assert "\x1b[" in st.mark("▀▀", ink=False)


# ---------------------------------------------------------------------------
# Live status line
# ---------------------------------------------------------------------------

def test_status_line_shows_firing_count() -> None:
    line = render_status_line(tick=3, clock="12:00:00", active=2)
    assert "2 firing" in line
    assert "no data" not in line


def test_status_line_no_data_does_not_imply_cleared() -> None:
    # A transient gap (all sources failed) must not read as "0 firing".
    line = render_status_line(tick=3, clock="12:00:00", no_data=True)
    assert "no data" in line
    assert "firing" not in line
