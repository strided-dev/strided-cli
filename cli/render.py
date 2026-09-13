"""Rendering for the strided CLI report.

The engine returns data; this module is the single source of truth for layout.
``render_report`` takes the merged ``DiagnosisInput`` and the engine's
``DiagnosisReport`` and returns a finished string — it never writes to stdout,
so it is trivially testable and the caller owns I/O.

Design notes:

- Alignment is always computed on the *plain* text, and styling is applied
  afterwards, so colour codes can never corrupt a column width.
- Every visible number comes from the input or the report. We never invent a
  metric to fill the frame; a field that is absent is simply not rendered.
"""

from __future__ import annotations

import re
import shutil
import textwrap
from typing import Optional

from cli import CLI_VERSION
from cli.style import Styler
from collect.tiers import is_temporal
from engine.runner import DiagnosisReport, RankedDiagnosis
from schema import DiagnosisInput

# Untrusted, parser-derived strings (model name, engine, GPU type, warnings) get
# interpolated into output. Strip C0/C1 control characters — crucially ESC (0x1b) —
# before they reach the terminal, so a crafted value can't smuggle an ANSI escape
# into the report. This also upholds the "plain output has no ANSI" contract on
# hostile input. Styling is applied to the *cleaned* value, never the raw one.
_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize(text: str) -> str:
    """Strip ANSI CSI escapes and C0/C1 control characters from an untrusted string.

    Whole escape sequences go first (so ``\\x1b[31m`` vanishes entirely rather than
    leaving a stray ``[31m``), then any residual control characters — including a
    lone ESC or BEL — are removed.
    """
    return _CONTROL_CHARS.sub("", _ANSI_CSI.sub("", text))


# Card layout constants.
_CARD_INDENT = "    "          # left margin for a diagnosis card's body
_FIELD_LABEL_W = 10            # width of the cause/fix/evidence label column
_MIN_WRAP_WIDTH = 24          # never wrap narrower than this, however small the term
_MAX_WIDTH = 100              # cap line width on very wide terminals for readability
_METER_W = 12                # visible width of a confidence meter

# What strided is, in one line. The loop it runs (observe, adjust, verify) is
# named in --help and in each command's summary; the banner stays a wordmark.
_TAGLINE = "local model hosting, tuned live"


def brand_header(st: Styler, width: int) -> list[str]:
    """The strided wordmark: name + version, tagline, and a faint rule.

    Shared by ``diagnose`` and ``watch`` so every surface opens the same way.
    The plain rendering is ``strided v<version>`` on line one, which both the
    renderer's own tests and a quick ``startswith`` check can rely on.
    """
    return [
        st.accent("strided") + " " + st.dim(f"v{CLI_VERSION}"),
        st.dim(_TAGLINE),
        st.rule(width),
    ]


# The strided mark, drawn for the terminal: four small, straight bars — top two in
# the logo purple, bottom two in ink, exactly as the vector mark's four bars are
# filled. Ink in a terminal is the surface's own default foreground, so those two
# are left unpainted and the mark inverts correctly on any theme. Odd rows stride
# right by a third of a bar, even rows sit flush left, exactly as the vector mark
# does. One row per bar with a blank line between, so the bars read as a clean
# staggered stack: block, gap, block, gap.
# Upper-half blocks: each bar fills the top of its row, leaving a thin gap below,
# so the staggered stack stays tight — no blank rows needed between bars.
_BAR = "▀" * 6
_MARK_STRIDE = "  "                          # ~1/3-bar horizontal offset
_MARK_COL = len(_MARK_STRIDE) + len(_BAR) + 3  # text column, right of the mark


def logo_banner(st: Styler, width: int) -> list[str]:
    """The mark-plus-wordmark lockup used as the start-up banner.

    Colour terminals get the painted blocks; piped / no-colour output falls back to
    the plain text :func:`brand_header`, so logs and pipes stay clean (there is
    nothing to gain from block art no one will see in colour).
    """
    if not st.enabled:
        return brand_header(st, width)

    # One bar per row; the half-block leaves its own gap below, so no blank lines.
    plains: list[str] = []
    painted: list[str] = []
    for ink, shifted in ((False, True), (False, False), (True, True), (True, False)):
        seg = (_MARK_STRIDE if shifted else "") + _BAR
        plains.append(seg)
        painted.append(st.mark(seg, ink=ink))

    # Lockup: set the wordmark and tagline against the vertical centre of the mark.
    text = {1: st.accent("strided") + " " + st.dim(f"v{CLI_VERSION}"), 2: st.dim(_TAGLINE)}

    lines = []
    for i, (plain, paint) in enumerate(zip(plains, painted)):
        if i in text:
            paint = paint + " " * (_MARK_COL - len(plain)) + text[i]
        lines.append(paint)
    lines.append(st.rule(width))
    return lines


def render_report(
    dx: DiagnosisInput,
    report: DiagnosisReport,
    *,
    color: bool = False,
    width: Optional[int] = None,
    elapsed_s: Optional[float] = None,
    brand: bool = True,
    one_shot: bool = False,
) -> str:
    """Render one diagnosis run to a finished, ready-to-print string.

    ``brand`` controls the opening wordmark; ``watch`` turns it off so each live
    event reads as an update under the one banner printed at start-up, not a
    fresh report.

    ``one_shot`` marks a single-dump run (``diagnose``): temporal rules — which
    observe over a sustained run and can never fire from one snapshot — have
    their COULD-NOT-EVALUATE rows collapsed into a single quiet pointer at
    ``watch``. Live rendering keeps them itemized: there, "needs more history"
    is a real, changing state.
    """
    st = Styler(color)
    width = _resolve_width(width)
    lines: list[str] = []

    if brand:
        lines.extend(brand_header(st, width))
        lines.append("")
    lines.extend(_loaded_block(st, dx))

    phase = _phase_block(st, dx)
    if phase:
        lines.append("")
        lines.extend(phase)

    for title, items in _warning_sections(dx, report):
        lines.append("")
        lines.extend(_bullet_block(st, title, items))

    lines.append("")
    lines.extend(_diagnosis_section(st, report, width))

    for title, items in _trailer_sections(report, one_shot=one_shot):
        lines.append("")
        lines.extend(_bullet_block(st, title, items))

    if one_shot:
        quiet = _temporal_quiet_line(st, report)
        if quiet:
            lines.append("")
            lines.append(quiet)

    lines.append("")
    lines.append(st.rule(width))
    lines.append(_footer(st, report, elapsed_s))

    nxt = _next_step(st, report, one_shot=one_shot)
    if nxt:
        lines.append("")
        lines.append(nxt)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Header blocks
# ---------------------------------------------------------------------------

def _loaded_block(st: Styler, dx: DiagnosisInput) -> list[str]:
    """The LOADED key/value block. Rows are omitted when the value is absent."""
    gpu = dx.gpu_type
    if dx.num_gpus:
        gpu = f"{gpu} × {dx.num_gpus}"

    # These come from parsed dumps (untrusted) — clean before display.
    rows: list[tuple[str, str]] = [
        ("model", _sanitize(dx.model_name)),
        ("engine", _sanitize(dx.inference_engine)),
        ("gpu", _sanitize(gpu)),
    ]
    if dx.batch_size is not None:
        rows.append(("batch_size", str(dx.batch_size)))

    return _kv_block(st, "loaded", rows)


def _phase_block(st: Styler, dx: DiagnosisInput) -> list[str]:
    """The PHASE BREAKDOWN block: duration + roofline class, when present."""
    rows: list[tuple[str, str]] = []
    for name, phase in (("prefill", dx.prefill), ("decode", dx.decode)):
        if phase is not None and phase.duration_ms is not None:
            value = f"{phase.duration_ms:>6.1f} ms"
            if phase.roofline_position:
                value += "   " + st.roofline(phase.roofline_position)
            rows.append((name, value))
    if not rows:
        return []
    return _kv_block(st, "phase breakdown", rows)


def _kv_block(st: Styler, title: str, rows: list[tuple[str, str]]) -> list[str]:
    """A titled, column-aligned key/value block. Values may already be styled."""
    if not rows:
        return []
    label_w = max(len(label) for label, _ in rows)
    out = [st.section(title)]
    for label, value in rows:
        out.append("  " + st.label(label.ljust(label_w)) + "   " + value)
    return out


# ---------------------------------------------------------------------------
# Diagnosis section
# ---------------------------------------------------------------------------

def _diagnosis_section(st: Styler, report: DiagnosisReport, width: int) -> list[str]:
    n = len(report.diagnoses)
    if n == 0:
        return [st.section("diagnosis") + st.dim(" / no rules fired")]

    summary = f"{n} rule{'s' if n != 1 else ''} fired, ranked by confidence"
    out = [st.section("diagnosis") + st.dim(" / " + summary)]
    for ranked in report.diagnoses:
        out.append("")
        out.extend(_card(st, ranked, width))
    return out


def _card(st: Styler, ranked: RankedDiagnosis, width: int) -> list[str]:
    d = ranked.diagnosis
    title = _rule_title(d.rule_id)
    conf = f"{ranked.adjusted_confidence:.0%}"

    # Header line: ▌ <title>   <rule_id> ....... <meter>  <confidence>
    # The meter + percentage are right-aligned; pad against the *plain* lengths
    # (the meter is always _METER_W glyphs wide) so styling never shifts a column.
    left_plain = f"▌ {title}   {d.rule_id}"
    right_plain = f"{'?' * _METER_W}  {conf}"
    pad = max(2, width - len(left_plain) - len(right_plain))
    header = (
        st.accent("▌")
        + " "
        + st.accent(title)
        + "   "
        + st.dim(d.rule_id)
        + " " * pad
        + st.meter(ranked.adjusted_confidence, _METER_W)
        + "  "
        + st.confidence(conf)
    )
    out = [header]

    extras = _confidence_extras(ranked)
    if extras:
        out.append(_CARD_INDENT + st.dim(" · ".join(extras)))

    out.extend(_field(st, "cause", d.cause, width))
    out.extend(_field(st, "fix", d.fix, width))
    if d.evidence:
        ev = ", ".join(f"{k}={_fmt_evidence(v)}" for k, v in d.evidence.items())
        out.extend(_field(st, "evidence", ev, width))
    return out


def _confidence_extras(ranked: RankedDiagnosis) -> list[str]:
    """Sub-line annotations: base confidence, corroboration, conflict."""
    extras: list[str] = []
    if ranked.adjusted_confidence != ranked.diagnosis.confidence:
        extras.append(f"base {ranked.diagnosis.confidence:.0%}")
    if ranked.corroborated_by:
        extras.append("corroborated by " + ", ".join(ranked.corroborated_by))
    if ranked.conflicts_with:
        extras.append("conflicts with " + ", ".join(ranked.conflicts_with))
    return extras


def _field(st: Styler, label: str, text: str, width: int) -> list[str]:
    """A labelled, wrapped field inside a card.

    The dim label sits in a fixed column; continuation lines hang under the
    text, not the label. Wrapping uses the plain text so colour never throws
    off the width maths (the label is the only styled part, and it is padded
    before styling).
    """
    indent = _CARD_INDENT + " " * _FIELD_LABEL_W
    wrap_w = max(_MIN_WRAP_WIDTH, width - len(indent))
    wrapped = textwrap.wrap(text, width=wrap_w) or [""]

    head = _CARD_INDENT + st.label(label.ljust(_FIELD_LABEL_W)) + wrapped[0]
    return [head] + [indent + cont for cont in wrapped[1:]]


# ---------------------------------------------------------------------------
# Warning / trailer sections
# ---------------------------------------------------------------------------

def _warning_sections(
    dx: DiagnosisInput, report: DiagnosisReport
) -> list[tuple[str, list[str]]]:
    """Sections shown above the diagnosis: input + parse warnings."""
    sections: list[tuple[str, list[str]]] = []
    if report.warnings:
        sections.append(("input warnings", list(report.warnings)))
    if dx.parse_warnings:
        sections.append(("parse warnings", list(dx.parse_warnings)))
    return sections


def _trailer_sections(
    report: DiagnosisReport, *, one_shot: bool = False
) -> list[tuple[str, list[str]]]:
    """Sections shown below the diagnosis: insufficient data, errors, suppressed.

    In one-shot mode, temporal rules' insufficiency rows are excluded here and
    surfaced by ``_temporal_quiet_line`` instead — a single dump can never feed
    them, so an itemized "missing data" row would read as a defect, not a fact.
    """
    sections: list[tuple[str, list[str]]] = []
    notes = list(report.insufficient_data)
    if one_shot:
        notes = [n for n in notes if not is_temporal(n.rule_id)]
    if notes:
        sections.append((
            "could not evaluate / missing data",
            [_insufficient_line(n) for n in notes],
        ))
    if report.errors:
        sections.append((
            "rule errors",
            [f"{e.rule_id}: {e.error}" for e in report.errors],
        ))
    if report.suppressed:
        sections.append((
            "suppressed by conflict resolution",
            [f"{s.diagnosis.rule_id} ({s.reason})" for s in report.suppressed],
        ))
    return sections


# ---------------------------------------------------------------------------
# Live (watch) rendering
# ---------------------------------------------------------------------------

def render_live_event(
    dx: DiagnosisInput,
    report: DiagnosisReport,
    *,
    clock: str,
    color: bool = False,
    width: Optional[int] = None,
) -> str:
    """A timestamped 'state changed' block for ``watch``, reusing the report layout.

    Printed only when the diagnosis state changes; between changes the caller
    shows ``render_status_line`` on a single rewritable line instead.
    """
    st = Styler(color)
    w = _resolve_width(width)
    bar = st.rule(w)
    fired = ", ".join(r.diagnosis.rule_id for r in report.diagnoses) or "none"
    header = (
        st.accent("▌")
        + " "
        + st.accent(clock)
        + "   "
        + st.state("state changed", "yellow")
        + st.dim(" · firing: ")
        + (st.confidence(fired) if report.diagnoses else st.dim(fired))
    )
    body = render_report(dx, report, color=color, width=w, brand=False)
    return "\n".join([bar, header, "", body])


def render_status_line(
    *,
    tick: int,
    clock: str,
    active: int = 0,
    gen_tok_s: Optional[float] = None,
    no_data: bool = False,
    color: bool = False,
) -> str:
    """A single rewritable status line (no trailing newline); caller prints with '\\r'.

    ``no_data`` marks a tick where every source failed or ran dry — rendered as
    "no data" rather than "0 firing", so a transient gap does not read as the
    diagnosis having just cleared.
    """
    st = Styler(color)
    parts = [st.dim(f"tick {tick}"), st.dim(clock)]
    if no_data:
        parts.append(st.warn("no data"))
    elif active:
        parts.append(st.confidence(f"{active} firing"))
    else:
        parts.append(st.dim("0 firing"))
    if gen_tok_s is not None:
        parts.append(st.dim(f"{gen_tok_s:,.0f} gen tok/s"))
    return st.dim(" · ").join(parts)


def _temporal_quiet_line(st: Styler, report: DiagnosisReport) -> Optional[str]:
    """One dim line for temporal rules idle in a one-shot run, or None.

    ``2 trend rules idle (r06, r08) — they observe over a run; use strided watch``
    """
    idle = [n for n in report.insufficient_data if is_temporal(n.rule_id)]
    if not idle:
        return None
    ids = ", ".join(n.rule_id for n in idle)
    noun, verb = (
        ("trend rule", "it observes") if len(idle) == 1 else ("trend rules", "they observe")
    )
    return st.dim(
        f"  {len(idle)} {noun} idle ({ids}): {verb} over a run, "
        "so use `strided watch`"
    )


def _insufficient_line(note) -> str:
    """One COULD-NOT-EVALUATE row: rule + title, plus the named gap when present.

    ``InsufficientDataNote`` now carries the missing schema fields (or a free-form
    reason); a legacy payload-free note renders just the rule and title.
    """
    base = f"{note.rule_id} {note.title}"
    if note.missing:
        return f"{base}, needs {', '.join(note.missing)}"
    if note.reason:
        return f"{base}, {note.reason}"
    return base


def _bullet_block(st: Styler, title: str, items: list[str]) -> list[str]:
    # Items include parser-emitted warnings (untrusted) — clean before display.
    out = [st.section(title)]
    out.extend("  " + st.dim(f"- {_sanitize(item)}") for item in items)
    return out


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

def _next_step(
    st: Styler, report: DiagnosisReport, *, one_shot: bool
) -> Optional[str]:
    """One quiet line handing the reader the next step in the loop, or None.

    Observing is only half of it: a diagnosis that no one acts on has not tuned
    anything. This sits last, where the eye comes to rest, and names the exact
    command that adjusts the top-ranked finding. Live rendering skips it, where a
    per-event footer would be noise rather than a prompt.

    The label is dim and the command is not, so the command is what you see.
    """
    if not one_shot or not report.diagnoses:
        return None
    top = report.diagnoses[0].diagnosis.rule_id
    return st.label("  next  ") + f"strided fix {top} --config <launch-file>"


def _footer(st: Styler, report: DiagnosisReport, elapsed_s: Optional[float]) -> str:
    n = report.rules_run
    parts: list[str] = []
    if elapsed_s is not None:
        parts.append(f"parsed in {elapsed_s:.2f}s")
    parts.append(f"{n} rule{'s' if n != 1 else ''} evaluated")
    return st.dim(" · ".join(parts))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _resolve_width(width: Optional[int]) -> int:
    if width is not None:
        return width
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(_MIN_WRAP_WIDTH + _FIELD_LABEL_W + len(_CARD_INDENT), min(cols, _MAX_WIDTH))


def _rule_title(rule_id: str) -> str:
    """Look up a rule's class-level title by id; fall back to the id itself."""
    from engine.registry import ALL_RULES  # local import: avoid a cycle at module load

    for cls in ALL_RULES:
        if cls.rule_id == rule_id:
            return cls.title
    return rule_id


def _fmt_evidence(v) -> str:
    if isinstance(v, float):
        if 0.0 <= v <= 1.0:
            return f"{v:.2f}"
        return f"{v:.3g}"
    return str(v)


__all__ = [
    "brand_header",
    "logo_banner",
    "render_report",
    "render_live_event",
    "render_status_line",
]
