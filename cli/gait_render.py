"""Presentation for the ``gait`` agent — clean, branded, and honest.

gait is the *adjust* step of strided's loop: observe the workload, adjust within the
limits you set, verify the result. Automatic adjustments, visible decisions. The
agent surfaces in the terminal the way a good pair-assistant does: it leads with a
one-line TLDR of what it intends to do (and how sure it is), asks for a yes/no, and
otherwise stays quiet. ``--verbose`` opens the curtain and narrates every step of the
state machine (resolve → propose → approve → apply → verify) with the full predicted
effect and before/after numbers.

All colour goes through :class:`cli.style.Styler`, so the brand accent (the logo
purple ``#4a0081``), the confidence meter, and the card rail are exactly the ones
``diagnose`` and ``watch`` use — gait reads as the same product, not a bolt-on. The
whole thing collapses to clean plain text the moment stdout is not a terminal.
"""

from __future__ import annotations

import shutil
import textwrap
from typing import Optional

from cli.style import Color, Styler
from gait import Abstained, Applied, Proposed, Verdict, Verified

_MAX_WIDTH = 92
_MIN_WIDTH = 40
_INDENT = "  "
_METER_W = 12

# Verdict → (glyph, colour, headline). Honesty is the point: only CONFIRMED is
# celebratory; the rest read as plainly as they mean. _DIM is a sentinel: the dim
# tone depends on the terminal surface, so it is resolved from the Styler at render
# time rather than pinned here at import.
_DIM: Optional[Color] = None
_VERDICT_STYLE: dict[Verdict, tuple[str, Optional[Color], str]] = {
    Verdict.CONFIRMED:         ("✓", "green",  "CONFIRMED"),
    Verdict.NO_CHANGE:         ("✗", "yellow", "NO CHANGE"),
    Verdict.INCONCLUSIVE:      ("~", "cyan",   "INCONCLUSIVE"),
    Verdict.INSUFFICIENT_DATA: ("?", _DIM,     "INSUFFICIENT DATA"),
}


def _width() -> int:
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(_MIN_WIDTH, min(cols, _MAX_WIDTH))


def _wrap(text: str, indent: str = _INDENT) -> list[str]:
    w = max(_MIN_WIDTH - len(indent), _width() - len(indent))
    return [indent + line for line in textwrap.wrap(text, width=w)] or [indent]


# --------------------------------------------------------------------------- #
# Header — the agent "popping up"
# --------------------------------------------------------------------------- #

def header(st: Styler, rule_id: str, title: str, confidence: float) -> str:
    """``▌ gait   <title>   <rule_id> ....... <meter>  NN%`` — mirrors a diagnosis card."""
    conf = f"{confidence:.0%}"
    left_plain = f"▌ gait   {title}   {rule_id}"
    right_plain = f"{'?' * _METER_W}  {conf}"
    pad = max(2, _width() - len(left_plain) - len(right_plain))
    return (
        st.accent("▌")
        + " "
        + st.accent("gait")
        + "   "
        + st.accent(title)
        + "   "
        + st.dim(rule_id)
        + " " * pad
        + st.meter(confidence, _METER_W)
        + "  "
        + st.confidence(conf)
    )


# --------------------------------------------------------------------------- #
# TLDR — the agent says, in one breath, what it wants to do
# --------------------------------------------------------------------------- #

def tldr(st: Styler, proposed: Proposed, *, confidence_threshold: float) -> str:
    """The spoken summary: the claim, the target, and an honest caveat if unsure."""
    lines: list[str] = []
    lines.extend(_wrap(proposed.prediction.summary + "."))
    lines.append("")
    change = (
        st.label("change  ")
        + st.accent(f"{proposed.param} {proposed.current_value!r} ")
        + st.dim("→")
        + st.accent(f" {proposed.proposed_value!r}")
    )
    lines.append(_INDENT + change)
    lines.append(_INDENT + st.label("target  ") + st.dim(proposed.target.ref))

    if proposed.diagnosis.confidence < confidence_threshold:
        lines.append("")
        caveat = (
            f"this diagnosis is low-confidence ({proposed.diagnosis.confidence:.0%}); "
            f"I won't auto-apply it, and I'll be cautious about claiming it worked."
        )
        lines.extend(st.warn(line) if i == 0 else st.dim(line)
                     for i, line in enumerate(_wrap(caveat)))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Verbose — narrate each state-machine step
# --------------------------------------------------------------------------- #

def step(st: Styler, name: str, *detail: str) -> str:
    """A single narrated step: an accent dot, a fixed-width label, then its detail."""
    head_label = st.dim("›") + " " + st.label(f"{name:<8}")
    if not detail:
        return _INDENT + head_label
    first = _INDENT + head_label + detail[0]
    cont = [_INDENT + " " * 10 + d for d in detail[1:]]
    return "\n".join([first] + cont)


def predicted_effect(st: Styler, proposed: Proposed) -> list[str]:
    """The recorded prediction, rendered as a checklist gait will verify against."""
    out = [_INDENT + st.dim("predicted effect (recorded now, checked after apply):")]
    for c in proposed.prediction.checks:
        note = f"({c.description}; advisory)" if c.advisory else f"({c.description})"
        out.append(
            _INDENT + "  " + st.dim("· ")
            + f"{c.field} {c.comparison.value} {c.threshold}  "
            + st.dim(note)
        )
    return out


def diagnosis_line(st: Styler, proposed: Proposed) -> list[str]:
    return [_INDENT + st.dim("diagnosis:")] + _wrap(proposed.diagnosis.cause, _INDENT * 2)


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #

def applied_line(st: Styler, applied: Applied) -> str:
    c = applied.change
    return (
        f"{_INDENT}{st.good('✓')} {st.good('applied')}  "
        + st.accent(f"{c.param} {c.prior_value!r} ")
        + st.dim("→")
        + st.accent(f" {c.new_value!r}")
        + st.dim(f"   ·   undo: strided undo {c.change_id}")
    )


def verdict_block(st: Styler, verified: Verified) -> str:
    glyph, color, label = _VERDICT_STYLE[verified.verdict]
    color = st.dim_color if color is _DIM else color
    head = _INDENT + st.state(f"{glyph} verify: {label}", color)
    lines = [head]
    lines.extend(_wrap(verified.detail))
    # before/after for the fields that carried a value on either side.
    rows = [
        (f, b, verified.after.get(f))
        for f, b in verified.before.items()
        if b is not None or verified.after.get(f) is not None
    ]
    if rows:
        lines.append("")
        for f, b, a in rows:
            lines.append(_INDENT + st.label(f"{f}  ") + f"{b}" + st.dim(" → ") + f"{a}")
    return "\n".join(lines)


def abstained(st: Styler, ab: Abstained) -> str:
    """The agent stopping, said plainly. Every stop names what was missing."""
    detail = ab.payload.get("detail", ab.message)
    out = [f"{_INDENT}{st.dim('■')} " + st.dim("gait stopped") + st.dim(f": {detail}")]
    if ab.payload.get("candidates"):
        out.append(_INDENT + "  " + st.dim("candidates: " + ", ".join(map(str, ab.payload["candidates"]))))
    return "\n".join(out)


def declined_line(st: Styler) -> str:
    return _INDENT + st.dim("okay, nothing changed.")


def rolled_back_line(st: Styler, param: str, value: object) -> str:
    return (
        f"{_INDENT}{st.info('↩')} "
        + st.dim("rolled back  ")
        + st.accent(f"{param} ")
        + st.dim("→")
        + st.accent(f" {value!r}")
    )


__all__ = [
    "header",
    "tldr",
    "step",
    "predicted_effect",
    "diagnosis_line",
    "applied_line",
    "verdict_block",
    "abstained",
    "declined_line",
    "rolled_back_line",
]
