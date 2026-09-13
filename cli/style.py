"""Terminal styling primitives for the strided CLI.

Colour is a pure presentation concern, isolated here so the renderer can ask
for *semantic* styles ("this is a section header", "this is a confidence
value") without ever touching an ANSI code, and so the whole thing collapses
to plain text the moment the output is not an interactive terminal.

Built on ``click.style`` (already a dependency) — no new packages, and
``click``'s colorama integration makes the colours work on Windows PowerShell
too. We honour the NO_COLOR convention (https://no-color.org) and FORCE_COLOR
(handy for CI/tests where stdout is not a tty).

Design language
---------------
One brand colour, used sparingly. strided's accent is the logo purple
(``#4a0081``); it marks the things that are *strided's voice* — the wordmark,
the card rail, a confidence value — and nothing else. Structure (section
labels, keys, dividers) is quiet grey; state (a roofline class, a verdict)
borrows the conventional traffic-light colours. Glyphs are emitted in both the
coloured and the plain paths so piped output keeps the same shape, only without
the paint — which keeps the renderer's "colour never changes the text" contract.

Every colour here comes from :mod:`cli.brand`, which holds the exported design
tokens and picks the light- or dark-surface variant. The palette is resolved
once per :class:`Styler`, so a caller can pin a surface for a test without
touching the environment.
"""

from __future__ import annotations

import os
from typing import IO, Optional, Union

import click

from cli import brand

# What ``click.style`` accepts as a colour: one of its named ANSI colours, or an
# RGB tuple for truecolor. The palette gives us the latter; instrument states are
# named, so both forms travel through the same methods.
Color = Union[str, tuple[int, int, int]]

# Glyph vocabulary — kept in one place so the whole CLI speaks the same symbols.
GLYPH_RAIL = "▌"      # the accent rail down the left of a diagnosis/agent card
GLYPH_SECTION = "▪"   # a small mark that leads every section header
GLYPH_STATE = "●"     # a filled dot carrying a state colour (roofline class…)
GLYPH_METER_FULL = "█"
GLYPH_METER_EMPTY = "░"
GLYPH_RULE = "─"

# Roofline position → colour. These are instrument states, not brand colour:
# the conventional traffic-light reading is more useful here than anything the
# palette could say, and the brand reserves its purple for strided's own voice.
# "unknown" is deliberately absent: it takes the Styler's dim tone, which depends
# on the terminal surface and so cannot be pinned in a module-level table.
_ROOFLINE_COLORS: dict[str, Color] = {
    "compute_bound": "cyan",
    "memory_bound": "yellow",
    "balanced": "green",
}


def should_color(stream: IO[str], flag: Optional[bool]) -> bool:
    """Decide whether to emit ANSI colour to ``stream``.

    Precedence, highest first:
      1. an explicit ``--color`` / ``--no-color`` flag,
      2. ``NO_COLOR`` set (any value) → never colour,
      3. ``FORCE_COLOR`` set → always colour (lets tests exercise the colour
         path even though their captured stdout is not a tty),
      4. otherwise: colour only when ``stream`` is an interactive terminal.
    """
    if flag is not None:
        return flag
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 — a stream without isatty is simply "not a tty"
        return False


class Styler:
    """Semantic style functions bound to one on/off decision.

    When ``enabled`` is False every method returns plain text, so the renderer's
    output is byte-for-byte the same shape, only without ANSI. That keeps piped
    output clean and the formatter's text contract trivial to assert on. Glyphs
    are intentionally emitted in *both* modes — they are part of the layout, not
    the colour — so stripping the ANSI from a coloured render yields exactly the
    plain render.

    ``surface`` selects the light or dark variant of the palette; left None it
    is resolved from the environment by :func:`cli.brand.resolve_surface`.
    """

    def __init__(self, enabled: bool, *, surface: Optional[str] = None) -> None:
        self.enabled = enabled
        self.surface = brand.resolve_surface() if surface is None else surface
        self._accent = brand.purple(self.surface)
        self._dim = brand.dim(self.surface)

    @property
    def dim_color(self) -> Color:
        """This Styler's resolved dim tone, for callers passing an explicit colour."""
        return self._dim

    def _style(self, text: str, **kwargs) -> str:
        return click.style(text, **kwargs) if self.enabled else text

    # -- structure ---------------------------------------------------------- #

    def section(self, text: str) -> str:
        """A section header — an accent tick plus a bright, bold label."""
        return self._style(GLYPH_SECTION, fg=self._accent) + " " + self._style(text, bold=True)

    def label(self, text: str) -> str:
        """A dim key in a key/value row."""
        return self._style(text, fg=self._dim)

    def dim(self, text: str) -> str:
        """Secondary / de-emphasised text."""
        return self._style(text, fg=self._dim)

    def rule(self, width: int) -> str:
        """A faint full-width horizontal divider."""
        return self._style(GLYPH_RULE * max(0, width), fg=self._dim)

    # -- brand -------------------------------------------------------------- #

    def accent(self, text: str) -> str:
        """The brand accent — diagnosis titles, the card rail, the wordmark."""
        return self._style(text, fg=self._accent, bold=True)

    def confidence(self, text: str) -> str:
        """A confidence percentage."""
        return self._style(text, fg=self._accent, bold=True)

    def mark(self, text: str, *, ink: bool) -> str:
        """A run of the logo mark — an ink bar or a purple one.

        The mark's lower bars are ink, which in a terminal is the surface's own
        default foreground: leaving them unpainted is what makes the mark invert
        correctly on a light theme and a dark one alike.
        """
        return text if ink else self._style(text, fg=self._accent)

    def meter(self, fraction: Optional[float], width: int = 12) -> str:
        """A fixed-width confidence bar: accent for the filled run, dim for the rest.

        The *visible* width is always ``width`` regardless of fill or colour, so
        callers can align around it using the plain length ``width``.
        """
        frac = 0.0 if fraction is None else max(0.0, min(1.0, fraction))
        filled = int(round(frac * width))
        full = GLYPH_METER_FULL * filled
        empty = GLYPH_METER_EMPTY * (width - filled)
        if not self.enabled:
            return full + empty
        return click.style(full, fg=self._accent) + click.style(empty, fg=self._dim)

    # -- state -------------------------------------------------------------- #

    def good(self, text: str) -> str:
        return self._style(text, fg="green", bold=True)

    def warn(self, text: str) -> str:
        return self._style(text, fg="yellow", bold=True)

    def info(self, text: str) -> str:
        return self._style(text, fg="cyan", bold=True)

    def state(self, text: str, color: Color, *, bold: bool = True) -> str:
        """A status word in an explicit colour (verdicts, live events)."""
        return self._style(text, fg=color, bold=bold)

    def dot(self, color: Color) -> str:
        """A filled state dot in ``color`` — pairs with :meth:`roofline`/verdicts."""
        return self._style(GLYPH_STATE, fg=color)

    def roofline(self, position: str, text: Optional[str] = None) -> str:
        """Colour a roofline position label by its class, led by a state dot."""
        label = position.replace("_", "-") if text is None else text
        color = _ROOFLINE_COLORS.get(position, self._dim)
        return self._style(GLYPH_STATE + " " + label, fg=color)


__all__ = [
    "Color",
    "should_color",
    "Styler",
    "GLYPH_RAIL",
    "GLYPH_SECTION",
    "GLYPH_STATE",
    "GLYPH_RULE",
]
