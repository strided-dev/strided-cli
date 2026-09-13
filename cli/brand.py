"""Canonical strided brand tokens, and how they map onto a terminal.

These are the tokens exported from the strided design system: the Porcelain
palette and the four-bar mark. This module holds the exported tokens and nothing
else. :mod:`cli.style` binds them to semantic styles.

Terminal mapping
----------------
A terminal is not Porcelain paper. We cannot repaint the surface, and we do not
try: the palette maps to *roles* instead.

===================  =======================================================
token                terminal role
===================  =======================================================
``bench``/``panel``  the terminal's own background. Never painted.
``ink``              the terminal's own default foreground. Never painted, so
                     the mark inverts correctly on any theme.
``inkSoft``          the dim tone on a light surface.
``inkFaint``         the dim tone on a dark surface.
``purple``           the one painted brand colour.
``accent``           no terminal role. On the web it is an ink button's hover
                     state against paper; there is no such surface here.
===================  =======================================================

Two surfaces
------------
The canonical purple is dark by design — it is meant to sit on Porcelain, where
it measures 11.8:1. Against a black terminal it measures 1.6:1 and is
effectively invisible, so a dark surface gets a lifted tint of the *same hue*
(274.4°); only lightness (25% → 62%) and saturation (100% → 70%) move. The
saturation is eased on purpose: held at 100% the lift lands on a neon
``#AC3DFF``, which reads as campaign art rather than an instrument, and the
brand's own visual rule is that "color behaves like an instrument state".

The dim tone splits the same way, and each half is the canonical token the
palette already intends for that surface: ``inkSoft`` is what the website uses
for small text on Porcelain, and ``inkFaint`` is the lighter grey that clears
4.6:1 against a black terminal where ``inkSoft`` would not.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# The exported palette — selected.palette.tokens, verbatim.
# --------------------------------------------------------------------------- #

BENCH = "#F7F7F5"
PANEL = "#FFFFFF"
BEZEL_DARK = "#C8CBC9"
BEZEL_LIGHT = "#FFFFFF"
INK = "#0B0D0C"
INK_SOFT = "#4D524F"
INK_FAINT = "#727774"
ACCENT = "#242826"

# selected.logo.bars — the mark's top two bars. Not part of the palette table;
# the logo carries its own colour.
PURPLE = "#4a0081"

# The dark-surface tint of PURPLE. Same hue, lifted for legibility (see above).
PURPLE_LIFTED = "#A85AE2"


def rgb(hex_color: str) -> tuple[int, int, int]:
    """``"#4a0081"`` → ``(74, 0, 129)``, the tuple form ``click.style`` wants."""
    value = hex_color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


# --------------------------------------------------------------------------- #
# Surface resolution
# --------------------------------------------------------------------------- #

LIGHT = "light"
DARK = "dark"


def resolve_surface(environ: dict[str, str] | None = None) -> str:
    """Decide whether we are painting onto a light or a dark terminal.

    Precedence, highest first:
      1. ``STRIDED_SURFACE=light|dark`` — the explicit escape hatch, for the
         many terminals that report nothing at all,
      2. ``COLORFGBG`` — set by rxvt, konsole and friends as ``fg;bg`` (or
         ``fg;<something>;bg``); a background in the 7-15 range is light,
      3. ``dark``, which is what most terminals are.
    """
    env = os.environ if environ is None else environ

    declared = env.get("STRIDED_SURFACE", "").strip().lower()
    if declared in (LIGHT, DARK):
        return declared

    fgbg = env.get("COLORFGBG", "")
    if fgbg:
        # The background is the last field; some terminals interpose a third.
        background = fgbg.rsplit(";", 1)[-1].strip()
        if background.isdigit():
            return LIGHT if 7 <= int(background) <= 15 else DARK

    return DARK


def purple(surface: str) -> tuple[int, int, int]:
    """The brand purple for ``surface``: canonical on light, lifted on dark."""
    return rgb(PURPLE if surface == LIGHT else PURPLE_LIFTED)


def dim(surface: str) -> tuple[int, int, int]:
    """The secondary-text grey for ``surface``: ink-soft on light, ink-faint on dark."""
    return rgb(INK_SOFT if surface == LIGHT else INK_FAINT)


__all__ = [
    "BENCH",
    "PANEL",
    "BEZEL_DARK",
    "BEZEL_LIGHT",
    "INK",
    "INK_SOFT",
    "INK_FAINT",
    "ACCENT",
    "PURPLE",
    "PURPLE_LIFTED",
    "LIGHT",
    "DARK",
    "rgb",
    "resolve_surface",
    "purple",
    "dim",
]
