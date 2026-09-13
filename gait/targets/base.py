"""The ``ConfigTarget`` protocol — how ``gait`` reads and writes one config surface.

A target is the *declared* place a user's config lives. ``gait`` never goes hunting
across arbitrary deployment shapes; the user points it at exactly one surface and
``gait`` reads and writes only within it. New surfaces are added later by
implementing this protocol — the same extension pattern as a collector or a rule.

``locate`` is deliberately three-valued. The single most dangerous thing this agent
can do is confidently edit the wrong knob, so a target reports RESOLVED only when it
found *exactly one* place the param lives; otherwise it says AMBIGUOUS or NOT_FOUND
and lets :mod:`gait.resolve` abstain rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable


class _Absent:
    """Sentinel for 'this param was not present in the surface'.

    Used as a prior value so rollback restores a param to *absent* (removes the
    flag) rather than writing back an implicit engine default as an explicit one —
    keeping a change byte-exactly reversible. A singleton; compare with ``is``.
    """

    _instance: "_Absent | None" = None

    def __new__(cls) -> "_Absent":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<absent>"


ABSENT = _Absent()


class ResolveOutcome(Enum):
    """The three outcomes of locating a param. Never a silent guess."""

    RESOLVED = auto()   # found exactly one place the param lives
    AMBIGUOUS = auto()  # found several candidates → stop, ask the human
    NOT_FOUND = auto()  # found none → stop, abstain with the param name


@dataclass(frozen=True)
class Location:
    """The result of ``ConfigTarget.locate``.

    ``value`` is the current value when RESOLVED. ``candidates`` lists what was seen
    when AMBIGUOUS (so the human can disambiguate). ``note`` carries context such as
    "implicit default" for a param that is unset but has a known engine default.
    """

    outcome: ResolveOutcome
    value: Any = None
    candidates: tuple[str, ...] = ()
    note: str = ""
    implicit: bool = False  # RESOLVED from a default — the param is not written in the surface


@runtime_checkable
class ConfigTarget(Protocol):
    """A read/write surface for one kind of config.

    Implementations must be reversible-friendly: ``locate`` before a write captures
    the prior value, and ``write`` must be exactly undoable by writing that prior
    value back. Prefer inspectable mechanisms (edit a file, emit a patch) over poking
    a live process.
    """

    @property
    def ref(self) -> str:
        """Human-readable identifier of what is being edited (a path or command)."""
        ...

    def locate(self, param: str) -> Location:
        """Find ``param`` in this surface. Three-valued; never guesses."""
        ...

    def write(self, param: str, value: Any) -> None:
        """Set ``param`` to ``value`` within this surface (mutates)."""
        ...


__all__ = ["ResolveOutcome", "Location", "ConfigTarget", "ABSENT"]
