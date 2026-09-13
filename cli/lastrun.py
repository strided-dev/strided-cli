"""The last run's fired rule ids, for the interactive session to offer back.

A presentation-layer cache, nothing more. ``diagnose`` and ``watch`` publish what
fired; :mod:`cli.home` reads it to complete ``fix <tab>`` against the rules that
actually fired rather than the whole registry, and to offer the next step as the
prompt's default. Nothing in the engine, the rules, or the report depends on it,
and a stale or empty value only costs a suggestion.
"""

from __future__ import annotations

from typing import Iterable

FIRED: tuple[str, ...] = ()


def record(rule_ids: Iterable[str]) -> None:
    """Publish the rule ids that fired in the run that just rendered."""
    global FIRED
    FIRED = tuple(rule_ids)


__all__ = ["FIRED", "record"]
