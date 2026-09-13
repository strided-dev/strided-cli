"""rollback — restore the prior value of an applied change.

``Applied`` carries a :class:`Change` whose ``rollback`` is already bound (it knows
the target and the prior value, and it marks the journal). This transition invokes
it and returns the terminal ``RolledBack`` state. ``strided undo`` uses this for an
in-session reversal; cross-process undo is reconstructed from the journal in the CLI.
"""

from __future__ import annotations

from gait.state import Applied, RolledBack


def rollback(s: Applied) -> RolledBack:
    """Reverse an applied change, restoring the prior value."""
    s.change.rollback()
    return RolledBack(change=s.change, restored_value=s.change.prior_value)


__all__ = ["rollback"]
