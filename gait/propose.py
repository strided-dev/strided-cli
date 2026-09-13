"""propose — render a reviewable change without performing it.

Produces a ``Proposed`` that fully describes a pending change: the diff (current →
proposed), the reasoning chain (which diagnosis, what confidence, why), and the
``Prediction`` — the concrete, checkable effect we commit to *now, before acting*,
so ``verify`` later checks a prior commitment rather than rationalizing after the
fact. If the recommended value already equals the current one, there is nothing to
do and propose abstains with NO_OP.

``fix --dry-run`` is exactly "run the machine to ``Proposed`` and print the preview,
then stop." Nothing downstream of here runs. READ-ONLY.
"""

from __future__ import annotations

import hashlib

from gait.fixes import fix_for
from gait.state import Proposed, Resolved
from gait.types import Abstained, AbstentionReason, Prediction


def propose(s: Resolved) -> Proposed | Abstained:
    """Build the full, non-mutating description of the pending change."""
    spec = fix_for(s.diagnosis.rule_id)
    # resolve() already guaranteed a spec exists; guard defensively.
    if spec is None:  # pragma: no cover - unreachable via the state machine
        return Abstained(
            AbstentionReason.NO_FIX_MAPPING,
            {"rule_id": s.diagnosis.rule_id, "detail": "fix registry changed mid-flight"},
        )

    proposed_value = s.recommended.proposed_value

    if proposed_value == s.current_value:
        return Abstained(
            AbstentionReason.NO_OP,
            {
                "param": s.param,
                "current_value": s.current_value,
                "detail": (
                    f"{s.param} is already at the recommended value "
                    f"{s.current_value!r}; nothing to change"
                ),
            },
        )

    prediction = spec.predict(s.diagnosis, s.snapshot, s.current_value, proposed_value)
    proposal_id = _proposal_id(s, proposed_value)
    preview = _render_preview(s, proposed_value, prediction, proposal_id)

    return Proposed(
        diagnosis=s.diagnosis,
        snapshot=s.snapshot,
        target=s.target,
        param=s.param,
        current_value=s.current_value,
        proposed_value=proposed_value,
        prediction=prediction,
        preview=preview,
        proposal_id=proposal_id,
    )


def _proposal_id(s: Resolved, proposed_value: object) -> str:
    raw = f"{s.diagnosis.rule_id}|{s.target.ref}|{s.param}|{s.current_value}|{proposed_value}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _render_preview(
    s: Resolved, proposed_value: object, prediction: Prediction, proposal_id: str
) -> str:
    d = s.diagnosis
    lines = [
        f"proposal {proposal_id} — {d.rule_id} (confidence {d.confidence:.0%})",
        "",
        "  diagnosis:",
        f"    {d.cause}",
        "",
        "  change:",
        f"    {s.param}: {s.current_value!r} → {proposed_value!r}",
        f"    target: {s.target.ref}",
        "",
        "  why:",
        f"    {s.recommended.rationale}",
        "",
        "  predicted effect (recorded now, checked after apply):",
    ]
    for c in prediction.checks:
        lines.append(f"    - {c.field} {c.comparison.value} {c.threshold}  ({c.description})")
    lines += ["", "  nothing has been changed. approve to apply."]
    return "\n".join(lines)


__all__ = ["propose"]
