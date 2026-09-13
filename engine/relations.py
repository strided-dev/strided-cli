"""Declarative cross-rule relations: conflict and corroboration.

A rule is, by contract, blind to every other rule — ``rules/base.py`` forbids
cross-rule knowledge, and rules are unit-tested in isolation. Anything
*relational* therefore belongs to the engine, never to a rule.

We keep the relations here as data: symmetric sets of ``rule_id``s, separate
from the arithmetic in ``confidence.py``, so the relationships can be reviewed
without reading any formula. Both tables are validated against the registry at
import time — a relation that names an unknown rule is a bug we want to fail on
immediately, not silently ignore.

v1 status: the conflict table ships empty; the corroboration table holds one live
relation, ``{r02, r08}`` — the same prefill↔decode contention pathology seen from
two independent sources (r02's static /metrics fingerprint and r08's timeline
measurement). It is symmetric reinforcement, exactly what CORROBORATION_SETS is
for, so it is encoded rather than left latent. The boost stays off by default
(``confidence.py``), so today this only annotates ("corroborated by r08") without
moving the scalar — the cautious default until real co-firing dumps validate a
boost. Several other relationships remain *latent*:

- r04 (one TP rank is a slow outlier) is the *upstream cause* of r05 (the NCCL
  collective dominates step time): the collective looks slow because each rank's
  local NCCL time includes busy-wait on the straggler. This asymmetric cause-of is
  not expressible as a CONFLICT_SET (mutual exclusion) or CORROBORATION_SET
  (symmetric reinforcement), and the runner has no cause-of mechanism — so rather
  than encode it here, **r05 self-guards**: it reads ``tp_rank_sm_clocks`` and steps
  aside (BELOW_THRESHOLD) when an isolated straggler is present, leaving r04 to own
  the diagnosis. r05 fires alone only where the ranks are balanced. The relation is
  therefore resolved in the rule, not in these tables.

- r02 (colocation contention) folds in what would have been a standalone
  chunked-prefill rule, so there is no r02↔chunked-prefill conflict to encode —
  the two fixes are one rule's two tiers.
- r02 is frequently the *upstream cause* of decode-underutilisation (r01) and
  KV-fragmentation (r03) symptoms; on a colocated dump with a positive contention
  fingerprint those should be demoted to contributing evidence. Likewise r03 is
  often the upstream cause of r01's memory-bound decode (fragmentation caps the
  batch). These are asymmetric *cause-of* relations, which CONFLICT_SETS (mutual
  exclusion) and CORROBORATION_SETS (symmetric reinforcement) cannot express, and
  the runner has no cause-of mechanism yet — so they remain documented here and
  in the rules' specs, not encoded, until that mechanism lands.

- r07 (unfused attention path) follows the r05 self-guard pattern for its
  boundary with the *future* r09 (long-context inefficiency): when fused
  attention kernels are already present, r07 steps aside (BELOW_THRESHOLD) —
  fused-and-still-dominant is workload shape, not an implementation gap. Its
  overlap with r01 is mechanism-level only (both describe memory-bound work,
  on different fields, with composable fixes), so co-firing is legitimate and
  nothing is encoded here.

The query helpers and their tests exercise the mechanism via synthetic
``rule_id``s so the behaviour is already correct when r03–r10 land.
"""

from __future__ import annotations

from collections.abc import Iterable

from engine.registry import rule_ids

# ---------------------------------------------------------------------------
# Relation tables (data only). Each group is a symmetric set of rule_ids.
# ---------------------------------------------------------------------------

# Rules that are mutually contradictory: at most one should be presented as the
# top cause. How a conflict is resolved (annotate / demote / suppress) is the
# runner's job; this table only declares the relationship.
CONFLICT_SETS: tuple[frozenset[str], ...] = ()

# Rules that reinforce one another — same root cause seen from different
# evidence. Co-occurrence raises joint confidence, but only when the engine's
# corroboration boost is explicitly enabled (off by default in v1, pending
# validation against real customer dumps).
#
# {r02, r08}: prefill↔decode contention. r02 infers it correlationally from a
# static /metrics fingerprint (preemptions + TPOT tail); r08 measures it directly
# from the nsys timeline (long prefills stalling the decode cadence). Different
# evidence, one pathology — so when both fire, each corroborates the other.
CORROBORATION_SETS: tuple[frozenset[str], ...] = (
    frozenset({"r02", "r08"}),
)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _related(
    sets: tuple[frozenset[str], ...], rule_id: str, present_ids: set[str]
) -> tuple[str, ...]:
    """rule_ids related to ``rule_id`` via ``sets`` and also present in input.

    Returns a sorted tuple (excluding ``rule_id`` itself) so the result is
    deterministic regardless of set iteration order.
    """
    related: set[str] = set()
    for group in sets:
        if rule_id in group:
            related |= group & present_ids
    related.discard(rule_id)
    return tuple(sorted(related))


def corroborators_of(rule_id: str, present_ids: Iterable[str]) -> tuple[str, ...]:
    """Rules that corroborate ``rule_id`` and are present in ``present_ids``."""
    return _related(CORROBORATION_SETS, rule_id, set(present_ids))


def conflicts_of(rule_id: str, present_ids: Iterable[str]) -> tuple[str, ...]:
    """Rules that conflict with ``rule_id`` and are present in ``present_ids``."""
    return _related(CONFLICT_SETS, rule_id, set(present_ids))


# ---------------------------------------------------------------------------
# Import-time validation
# ---------------------------------------------------------------------------

def _validate(known: set[str] | None = None) -> None:
    """Fail loudly if a relation is malformed or names an unknown rule.

    A relation needs at least two members to mean anything, and every member
    must be a registered rule. ``known`` is injectable for tests; it defaults
    to the live registry.
    """
    known = set(rule_ids()) if known is None else known
    for label, sets in (
        ("CONFLICT_SETS", CONFLICT_SETS),
        ("CORROBORATION_SETS", CORROBORATION_SETS),
    ):
        for group in sets:
            if len(group) < 2:
                raise ValueError(
                    f"{label} contains a group with fewer than two rule_ids: "
                    f"{set(group)!r}. A relation needs at least two members."
                )
            unknown = group - known
            if unknown:
                raise ValueError(
                    f"{label} references unknown rule_id(s) {sorted(unknown)!r}; "
                    f"not in the registry."
                )


_validate()


__all__ = [
    "CONFLICT_SETS",
    "CORROBORATION_SETS",
    "corroborators_of",
    "conflicts_of",
]
