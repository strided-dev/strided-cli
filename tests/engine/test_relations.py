"""Tests for the cross-rule relation tables and their validation.

The live tables are empty in v1, so these exercise the query helpers and the
import-time validator with synthetic rule_ids.
"""

from __future__ import annotations

import pytest

from engine import relations


class TestQueries:
    def test_corroborators_filtered_to_present(self) -> None:
        sets = (frozenset({"r01", "r03", "r05"}),)
        result = relations._related(sets, "r01", {"r01", "r03"})
        assert result == ("r03",)  # r05 absent; r01 excluded from its own result

    def test_no_relation_returns_empty(self) -> None:
        result = relations._related((frozenset({"r02", "r04"}),), "r01", {"r01", "r02"})
        assert result == ()

    def test_result_is_sorted(self) -> None:
        sets = (frozenset({"r01", "r09", "r03"}),)
        assert relations._related(sets, "r01", {"r03", "r09"}) == ("r03", "r09")


class TestValidation:
    def test_unknown_rule_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown rule_id"):
            _validate_with(relations, conflict=(frozenset({"r01", "r99"}),), known={"r01"})

    def test_undersized_group_rejected(self) -> None:
        with pytest.raises(ValueError, match="fewer than two"):
            _validate_with(relations, corroboration=(frozenset({"r01"}),), known={"r01"})

    def test_valid_tables_pass(self) -> None:
        _validate_with(
            relations,
            conflict=(frozenset({"r01", "r02"}),),
            known={"r01", "r02"},
        )  # no raise


def _validate_with(mod, *, conflict=(), corroboration=(), known) -> None:
    """Run _validate against patched tables without monkeypatch fixtures."""
    orig_c, orig_k = mod.CONFLICT_SETS, mod.CORROBORATION_SETS
    mod.CONFLICT_SETS, mod.CORROBORATION_SETS = conflict, corroboration
    try:
        mod._validate(known=known)
    finally:
        mod.CONFLICT_SETS, mod.CORROBORATION_SETS = orig_c, orig_k
