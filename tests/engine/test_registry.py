"""Tests for the rule registry.

These enforce that the explicit registry stays consistent with the rule files
on disk — the safety net that lets us reject filesystem auto-discovery without
risking a silently-unregistered rule.
"""

from __future__ import annotations

import re
from pathlib import Path

from engine.registry import ALL_RULES, rule_ids
from rules.base import Rule

_RULES_DIR = Path(__file__).resolve().parents[2] / "rules"
_RULE_FILE = re.compile(r"^(r\d{2})_.+\.py$")


def _rule_ids_on_disk() -> set[str]:
    return {
        m.group(1)
        for path in _RULES_DIR.glob("r[0-9][0-9]_*.py")
        if (m := _RULE_FILE.match(path.name))
    }


class TestRegistry:
    def test_non_empty(self) -> None:
        assert ALL_RULES, "registry must contain at least one rule"

    def test_all_are_rule_subclasses(self) -> None:
        assert all(issubclass(r, Rule) for r in ALL_RULES)

    def test_rule_ids_unique(self) -> None:
        ids = rule_ids()
        assert len(ids) == len(set(ids)), f"duplicate rule_id in registry: {ids}"

    def test_rule_ids_well_formed(self) -> None:
        assert all(re.fullmatch(r"r\d{2}", rid) for rid in rule_ids())

    def test_registry_matches_files_on_disk(self) -> None:
        """Every rNN_*.py file is registered, and the registry invents none."""
        assert set(rule_ids()) == _rule_ids_on_disk()
