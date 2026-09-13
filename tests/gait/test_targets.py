"""VllmArgsTarget — the three-valued locate and reversible write."""

from __future__ import annotations

import pytest

from gait import ABSENT, ResolveOutcome, VllmArgsTarget


def test_locate_explicit_flag_is_resolved(target):
    loc = target.locate("block-size")
    assert loc.outcome is ResolveOutcome.RESOLVED
    assert loc.value == 16
    assert isinstance(loc.value, int)
    assert loc.implicit is False


def test_locate_unset_known_param_resolves_to_default():
    t = VllmArgsTarget.from_command("python -m vllm --model X")
    loc = t.locate("block-size")
    assert loc.outcome is ResolveOutcome.RESOLVED
    assert loc.value == 16
    assert "default" in loc.note
    assert loc.implicit is True  # value is a default, the flag is not in the command


def test_locate_unknown_param_is_not_found():
    t = VllmArgsTarget.from_command("python -m vllm --model X")
    loc = t.locate("definitely-not-a-vllm-flag")
    assert loc.outcome is ResolveOutcome.NOT_FOUND


def test_locate_duplicate_flag_is_ambiguous():
    t = VllmArgsTarget.from_command("python -m vllm --block-size 16 --block-size 32")
    loc = t.locate("block-size")
    assert loc.outcome is ResolveOutcome.AMBIGUOUS
    assert set(loc.candidates) == {"16", "32"}


def test_write_replaces_existing_flag(target):
    target.write("block-size", 8)
    assert target.locate("block-size").value == 8
    assert "--block-size 8" in target.command()


def test_write_appends_absent_flag():
    t = VllmArgsTarget.from_command("python -m vllm --model X")
    t.write("swap-space", 8)
    assert "--swap-space 8" in t.command()


def test_write_is_reversible(target):
    prior = target.locate("block-size").value
    target.write("block-size", 4)
    target.write("block-size", prior)
    assert target.locate("block-size").value == prior


def test_write_refuses_ambiguous_param():
    t = VllmArgsTarget.from_command("python -m vllm --block-size 16 --block-size 32")
    with pytest.raises(ValueError):
        t.write("block-size", 8)


def test_write_absent_removes_an_added_flag():
    # ABSENT means restore-to-not-present: a param that was implicit, set, then
    # cleared must leave the command byte-identical to the original.
    original = "python -m vllm --model X"
    t = VllmArgsTarget.from_command(original)
    t.write("block-size", 8)            # apply: adds the flag
    assert "--block-size 8" in t.command()
    t.write("block-size", ABSENT)       # rollback: removes it
    assert "block-size" not in t.command()
    assert t.command() == original


def test_write_absent_is_noop_when_flag_absent():
    t = VllmArgsTarget.from_command("python -m vllm --model X")
    t.write("block-size", ABSENT)
    assert t.command() == "python -m vllm --model X"


def test_file_backing_persists_and_reloads(tmp_path):
    cfg = tmp_path / "launch.txt"
    cfg.write_text("python -m vllm --model X --block-size 16\n")
    t = VllmArgsTarget.from_file(cfg)
    t.write("block-size", 8)
    # Reload from disk: the edit persisted.
    assert VllmArgsTarget.from_file(cfg).locate("block-size").value == 8
