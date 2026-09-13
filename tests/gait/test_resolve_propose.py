"""resolve + propose — the read-only stages, and their abstentions."""

from __future__ import annotations

from gait import (
    Abstained,
    AbstentionReason,
    Diagnosed,
    Proposed,
    Resolved,
    VllmArgsTarget,
    propose,
    resolve,
)

from gait_builders import make_diagnosis, make_snapshot


def _diagnosed(diagnosis=None, snapshot=None):
    return Diagnosed(diagnosis or make_diagnosis(), snapshot or make_snapshot())


# -- resolve ---------------------------------------------------------------- #

def test_resolve_finds_the_param(target, diagnosis, snapshot):
    r = resolve(Diagnosed(diagnosis, snapshot), target)
    assert isinstance(r, Resolved)
    assert r.param == "block-size"
    assert r.current_value == 16
    assert r.recommended.proposed_value == 8


def test_resolve_abstains_for_unknown_rule(target, snapshot):
    r = resolve(Diagnosed(make_diagnosis(rule_id="r99"), snapshot), target)
    assert isinstance(r, Abstained)
    assert r.reason is AbstentionReason.NO_FIX_MAPPING


def test_resolve_abstains_for_non_vllm_engine(target):
    snap = make_snapshot(engine="trt-llm")
    r = resolve(Diagnosed(make_diagnosis(), snap), target)
    assert isinstance(r, Abstained)
    assert r.reason is AbstentionReason.NO_FIX_MAPPING
    assert "trt-llm" in r.message


def test_resolve_abstains_when_param_not_found(diagnosis, snapshot):
    # A command with no block-size and no default? block-size always has a default,
    # so force NOT_FOUND by pointing the fix at a surface that lacks the knob: use a
    # target whose param genuinely isn't known — here we simulate via a command and a
    # rule whose param is unknown is covered elsewhere; instead delete the default by
    # using a target that reports the param's flag with no value.
    target = VllmArgsTarget.from_command("python -m vllm --model X --block-size")
    r = resolve(Diagnosed(diagnosis, snapshot), target)
    assert isinstance(r, Abstained)
    assert r.reason is AbstentionReason.PARAM_NOT_FOUND
    assert r.payload["param"] == "block-size"


def test_resolve_abstains_when_ambiguous(diagnosis, snapshot):
    target = VllmArgsTarget.from_command("python -m vllm --block-size 16 --block-size 32")
    r = resolve(Diagnosed(diagnosis, snapshot), target)
    assert isinstance(r, Abstained)
    assert r.reason is AbstentionReason.AMBIGUOUS_CONFIG
    assert set(r.payload["candidates"]) == {"16", "32"}


# -- propose ---------------------------------------------------------------- #

def test_propose_builds_prediction_and_preview(target, diagnosis, snapshot):
    p = propose(resolve(Diagnosed(diagnosis, snapshot), target))
    assert isinstance(p, Proposed)
    assert p.proposed_value == 8
    assert p.proposal_id  # stable, non-empty
    # Prediction is recorded BEFORE acting and names checkable fields.
    fields = {c.field for c in p.prediction.checks}
    assert "kv_cache_fragmentation" in fields
    assert "block-size: 16" in p.preview and "→ 8" in p.preview


def test_propose_abstains_on_noop(diagnosis, snapshot):
    # block-size already 1 → halving floors at 1 → proposed == current → NO_OP.
    target = VllmArgsTarget.from_command("python -m vllm --model X --block-size 1")
    p = propose(resolve(Diagnosed(diagnosis, snapshot), target))
    assert isinstance(p, Abstained)
    assert p.reason is AbstentionReason.NO_OP


def test_proposal_id_is_deterministic(target, diagnosis, snapshot):
    a = propose(resolve(Diagnosed(diagnosis, snapshot), target))
    b = propose(resolve(Diagnosed(diagnosis, snapshot),
                        VllmArgsTarget.from_command(target.command())))
    assert a.proposal_id == b.proposal_id
