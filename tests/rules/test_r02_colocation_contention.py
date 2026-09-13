"""
Tests for r02 — Colocation contention (prefill ↔ decode).

The acceptance matrix for r02, plus
helper-level and evidence coverage. The abstention cases (3, 4, 6, 8, 9, 10) are
the point of the rule: they are what stop it from confidently telling a Tier-1
engineer to re-architect a stack that is fine.

`TestTailGateG5` covers case 10 and, more importantly, the *cost* of that gate.
G5 is the second predicate change in a change whose stated job was the inverted-arms finding, and
it is the one that narrows rather than widens: it silences a region `main` fires
on, and no field capture reaches that region. Read its class docstring before
touching TPOT_TAIL_MILD.

`TestInvertedArmsRegression` pins the inverted-arms finding with
the numbers actually measured on a field capture. Read it before touching
the fingerprint: the field long-context arm outscores the field contended arm on
every signal the score blends, so those tests fail the moment the concurrency
gate is weakened back into a re-weighting exercise. The old synthetic fixtures
could not catch that, because they were built from the same inverted mental model
as the guard — they asserted long context has a *low* prefill share, and the
field measured 0.941.
"""

from __future__ import annotations

import pytest

from rules.base import Abstention, Diagnosis, InsufficientData
from rules.r02_colocation_contention import (
    CONCURRENCY_MIN,
    CONTENTION_MIN,
    LONG_CONTEXT_SHARE,
    PD_WORK_MIN,
    PREEMPT_MILD,
    TPOT_TAIL_MILD,
    ColocationContentionRule,
    SloProfile,
    infer_topology,
    roofline_corroborates,
    score_contention,
)
from schema import DiagnosisInput, Distribution, PhaseMetrics, VllmServingMetrics


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def _input(
    *,
    # serving counters
    preemptions: int | None = 4,
    successes: int | None = 200,
    prompt_tokens: int | None = 300,
    generation_tokens: int | None = 700,
    # Default concurrency is a CONTENDED server: gate G4 requires
    # num_requests_running >= 16, and the field contended arm measured 32.6 mean.
    # Before the inverted-arms finding this defaulted to 8, which the
    # rule never looked at; every firing case in this file implicitly assumed a
    # concurrency the fingerprint could not see.
    running: int | None = 32,
    waiting: int | None = 2,
    connector: str | None = None,
    serving_present: bool = True,
    # latency
    tpot_p50: float | None = 10.0,
    tpot_p99: float | None = 24.0,
    tpot_present: bool = True,
    # phases / topology
    prefill_present: bool = True,
    decode_present: bool = True,
    prefill_roofline: str | None = None,
    decode_roofline: str | None = None,
    # scale
    model_params_b: float | None = 13.0,
    num_gpus: int | None = 1,
) -> DiagnosisInput:
    serving = (
        VllmServingMetrics(
            num_preemptions_total=preemptions,
            request_success_total=successes,
            prompt_tokens_total=prompt_tokens,
            generation_tokens_total=generation_tokens,
            num_requests_running=running,
            num_requests_waiting=waiting,
            kv_transfer_connector=connector,
        )
        if serving_present
        else None
    )
    tpot = (
        Distribution(mean=tpot_p50 or 0.0, p50=tpot_p50, p99=tpot_p99)
        if tpot_present
        else None
    )
    prefill = (
        PhaseMetrics(roofline_position=prefill_roofline) if prefill_present else None
    )
    decode = PhaseMetrics(roofline_position=decode_roofline) if decode_present else None
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-13B",
        gpu_type="H100-SXM",
        inference_engine="vllm",
        model_params_b=model_params_b,
        num_gpus=num_gpus,
        prefill=prefill,
        decode=decode,
        tpot_ms=tpot,
        vllm_serving=serving,
    )


@pytest.fixture
def rule() -> ColocationContentionRule:
    return ColocationContentionRule()


# --------------------------------------------------------------------------- #
# Acceptance matrix
# --------------------------------------------------------------------------- #

class TestAcceptanceMatrix:
    def test_1_true_positive_mild(self, rule: ColocationContentionRule) -> None:
        # colocated, preempt 2%, tail 2.4x, share 0.30, 13B / 1 GPU
        result = rule.evaluate(
            _input(preemptions=4, successes=200, prompt_tokens=300,
                   generation_tokens=700, tpot_p50=10.0, tpot_p99=24.0)
        )
        assert isinstance(result, Diagnosis)
        assert result.rule_id == "r02"
        assert result.evidence["fix_tier"] == "chunked_prefill"
        assert 0.55 <= result.confidence <= 0.65
        assert "chunked prefill" in result.fix.lower()

    def test_2_true_positive_severe_at_scale(self) -> None:
        # preempt 7%, tail 5x, 70B / 16 GPU, latency_bound, tpot SLO violated
        slo = SloProfile(kind="latency_bound", ttft_slo_ms=500.0, tpot_slo_ms=20.0)
        rule = ColocationContentionRule(slo=slo)
        result = rule.evaluate(
            _input(preemptions=14, successes=200, tpot_p50=10.0, tpot_p99=50.0,
                   model_params_b=70.0, num_gpus=16)
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["fix_tier"] == "disaggregate"
        assert result.confidence >= 0.65
        assert "disaggregation" in result.fix.lower()

    def test_3_already_disaggregated_abstains(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(connector="NixlConnector"))
        assert result is Abstention.BELOW_THRESHOLD

    def test_4_decode_dominated_workload_abstains(self, rule: ColocationContentionRule) -> None:
        """Tail 4x but prefill is 5% of token work -> admission gate G2 suppresses.

        Renamed from `test_4_long_context_no_contention_abstains`. This input is
        NOT a long-context workload: a 50/950 prompt/generation split is
        decode-dominated (short prompts, long answers — chat/agent traffic). The
        old name encoded the inverted assumption the inverted-arms finding disproved, and the case
        it claimed to cover — real long context, prefill share 0.941 — is in
        `TestInvertedArmsRegression` below, where it lands on the opposite side
        of G2 and has to be stopped by concurrency instead.
        """
        result = rule.evaluate(
            _input(preemptions=0, prompt_tokens=50, generation_tokens=950,
                   tpot_p50=10.0, tpot_p99=40.0)
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_5_throughput_bound_forces_chunked(self) -> None:
        # severe fingerprint, but throughput_bound -> never disaggregate
        slo = SloProfile(kind="throughput_bound")
        rule = ColocationContentionRule(slo=slo)
        result = rule.evaluate(
            _input(preemptions=14, successes=200, tpot_p50=10.0, tpot_p99=50.0,
                   model_params_b=70.0, num_gpus=16)
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["fix_tier"] == "chunked_prefill"

    def test_6_too_few_samples_abstains(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(successes=40, preemptions=2))
        assert isinstance(result, InsufficientData)
        assert "40" in result.reason and "200" in result.reason

    def test_7_small_scale_real_contention_routes_chunked(self, rule: ColocationContentionRule) -> None:
        # severe fingerprint but 7B / 1 GPU -> scale gate blocks disaggregate
        result = rule.evaluate(
            _input(preemptions=14, successes=200, tpot_p50=10.0, tpot_p99=50.0,
                   model_params_b=7.0, num_gpus=1)
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["fix_tier"] == "chunked_prefill"

    def test_8_malformed_dump_abstains(self, rule: ColocationContentionRule) -> None:
        # prefill labelled memory_bound, decode compute_bound -> inverted roofline
        result = rule.evaluate(
            _input(prefill_roofline="memory_bound", decode_roofline="compute_bound")
        )
        assert isinstance(result, InsufficientData)
        assert "roofline" in result.reason

    def test_9_low_concurrency_abstains(self, rule: ColocationContentionRule) -> None:
        # Severe-looking fingerprint, but only 5 requests resident: nothing to
        # contend with, so the tail is not interference (gate G4).
        result = rule.evaluate(
            _input(preemptions=14, tpot_p50=10.0, tpot_p99=50.0, running=5)
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_10_flat_tail_abstains(self, rule: ColocationContentionRule) -> None:
        """Preempt 5%, tail 1.5x, 32 running -> BELOW_THRESHOLD (G5).

        The matrix row for the gate this change made explicit. See
        `TestTailGateG5` for why this narrowing is deliberate and what it costs:
        the same input fired at capped confidence before G5 existed.
        """
        result = rule.evaluate(
            _input(preemptions=10, successes=200, tpot_p50=10.0, tpot_p99=15.0)
        )
        assert result is Abstention.BELOW_THRESHOLD


# --------------------------------------------------------------------------- #
# Inverted-arms regression
#
# Numbers are the ones measured on a field capture (A100-PCIE-40GB, vLLM
# 0.10.2, Qwen2.5-7B-Instruct), not invented ones. The shipped rule scored the
# long-context arm at 0.541, ABOVE the genuinely contended arm at 0.526, and
# fired on it at full confidence.
# --------------------------------------------------------------------------- #

class TestInvertedArmsRegression:
    # 4096-token prompts generating 256 tokens: prefill_share = 0.941.
    _LONGCTX_PROMPT, _LONGCTX_GEN = 4096 * 100, 256 * 100
    # The contended arm's measured share, 0.842, at the same token scale.
    _CONTEND_PROMPT, _CONTEND_GEN = 842_000, 158_000

    def _longctx(self, **over):
        """Long prompts, low concurrency: tail 10.54, preempt 0, running 5.5."""
        kw = dict(preemptions=0, successes=300,
                  prompt_tokens=self._LONGCTX_PROMPT,
                  generation_tokens=self._LONGCTX_GEN,
                  tpot_p50=10.0, tpot_p99=105.4, running=5)
        kw.update(over)
        return _input(**kw)

    def _contend(self, **over):
        """Heavy load: tail 6.04, preempt 0, running 32.6."""
        kw = dict(preemptions=0, successes=300,
                  prompt_tokens=self._CONTEND_PROMPT,
                  generation_tokens=self._CONTEND_GEN,
                  tpot_p50=10.0, tpot_p99=60.4, running=33)
        kw.update(over)
        return _input(**kw)

    def test_field_longctx_does_not_fire(self, rule: ColocationContentionRule) -> None:
        """The disqualifying false positive. This is the whole finding."""
        result = rule.evaluate(self._longctx())
        assert not isinstance(result, Diagnosis)
        assert result is Abstention.BELOW_THRESHOLD

    def test_field_contend_still_fires(self, rule: ColocationContentionRule) -> None:
        """The true positive the rule got right must stay right."""
        result = rule.evaluate(self._contend())
        assert isinstance(result, Diagnosis)
        assert result.evidence["num_requests_running"] == 33

    def test_field_healthy_does_not_fire(self, rule: ColocationContentionRule) -> None:
        # tail 1.42, share 0.571, 2.6 concurrent — the true negative.
        result = rule.evaluate(
            _input(preemptions=0, successes=300, prompt_tokens=571_000,
                   generation_tokens=429_000, tpot_p50=10.0, tpot_p99=14.2,
                   running=3)
        )
        assert not isinstance(result, Diagnosis)

    def test_score_ranks_the_true_positive_above_the_lookalike(self) -> None:
        """The ranking inversion at the heart of the inverted-arms finding, pinned in both directions.

        Shipped, the blend read `prefill_share` as its third term and scored the
        long-context arm 0.541 against the contended arm's 0.526 — the documented
        lookalike ABOVE the real case. That could not be re-weighted away: both
        arms saturate the tail term (6.04 and 10.54 each clamp to 1.0) and both
        measured 0.0 preemptions, so the third term was the only one that could
        order them, and it ran backwards (0.941 > 0.842).

        With concurrency in that slot the ordering is correct and the margin comes
        from the signal that causally belongs there. Assert the ORDER, not just
        the values: the order is the finding.
        """
        contend = score_contention(0.0, 6.04, 32.6)
        longctx = score_contention(0.0, 10.54, 5.5)
        healthy = score_contention(0.0, 1.42, 2.6)

        assert contend > longctx, "the lookalike must never outscore the real case"
        assert longctx > healthy
        assert contend == pytest.approx(0.550, abs=0.005)
        assert longctx == pytest.approx(0.426, abs=0.005)
        assert healthy == pytest.approx(0.068, abs=0.005)

    def test_tail_and_preemptions_alone_cannot_order_the_arms(self) -> None:
        """The proof that the replacement had to be a *new* signal.

        Holding concurrency equal, the two field arms are indistinguishable to
        the rest of the fingerprint — the long-context arm is if anything worse.
        This is why the inverted-arms finding is a design finding and not a calibration target: no
        weighting of preemption rate and TPOT tail separates these two.
        """
        same_conc = 20.0
        contend = score_contention(0.0, 6.04, same_conc)
        longctx = score_contention(0.0, 10.54, same_conc)
        assert longctx >= contend

    def test_prefill_share_cannot_separate_the_arms(self) -> None:
        """The retired guard's band is one no long-context workload can enter.

        Measured share is 0.941 for long context against 0.842 for contention:
        both far above LONG_CONTEXT_SHARE (0.20), and ordered the *opposite* way
        to what the guard assumed. There is no cut on this variable that keeps
        `contend` and drops `longctx`.
        """
        longctx_share = 4096 / (4096 + 256)
        contend_share = 0.842
        assert longctx_share == pytest.approx(0.941, abs=0.001)
        assert longctx_share > contend_share > LONG_CONTEXT_SHARE

    def test_discriminator_separates_the_arms_by_at_least_3x(self) -> None:
        """The acceptance bar for a replacement discriminator.

        A signal earns its place in the predicate by separating the arms it is
        asked to tell apart. `prefill_share` separated them by 1.12x — and in the
        wrong direction. `num_requests_running` separates them by 5.9x in the
        right one (32.6 vs 5.5, a field capture). 3x is the floor the
        replacement had to clear; the field margin is nearly double it.

        Stated as a ratio rather than a threshold on purpose: CONCURRENCY_MIN is
        provisional and will move under calibration, but if a recalibration ever
        leaves these two arms closer than 3x apart, the signal has stopped
        discriminating and the rule is back where the inverted-arms finding found it.
        """
        contend_running, longctx_running = 32.6, 5.5
        separation = contend_running / longctx_running

        assert separation >= 3.0
        assert separation == pytest.approx(5.93, abs=0.05)
        # The floor must sit strictly between the arms, or it separates nothing.
        assert longctx_running < CONCURRENCY_MIN < contend_running

        # For contrast: the retired discriminator, on the same two arms.
        assert (0.941 / 0.842) < 1.2

    def test_long_context_at_high_concurrency_does_fire(
        self, rule: ColocationContentionRule
    ) -> None:
        """Not a false positive: that server really is contended.

        Same long prompts, but 40 requests resident at once. The rule is meant to
        fire here — the guard being removed suppressed long context as a
        *category*, when the thing that makes it benign is the absence of
        competition, not the length of the prompt.
        """
        result = rule.evaluate(self._longctx(running=40))
        assert isinstance(result, Diagnosis)


class TestConcurrencyGate:
    def test_missing_num_requests_running_is_insufficient(
        self, rule: ColocationContentionRule
    ) -> None:
        """Abstain loudly, never silently fire, when the discriminator is absent."""
        result = rule.evaluate(_input(running=None))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("vllm_serving.num_requests_running",)

    def test_missing_serving_block_names_the_concurrency_gauge(
        self, rule: ColocationContentionRule
    ) -> None:
        result = rule.evaluate(_input(serving_present=False))
        assert isinstance(result, InsufficientData)
        assert "vllm_serving.num_requests_running" in result.missing

    def test_boundary_exactly_at_the_negative_arms_max_abstains(
        self, rule: ColocationContentionRule
    ) -> None:
        """CONCURRENCY_MIN is the long-context arm's observed *maximum* (16).

        This test previously asserted that the boundary FIRES, on the reasoning
        that the gate was `< CONCURRENCY_MIN` so `==` passes. That was a
        description of the code, not a claim about correctness: it pinned a
        full-confidence false positive on a measured field value, since 16 is the
        worst tick the *negative* arm actually produced. The gate is now exclusive
        (`<= CONCURRENCY_MIN` abstains) and this asserts the silence instead.

        The constant itself is deliberately unchanged: it also normalises the
        score, so moving the value would shift the 0.550/0.426/0.068 arm ordering
        that G2 measured. Moving the boundary does not.
        """
        result = rule.evaluate(
            _input(preemptions=14, tpot_p50=10.0, tpot_p99=50.0,
                   running=CONCURRENCY_MIN)
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_boundary_one_above_the_negative_arms_max_fires(
        self, rule: ColocationContentionRule
    ) -> None:
        """The first concurrency no long-context tick reached must still fire.

        Guards the other direction: an exclusive gate must not cost the contended
        arm anything. Its field mean is 32.6, far above this.
        """
        result = rule.evaluate(
            _input(preemptions=14, tpot_p50=10.0, tpot_p99=50.0,
                   running=CONCURRENCY_MIN + 1)
        )
        assert isinstance(result, Diagnosis)

    def test_boundary_one_below_floor_abstains(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(
            _input(preemptions=14, tpot_p50=10.0, tpot_p99=50.0,
                   running=CONCURRENCY_MIN - 1)
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_concurrency_gate_precedes_the_score(self, rule: ColocationContentionRule) -> None:
        """A maximal fingerprint at zero concurrency is still silent."""
        result = rule.evaluate(
            _input(preemptions=200, successes=200, prompt_tokens=900,
                   generation_tokens=100, tpot_p50=10.0, tpot_p99=200.0,
                   model_params_b=70.0, num_gpus=16, running=0)
        )
        assert result is Abstention.BELOW_THRESHOLD


# --------------------------------------------------------------------------- #
# Gate G5 — the TPOT tail band, made explicit
# --------------------------------------------------------------------------- #

class TestTailGateG5:
    """`tpot_tail_ratio < TPOT_TAIL_MILD` abstains — a DELIBERATE narrowing.

    Before this change G5 was implicit: the tail reached the verdict only through
    `score_contention` (0.40 weight) and a ±0.15 confidence term, so a flat-tailed
    server could be carried over the line by preemptions and concurrency alone.
    It is now an explicit gate, which is what lets the module docstring state the
    predicate as the code actually runs it.

    **What it costs, stated plainly.** Preemption carries the largest weight
    (0.45), so a server preempting hard used to fire on preemptions alone. Under
    G5 it cannot fire at *any* preemption rate unless the decode tail is also
    above band. Measured:

        preempt 0.05, tail 1.50, 32 running -> score 0.667, now silent
        preempt 0.10, tail 1.90, 32 running -> score 0.720, now silent

    Both clear CONTENTION_MIN and both cleared the confidence floor; G5 alone is
    what stops them. This is a new false-NEGATIVE surface opened by a fix whose
    stated job was closing a false positive, so it is pinned here in both
    directions rather than left to the score.

    **Why the narrowing is believed correct.** The rule's claim is that prefill is
    stalling *decode*. A preemption storm that never shows up in the decode tail
    is not evidence for that claim — the scheduler is thrashing, but decode is not
    demonstrably the victim, and r12 (queue growth) is the rule that owns pure
    scheduler pressure. TPOT_TAIL_MILD is also the one ANCHORED threshold in this
    rule: a field capture read 6.04 on the contended arm against 1.42 on the healthy
    one, so 2.0 sits inside a measured gap.

    **What is NOT verified, and cannot be from the existing captures.** All three
    field arms measured preemption rate 0.0, so
    no capture lands anywhere near the region G5 silences. The band's *placement*
    is field-anchored; the *cost* of the narrowing is reasoned, not measured. A
    calibration capture with high preemptions and a flat tail is the outstanding
    work.
    """

    def _flat_tail(self, rule: ColocationContentionRule, preempt: int, tail: float):
        return rule.evaluate(
            _input(preemptions=preempt, successes=200,
                   tpot_p50=10.0, tpot_p99=10.0 * tail)
        )

    @pytest.mark.parametrize(
        "preempt_rate, tail, score",
        [
            (0.05, 1.50, 0.667),   # review case: fired at 0.65 before G5
            (0.10, 1.90, 0.720),   # further in, and still silent
        ],
    )
    def test_flat_tail_abstains_although_the_score_clears_the_floor(
        self, rule: ColocationContentionRule, preempt_rate: float, tail: float,
        score: float,
    ) -> None:
        """The silenced region, with the score that proves G5 is what silences it.

        Asserting only "abstains" would not distinguish G5 from the score floor or
        the confidence floor, so the score is pinned too: it is comfortably ABOVE
        CONTENTION_MIN, which leaves G5 as the only gate that can be returning.
        """
        assert score_contention(preempt_rate, tail, 32) == pytest.approx(score, abs=0.005)
        assert score_contention(preempt_rate, tail, 32) > CONTENTION_MIN

        result = self._flat_tail(rule, int(preempt_rate * 200), tail)
        assert result is Abstention.BELOW_THRESHOLD

    def test_preemptions_alone_cannot_fire_the_rule_at_any_rate(
        self, rule: ColocationContentionRule
    ) -> None:
        """The narrowing's full extent: the whole preemption axis, flat tail.

        Sweeps preemption from just-above-PREEMPT_MILD to every request preempted,
        at a tail of 1.5x and the contended arm's field concurrency. Not one of
        them fires. Before G5 the first entry already did, at capped confidence,
        because 0.45·norm_preempt saturates at PREEMPT_SEVERE and carries the
        score on its own.
        """
        for preempt in (3, 10, 20, 60, 120, 200):   # 1.5% .. 100% of 200 requests
            result = self._flat_tail(rule, preempt, tail=1.5)
            assert result is Abstention.BELOW_THRESHOLD, f"fired at {preempt} preemptions"

    def test_tail_gate_precedes_the_score(self, rule: ColocationContentionRule) -> None:
        """Every request preempted, 64 resident, flat tail — still silent.

        The mirror of `test_concurrency_gate_precedes_the_score`. Score is 0.613,
        twice CONTENTION_MIN, so this fixture cannot be abstaining on magnitude.
        """
        assert score_contention(1.0, 1.1, 64) > 2 * CONTENTION_MIN
        result = rule.evaluate(
            _input(preemptions=200, successes=200, tpot_p50=10.0, tpot_p99=11.0,
                   model_params_b=70.0, num_gpus=16, running=64)
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_tail_exactly_at_the_band_fires(self, rule: ColocationContentionRule) -> None:
        """The gate is `<`, so `== TPOT_TAIL_MILD` is admitted.

        Deliberately inclusive, unlike G4's exclusive concurrency floor: 16 was a
        value the *negative* arm actually produced, whereas 2.0 sits in a gap no
        arm occupies (healthy 1.42, contended 6.04). Nothing measured lands on this
        boundary, so there is no zero-margin false positive to close here.
        """
        result = rule.evaluate(
            _input(preemptions=10, successes=200,
                   tpot_p50=10.0, tpot_p99=10.0 * TPOT_TAIL_MILD)
        )
        assert isinstance(result, Diagnosis)
        assert result.evidence["tpot_tail_ratio"] == pytest.approx(TPOT_TAIL_MILD)

    def test_tail_just_below_the_band_abstains(self, rule: ColocationContentionRule) -> None:
        result = self._flat_tail(rule, preempt=10, tail=1.99)
        assert result is Abstention.BELOW_THRESHOLD

    def test_field_arms_all_sit_off_this_boundary(self) -> None:
        """What the captures can and cannot say about G5.

        They anchor the band's *placement* — the healthy arm is below it and both
        fat-tailed arms are well above — and they say nothing at all about the
        narrowing, because every arm measured preemption rate 0.0 and so none of
        them lives in the region G5 removed. Recorded as a test so a later
        recalibration cannot quietly claim field support it does not have.
        """
        contend_tail, longctx_tail, healthy_tail = 6.04, 10.54, 1.42
        assert healthy_tail < TPOT_TAIL_MILD < contend_tail < longctx_tail
        # Margin below the band is 0.58x on a single-session single value; the
        # gap above it is 4.04x. The band is not centred, and the healthy side is
        # one measurement, not a distribution.
        assert (TPOT_TAIL_MILD - healthy_tail) < (contend_tail - TPOT_TAIL_MILD)

        # Every field arm had preempt == 0.0, so all three land in the corner of
        # the space where G5 changes nothing: the two abstentions are already
        # taken by G4/G5 agreement, and none tests preemption-without-tail.
        field_preempt_rates = (0.0, 0.0, 0.0)
        assert all(p < PREEMPT_MILD for p in field_preempt_rates)


# --------------------------------------------------------------------------- #
# Additional gate / abstention coverage
# --------------------------------------------------------------------------- #

class TestGates:
    def test_unknown_topology_is_insufficient(self, rule: ColocationContentionRule) -> None:
        # neither phase present -> topology unknown
        result = rule.evaluate(_input(prefill_present=False, decode_present=False))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("prefill", "decode")

    def test_single_phase_dump_is_disaggregated(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(decode_present=False))
        assert result is Abstention.BELOW_THRESHOLD

    def test_missing_serving_block_is_insufficient(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(serving_present=False))
        assert isinstance(result, InsufficientData)
        assert "vllm_serving.num_preemptions_total" in result.missing

    def test_missing_counter_is_insufficient(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(preemptions=None))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("vllm_serving.num_preemptions_total",)

    def test_missing_tpot_percentiles_is_insufficient(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(tpot_present=False))
        assert isinstance(result, InsufficientData)
        assert result.missing == ("tpot_ms.p50", "tpot_ms.p99")

    def test_below_contention_min_abstains(self, rule: ColocationContentionRule) -> None:
        """The score floor itself decides — every earlier gate is cleared first.

        The previous fixture (share 0.16, preempt 0, tail **1.05**) no longer
        reached this branch. G5 returns on `tpot_tail_ratio < 2.0` before
        `contention_score` is computed at all, so from the moment G5 landed the
        test passed through the new gate and its name was a claim about coverage
        that was no longer true. Inputs moved into the band where CONTENTION_MIN
        is genuinely the deciding gate.

        That band is narrow after G2/G4/G5, so the preconditions are asserted
        rather than assumed — otherwise this silently rots the same way again.
        share 0.16 (clears G2), 17 running (clears G4 by one), tail 2.4 (clears
        G5), preempt 0 -> score 0.266 against the 0.30 floor.
        """
        share, running, tail = 160 / 1000, 17, 2.4
        assert share >= PD_WORK_MIN              # G2 admits
        assert running > CONCURRENCY_MIN         # G4 admits
        assert tail >= TPOT_TAIL_MILD            # G5 admits
        score = score_contention(0.0, tail, running)
        assert score == pytest.approx(0.266, abs=0.005)
        assert score < CONTENTION_MIN            # ...so this is what returns

        result = rule.evaluate(
            _input(preemptions=0, prompt_tokens=160, generation_tokens=840,
                   tpot_p50=10.0, tpot_p99=24.0, running=17)
        )
        assert result is Abstention.BELOW_THRESHOLD

    def test_contention_min_only_decides_below_the_mild_preemption_rate(self) -> None:
        """How little room the score floor has left, pinned as an exact statement.

        Raised by review alongside the stale test above. G4 and G5 now put hard
        floors under two of the three score terms, so the lowest score any
        admitted input can carry is

            0.45·0 + 0.40·(2.0−1)/3 + 0.15·(17/32)  =  0.213

        leaving CONTENTION_MIN a live window of only [0.213, 0.30). One
        consequence is sharp enough to assert: at `preemption_rate >=
        PREEMPT_MILD` the preemption term alone contributes 0.09, which lifts even
        that minimum to 0.303 — so **the score floor can never be the deciding
        gate once preemptions are above PREEMPT_MILD.** It still decides below
        that rate (confidence there is 0.65, well clear of its own floor), so this
        is a narrowed gate, not a dead one — but any recalibration that raises
        TPOT_TAIL_MILD or CONCURRENCY_MIN kills it outright, and this test is what
        will say so.
        """
        floor_case = score_contention(0.0, TPOT_TAIL_MILD, CONCURRENCY_MIN + 1)
        assert floor_case == pytest.approx(0.213, abs=0.005)
        assert floor_case < CONTENTION_MIN, "the score floor is unreachable — G4/G5 subsume it"

        at_mild_preempt = score_contention(PREEMPT_MILD, TPOT_TAIL_MILD, CONCURRENCY_MIN + 1)
        assert at_mild_preempt == pytest.approx(0.303, abs=0.005)
        assert at_mild_preempt > CONTENTION_MIN


# --------------------------------------------------------------------------- #
# Evidence & fix-tier exclusivity
# --------------------------------------------------------------------------- #

class TestEvidence:
    def test_evidence_populated(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(preemptions=4, successes=200))
        assert isinstance(result, Diagnosis)
        ev = result.evidence
        assert ev["preemption_rate"] == pytest.approx(0.02, abs=1e-3)
        assert ev["tpot_tail_ratio"] == pytest.approx(2.4, abs=1e-2)
        assert ev["prefill_share"] == pytest.approx(0.30, abs=1e-2)
        # The discriminator is reported, and prefill_share stays alongside it as
        # workload context (the inverted-arms finding: informative, but not a discriminator).
        assert ev["num_requests_running"] == 32
        assert ev["slo_profile"] == "unknown"
        assert ev["fix_tier"] == "chunked_prefill"
        assert ev["scale"] == {"model_params_b": 13.0, "gpu_count": 1}

    def test_exactly_one_fix_tier_in_fix_text(self) -> None:
        # The disaggregate fix must not also pitch the chunked-prefill config flag.
        slo = SloProfile(kind="latency_bound", tpot_slo_ms=20.0)
        rule = ColocationContentionRule(slo=slo)
        result = rule.evaluate(
            _input(preemptions=14, tpot_p50=10.0, tpot_p99=50.0,
                   model_params_b=70.0, num_gpus=16)
        )
        assert isinstance(result, Diagnosis)
        assert "--enable-chunked-prefill" not in result.fix
        assert "PD disaggregation" in result.fix

    def test_confidence_breakdown_signal_strength_tracks_score(self, rule: ColocationContentionRule) -> None:
        result = rule.evaluate(_input(preemptions=14, tpot_p50=10.0, tpot_p99=50.0,
                                       model_params_b=70.0, num_gpus=16))
        assert isinstance(result, Diagnosis)
        bd = result.confidence_breakdown
        assert bd.signal_strength == pytest.approx(result.evidence["contention_score"], abs=0.02)
        assert 0.0 <= bd.data_completeness <= 1.0


# --------------------------------------------------------------------------- #
# Helper-level unit tests
# --------------------------------------------------------------------------- #

class TestHelpers:
    def test_infer_topology_colocated(self) -> None:
        assert infer_topology(_input()) == "colocated"

    def test_infer_topology_connector(self) -> None:
        assert infer_topology(_input(connector="LMCacheConnectorV1")) == "disaggregated"

    def test_infer_topology_single_phase(self) -> None:
        assert infer_topology(_input(prefill_present=False)) == "disaggregated"

    def test_infer_topology_unknown(self) -> None:
        assert infer_topology(_input(prefill_present=False, decode_present=False)) == "unknown"

    def test_roofline_passes_when_absent(self) -> None:
        assert roofline_corroborates(None, None) is True
        assert roofline_corroborates(PhaseMetrics(), PhaseMetrics()) is True

    def test_roofline_rejects_inverted(self) -> None:
        prefill = PhaseMetrics(roofline_position="memory_bound")
        decode = PhaseMetrics(roofline_position="compute_bound")
        assert roofline_corroborates(prefill, decode) is False

    def test_roofline_accepts_healthy(self) -> None:
        prefill = PhaseMetrics(roofline_position="compute_bound")
        decode = PhaseMetrics(roofline_position="memory_bound")
        assert roofline_corroborates(prefill, decode) is True

    def test_score_monotonic_in_signals(self) -> None:
        # Third argument is now concurrency, not prefill share (the inverted-arms finding), so the
        # operands are request counts: an idle server against a loaded one.
        low = score_contention(0.0, 1.0, 2.0)
        high = score_contention(0.05, 4.0, 32.0)
        assert high > low
        assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0

    def test_score_monotonic_in_concurrency_alone(self) -> None:
        """Holding the tail fixed, more resident requests must never score lower."""
        fixed_tail = 6.04
        scores = [score_contention(0.0, fixed_tail, n) for n in (2, 5, 16, 33, 64)]
        assert scores == sorted(scores)
        assert scores[-1] > scores[0]
