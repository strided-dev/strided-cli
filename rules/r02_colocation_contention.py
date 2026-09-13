"""
r02 — Colocation contention (prefill ↔ decode), tiered fix.

Supersedes the two never-shipped alternatives "PD disaggregation" and "chunked
prefill": they were mutually-exclusive fixes for the *same* pathology, so they
are one rule with two fix tiers.

What this rule does NOT do: it does not fire on "prefill is compute-bound and
decode is memory-bound." That divergence is universal transformer physics, true
of every healthy colocated dump, and worthless as a signal. The roofline is used
only to *reject malformed dumps* (inverted labels), never to diagnose.

What it DOES do: on a confirmed-colocated deployment carrying enough concurrent
requests for contention to be possible at all, it detects prefill↔decode
contention from a serving-metrics fingerprint (preemption rate + TPOT tail ratio
+ concurrent residency), quantifies the interference tax, and routes to a graded
fix — tune chunked prefill (mild) or consider PD disaggregation (severe at scale).

The default predicate is:

    TPOT p99/p50 above band (G5, anchored)  AND  mean concurrency >= 16 (G4,
    provisional)

The signal is correlational, not causal: a static /metrics scrape gives
distributions and cumulative counters, not a paired "TPOT with vs without
concurrent prefill" counterfactual. The dominant false positive is a long-context
workload whose TPOT tail comes from attention cost, not contention; it is
separated by gate G4, a floor on the number of concurrently-running requests.

Why G4 and not the prefill share (the inverted-arms field finding). The
rule used to suppress long context by testing `prefill_share < 0.20`, encoding
"long-context workloads have a low prefill share." That is backwards: a
long-context request is one whose *prompt* dwarfs what it generates, so its
prefill share is HIGH — measured 0.941 on a 4096-token prompt generating 256
tokens (A100-PCIE-40GB, vLLM 0.10.2, Qwen2.5-7B). The
guard's band was exactly where long-context work never lands, so it never fired,
and the long-context control scored 0.541 against the genuinely contended
server's 0.526 — the fingerprint ranked its own false positive ABOVE its true
positive, on every axis it read. No reweighting of preemption rate, TPOT tail and
prefill share can separate those cases; the arms are inverted, which makes this a
design defect rather than a calibration target.

The missing signal is concurrency, and it was already in the same captures:
contention is *competition*, so it needs multiple requests resident at once,
while a long-context tail comes from attention cost inside one sequence.
`num_requests_running` measured 32.6 mean on the contended arm against 5.5 on the
long-context arm and 2.6 on the healthy one — a ~6x separation in the causally
correct direction. It is used twice, for two different jobs: as gate G4 (does
contention exist at all) and as the third term of `score_contention`, replacing
the prefill share there so the blend stops ranking the lookalike above the real
case (0.550 vs 0.426 now, against the shipped 0.526 vs 0.541).

`prefill_share` survives as the admission gate G2 ("prefill is a non-trivial
share of the work") and as reported evidence; it is simply not a discriminator,
and one variable cannot be both the admission gate (high => fire) and the
suppressor (low => silence). It no longer appears in the score or the confidence
model — confidence is part of the firing decision, so a share-derived term there
would be a share-derived term in the predicate.

Confidence is capped while THRESHOLDS_UNCALIBRATED is True: the severity bands
below are calibration seeds, not literature values — there is no published
"preemption rate that means contention." Flip the flag only after validating the
fingerprint against real colocated dumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput, PhaseMetrics

# --------------------------------------------------------------------------- #
# Threshold constants — CALIBRATION SEEDS, NOT LITERATURE VALUES.
# There is no published threshold for "preemption rate that means contention."
# Calibrate on real colocated dumps before trusting the severity bands; until
# then THRESHOLDS_UNCALIBRATED caps confidence (see _confidence).
# --------------------------------------------------------------------------- #
MIN_REQUESTS = 200              # below this, histograms are noise
CONTENTION_MIN = 0.30           # below this, don't diagnose
PREEMPT_MILD = 0.01             # >=1% of requests preempted = contention present
PREEMPT_SEVERE = 0.05           # >=5% = severe KV/scheduler pressure

# TPOT tail bands. Derivation: ANCHORED. Chosen as seeds, but since retro-anchored
# by measurement: a field capture read p99/p50 = 6.04 on the contended arm
# against 1.42 on the healthy one, so TPOT_TAIL_MILD lands inside the measured gap
# and TPOT_TAIL_SEVERE below the contended reading — the bands separate stalled
# decode from healthy decode on real telemetry, in the right direction, without
# being moved. What the tail is NOT is a discriminator of
# contention-vs-long-context: the long-context arm measured 10.54, *larger* than
# the contended one. That job belongs to CONCURRENCY_MIN below.
TPOT_TAIL_MILD = 2.0            # p99/p50 >= 2x = noticeable tail
TPOT_TAIL_SEVERE = 4.0          # p99/p50 >= 4x = decode badly stalled

PD_WORK_MIN = 0.15              # prefill must be >=15% of token work (gate G2)
SCALE_LARGE_PARAMS_B = 30       # disaggregation pays off at scale
SCALE_LARGE_GPUS = 8

# Concurrency floor (gate G4) — the discriminator that separates real contention
# from a long-context TPOT tail. Derivation: PROVISIONAL. It is a single-site,
# single-model, single-session measurement (one A100, one
# 7B model, three arms), placed between the two field means rather than derived
# from a queueing model or a published result:
#
#     arm       mean num_requests_running    max
#     contend   32.6                         67
#     longctx    5.5                         16
#     healthy    2.6                          9
#
# Contention is competition: prefill can only stall decode when several requests
# are resident at once. A long-context tail needs no such company — it is
# attention cost inside one sequence — which is why this signal separates the two
# arms (~6x) when nothing in the old fingerprint did. Recalibrate with the rest
# of the severity bands; a long-context workload at genuinely high concurrency
# will pass this floor, and firing there is CORRECT (that server is contended).
#
# Caveat carried by the schema, not fixable here: `num_requests_running` is a
# point-in-time gauge and the schema carries no series of it, so the rule applies
# the floor to a single scrape while the numbers above are means over ~100
# scrapes. 16 is also exactly the long-context arm's observed *maximum*, which is
# why the gate is EXCLUSIVE (`<= CONCURRENCY_MIN` abstains): at `<` the single
# worst long-context tick sat on the firing side and produced a full-confidence
# false positive on a measured field value — a threshold with no margin, which is
# one of this project's six recurring defect shapes. The constant itself is
# unchanged on purpose: it also normalises the score (`norm_conc`, below), so
# moving the VALUE would shift every score and invalidate the 0.550/0.426/0.068
# arm ordering that G2 measured. Moving the BOUNDARY does not. A windowed mean
# would still need a schema addition.
CONCURRENCY_MIN = 16

# RETIRED AS A GUARD — the inverted-arms finding. Kept, not deleted,
# because the reasoning trail is the point (same convention as PERSISTENCE_TICKS
# in rules/_stats.py): this constant is the artefact of a discriminator that could
# never fire. It encoded "long-context workloads have a LOW prefill share"; the
# field measured 0.941 on the actual long-context arm, so the band [0, 0.20) is
# precisely where such workloads never land. No threshold on `prefill_share`
# separates the arms — they are inverted — so this is not re-tuned, it is removed
# from the firing predicate and replaced by CONCURRENCY_MIN above. `prefill_share`
# itself remains: as admission gate G2 (PD_WORK_MIN) and as reported evidence.
LONG_CONTEXT_SHARE = 0.20       # unused in the predicate; retained as history

# Derivation labels above ("ANCHORED", "PROVISIONAL") are written in comments
# for now. They do not change behaviour; THRESHOLDS_UNCALIBRATED stays True and keeps
# capping confidence, since the concurrency floor rests on one field session and the
# severity bands are still seeds.
THRESHOLDS_UNCALIBRATED = True  # flip to False only after calibration
_UNCALIBRATED_CEILING = 0.65    # confidence cap while uncalibrated

# Confidence contract (see rules/base.py): a firing Diagnosis must report
# confidence in [0.5, 0.9]; anything weaker abstains. The spec's additive model
# (base 0.45 + signal terms − guards) is reshaped to that contract — sub-0.50
# outcomes abstain via BELOW_THRESHOLD rather than being emitted.
_CONFIDENCE_FLOOR = 0.50

# KV-transfer connector signatures that mark a *disaggregated* stack. Presence of
# any of these in the launch config means prefill and decode are already split.
_KV_CONNECTOR_SIGNATURES = (
    "NixlConnector",
    "LMCacheConnectorV1",
    "PyNcclConnector",
    "kv_producer",
    "kv_consumer",
)

Topology = Literal["colocated", "disaggregated", "unknown"]


# --------------------------------------------------------------------------- #
# SLO sidecar — customer policy, NOT parsed telemetry.
#
# Latency SLOs live in no dump; they are customer configuration. They are a
# refinement, never a gate: with them the recommendation sharpens, without them
# the rule still reports a measured tax and a conditional fix. The sidecar is
# passed to the rule at construction; the engine constructs rules with no args,
# so in a normal engine run `slo` is None and the rule emits the conditional
# version. This is the documented seam for a future config sidecar.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SloProfile:
    kind: Literal["latency_bound", "throughput_bound", "unknown"] = "unknown"
    ttft_slo_ms: Optional[float] = None
    tpot_slo_ms: Optional[float] = None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #

def infer_topology(dx: DiagnosisInput) -> Topology:
    """Infer deployment topology from dump shape + connector signature (§3).

    - disaggregated: a KV-transfer connector is configured, OR the dump carries
      only one phase (prefill XOR decode).
    - colocated: a single dump carries BOTH phases and no connector signature.
    - unknown: neither phase is present — we cannot tell.

    The topology label is computed here, never stored on the schema.
    """
    serving = dx.vllm_serving
    if serving is not None and serving.kv_transfer_connector:
        connector = serving.kv_transfer_connector
        if any(sig in connector for sig in _KV_CONNECTOR_SIGNATURES):
            return "disaggregated"
        # An unrecognised connector string still indicates a transfer path.
        return "disaggregated"

    prefill_present = dx.prefill is not None
    decode_present = dx.decode is not None
    if prefill_present and decode_present:
        return "colocated"
    if prefill_present or decode_present:  # single-phase dump
        return "disaggregated"
    return "unknown"


def roofline_corroborates(
    prefill: Optional[PhaseMetrics], decode: Optional[PhaseMetrics]
) -> bool:
    """Reject malformed dumps only. Corroboration, never diagnosis.

    A healthy colocated dump has compute-bound prefill and memory-bound decode.
    If the labels are inverted (prefill reads memory_bound or decode reads
    compute_bound) the dump is mislabeled and untrustworthy. When the roofline
    data is absent (e.g. a vLLM-only scrape with no Nsight) we pass: the rule
    must still fire on the serving fingerprint alone.
    """
    if prefill is None or decode is None:
        return True
    p, d = prefill.roofline_position, decode.roofline_position
    if p is None or d is None:
        return True
    if p == "memory_bound" or d == "compute_bound":
        return False
    return True


def score_contention(
    preemption_rate: float, tpot_tail_ratio: float, concurrency: float
) -> float:
    """Weighted blend of the fingerprint signals, normalised to [0, 1].

    Preemptions and the TPOT tail are the primary fingerprint; concurrency is the
    third term, carrying the same 0.15 the prefill share used to.

    Why the third term changed (the inverted-arms finding). This blend read
    `prefill_share` and so ranked the long-context arm at 0.541 ABOVE the
    genuinely contended arm at 0.526 — the false positive outscored the true
    positive. That was not a weighting accident. Both arms saturate the tail term
    (6.04 and 10.54 both clamp to 1.0 against TPOT_TAIL_SEVERE), preemptions were
    0.0 on both, so the ONLY term left to order them was the prefill share, which
    is *higher* for long context (0.941 vs 0.842) — i.e. the one live term ran
    backwards. Re-weighting could not fix that; the term had to be replaced by one
    that runs the right way.

    Concurrency does run the right way, because contention IS competition: at the
    same saturated tail, the arm with more requests resident is the more contended
    one. Substituting it restores the ordering the finding demands:

        arm       preempt   tail   running   score
        contend   0.0       6.04     32.6   0.550   <- true positive, now highest
        longctx   0.0      10.54      5.5   0.426
        healthy   0.0       1.42      2.6   0.068

    Note this is a magnitude, not the admission decision: `longctx` still scores
    above CONTENTION_MIN, and is rejected by gate G4's hard floor rather than by
    this number. The score says "how bad", G4 says "whether" — but the score is no
    longer allowed to rank a lookalike above the real thing.
    """
    norm_preempt = _clamp01(preemption_rate / PREEMPT_SEVERE)
    norm_tail = _clamp01((tpot_tail_ratio - 1.0) / (TPOT_TAIL_SEVERE - 1.0))
    # Normalised against the gate itself: at the floor the term is half-weight,
    # at twice the floor it saturates. No new magic number is introduced.
    norm_conc = _clamp01(concurrency / (2.0 * CONCURRENCY_MIN))
    return 0.45 * norm_preempt + 0.40 * norm_tail + 0.15 * norm_conc


def _confidence(
    *,
    preemption_rate: float,
    tpot_tail_ratio: float,
    concurrency: float,
    topology_colocated: bool,
    samples_ok: bool,
    long_context_suspected: bool,
    slo: Optional[SloProfile],
    tpot_slo_violated: bool,
) -> float:
    """Additive confidence model (§5), clamped to the codebase contract.

    The gates in `evaluate` already guarantee the firing path is colocated, has
    enough samples, and clears G2, G4 and G5; the guard terms below are kept for
    parity and defence-in-depth. The result is clamped to [0, 0.9], then capped
    at 0.65 while the thresholds are uncalibrated. Callers abstain when it lands
    below the 0.50 contract floor.

    No term reads `prefill_share` any more. Confidence is part of the
    firing decision — sub-0.50 abstains — so a share-derived term here would be
    a share-derived term in the predicate, which is exactly what the finding
    rules out.

    `long_context_suspected` is now derived from *concurrency*, never from the
    prefill share. Read from the share it was unreachable — long-context
    work has a high share, not a low one — so this −0.40 penalty had never once
    applied on real telemetry. Gate G4 makes it defence-in-depth rather than the
    load-bearing suppressor it was documented to be.
    """
    c = 0.45
    c += 0.20 if preemption_rate >= PREEMPT_MILD else -0.10
    c += 0.15 if tpot_tail_ratio >= TPOT_TAIL_MILD else -0.05
    # Parity term for the concurrency gate, in the slot the prefill-share term
    # used to occupy. Same shape (G4 guarantees it true on the firing path, as G2
    # guaranteed the old one), same 0.10, so firing confidences are unchanged —
    # but the model no longer takes credit for a signal that field testing showed
    # points the wrong way. `prefill_share` is now evidence only.
    c += 0.10 if concurrency >= CONCURRENCY_MIN else 0.0
    c += 0.10 if topology_colocated else 0.0
    if slo is not None and slo.kind == "latency_bound" and tpot_slo_violated:
        c += 0.10
    # False-positive guards (the important negatives).
    if long_context_suspected and preemption_rate < PREEMPT_MILD:
        c -= 0.40
    if not topology_colocated:
        c -= 0.50
    if not samples_ok:
        c -= 0.30
    c = max(0.0, min(0.90, c))
    if THRESHOLDS_UNCALIBRATED:
        c = min(c, _UNCALIBRATED_CEILING)
    return c


# --------------------------------------------------------------------------- #
# Fix strings (one per tier — never both)
# --------------------------------------------------------------------------- #

_FIX_CHUNKED = (
    "Enable/tune chunked prefill to smooth interference within your current "
    "pool. In vLLM, set `--enable-chunked-prefill` and tune "
    "`--max-num-batched-tokens` (start near 512–2048 and tune to your TPOT "
    "target). This is a config change, not a re-architecture, and is the right "
    "tier given your scale/throughput profile."
)

_FIX_DISAGGREGATE = (
    "Interference is severe enough that chunked prefill will not fully resolve "
    "it at your scale. Consider PD disaggregation — separate prefill and decode "
    "pools (vLLM disaggregated prefilling, NVIDIA Dynamo, SGLang PD, or llm-d) — "
    "*if you are bound by both a TTFT and a TPOT SLO*. Size the prefill:decode "
    "pool ratio from your prefill-vs-decode time share and keep KV transfer on a "
    "high-bandwidth intra-node link (NVLink/NIXL)."
)


def _fix_for_tier(fix_tier: str) -> str:
    return _FIX_DISAGGREGATE if fix_tier == "disaggregate" else _FIX_CHUNKED


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

class ColocationContentionRule(Rule):
    """Colocated prefill and decode contend on one GPU pool; route to a fix tier."""

    rule_id = "r02"
    title = "Colocation contention (prefill ↔ decode)"
    references = (
        "Zhong et al., 'DistServe: Disaggregating Prefill and Decoding for "
        "Goodput-optimized LLM Serving,' OSDI 2024. arXiv:2401.09670.",
        "Patel et al., 'Splitwise: Efficient Generative LLM Inference Using "
        "Phase Splitting,' ISCA 2024. arXiv:2311.18677.",
        "Agrawal et al., 'Taming Throughput-Latency Tradeoff in LLM Inference "
        "with Sarathi-Serve,' OSDI 2024. arXiv:2403.02310.",
        "Hao AI Lab, 'Disaggregated Inference: 18 Months Later,' Nov 2025. "
        "https://haoailab.com/blogs/distserve-retro",
        "vLLM Metrics design. https://docs.vllm.ai/en/latest/design/metrics/",
    )

    def __init__(self, slo: Optional[SloProfile] = None) -> None:
        # SLO is a customer-policy sidecar, never parsed telemetry, never a gate.
        # The engine constructs rules with no args, so a normal run has slo=None
        # and emits the conditional fix. Tests inject an SloProfile to exercise
        # the refinement path.
        self.slo = slo

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- GATE G1: topology must be colocated --------------------------
        topology = infer_topology(dx)
        if topology == "disaggregated":
            # The stack already separates the phases — the rule does not apply.
            return Abstention.BELOW_THRESHOLD
        if topology == "unknown":
            # No phase data to confirm colocation; surface the gap to the user.
            return InsufficientData(
                missing=("prefill", "decode"),
                reason="no phase data present to confirm colocated topology",
            )

        # ---- GATE G3: serving metrics present and trustworthy -------------
        s = dx.vllm_serving
        # `num_requests_running` was promoted from optional corroborator to
        # REQUIRED input by the inverted-arms finding: it is the only field that separates
        # contention from a long-context tail, so without it the rule cannot tell
        # the two apart and must say so rather than guess. Abstaining loudly here
        # is the whole point — the failure this closes was a silent full-strength
        # fire on the workload the rule documents as its dominant false positive.
        _serving_fields = (
            "num_preemptions_total",
            "request_success_total",
            "prompt_tokens_total",
            "generation_tokens_total",
            "num_requests_running",
        )
        if s is None:
            return InsufficientData(
                missing=tuple(f"vllm_serving.{f}" for f in _serving_fields)
            )
        missing_serving = [
            f"vllm_serving.{f}" for f in _serving_fields if getattr(s, f) is None
        ]
        if missing_serving:
            return InsufficientData(missing=tuple(missing_serving))
        if s.request_success_total < MIN_REQUESTS:
            # Below this, the histograms are noise — we cannot trust the fingerprint.
            return InsufficientData(
                reason=f"only {s.request_success_total} successful requests; "
                f"need >={MIN_REQUESTS} for a trustworthy fingerprint"
            )

        # TPOT percentiles come from the existing source of truth, not a dup field.
        if dx.tpot_ms is None:
            return InsufficientData(missing=("tpot_ms.p50", "tpot_ms.p99"))
        missing_tpot = [
            f"tpot_ms.{p}" for p in ("p50", "p99") if getattr(dx.tpot_ms, p) is None
        ]
        if missing_tpot:
            return InsufficientData(missing=tuple(missing_tpot))
        tpot_p50 = dx.tpot_ms.p50
        tpot_p99 = dx.tpot_ms.p99
        if tpot_p50 <= 0:
            return InsufficientData(
                reason=f"tpot p50 is {tpot_p50} (non-positive); cannot form a tail ratio"
            )

        # ---- GATE G2: prefill must be a non-trivial share of token work ----
        # ADMISSION gate only: there must be enough prefill work for prefill to
        # plausibly be stalling decode. It is NOT the long-context guard — that
        # reading is what field testing disproved, and it now lives in G4 below. One
        # variable cannot be both "high => fire" and "low => silence".
        total_tokens = s.prompt_tokens_total + s.generation_tokens_total
        if total_tokens <= 0:
            return InsufficientData(
                reason="no prompt/generation tokens recorded; cannot compute prefill share"
            )
        prefill_share = s.prompt_tokens_total / total_tokens
        pd_work_ratio = s.prompt_tokens_total / max(s.generation_tokens_total, 1)
        if prefill_share < PD_WORK_MIN:
            return Abstention.BELOW_THRESHOLD

        # ---- Malformed-dump check (roofline corroboration) ----------------
        if not roofline_corroborates(dx.prefill, dx.decode):
            return InsufficientData(
                reason="roofline labels inverted (prefill memory-bound or decode "
                "compute-bound); dump is malformed and untrustworthy"
            )

        # ---- GATE G4: concurrency floor (the long-context discriminator) ---
        # Contention is competition. Prefill can only stall decode when several
        # requests are resident at once; a long-context TPOT tail needs no such
        # company, because it is attention cost inside one sequence. Field means:
        # contended 32.6, long-context 5.5, healthy 2.6. This is the
        # signal the old fingerprint lacked entirely, and the reason that
        # fingerprint ranked its own false positive above its true positive.
        concurrency = s.num_requests_running
        if concurrency <= CONCURRENCY_MIN:
            # Data present, signal absent — silent, not a data complaint.
            return Abstention.BELOW_THRESHOLD

        # ---- FINGERPRINT --------------------------------------------------
        preemption_rate = s.num_preemptions_total / max(s.request_success_total, 1)
        tpot_tail_ratio = tpot_p99 / tpot_p50

        # ---- GATE G5: TPOT tail must be above band ------------------------
        # The corrected default predicate is
        # "TPOT p99/p50 above band AND concurrency >= CONCURRENCY_MIN". The tail
        # leg used to be implicit — it reached the verdict only through the score
        # and a confidence term, so a flat-tailed server could in principle be
        # carried over the line by the other terms. Making it an explicit gate is
        # what lets the module docstring state the predicate as the code runs it.
        # Derivation: ANCHORED (see TPOT_TAIL_MILD).
        if tpot_tail_ratio < TPOT_TAIL_MILD:
            return Abstention.BELOW_THRESHOLD

        contention_score = score_contention(preemption_rate, tpot_tail_ratio, concurrency)
        if contention_score < CONTENTION_MIN:
            return Abstention.BELOW_THRESHOLD

        # ---- SEVERITY ROUTING (to exactly one tier) -----------------------
        scale_large = (
            (dx.model_params_b is not None and dx.model_params_b >= SCALE_LARGE_PARAMS_B)
            or (dx.num_gpus is not None and dx.num_gpus >= SCALE_LARGE_GPUS)
        )
        severe = (
            preemption_rate >= PREEMPT_SEVERE
            and tpot_tail_ratio >= TPOT_TAIL_SEVERE
            and scale_large
        )

        fix_tier = "chunked_prefill"
        if self.slo is not None and self.slo.kind == "throughput_bound":
            # Disaggregation's KV-transfer overhead can hurt pure throughput.
            fix_tier = "chunked_prefill"
        elif severe:
            fix_tier = "disaggregate"

        # ---- Confidence ---------------------------------------------------
        tpot_slo_violated = bool(
            self.slo is not None
            and self.slo.tpot_slo_ms is not None
            and tpot_p99 > self.slo.tpot_slo_ms
        )
        # The long-context hypothesis, read from the signal that can actually
        # test it: a fat tail with no preemptions and no concurrent company. Gate
        # G4 has already returned in that case, so this is defence-in-depth (see
        # `_confidence`). It used to read `prefill_share < LONG_CONTEXT_SHARE`,
        # which was unreachable by construction.
        long_context_suspected = (
            preemption_rate < PREEMPT_MILD
            and tpot_tail_ratio >= TPOT_TAIL_MILD
            and concurrency <= CONCURRENCY_MIN
        )
        confidence = _confidence(
            preemption_rate=preemption_rate,
            tpot_tail_ratio=tpot_tail_ratio,
            concurrency=concurrency,
            topology_colocated=True,
            samples_ok=True,
            long_context_suspected=long_context_suspected,
            slo=self.slo,
            tpot_slo_violated=tpot_slo_violated,
        )
        if confidence < _CONFIDENCE_FLOOR:
            return Abstention.BELOW_THRESHOLD

        # ---- Build the diagnosis ------------------------------------------
        slo_kind = self.slo.kind if self.slo is not None else "unknown"
        data_completeness = self._data_completeness(dx, s)

        cause = (
            "Colocated prefill and decode are contending on the same GPU pool. "
            f"With {concurrency} requests resident at once, decode latency is "
            "being stalled by prefill: measured TPOT p99/p50 = "
            f"{tpot_tail_ratio:.1f}x with a preemption rate of {preemption_rate:.1%}, "
            "while prefill accounts for a substantial share of token work "
            f"({prefill_share:.0%}). This is the interference cost of running both "
            "phases together."
        )

        notes = (
            "additive fingerprint model (preemption rate + TPOT tail + "
            f"concurrency) gated on concurrency > {CONCURRENCY_MIN}; "
            f"fix_tier={fix_tier}"
        )
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=_fix_for_tier(fix_tier),
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=_clamp01(contention_score),
                data_completeness=data_completeness,
                notes=notes,
            ),
            evidence={
                "deployment_topology": "colocated (single dump, both phases, no kv-transfer connector)",
                "preemption_rate": round(preemption_rate, 4),
                "tpot_tail_ratio": round(tpot_tail_ratio, 2),
                "tpot_p50_ms": round(tpot_p50, 2),
                "tpot_p99_ms": round(tpot_p99, 2),
                "pd_work_ratio": round(pd_work_ratio, 2),
                # The discriminator (gate G4) and the signal that is NOT one.
                # `prefill_share` stays reported because it is still informative
                # about the workload's shape — it just does not separate
                # contention from long context.
                "num_requests_running": concurrency,
                "prefill_share": round(prefill_share, 2),
                "scale": {"model_params_b": dx.model_params_b, "gpu_count": dx.num_gpus},
                "slo_profile": slo_kind,
                "fix_tier": fix_tier,
                "contention_score": round(contention_score, 2),
            },
        )

    @staticmethod
    def _data_completeness(dx: DiagnosisInput, s) -> float:
        """Fraction of optional corroborating fields the rule had available.

        `num_requests_running` used to be counted here. It is no longer optional
        — G4 requires it, so on any path that reaches a Diagnosis it is present
        by construction, and leaving it in would have inflated completeness with
        a term that can never be False.
        """
        optional = [
            dx.ttft_ms is not None,
            s.num_requests_waiting is not None,
            dx.kv_cache_util is not None,
            dx.prefill is not None and dx.prefill.roofline_position is not None,
            dx.decode is not None and dx.decode.roofline_position is not None,
        ]
        return sum(optional) / len(optional)
