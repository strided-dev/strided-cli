"""
r07 — Attention bottleneck (unfused attention path).

Fires when a kernel trace carries the signature of an UNFUSED attention
implementation: standalone softmax kernels holding a material share of GPU time
(leg A), or generically-named attention kernels dominating the step (leg B),
while no fused attention kernel (FlashAttention / SDPA / fMHA / paged
attention) appears. Unfused attention materialises the s×s score matrix to HBM
and re-reads it through a separate memory-bound softmax pass; the fused kernel
eliminates exactly that traffic for 2–4× kernel-level speedups (Dao et al.,
NeurIPS 2022). The fix is a backend flag, not new hardware.

Two structural facts shape the signal (spec §Signal):

1. **Naive attention's GEMMs hide as "mlp" by name.** Q·K^T and P·V run as
   generic GEMM kernels indistinguishable from MLP GEMMs, so the unfused path's
   "attention share" reads near zero — the standalone softmax IS the observable
   part of its cost. Leg A therefore never consults the attention share.
2. **Fused kernels make the standalone softmax vanish** (online softmax keeps
   the chain in SRAM), leaving only the tiny logits/sampling softmax — which is
   why leg A's bar is a share, not presence.

Self-guard: if fused-named kernels hold more than a minor share, the advice is
already taken — abstain. A fused-and-still-dominant workload is legitimate
long-context cost (future r09's scope), and "enable FlashAttention" would be
wrong. Roofline corroboration is a confidence boost, NEVER a gate: healthy
paged-attention decode is inherently memory-bound (Kwon et al., SOSP 2023).

This is a dump-only rule (like r05): it reads the per-kernel ``layers``
breakdown only the Nsight parser populates, and abstains with
INSUFFICIENT_DATA on the live path.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from rules.base import (
    Abstention,
    ConfidenceBreakdown,
    Diagnosis,
    InsufficientData,
    Rule,
    RuleResult,
)
from schema import DiagnosisInput, LayerMetrics

# --------------------------------------------------------------------------- #
# Thresholds — CALIBRATION SEEDS, not literature values.
# Fused traces keep standalone softmax near zero (only the logits/sampling
# pass), while an unfused matmul→softmax→matmul block measured ~14.5%
# standalone-softmax share on the local RTX 3050 preflight — 8% sits between
# the two populations. ATTN_DOMINANT leans on Pope et al. (arXiv:2211.05102):
# attention should hold well under half of step time at moderate context on a
# well-implemented stack. Validate with the naive/fused sweep + real dumps
# before trusting the bands; until then THRESHOLDS_UNCALIBRATED caps confidence.
# --------------------------------------------------------------------------- #
MIN_KERNELS = 5              # below this, time shares are noise
MIN_TOTAL_MS = 1.0           # minimum profiled kernel time to judge shares
SOFTMAX_SHARE_FIRE = 0.08    # standalone softmax ≥ 8% of total → unfused signature
SOFTMAX_SHARE_SEVERE = 0.25  # counted as full-strength
ATTN_DOMINANT = 0.40         # non-fused attention-named kernels dominate (leg B)
ATTN_SEVERE = 0.80           # leg B full-strength point
FUSED_MINOR = 0.05           # fused share below this = absent/minor (guard bar)

CORROBORATION_BUMP = 0.05    # additive boost, never a gate (r05 penalty precedent)
CORROB_MIN_FRAC = 0.5        # ≥ half of attention-path time memory_bound → boost

# Confidence model (mirrors the r01/r05/r06 skeleton).
_CONFIDENCE_FLOOR = 0.50
_CONFIDENCE_SCALE = 0.80
_CONFIDENCE_CEILING = 0.90
THRESHOLDS_UNCALIBRATED = True
_UNCALIBRATED_CEILING = 0.65

# Fused attention kernel families, matched against layer_name. This finer
# sub-classification lives in the rule, not the parser — the parser writes the
# raw layer_type; deriving "fused vs generic" is rule logic (same split as
# slowness_from_sm_clocks). Triton's bare "_fwd_kernel" is deliberately absent
# (generic suffix; would swallow unrelated kernels) — spec §Open questions.
_FUSED_ATTENTION_RE = re.compile(
    r"flash_fwd|flash_attn|flash::|fmha|sdpa|mem_eff|paged_attention", re.I
)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Helpers (pure)
# --------------------------------------------------------------------------- #

def _timed(layers: list[LayerMetrics]) -> list[LayerMetrics]:
    """Layers carrying a positive duration — the share denominator population."""
    return [l for l in layers if l.duration_ms is not None and l.duration_ms > 0]


def is_fused_attention(layer: LayerMetrics) -> bool:
    """True when this attention-classified kernel belongs to a fused family."""
    return (
        layer.layer_type == "attention"
        and _FUSED_ATTENTION_RE.search(layer.layer_name) is not None
    )


def _memory_bound_frac(attention_path: list[LayerMetrics]) -> Optional[float]:
    """Duration-weighted fraction of attention-path time that is memory_bound.

    None when no attention-path kernel carries a known roofline — "could not
    be evaluated" must stay distinct from "evaluated and low" (r05 precedent).
    """
    known = [
        l for l in attention_path
        if l.roofline_position is not None and l.roofline_position != "unknown"
    ]
    if not known:
        return None
    total = sum(l.duration_ms for l in known)
    if total <= 0:
        return None
    bound = sum(l.duration_ms for l in known if l.roofline_position == "memory_bound")
    return bound / total


class AttentionBottleneckRule(Rule):
    """Standalone-softmax / unfused-attention signature; route to a fused backend."""

    rule_id = "r07"
    title = "Attention bottleneck (unfused attention path)"
    references = (
        "Dao et al., 'FlashAttention: Fast and Memory-Efficient Exact Attention "
        "with IO-Awareness,' NeurIPS 2022, arXiv:2205.14135. (Memory-bound "
        "softmax dominates unfused attention; fusion removes the S×S "
        "materialisation for 2–4× kernel speedups.)",
        "Dao, 'FlashAttention-2: Faster Attention with Better Parallelism and "
        "Work Partitioning,' arXiv:2307.08691.",
        "Milakov & Gimelshein, 'Online normalizer calculation for softmax,' "
        "arXiv:1805.02867. (The online-softmax primitive that makes the fusion exact.)",
        "Hong et al., 'FlashDecoding++: Faster Large Language Model Inference "
        "on GPUs,' arXiv:2311.01282. (The same fusion economics in decode.)",
        "Pope et al., 'Efficiently Scaling Transformer Inference,' "
        "arXiv:2211.05102. (Attention:MLP cost analytics behind ATTN_DOMINANT.)",
        "Kwon et al., 'Efficient Memory Management for LLM Serving with "
        "PagedAttention,' SOSP 2023, arXiv:2309.06180. (Healthy paged decode is "
        "memory-bound — roofline must never gate this rule.)",
    )

    def evaluate(self, dx: DiagnosisInput) -> RuleResult:
        # ---- Required field ------------------------------------------------
        if not dx.layers:
            return InsufficientData(missing=("layers",))

        timed = _timed(dx.layers)
        total_ms = sum(l.duration_ms for l in timed)
        if len(timed) < MIN_KERNELS or total_ms < MIN_TOTAL_MS:
            return InsufficientData(
                reason=(
                    f"per-kernel time shares need ≥{MIN_KERNELS} timed kernels and "
                    f"≥{MIN_TOTAL_MS:.0f}ms of profiled time (got {len(timed)} "
                    f"kernels, {total_ms:.2f}ms) — shares over a handful of "
                    f"launches are noise, not evidence"
                )
            )

        # ---- Shares ----------------------------------------------------------
        softmax_layers = [l for l in timed if l.layer_type == "softmax"]
        attn_layers = [l for l in timed if l.layer_type == "attention"]
        fused_layers = [l for l in attn_layers if is_fused_attention(l)]
        unfused_attn_layers = [l for l in attn_layers if not is_fused_attention(l)]

        softmax_share = sum(l.duration_ms for l in softmax_layers) / total_ms
        attention_share = sum(l.duration_ms for l in attn_layers) / total_ms
        fused_share = sum(l.duration_ms for l in fused_layers) / total_ms
        unfused_attn_share = sum(l.duration_ms for l in unfused_attn_layers) / total_ms
        mlp_share = sum(l.duration_ms for l in timed if l.layer_type == "mlp") / total_ms

        # ---- Self-guard: the advice is already taken -------------------------
        # Fused kernels beyond a minor share → this workload already runs a
        # fused path. If it still dominates, that is long-context cost (future
        # r09's scope), not an implementation gap — "enable FlashAttention"
        # would be wrong advice. Includes the partial-migration mixed case,
        # abstained on conservatively (spec §Open questions).
        if fused_share >= FUSED_MINOR:
            return Abstention.BELOW_THRESHOLD

        # ---- Firing legs ------------------------------------------------------
        if softmax_share >= SOFTMAX_SHARE_FIRE:
            firing_leg = "softmax_signature"
            primary = _clamp01(
                (softmax_share - SOFTMAX_SHARE_FIRE)
                / (SOFTMAX_SHARE_SEVERE - SOFTMAX_SHARE_FIRE)
            )
            offender_pool = softmax_layers
        elif unfused_attn_share >= ATTN_DOMINANT:
            firing_leg = "unfused_attention_dominant"
            primary = _clamp01(
                (unfused_attn_share - ATTN_DOMINANT) / (ATTN_SEVERE - ATTN_DOMINANT)
            )
            offender_pool = unfused_attn_layers
        else:
            return Abstention.BELOW_THRESHOLD

        # ---- Signal strength --------------------------------------------------
        # Geometric mean (r01 rationale): the firing share must be clear AND
        # fused kernels genuinely absent — a strong softmax share with fused
        # kernels creeping toward the guard bar is a weaker unfused verdict.
        fused_absence = _clamp01((FUSED_MINOR - fused_share) / FUSED_MINOR)
        signal_strength = math.sqrt(primary * fused_absence)

        confidence = min(
            _CONFIDENCE_CEILING, _CONFIDENCE_FLOOR + _CONFIDENCE_SCALE * signal_strength
        )

        # Corroborator: attention-path kernels measured memory-bound. A boost,
        # never a gate — paged decode is memory-bound on a healthy server too.
        mem_frac = _memory_bound_frac(softmax_layers + attn_layers)
        corroborated = mem_frac is not None and mem_frac >= CORROB_MIN_FRAC
        if corroborated:
            confidence = min(_CONFIDENCE_CEILING, confidence + CORROBORATION_BUMP)
        if THRESHOLDS_UNCALIBRATED:
            confidence = min(confidence, _UNCALIBRATED_CEILING)

        # ---- Build the diagnosis ----------------------------------------------
        top = max(offender_pool, key=lambda l: l.duration_ms)

        if firing_leg == "softmax_signature":
            cause = (
                f"Standalone softmax kernels consumed {softmax_share:.0%} of "
                f"profiled GPU time and no fused attention kernel (FlashAttention "
                f"/ SDPA / fMHA / paged attention) appeared in the trace — the "
                f"attention path ran unfused: Q·K^T and P·V executed as generic "
                f"GEMMs with the S×S score matrix materialised to HBM and re-read "
                f"by a separate, memory-bound softmax pass whose cost grows "
                f"quadratically with context length."
            )
        else:
            cause = (
                f"Attention kernels outside the fused families "
                f"('{top.layer_name}' and peers) consumed "
                f"{unfused_attn_share:.0%} of profiled GPU time — disproportionate "
                f"to the model's attention:MLP cost ratio at moderate context — "
                f"and no fused attention kernel appeared in the trace: the "
                f"attention path ran through a custom or legacy implementation "
                f"rather than an IO-aware fused kernel."
            )

        notes = (
            f"leg={firing_leg}; softmax_share={softmax_share:.1%}, "
            f"unfused_attn_share={unfused_attn_share:.1%}, "
            f"fused_share={fused_share:.1%} (guard bar {FUSED_MINOR:.0%})"
        )
        if mem_frac is None:
            notes += "; no roofline data — memory-bound corroboration not evaluated"
        elif corroborated:
            notes += f"; corroborated: {mem_frac:.0%} of attention-path time memory-bound (+{CORROBORATION_BUMP})"
        if THRESHOLDS_UNCALIBRATED:
            notes += f"; thresholds uncalibrated, confidence capped at {_UNCALIBRATED_CEILING}"

        return Diagnosis(
            rule_id=self.rule_id,
            cause=cause,
            fix=self._fix(),
            confidence=confidence,
            confidence_breakdown=ConfidenceBreakdown(
                signal_strength=signal_strength,
                data_completeness=self._data_completeness(timed),
                notes=notes,
            ),
            evidence={
                "softmax_share": round(softmax_share, 3),
                "attention_share": round(attention_share, 3),
                "fused_attention_share": round(fused_share, 3),
                "mlp_share": round(mlp_share, 3),
                "firing_leg": firing_leg,
                "total_kernel_time_ms": round(total_ms, 3),
                "num_kernels": len(timed),
                "top_offender": top.layer_name,
                "top_offender_ms": round(top.duration_ms, 3),
                # None = rooflines unknown, corroboration NOT evaluated — which
                # is not the same as evaluated-and-low (r05 precedent).
                "attention_path_memory_bound_frac": (
                    round(mem_frac, 3) if mem_frac is not None else None
                ),
                "signal_strength": round(signal_strength, 2),
            },
        )

    @staticmethod
    def _fix() -> str:
        return (
            "Enable a fused attention backend — this is a config change, not new "
            "hardware. vLLM: set VLLM_ATTENTION_BACKEND=FLASH_ATTN (or upgrade to "
            "a build with FlashAttention-2 / FlashInfer). PyTorch: replace the "
            "explicit matmul → softmax → matmul chain with "
            "torch.nn.functional.scaled_dot_product_attention, pinning the "
            "backend via torch.nn.attention.sdpa_kernel([SDPBackend."
            "FLASH_ATTENTION]). HF Transformers: load the model with "
            "attn_implementation=\"flash_attention_2\". The fused kernel keeps "
            "the score matrix in SRAM (online softmax), eliminating the "
            "standalone softmax pass and its O(seq_len²) HBM traffic — 2–4× on "
            "the attention kernels in the literature (FlashAttention, "
            "arXiv:2205.14135; FlashAttention-2, arXiv:2307.08691)."
        )

    @staticmethod
    def _data_completeness(timed: list[LayerMetrics]) -> float:
        """Fraction of timed kernels carrying the optional corroborating metrics.

        Duration-only captures (nsys / torch.profiler / ncu without counter
        permission) fire the duration-share legs with visibly reduced
        completeness — the honest label for a degraded capture.
        """
        if not timed:
            return 0.0
        rich = [
            l for l in timed
            if l.sm_occupancy is not None
            and l.roofline_position is not None
            and l.roofline_position != "unknown"
        ]
        return len(rich) / len(timed)
