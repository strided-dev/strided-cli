"""Builders shared across the gait test suite.

Imported directly (``from gait_builders import ...``): with no ``__init__.py`` under
``tests/``, pytest prepends each test directory to ``sys.path``, so this sibling
module is importable by name from the test files and conftest in this directory.
"""

from __future__ import annotations

from rules.base import ConfidenceBreakdown, Diagnosis
from schema import DiagnosisInput, Distribution


def make_diagnosis(rule_id: str = "r03", confidence: float = 0.65) -> Diagnosis:
    return Diagnosis(
        rule_id=rule_id,
        cause="KV cache fragmentation was 47% near capacity.",
        fix="reduce --block-size toward the workload",
        confidence=confidence,
        confidence_breakdown=ConfidenceBreakdown(0.8, 0.5, "test"),
    )


def make_snapshot(
    *,
    frag: float | None = 0.47,
    util: float | None = 0.86,
    throughput: float | None = 10.0,
    seqlen_mean: float | None = 512.0,
    engine: str = "vllm",
    block_size: int | None = 16,
) -> DiagnosisInput:
    return DiagnosisInput(
        model_name="meta-llama/Llama-3-8B",
        gpu_type="H100-SXM",
        inference_engine=engine,
        kv_cache_fragmentation=frag,
        kv_cache_util=util,
        kv_block_size=block_size,
        request_throughput=throughput,
        seq_len_distribution=Distribution(mean=seqlen_mean) if seqlen_mean is not None else None,
    )
