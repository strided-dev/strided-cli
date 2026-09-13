"""Unit tests for the CLI's merge logic and parser error boundary.

The end-to-end tests in test_main.py prove the happy path; these isolate the two
pieces of plumbing that are easy to get subtly wrong:

  * `merge_inputs` — first-non-None precedence, deep-merge of nested
    PhaseMetrics (enrich, don't clobber), provenance concatenation, and the
    override path.
  * the ingestion boundary — a parser that raises must become a clean CLI error,
    not a stack trace, and `--strict` must re-raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.main import cli
from schema import DiagnosisInput, Distribution, PhaseMetrics
from schema.merge import _merge_field, merge_inputs


def _vllm_like() -> DiagnosisInput:
    """A vLLM-shaped input: workload context + latency-only decode."""
    return DiagnosisInput(
        model_name="from-vllm",
        gpu_type="from-vllm-gpu",
        inference_engine="vllm",
        batch_size=4,
        decode=PhaseMetrics(latency_ms=20.0),
        kv_cache_util=0.5,
        source_files=["vllm.prom"],
        parse_warnings=["vllm warning"],
    )


def _nsight_like() -> DiagnosisInput:
    """An Nsight-shaped input: kernel-level decode metrics, no workload context."""
    return DiagnosisInput(
        model_name="unknown",
        gpu_type="unknown",
        inference_engine="unknown",
        decode=PhaseMetrics(
            duration_ms=2.0, sm_occupancy=0.18, hbm_bandwidth_util=0.85,
        ),
        source_files=["report.csv"],
        parse_warnings=["nsight warning"],
    )


# --------------------------------------------------------------------------- #
# _merge_field unit behaviour
# --------------------------------------------------------------------------- #

class TestMergeField:
    def test_first_non_none_wins_for_scalars(self) -> None:
        assert _merge_field("batch_size", 4, 8) == 4
        assert _merge_field("batch_size", None, 8) == 8
        assert _merge_field("batch_size", 4, None) == 4

    def test_provenance_lists_concatenate(self) -> None:
        assert _merge_field("source_files", ["a"], ["b"]) == ["a", "b"]
        assert _merge_field("parse_warnings", None, ["b"]) == ["b"]

    def test_non_provenance_list_is_first_wins(self) -> None:
        # tp_rank_sm_clocks is data, not provenance: it must not concatenate.
        assert _merge_field("tp_rank_sm_clocks", [1.0], [2.0]) == [1.0]

    def test_nested_basemodel_deep_merges(self) -> None:
        a = PhaseMetrics(latency_ms=20.0)
        b = PhaseMetrics(sm_occupancy=0.18, latency_ms=99.0)
        merged = _merge_field("decode", a, b)
        assert merged.latency_ms == 20.0       # a wins the overlap
        assert merged.sm_occupancy == 0.18     # b enriches the gap


# --------------------------------------------------------------------------- #
# merge_inputs integration
# --------------------------------------------------------------------------- #

class TestMergeInputs:
    def test_single_input_passthrough(self) -> None:
        v = _vllm_like()
        assert merge_inputs([v], model_name=None, gpu_type=None) is v

    def test_empty_raises(self) -> None:
        with pytest.raises(Exception):  # click.UsageError
            merge_inputs([], model_name=None, gpu_type=None)

    def test_deep_merge_enriches_decode_without_clobbering(self) -> None:
        merged = merge_inputs([_vllm_like(), _nsight_like()], model_name=None, gpu_type=None)
        # vLLM owns the per-token latency; Nsight owns the kernel-measured
        # duration and the SM/HBM. Neither side loses its data.
        assert merged.decode.latency_ms == pytest.approx(20.0)
        assert merged.decode.duration_ms == pytest.approx(2.0)
        assert merged.decode.sm_occupancy == pytest.approx(0.18)
        assert merged.decode.hbm_bandwidth_util == pytest.approx(0.85)

    def test_workload_context_takes_first_source(self) -> None:
        merged = merge_inputs([_vllm_like(), _nsight_like()], model_name=None, gpu_type=None)
        assert merged.model_name == "from-vllm"      # vLLM precedes Nsight
        assert merged.inference_engine == "vllm"
        assert merged.batch_size == 4

    def test_provenance_concatenated_in_order(self) -> None:
        merged = merge_inputs([_vllm_like(), _nsight_like()], model_name=None, gpu_type=None)
        assert merged.source_files == ["vllm.prom", "report.csv"]
        assert "vllm warning" in merged.parse_warnings
        assert "nsight warning" in merged.parse_warnings

    def test_overrides_win_over_every_source(self) -> None:
        merged = merge_inputs(
            [_vllm_like(), _nsight_like()], model_name="OVERRIDE", gpu_type="OVERRIDE-GPU"
        )
        assert merged.model_name == "OVERRIDE"
        assert merged.gpu_type == "OVERRIDE-GPU"

    def test_override_applies_to_single_input_too(self) -> None:
        merged = merge_inputs([_vllm_like()], model_name="OVERRIDE", gpu_type=None)
        assert merged.model_name == "OVERRIDE"
        assert merged.gpu_type == "from-vllm-gpu"  # untouched

    def test_nested_distribution_merges(self) -> None:
        a = DiagnosisInput(model_name="a", gpu_type="g", ttft_ms=Distribution(mean=1.0))
        b = DiagnosisInput(model_name="b", gpu_type="g", ttft_ms=Distribution(mean=9.0, p95=5.0))
        merged = merge_inputs([a, b], model_name=None, gpu_type=None)
        assert merged.ttft_ms.mean == 1.0    # a wins the overlap
        assert merged.ttft_ms.p95 == 5.0     # b fills the gap


# --------------------------------------------------------------------------- #
# Parser error boundary (via the CLI)
# --------------------------------------------------------------------------- #

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestParseBoundary:
    def test_ncu_rep_is_clean_usage_error(self, runner: CliRunner, tmp_path: Path) -> None:
        rep = tmp_path / "trace.ncu-rep"
        rep.write_bytes(b"not really a binary report")
        result = runner.invoke(cli, ["diagnose", "--nsight", str(rep)])
        assert result.exit_code == 2                  # UsageError, not a crash
        assert "not yet implemented" in result.output
        assert "Traceback" not in result.output

    def test_malformed_file_degrades_to_clean_error(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Invalid UTF-8 makes the parser's open().read() raise — the boundary
        # must catch it and report cleanly rather than dumping a traceback.
        bad = tmp_path / "broken.csv"
        bad.write_bytes(b"\xff\xfe\x00bad")
        result = runner.invoke(cli, ["diagnose", "--nsight", str(bad)])
        assert result.exit_code == 1                  # ClickException
        assert "Failed to parse Nsight input" in result.output
        assert "Traceback" not in result.output

    def test_strict_reraises_parser_error(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "broken.csv"
        bad.write_bytes(b"\xff\xfe\x00bad")
        result = runner.invoke(
            cli, ["diagnose", "--nsight", str(bad), "--strict"], catch_exceptions=True
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, UnicodeDecodeError)  # raw exception surfaced
