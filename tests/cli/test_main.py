"""End-to-end tests for the strided CLI.

These exercise the full pipeline — parser → merge → engine → formatter — with
real components, no mocks. Each test is the regression form of one manual
smoke-check from CLI v0.1 bring-up.

Layout:

  - Fixture helpers build small, deliberately-crafted dumps on disk so the
    parsers are exercised on their real file APIs (not bypassed via direct
    DiagnosisInput construction).
  - Each test invokes the CLI with `click.testing.CliRunner` and asserts on
    the rendered stdout. We assert on observable user-facing strings so the
    tests double as documentation of the output contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.main import cli


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

# vLLM /metrics with KV cache + 4 in-flight requests, TTFT mean ~62ms,
# TPOT mean ~18ms. Sm/HBM are not exposed by vLLM, so r01 must abstain
# on this input alone.
_VLLM_TEXT = """\
# HELP vllm:gpu_cache_usage_perc Gauge of GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="x"} 0.86
# HELP vllm:num_requests_running Requests currently running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="x"} 4
# HELP vllm:time_to_first_token_seconds TTFT histogram
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.01"} 0
vllm:time_to_first_token_seconds_bucket{le="0.05"} 50
vllm:time_to_first_token_seconds_bucket{le="0.10"} 200
vllm:time_to_first_token_seconds_bucket{le="0.50"} 300
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 300
vllm:time_to_first_token_seconds_count 300
vllm:time_to_first_token_seconds_sum 18.5
# HELP vllm:time_per_output_token_seconds TPOT histogram
# TYPE vllm:time_per_output_token_seconds histogram
vllm:time_per_output_token_seconds_bucket{le="0.005"} 0
vllm:time_per_output_token_seconds_bucket{le="0.010"} 60
vllm:time_per_output_token_seconds_bucket{le="0.025"} 250
vllm:time_per_output_token_seconds_bucket{le="0.050"} 300
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 300
vllm:time_per_output_token_seconds_count 300
vllm:time_per_output_token_seconds_sum 5.4
"""

# Nsight raw-page CSV crafted to be clearly decode-memory-bound on A100-80G:
#   peak HBM = 2.039 TB/s; 1.73 GB / 1ms ≈ 1.73 TB/s ≈ 85% util.
#   SM occupancy ~20% (below r01's 30% threshold).
_NSIGHT_MEMBOUND_CSV = (
    '"Kernel Name","Metric Name","Metric Value"\n'
    '"flash_attn_decode_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","18.5"\n'
    '"flash_attn_decode_kernel","dram__bytes.sum","1730000000"\n'
    '"flash_attn_decode_kernel","gpu__time_duration.sum","1000000"\n'
    '"gemv_decode_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","22.0"\n'
    '"gemv_decode_kernel","dram__bytes.sum","1650000000"\n'
    '"gemv_decode_kernel","gpu__time_duration.sum","1000000"\n'
)

# Nsight CSV crafted NOT to trigger r01: SM occupancy comfortably above 30%.
_NSIGHT_COMPUTE_BOUND_CSV = (
    '"Kernel Name","Metric Name","Metric Value"\n'
    '"gemm_prefill_kernel","sm__warps_active.avg.pct_of_peak_sustained_active","72.0"\n'
    '"gemm_prefill_kernel","dram__bytes.sum","400000000"\n'
    '"gemm_prefill_kernel","gpu__time_duration.sum","1000000"\n'
)


# nsys NVTX push/pop CSV (nanoseconds) crafted to fire r08: 32 decode steps at
# 10ms and 8 prefill steps at 120ms (12× the decode baseline). Built here so the
# CLI exercises the real parsers/nsight_systems.py file API.
def _nsys_interference_csv() -> str:
    rows = ['"Start (ns)","End (ns)","Duration (ns)","Name"']
    t = 0
    for _ in range(32):
        rows.append(f'"{t}","{t + 10_000_000}","10000000","decode"')
        t += 10_000_000
    for _ in range(8):
        rows.append(f'"{t}","{t + 120_000_000}","120000000","prefill tokens=8192"')
        t += 120_000_000
    return "\n".join(rows) + "\n"


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.fixture
def vllm_file(tmp_path: Path) -> Path:
    return _write(tmp_path / "vllm.prom", _VLLM_TEXT)


@pytest.fixture
def nsight_membound_csv(tmp_path: Path) -> Path:
    return _write(tmp_path / "membound.csv", _NSIGHT_MEMBOUND_CSV)


@pytest.fixture
def nsight_compute_bound_csv(tmp_path: Path) -> Path:
    return _write(tmp_path / "compute_bound.csv", _NSIGHT_COMPUTE_BOUND_CSV)


@pytest.fixture
def nsys_interference_csv(tmp_path: Path) -> Path:
    return _write(tmp_path / "timeline.csv", _nsys_interference_csv())


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #

class TestArguments:
    def test_no_sources_errors(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["diagnose"])
        # click translates UsageError into exit code 2.
        assert result.exit_code == 2
        assert "at least one of --vllm" in result.output

    def test_unrecognised_nsight_extension_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        bad = _write(tmp_path / "trace.txt", "irrelevant\n")
        result = runner.invoke(cli, ["diagnose", "--nsight", str(bad)])
        assert result.exit_code == 2
        assert "Unrecognised Nsight extension" in result.output

    def test_unrecognised_nsys_extension_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        bad = _write(tmp_path / "trace.txt", "irrelevant\n")
        result = runner.invoke(cli, ["diagnose", "--nsys", str(bad)])
        assert result.exit_code == 2
        assert "Unrecognised nsys extension" in result.output

    def test_missing_file_errors(self, runner: CliRunner) -> None:
        # click validates path-exists before our code runs.
        result = runner.invoke(cli, ["diagnose", "--vllm", "/nope/does-not-exist.prom"])
        assert result.exit_code == 2
        assert "does not exist" in result.output.lower() or "no such file" in result.output.lower()


# --------------------------------------------------------------------------- #
# Single-source flows
# --------------------------------------------------------------------------- #

class TestSingleSource:
    def test_vllm_only_abstains_r01(
        self, runner: CliRunner, vllm_file: Path
    ) -> None:
        """vLLM cannot supply SM/HBM, so r01 must surface as insufficient-data."""
        result = runner.invoke(
            cli, ["diagnose", "--vllm", str(vllm_file), "--model", "Llama-3-8B", "--gpu", "A100-80G"]
        )
        assert result.exit_code == 0, result.output
        assert "Llama-3-8B" in result.output
        assert "batch_size" in result.output and "4" in result.output
        assert "no rules fired" in result.output
        assert "could not evaluate" in result.output
        assert "r01" in result.output

    def test_nsight_only_fires_r01(
        self, runner: CliRunner, nsight_membound_csv: Path
    ) -> None:
        """Pure Nsight memory-bound input must fire r01 with confidence ≥ 0.5."""
        result = runner.invoke(
            cli,
            ["diagnose", "--nsight", str(nsight_membound_csv),
             "--model", "Llama-3-8B", "--gpu", "A100-80G"],
        )
        assert result.exit_code == 0, result.output
        assert "1 rule fired" in result.output
        assert "Decode memory-bound at low batch" in result.output
        assert "r01" in result.output
        # vLLM not supplied; batch_size must therefore be None in the evidence.
        assert "batch_size=None" in result.output

    def test_nsys_only_fires_r08(
        self, runner: CliRunner, nsys_interference_csv: Path
    ) -> None:
        """A spiky nsys timeline must fire r08 with a computed chunk budget."""
        result = runner.invoke(
            cli,
            ["diagnose", "--nsys", str(nsys_interference_csv),
             "--model", "Llama-3-8B", "--gpu", "H100-SXM"],
        )
        assert result.exit_code == 0, result.output
        assert "1 rule fired" in result.output
        assert "r08" in result.output
        # The computed budget reaches the rendered output (wrap-safe substrings;
        # the renderer word-wraps, so don't assert on the hyphenated flag name).
        assert "384" in result.output
        assert "interference" in result.output

    def test_nsight_compute_bound_does_not_fire(
        self, runner: CliRunner, nsight_compute_bound_csv: Path
    ) -> None:
        """SM occupancy above the 30% threshold should NOT trigger r01."""
        result = runner.invoke(
            cli,
            ["diagnose", "--nsight", str(nsight_compute_bound_csv),
             "--model", "Llama-3-8B", "--gpu", "A100-80G"],
        )
        assert result.exit_code == 0, result.output
        assert "no rules fired" in result.output
        # r01 had the data but the signal was below threshold — silent by contract,
        # so it must not appear at all (not fired, not in the insufficient-data list).
        assert "r01" not in result.output
        # r03 *does* surface under could not evaluate: a Nsight-only dump carries no
        # KV-cache fields, so it correctly abstains with INSUFFICIENT_DATA.
        assert "r03 KV cache fragmentation" in result.output


# --------------------------------------------------------------------------- #
# Merging two sources
# --------------------------------------------------------------------------- #

class TestMerge:
    def test_vllm_plus_nsight_carries_batch_size_into_cause(
        self,
        runner: CliRunner,
        vllm_file: Path,
        nsight_membound_csv: Path,
    ) -> None:
        """Merge must enrich the Nsight-side r01 evidence with the vLLM batch_size."""
        result = runner.invoke(
            cli,
            [
                "diagnose",
                "--vllm", str(vllm_file),
                "--nsight", str(nsight_membound_csv),
                "--model", "Llama-3-8B",
                "--gpu", "A100-80G",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "1 rule fired" in result.output
        # vLLM contributed batch_size=4; r01 should reference it in the cause.
        # Wrap-safe substrings: the renderer word-wraps the cause, and r01's is
        # long enough that "At batch size 4" can straddle a line break.
        assert "batch size" in result.output
        assert "batch_size=4" in result.output
        # Nsight contributed SM/HBM; both should be present in evidence. The HBM
        # key is locus-qualified since the r01 kernel-scope fix — an Nsight
        # dump carries `layers`, so the reading comes from the dominant kernel,
        # not from the collapsed `decode` aggregate, and the key says which.
        assert "layers.dominant.hbm_bandwidth_util=" in result.output
        assert "hbm_read_locus=layers.dominant_by_total_ms" in result.output
        # Both legs are scoped now, so both report the dominant kernel's own
        # value. This previously pinned the COLLAPSED occupancy key — the same
        # shape of stale assertion the locus fix had to correct for bandwidth,
        # one level down. A guard reading an aggregate while its partner leg
        # reads a scoped kernel is the guard fail-open.
        assert "layers.dominant.sm_occupancy=" in result.output
        assert "occupancy_read_locus=layers.dominant_by_total_ms" in result.output

    def test_overrides_win_over_parser_defaults(
        self, runner: CliRunner, vllm_file: Path
    ) -> None:
        result = runner.invoke(
            cli,
            ["diagnose", "--vllm", str(vllm_file),
             "--model", "OverrideModel", "--gpu", "OverrideGPU"],
        )
        assert result.exit_code == 0, result.output
        assert "OverrideModel" in result.output
        assert "OverrideGPU" in result.output


# --------------------------------------------------------------------------- #
# Output structure
# --------------------------------------------------------------------------- #

class TestOutput:
    def test_header_includes_version_and_phase_breakdown(
        self, runner: CliRunner, vllm_file: Path
    ) -> None:
        result = runner.invoke(
            cli, ["diagnose", "--vllm", str(vllm_file),
                  "--model", "Llama-3-8B", "--gpu", "A100-80G"]
        )
        assert result.exit_code == 0, result.output
        assert result.output.startswith("strided v")
        assert "phase breakdown" in result.output
        # The phase breakdown reports true phase wall-clocks only. vLLM supplies
        # prefill (≈TTFT); it cannot supply a decode-phase duration (TPOT is
        # per-token), so decode is absent from the breakdown until Nsight is
        # merged in.
        phase_lines = [
            line for line in result.output.splitlines()
            if line.lstrip().startswith(("prefill", "decode"))
        ]
        assert any(line.lstrip().startswith("prefill") for line in phase_lines)
        assert not any(line.lstrip().startswith("decode") for line in phase_lines)
