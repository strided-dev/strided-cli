"""Unit tests for parsers/nsight.py.

Nsight Compute CSV is the primary sprint data source. The parser must handle
both the raw-page (one-metric-per-row) and details-page (metric-as-column)
exports, classify kernels, and — critically — convert byte counts into a [0,1]
HBM utilisation using a *gpu_type-specific* peak bandwidth. These tests pin that
gpu_type dependency (a latent foot-gun: pass the wrong GPU and the utilisation,
hence the roofline classification, silently shifts).
"""

from __future__ import annotations

import pytest

from parsers.nsight import (
    _peak_hbm_bps,
    _classify_layer,
    _roofline_from_metrics,
    parse_nsight_csv,
    parse_nsight_csv_file,
    parse_nsight_ncu_rep,
)
from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# Raw-page format: memory-bound attention kernel (1.73 GB in 1 ms) + an
# allreduce. On A100-80G (peak 2.039 TB/s) the attention kernel sits at ~85%.
_RAW = (
    '"Kernel Name","Metric Name","Metric Value"\n'
    '"flash_attn_fwd","sm__warps_active.avg.pct_of_peak_sustained_active","18.5"\n'
    '"flash_attn_fwd","dram__bytes.sum","1730000000"\n'
    '"flash_attn_fwd","gpu__time_duration.sum","1000000"\n'
    '"nccl_all_reduce","gpu__time_duration.sum","1000000"\n'
)

# Details-page format: metrics are columns, one kernel per row.
_DETAILS = (
    "Kernel Name,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum,gpu__time_duration.sum\n"
    "gemm_mlp_kernel,78.0,1000000000,1500000\n"
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

class TestClassifyLayer:
    @pytest.mark.parametrize("name,expected", [
        ("flash_attn_fwd", "attention"),
        ("fmha_kernel", "attention"),
        # Real fused-attention kernel names from the wild (r07's inputs): the
        # flash-attn library emits flash_fwd_* (no "attn" substring), PyTorch
        # SDPA's mem-efficient path emits fmha_cutlass*, Triton attention
        # kernels live under a flash:: C++ namespace, vLLM's paged decode is
        # paged_attention_v*.
        ("flash_fwd_splitkv_kernel", "attention"),
        ("void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64>>", "attention"),
        ("fmha_cutlassF_f16_aligned_64x64_rf_sm80", "attention"),
        ("paged_attention_v2_kernel", "attention"),
        ("sdpa_forward", "attention"),
        ("gemm_mlp_kernel", "mlp"),
        # Naive attention's Q.K^T / P.V run as generic GEMMs — by name they are
        # indistinguishable from MLP GEMMs. Pinned so r07's spec stays honest:
        # the unfused path's signature is standalone softmax time, not
        # "attention share".
        ("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_64x3_nn", "mlp"),
        ("rmsnorm", "norm"),
        ("nccl_all_reduce", "allreduce"),
        ("embedding_lookup", "embedding"),
        ("softmax_warp", "softmax"),
        ("void (anonymous namespace)::softmax_warp_forward<c10::Half, 11>", "softmax"),
        ("mystery_kernel", "other"),
        # Triton's generic suffix is deliberately NOT matched (would swallow
        # unrelated kernels) — a known naming miss recorded in the r07 doc.
        ("_fwd_kernel", "other"),
    ])
    def test_classification(self, name: str, expected: str) -> None:
        assert _classify_layer(name) == expected


class TestRoofline:
    def test_unknown_when_inputs_missing(self) -> None:
        assert _roofline_from_metrics(None, 0.9, None) == "unknown"

    def test_stall_dominates(self) -> None:
        # High memory-stall pct forces memory_bound even with mid utilisation.
        assert _roofline_from_metrics(0.5, 0.5, 50.0) == "memory_bound"

    def test_memory_bound_by_util(self) -> None:
        assert _roofline_from_metrics(0.30, 0.90, None) == "memory_bound"

    def test_compute_bound_by_util(self) -> None:
        assert _roofline_from_metrics(0.80, 0.30, None) == "compute_bound"


# --------------------------------------------------------------------------- #
# Format handling
# --------------------------------------------------------------------------- #

class TestFormats:
    def test_raw_page_parses(self) -> None:
        out = parse_nsight_csv(_RAW, gpu_type="A100-80G")
        assert isinstance(out, DiagnosisInput)
        assert len(out.layers) == 2

    def test_details_page_parses(self) -> None:
        out = parse_nsight_csv(_DETAILS, gpu_type="A100-80G")
        assert len(out.layers) == 1
        assert out.layers[0].sm_occupancy == pytest.approx(0.78)

    def test_missing_kernel_column_warns(self) -> None:
        out = parse_nsight_csv("Foo,Bar\n1,2\n")
        assert any("Kernel Name" in w for w in out.parse_warnings)
        assert out.layers is None

    def test_empty_csv_warns(self) -> None:
        out = parse_nsight_csv("")
        assert any("empty or invalid" in w or "no kernels" in w for w in out.parse_warnings)


# --------------------------------------------------------------------------- #
# Launch-instance aggregation — ncu profiles each *launch*, so one kernel name
# recurs. Durations must SUM (last-wins silently skewed every time-share
# signal: r05's nccl_time_pct, r07's attention/softmax shares, the aggregate
# phase); percentages take the duration-weighted mean.
# --------------------------------------------------------------------------- #

class TestInstanceAggregation:
    _RAW_TWO_LAUNCHES = (
        '"ID","Kernel Name","Metric Name","Metric Value"\n'
        '"0","gemm_k","gpu__time_duration.sum","1000000"\n'
        '"0","gemm_k","sm__warps_active.avg.pct_of_peak_sustained_active","20.0"\n'
        '"1","gemm_k","gpu__time_duration.sum","2000000"\n'
        '"1","gemm_k","sm__warps_active.avg.pct_of_peak_sustained_active","80.0"\n'
    )

    def test_raw_page_durations_sum_across_ids(self) -> None:
        out = parse_nsight_csv(self._RAW_TWO_LAUNCHES)
        assert len(out.layers) == 1
        assert out.layers[0].duration_ms == pytest.approx(3.0)  # 1 ms + 2 ms

    def test_raw_page_occupancy_is_duration_weighted(self) -> None:
        out = parse_nsight_csv(self._RAW_TWO_LAUNCHES)
        # (20% * 1 ms + 80% * 2 ms) / 3 ms = 60%
        assert out.layers[0].sm_occupancy == pytest.approx(0.60)

    def test_raw_page_without_id_column_still_splits_on_key_collision(self) -> None:
        # Older exports may lack ID; a repeated metric key means a new launch.
        text = (
            '"Kernel Name","Metric Name","Metric Value"\n'
            '"gemm_k","gpu__time_duration.sum","1000000"\n'
            '"gemm_k","gpu__time_duration.sum","2000000"\n'
        )
        out = parse_nsight_csv(text)
        assert out.layers[0].duration_ms == pytest.approx(3.0)

    def test_details_page_duplicate_rows_sum(self) -> None:
        text = (
            "Kernel Name,gpu__time_duration.sum\n"
            "softmax_k,500000\n"
            "softmax_k,700000\n"
        )
        out = parse_nsight_csv(text)
        assert len(out.layers) == 1
        assert out.layers[0].duration_ms == pytest.approx(1.2)

    def test_nccl_fraction_counts_every_launch(self) -> None:
        # The last-wins regression: 3 allreduce launches of 1 ms each against a
        # single 3 ms gemm must read 50% NCCL, not 25% (one launch surviving).
        text = (
            "Kernel Name,gpu__time_duration.sum\n"
            "gemm_k,3000000\n"
            "nccl_all_reduce,1000000\n"
            "nccl_all_reduce,1000000\n"
            "nccl_all_reduce,1000000\n"
        )
        out = parse_nsight_csv(text)
        assert out.nccl_time_pct == pytest.approx(0.5)

    def test_aggregation_warns_with_launch_counts(self) -> None:
        out = parse_nsight_csv(self._RAW_TWO_LAUNCHES)
        assert any("2 kernel launches aggregated into 1" in w for w in out.parse_warnings)

    def test_single_launch_per_kernel_does_not_warn(self) -> None:
        out = parse_nsight_csv(_RAW)
        assert not any("launches aggregated" in w for w in out.parse_warnings)


# --------------------------------------------------------------------------- #
# HBM utilisation depends on gpu_type — the latent foot-gun
# --------------------------------------------------------------------------- #

class TestGpuTypeDependentBandwidth:
    def test_same_bytes_differ_by_gpu(self) -> None:
        a100 = parse_nsight_csv(_RAW, gpu_type="A100-80G").layers[0].hbm_bandwidth_util
        h100 = parse_nsight_csv(_RAW, gpu_type="H100-SXM").layers[0].hbm_bandwidth_util
        # 1.73 TB/s against A100's 2.039 vs H100's 3.35 TB/s peak.
        assert a100 == pytest.approx(1.73e12 / 2.039e12, rel=1e-3)
        assert h100 == pytest.approx(1.73e12 / 3.35e12, rel=1e-3)
        assert a100 > h100  # same workload reads as more memory-bound on A100

    def test_unknown_gpu_uses_h100_default(self) -> None:
        unknown = parse_nsight_csv(_RAW, gpu_type="definitely-not-a-gpu").layers[0]
        h100 = parse_nsight_csv(_RAW, gpu_type="H100-SXM").layers[0]
        assert unknown.hbm_bandwidth_util == pytest.approx(h100.hbm_bandwidth_util)

    def test_rtx_3050_laptop_peak_registered(self) -> None:
        # The local-validation card (r06/r07 harnesses pass RTX-3050-Laptop).
        # Without its own entry the trace would be scored against the H100
        # default and read ~17x under-utilised. 1.73 GB in 1 ms = 1.73 TB/s,
        # clamped to 1.0 against the 0.192 TB/s laptop-3050 peak.
        out = parse_nsight_csv(_RAW, gpu_type="RTX-3050-Laptop").layers[0]
        assert out.hbm_bandwidth_util == pytest.approx(1.0)

    def test_util_clamped_to_one(self) -> None:
        # Absurd byte count must not produce util > 1.
        text = (
            '"Kernel Name","Metric Name","Metric Value"\n'
            '"k","dram__bytes.sum","999999999999999"\n'
            '"k","gpu__time_duration.sum","1000"\n'
        )
        out = parse_nsight_csv(text, gpu_type="A100-80G")
        assert out.layers[0].hbm_bandwidth_util == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Aggregation & derived fields
# --------------------------------------------------------------------------- #

class TestAggregation:
    def test_duration_weighted_mean(self) -> None:
        # Two kernels with different occupancy and duration; the longer one
        # dominates the weighted aggregate.
        text = (
            '"Kernel Name","Metric Name","Metric Value"\n'
            '"long","sm__warps_active.avg.pct_of_peak_sustained_active","10"\n'
            '"long","gpu__time_duration.sum","9000000"\n'
            '"short","sm__warps_active.avg.pct_of_peak_sustained_active","90"\n'
            '"short","gpu__time_duration.sum","1000000"\n'
        )
        out = parse_nsight_csv(text, gpu_type="A100-80G")
        # (0.10*9 + 0.90*1) / 10 = 0.18, not the unweighted 0.50.
        assert out.decode.sm_occupancy == pytest.approx(0.18)

    def test_nccl_time_fraction(self) -> None:
        out = parse_nsight_csv(_RAW, gpu_type="A100-80G")
        # attention 1ms + allreduce 1ms → allreduce is half the time.
        assert out.nccl_time_pct == pytest.approx(0.5)

    def test_aggregate_goes_to_decode_prefill_none(self) -> None:
        out = parse_nsight_csv(_RAW, gpu_type="A100-80G")
        assert out.decode is not None
        assert out.prefill is None  # CSV cannot tag phases; never guesses prefill
        assert any("does not tag prefill/decode" in w for w in out.parse_warnings)


# --------------------------------------------------------------------------- #
# Binary stub & file wrapper
# --------------------------------------------------------------------------- #

class TestEntryPoints:
    def test_ncu_rep_stub_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            parse_nsight_ncu_rep("whatever.ncu-rep")

    def test_file_wrapper_handles_bom(self, tmp_path) -> None:
        # Written with a BOM; utf-8-sig read + header strip must still work.
        p = tmp_path / "report.csv"
        p.write_text(_RAW, encoding="utf-8-sig")
        out = parse_nsight_csv_file(str(p), gpu_type="A100-80G")
        assert out.source_files == [str(p)]
        assert len(out.layers) == 2


# --------------------------------------------------------------------------- #
# Log preamble before the header
# --------------------------------------------------------------------------- #

class TestPreambleStripping:
    """`ncu --csv` shares stdout with the application it profiles.

    The documented capture command redirects that stdout to a file, so anything
    the workload logs lands *above* the real CSV header. csv.DictReader treats
    line 1 as the header, so before this was handled the whole file parsed as a
    single garbage column and every metric silently went missing. Regression
    from a field capture where 33 lines of vLLM startup logging preceded the
    ncu table and r01 read `insufficient` off a complete 202-row capture.
    """

    _PREAMBLE = (
        "INFO 08-02 21:14:49 [__init__.py:216] Automatically detected platform cuda.\n"
        "WARNING 08-02 21:15:05 [topk_topp_sampler.py:69] FlashInfer is not available.\n"
        "==PROF== Connected to process 2777086 (/usr/bin/python3.12)\n"
    )

    def test_leading_log_lines_are_skipped(self) -> None:
        out = parse_nsight_csv(self._PREAMBLE + _RAW, gpu_type="A100-80G")
        assert len(out.layers) == 2
        assert out.decode is not None
        assert any("preamble" in w for w in out.parse_warnings)

    def test_metrics_match_the_unpolluted_parse(self) -> None:
        polluted = parse_nsight_csv(self._PREAMBLE + _RAW, gpu_type="A100-80G")
        clean = parse_nsight_csv(_RAW, gpu_type="A100-80G")
        assert polluted.decode is not None and clean.decode is not None
        assert polluted.decode.hbm_bandwidth_util == pytest.approx(
            clean.decode.hbm_bandwidth_util
        )
        assert polluted.decode.sm_occupancy == pytest.approx(clean.decode.sm_occupancy)

    def test_details_page_preamble_also_skipped(self) -> None:
        out = parse_nsight_csv(self._PREAMBLE + _DETAILS, gpu_type="A100-80G")
        assert len(out.layers) == 1

    def test_clean_csv_reports_no_preamble(self) -> None:
        out = parse_nsight_csv(_RAW, gpu_type="A100-80G")
        assert not any("preamble" in w for w in out.parse_warnings)

    def test_no_header_anywhere_still_warns_about_kernel_name(self) -> None:
        # A file with no header at all must keep its original diagnostic rather
        # than being silently emptied by the preamble scan.
        out = parse_nsight_csv("just\nsome\nlog lines\n", gpu_type="A100-80G")
        assert not out.layers
        assert any("'Kernel Name' column not found" in w for w in out.parse_warnings)


# --------------------------------------------------------------------------- #
# gpu_type recognition
# --------------------------------------------------------------------------- #

class TestGpuTypeRecognition:
    """An unknown gpu_type rescales every kernel instead of failing.

    HBM utilisation is `bytes / (duration x peak)`, and the fallback peak is the
    fastest card in the table — so an unrecognised name always reads
    *under*-utilised and biases the roofline toward compute-bound, i.e. toward
    rules abstaining. Regression from a field capture where `A100-PCIE-40GB`
    (what nvidia-smi actually reports) missed the `A100-40G` key and scored a
    1.555 TB/s card against 3.35 TB/s, understating every reading by 2.15x.
    """

    def test_nvidia_smi_style_name_is_recognised(self) -> None:
        assert _peak_hbm_bps("A100-PCIE-40GB") == pytest.approx(1.555e12)

    def test_vendor_prefix_and_case_tolerated(self) -> None:
        assert _peak_hbm_bps("NVIDIA A100-PCIE-40GB") == pytest.approx(1.555e12)
        assert _peak_hbm_bps("a100-pcie-40gb") == pytest.approx(1.555e12)

    def test_unknown_gpu_warns(self) -> None:
        out = parse_nsight_csv(_RAW, gpu_type="Totally-Made-Up-GPU")
        assert any("unrecognised gpu_type" in w for w in out.parse_warnings)

    def test_known_gpu_does_not_warn(self) -> None:
        out = parse_nsight_csv(_RAW, gpu_type="A100-PCIE-40GB")
        assert not any("unrecognised gpu_type" in w for w in out.parse_warnings)

    def test_wrong_gpu_changes_utilisation(self) -> None:
        # The foot-gun itself, pinned: same bytes, different card, different
        # verdict-driving number.
        a100 = parse_nsight_csv(_RAW, gpu_type="A100-PCIE-40GB")
        h100 = parse_nsight_csv(_RAW, gpu_type="H100-SXM")
        assert a100.decode is not None and h100.decode is not None
        assert a100.decode.hbm_bandwidth_util > h100.decode.hbm_bandwidth_util
