"""Unit tests for parsers/dcgm.py.

DCGM is the cluster-level source. The parser must accept both the ``dcgmi dmon``
dict format and the ``dcgm-exporter`` list format, normalise percentages,
aggregate across ranks, and never crash on malformed JSON — it degrades to a
warning instead. These tests pin all of that plus the roofline heuristic.
"""

from __future__ import annotations

import json

import pytest

from parsers.dcgm import parse_dcgm_json, parse_dcgm_json_file, parse_dcgm_prometheus
from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

# dmon: two ranks, profiling counters present (already [0,1]).
_DMON = json.dumps({
    "DCGM_FI_PROF_DRAM_ACTIVE": {"0": 0.91, "1": 0.88},
    "DCGM_FI_PROF_SM_OCCUPANCY": {"0": 0.20, "1": 0.25},
    "DCGM_FI_DEV_GPU_UTIL": {"0": 95, "1": 80},
})

# exporter: list of metric objects with gpu labels.
_EXPORTER = json.dumps([
    {"name": "DCGM_FI_PROF_DRAM_ACTIVE", "labels": {"gpu": "0"}, "value": 0.90},
    {"name": "DCGM_FI_PROF_SM_OCCUPANCY", "labels": {"gpu": "0"}, "value": 0.30},
    {"name": "DCGM_FI_PROF_DRAM_ACTIVE", "labels": {"gpu": "1"}, "value": 0.92},
    {"name": "DCGM_FI_PROF_SM_OCCUPANCY", "labels": {"gpu": "1"}, "value": 0.28},
])


# --------------------------------------------------------------------------- #
# Format detection
# --------------------------------------------------------------------------- #

class TestFormatDetection:
    def test_dmon_dict_format(self) -> None:
        out = parse_dcgm_json(_DMON, gpu_type="H100-SXM")
        assert isinstance(out, DiagnosisInput)
        assert out.num_gpus == 2
        assert out.tensor_parallel_size == 2

    def test_exporter_list_format(self) -> None:
        out = parse_dcgm_json(_EXPORTER, gpu_type="H100-SXM")
        assert out.num_gpus == 2
        assert out.decode.hbm_bandwidth_util == pytest.approx(0.91)  # mean(0.90, 0.92)

    def test_numeric_string_field_id_maps(self) -> None:
        # "203" is the numeric DCGM_FI_DEV_GPU_UTIL id as a JSON string key.
        out = parse_dcgm_json(json.dumps({"203": {"0": 50, "1": 70}}))
        # gpu_util 60% → sm_occupancy fallback 0.60
        assert out.decode.sm_occupancy == pytest.approx(0.60)


# --------------------------------------------------------------------------- #
# Aggregation & normalisation
# --------------------------------------------------------------------------- #

class TestAggregation:
    def test_hbm_prefers_dram_active(self) -> None:
        out = parse_dcgm_json(_DMON)
        assert out.decode.hbm_bandwidth_util == pytest.approx((0.91 + 0.88) / 2)

    def test_sm_prefers_prof_occupancy(self) -> None:
        out = parse_dcgm_json(_DMON)
        assert out.decode.sm_occupancy == pytest.approx((0.20 + 0.25) / 2)

    def test_mem_copy_util_fallback_is_normalised(self) -> None:
        # No DRAM_ACTIVE → fall back to MEM_COPY_UTIL (a 0-100 percentage).
        data = json.dumps({"DCGM_FI_DEV_MEM_COPY_UTIL": {"0": 70, "1": 90}})
        out = parse_dcgm_json(data)
        assert out.decode.hbm_bandwidth_util == pytest.approx(0.80)  # mean(70,90)/100

    def test_no_sm_clocks_from_util_only(self) -> None:
        out = parse_dcgm_json(_DMON)
        # util-only (no SM clocks): the parser captures no per-rank slowness signal.
        # gpu_util is not used (ambiguous straggler direction under the NCCL barrier),
        # so r04 has nothing to read and abstains rather than risk an inverted call.
        assert out.tp_rank_sm_clocks is None


# --------------------------------------------------------------------------- #
# Per-rank slowness telemetry — RAW SM clock + temperature capture.
# The slowness *ratio* and outlier ranking are derived in r04, not the parser.
# --------------------------------------------------------------------------- #

class TestTpRankSignal:
    def test_raw_clocks_and_temps_captured(self) -> None:
        # The parser surfaces the raw per-rank clocks and temps as-is; r04 derives
        # the slowness ratio (max_clock / clock) from them.
        data = json.dumps({
            "DCGM_FI_DEV_SM_CLOCK": {"0": 2000, "1": 1500, "2": 2000},
            "DCGM_FI_DEV_GPU_TEMP": {"0": 60, "1": 75, "2": 61},
            "DCGM_FI_DEV_GPU_UTIL": {"0": 99, "1": 99, "2": 99},
        })
        out = parse_dcgm_json(data)
        assert out.tp_rank_sm_clocks == [2000.0, 1500.0, 2000.0]
        assert out.tp_rank_temps == [60.0, 75.0, 61.0]

    def test_clocks_captured_when_util_present(self) -> None:
        # gpu_util is not a competing slowness signal; only the raw clocks are kept.
        data = json.dumps({
            "DCGM_FI_DEV_SM_CLOCK": {"0": 2000, "1": 2000, "2": 1400},
            "DCGM_FI_DEV_GPU_UTIL": {"0": 100, "1": 80, "2": 80},
        })
        out = parse_dcgm_json(data)
        assert out.tp_rank_sm_clocks == [2000.0, 2000.0, 1400.0]

    def test_no_clocks_from_util_only(self) -> None:
        out = parse_dcgm_json(_DMON)  # util-only, no SM clock
        assert out.tp_rank_sm_clocks is None
        assert out.tp_rank_temps is None

    def test_partial_clock_coverage_ignored(self) -> None:
        # Clock present for only some ranks → misaligned, so it is dropped (rank
        # indices must stay aligned); the parser surfaces no clocks.
        data = json.dumps({
            "DCGM_FI_DEV_SM_CLOCK": {"0": 2000},               # rank 1 missing
            "DCGM_FI_DEV_GPU_UTIL": {"0": 90, "1": 95},
        })
        out = parse_dcgm_json(data)
        assert out.tp_rank_sm_clocks is None                    # incomplete → not used


class TestRoofline:
    def test_memory_bound(self) -> None:
        out = parse_dcgm_json(_DMON)  # hbm ~0.895 > 0.85, sm ~0.225 < 0.4
        assert out.decode.roofline_position == "memory_bound"

    def test_compute_bound(self) -> None:
        data = json.dumps({
            "DCGM_FI_PROF_DRAM_ACTIVE": {"0": 0.30},
            "DCGM_FI_PROF_SM_OCCUPANCY": {"0": 0.80},
        })
        out = parse_dcgm_json(data)
        assert out.decode.roofline_position == "compute_bound"

    def test_balanced(self) -> None:
        data = json.dumps({
            "DCGM_FI_PROF_DRAM_ACTIVE": {"0": 0.70},
            "DCGM_FI_PROF_SM_OCCUPANCY": {"0": 0.65},
        })
        out = parse_dcgm_json(data)
        assert out.decode.roofline_position == "balanced"


# --------------------------------------------------------------------------- #
# Phase placement contract
# --------------------------------------------------------------------------- #

class TestPhasePlacement:
    def test_aggregate_goes_to_decode_with_warning(self) -> None:
        out = parse_dcgm_json(_DMON)
        assert out.decode is not None
        assert out.prefill is None
        assert any("does not distinguish prefill from decode" in w for w in out.parse_warnings)


# --------------------------------------------------------------------------- #
# Malformed / degenerate input
# --------------------------------------------------------------------------- #

class TestDegenerate:
    def test_invalid_json_degrades_to_warning(self) -> None:
        out = parse_dcgm_json("not valid json {[")
        assert isinstance(out, DiagnosisInput)
        assert any("JSON decode failed" in w for w in out.parse_warnings)
        assert out.decode is None

    def test_non_container_json_warns(self) -> None:
        out = parse_dcgm_json("42")
        assert any("unrecognised JSON structure" in w for w in out.parse_warnings)

    def test_unrecognised_fields_warn(self) -> None:
        out = parse_dcgm_json(json.dumps({"DCGM_FI_DEV_SOMETHING_ELSE": {"0": 1}}))
        assert any("no recognised fields" in w for w in out.parse_warnings)

    def test_file_wrapper_labels_source(self, tmp_path) -> None:
        p = tmp_path / "dcgm.json"
        p.write_text(_DMON)
        out = parse_dcgm_json_file(str(p))
        assert out.source_files == [str(p)]
        assert out.num_gpus == 2


# --------------------------------------------------------------------------- #
# Prometheus-text path (dcgm-exporter) — parity with the JSON path
# --------------------------------------------------------------------------- #

class TestPrometheusParity:
    """dcgm-exporter serves Prometheus text; it must produce the same
    DiagnosisInput as the equivalent exporter-JSON, since both feed the shared
    _build_dcgm_input."""

    def test_prometheus_matches_json_field_for_field(self) -> None:
        # 4 GPUs, PROF fields + gpu_util with one straggler (high util).
        fields = {
            "DCGM_FI_PROF_SM_OCCUPANCY": [0.15, 0.16, 0.14, 0.15],
            "DCGM_FI_PROF_DRAM_ACTIVE": [0.88, 0.90, 0.87, 0.89],
            "DCGM_FI_DEV_GPU_UTIL": [95, 80, 82, 81],
            "DCGM_FI_DEV_SM_CLOCK": [1980, 1975, 1400, 1982],
        }
        json_data = {name: {str(i): v for i, v in enumerate(vals)}
                     for name, vals in fields.items()}
        prom_lines = [
            f'{name}{{gpu="{i}"}} {v}'
            for name, vals in fields.items()
            for i, v in enumerate(vals)
        ]

        j = parse_dcgm_json(json.dumps(json_data))
        p = parse_dcgm_prometheus("\n".join(prom_lines))

        assert p.num_gpus == j.num_gpus == 4
        assert p.decode.sm_occupancy == j.decode.sm_occupancy
        assert p.decode.hbm_bandwidth_util == j.decode.hbm_bandwidth_util
        assert p.decode.roofline_position == j.decode.roofline_position == "memory_bound"
        assert p.tp_rank_sm_clocks == j.tp_rank_sm_clocks == [1980.0, 1975.0, 1400.0, 1982.0]

    def test_prometheus_empty_degrades_to_warning(self) -> None:
        out = parse_dcgm_prometheus("# only comments\n")
        assert out.decode is None
        assert any("no recognised" in w for w in out.parse_warnings)

    def test_prometheus_memory_bound_snapshot_fires_r01(self) -> None:
        # The live form must carry the fields r01 needs into decode.*
        text = "\n".join(
            f'DCGM_FI_PROF_SM_OCCUPANCY{{gpu="{i}"}} 0.15\n'
            f'DCGM_FI_PROF_DRAM_ACTIVE{{gpu="{i}"}} 0.88'
            for i in range(2)
        )
        out = parse_dcgm_prometheus(text)
        assert out.decode.sm_occupancy == pytest.approx(0.15)
        assert out.decode.hbm_bandwidth_util == pytest.approx(0.88)
