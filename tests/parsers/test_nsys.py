"""Unit tests for parsers/nsight_systems.py (the nsys timeline parser).

The nsys NVTX push/pop CSV is r08's data source. The parser must: map range
*names* to phases (prefill/decode/mixed, mixed-first), convert the native time
unit to ms, pull optional token/seq counts from the name, sort by start time, and
degrade to an empty-steps timeline + a warning (never raise) on a benign export.
"""

from __future__ import annotations

import pytest

from parsers.nsight_systems import (
    classify_phase,
    parse_nsys_rep,
    parse_nsys_timeline_csv,
    parse_nsys_timeline_csv_file,
)
from schema import DiagnosisInput


# --------------------------------------------------------------------------- #
# Fixtures — nvtx_pushpop_trace shape (Start/End/Duration in ns + Name).
# --------------------------------------------------------------------------- #

# 10ms decode, 120ms prefill (carrying a token count), a mixed step, and a
# non-step range that must be ignored. Rows are out of start-time order on
# purpose, to exercise the sort.
_NVTX_NS = (
    '"Start (ns)","End (ns)","Duration (ns)","Name"\n'
    '"12000000","132000000","120000000","prefill tokens=8192"\n'
    '"0","10000000","10000000","decode"\n'
    '"132000000","145000000","13000000","mixed prefill_tokens=512 decode_seqs=24"\n'
    '"145000000","145500000","500000","cudaLaunchKernel"\n'  # non-step → skipped
)


def test_classify_phase_mixed_first() -> None:
    assert classify_phase("mixed_prefill_decode") == "mixed"
    assert classify_phase("Prefill step") == "prefill"
    assert classify_phase("DECODE") == "decode"
    assert classify_phase("cudaMemcpyAsync") is None


class TestParse:
    def test_phases_and_units(self) -> None:
        out = parse_nsys_timeline_csv(_NVTX_NS, model_name="m", gpu_type="H100-SXM")
        assert isinstance(out, DiagnosisInput)
        steps = out.nsys_timeline.steps
        # 3 step ranges; the cudaLaunchKernel row is dropped.
        assert [s.phase for s in steps] == ["decode", "prefill", "mixed"]
        # ns → ms conversion.
        assert steps[0].duration_ms == pytest.approx(10.0)
        assert steps[1].duration_ms == pytest.approx(120.0)
        # sorted by start_ms even though the prefill row came first in the file.
        assert steps[0].start_ms < steps[1].start_ms < steps[2].start_ms

    def test_token_and_seq_extraction(self) -> None:
        out = parse_nsys_timeline_csv(_NVTX_NS)
        prefill = out.nsys_timeline.steps[1]
        mixed = out.nsys_timeline.steps[2]
        assert prefill.num_prefill_tokens == 8192
        assert prefill.num_decode_seqs is None
        assert mixed.num_prefill_tokens == 512
        assert mixed.num_decode_seqs == 24

    def test_trace_duration_spans_steps(self) -> None:
        out = parse_nsys_timeline_csv(_NVTX_NS)
        # max end (145ms) - min start (0) = 145ms.
        assert out.nsys_timeline.trace_duration_ms == pytest.approx(145.0)

    def test_duration_fallback_when_no_end_column(self) -> None:
        text = (
            '"Start (ns)","Duration (ns)","Name"\n'
            '"0","10000000","decode"\n'
            '"10000000","120000000","prefill"\n'
        )
        out = parse_nsys_timeline_csv(text)
        assert [round(s.duration_ms) for s in out.nsys_timeline.steps] == [10, 120]

    def test_millisecond_unit_header(self) -> None:
        text = (
            '"Start (ms)","End (ms)","Name"\n'
            '"0.0","10.0","decode"\n'
            '"10.0","130.0","prefill"\n'
        )
        out = parse_nsys_timeline_csv(text)
        assert out.nsys_timeline.steps[1].duration_ms == pytest.approx(120.0)


class TestDegradation:
    def test_missing_name_column_warns(self) -> None:
        out = parse_nsys_timeline_csv('"Start (ns)","End (ns)"\n"0","1"\n')
        assert out.nsys_timeline.steps == []
        assert any("range-name column" in w for w in out.parse_warnings)

    def test_missing_timing_columns_warn(self) -> None:
        out = parse_nsys_timeline_csv('"Name"\n"decode"\n')
        assert out.nsys_timeline.steps == []
        assert any("Start" in w for w in out.parse_warnings)

    def test_no_step_ranges_warns(self) -> None:
        text = (
            '"Start (ns)","End (ns)","Name"\n'
            '"0","1000","cudaLaunchKernel"\n'
            '"1000","2000","cudaMemcpyAsync"\n'
        )
        out = parse_nsys_timeline_csv(text)
        assert out.nsys_timeline.steps == []
        assert any("no prefill/decode/mixed step ranges" in w for w in out.parse_warnings)

    def test_empty_csv_warns(self) -> None:
        out = parse_nsys_timeline_csv("")
        assert any("empty or invalid" in w for w in out.parse_warnings)

    def test_bad_rows_skipped_with_warning(self) -> None:
        text = (
            '"Start (ns)","End (ns)","Name"\n'
            '"0","10000000","decode"\n'
            '"abc","xyz","prefill"\n'          # unparseable numbers
            '"30000000","20000000","prefill"\n'  # end before start
        )
        out = parse_nsys_timeline_csv(text)
        assert [s.phase for s in out.nsys_timeline.steps] == ["decode"]
        assert any("unparseable step rows" in w for w in out.parse_warnings)

    def test_nan_cell_is_skipped_not_a_crash(self) -> None:
        # float() accepts a literal "nan"; pre-guard it crashed EngineStep's
        # ge=0 validator with an uncaught pydantic ValidationError.
        text = (
            '"Start (ns)","End (ns)","Name"\n'
            '"0","10000000","decode"\n'
            '"nan","5000000","decode"\n'
        )
        out = parse_nsys_timeline_csv(text)
        assert len(out.nsys_timeline.steps) == 1
        assert any("unparseable step rows" in w for w in out.parse_warnings)

    def test_inf_cell_is_skipped_not_poisoning_durations(self) -> None:
        # inf passes the end<start check (inf < inf is False); pre-guard it
        # entered the schema and made every downstream duration stat non-finite.
        text = (
            '"Start (ns)","End (ns)","Name"\n'
            '"0","10000000","decode"\n'
            '"20000000","inf","prefill tokens=4096"\n'
        )
        out = parse_nsys_timeline_csv(text)
        assert [s.phase for s in out.nsys_timeline.steps] == ["decode"]
        import math
        assert all(
            math.isfinite(s.start_ms) and math.isfinite(s.end_ms)
            for s in out.nsys_timeline.steps
        )


class TestEntryPoints:
    def test_rep_stub_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            parse_nsys_rep("whatever.nsys-rep")

    def test_file_wrapper_handles_bom(self, tmp_path) -> None:
        p = tmp_path / "trace.csv"
        p.write_text(_NVTX_NS, encoding="utf-8-sig")
        out = parse_nsys_timeline_csv_file(str(p), model_name="m", gpu_type="H100-SXM")
        assert out.source_files == [str(p)]
        assert len(out.nsys_timeline.steps) == 3
