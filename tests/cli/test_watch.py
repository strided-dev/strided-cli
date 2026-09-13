"""End-to-end tests for `strided watch` via the click CliRunner.

Driven entirely off the offline replay fixtures, so no live server is needed.
Under CliRunner stdout is not a TTY, so the rewritable status line is suppressed
and only the event blocks (and banner/summary) appear — which is exactly what we
assert on.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from cli.main import _natural_key, _run_watch_loop, cli
from collect.session import TickResult

_REPLAY = Path(__file__).resolve().parents[1] / "fixtures" / "collect" / "replay"


def test_watch_replay_fires_all_three_and_emits_on_change() -> None:
    result = CliRunner().invoke(cli, ["watch", "--replay", str(_REPLAY), "--interval", "0"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Two CHANGED events: tick 0 (all three fire), tick 2 (r03 clears). Tick 1 is
    # identical state → quiet.
    assert out.count("state changed") == 2
    assert "firing: r01, r02, r03" in out
    assert "firing: r01, r02\n" in out

    # The flagship r01 came from the paired DCGM replay stream — offline.
    assert "Decode memory-bound at low batch" in out
    assert "KV cache fragmentation" in out  # r03 was diagnosed at least once


def test_watch_summary_is_printed() -> None:
    result = CliRunner().invoke(cli, ["watch", "--replay", str(_REPLAY), "--interval", "0"])
    assert "watch stopped. Last state:" in result.output


def test_watch_requires_a_source() -> None:
    result = CliRunner().invoke(cli, ["watch"])
    assert result.exit_code != 0
    assert "at least one" in result.output


def test_watch_rejects_negative_interval() -> None:
    result = CliRunner().invoke(cli, ["watch", "--replay", str(_REPLAY), "--interval", "-1"])
    assert result.exit_code != 0
    assert "non-negative" in result.output


def test_watch_rejects_replay_combined_with_live() -> None:
    result = CliRunner().invoke(
        cli, ["watch", "--replay", str(_REPLAY), "--vllm", "http://localhost:8000/metrics"])
    assert result.exit_code != 0
    assert "cannot be combined" in result.output


class _StubSession:
    """Returns a fixed TickResult every tick — drives _run_watch_loop directly."""

    def __init__(self, result: TickResult) -> None:
        self._result = result

    def tick(self, _now: float) -> TickResult:
        return self._result


def test_persistent_warning_echoes_once(capsys) -> None:
    # A down source re-records the same last_error every poll; the loop must
    # surface it once, not flood one line per tick.
    warning = "[dcgm] failed to scrape http://x: connection refused"
    session = _StubSession(
        TickResult(merged=None, report=None, changed=False,
                   warnings=(warning,), exhausted=False)
    )
    _run_watch_loop(session, interval=0, max_ticks=3,
                    use_color=False, interactive=False)
    out = capsys.readouterr().out
    assert out.count("connection refused") == 1


def test_natural_key_orders_numeric_filenames() -> None:
    paths = [Path("d/10.prom"), Path("d/2.prom"), Path("d/1.prom")]
    assert [p.name for p in sorted(paths, key=_natural_key)] == ["1.prom", "2.prom", "10.prom"]


def test_replay_flat_vllm_beside_dcgm_subdir_fires_all_three(tmp_path) -> None:
    # vLLM captures at the top level, DCGM in a subdir: both streams must load.
    (tmp_path / "00.prom").write_text((_REPLAY / "vllm" / "00.prom").read_text())
    (tmp_path / "dcgm").mkdir()
    (tmp_path / "dcgm" / "00.prom").write_text((_REPLAY / "dcgm" / "00.prom").read_text())

    result = CliRunner().invoke(cli, ["watch", "--replay", str(tmp_path), "--interval", "0"])
    assert result.exit_code == 0, result.output
    assert "firing: r01, r02, r03" in result.output


def test_watch_vllm_only_flags_dcgm_not_connected(tmp_path) -> None:
    # A replay dir with only a vLLM stream: r01 cannot fire, banner says so.
    vdir = tmp_path / "vllm"
    vdir.mkdir()
    (vdir / "00.prom").write_text((_REPLAY / "vllm" / "00.prom").read_text())

    result = CliRunner().invoke(cli, ["watch", "--replay", str(tmp_path), "--interval", "0"])
    assert result.exit_code == 0, result.output
    assert "not connected" in result.output
    # r01 does not fire (no DCGM); it is surfaced honestly with the fields it needs,
    # which together with the banner tells the user exactly what to connect.
    assert "firing: r02, r03" in result.output
    assert "r01 Decode memory-bound at low batch, needs decode" in result.output
