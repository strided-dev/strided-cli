"""CLI surface for the gait agent — the TLDR/verbose UX and clean plain-text output.

Driven through click's CliRunner against a real r03-firing replay fixture, with the
journal redirected to a tmp file so the suite never touches ``~/.strided``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.main import cli

_REPLAY_VLLM = Path(__file__).resolve().parents[1] / "fixtures" / "collect" / "replay" / "vllm" / "00.prom"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def launch(tmp_path) -> Path:
    p = tmp_path / "launch.txt"
    p.write_text(
        "python -m vllm.entrypoints.openai.api_server "
        "--model meta-llama/Llama-3-8B --block-size 16 --max-num-seqs 256\n"
    )
    return p


@pytest.fixture
def runner(tmp_path, monkeypatch) -> CliRunner:
    monkeypatch.setenv("STRIDED_GAIT_JOURNAL", str(tmp_path / "journal.jsonl"))
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")  # deterministic plain text under capture
    return CliRunner()


def _invoke(runner, launch, *args, input=None):
    return runner.invoke(
        cli,
        ["fix", "r03", "--vllm", str(_REPLAY_VLLM), "--config", str(launch), *args],
        input=input,
    )


def test_dry_run_states_tldr_and_changes_nothing(runner, launch):
    before = launch.read_text()
    result = _invoke(runner, launch, "--dry-run")
    assert result.exit_code == 0
    assert "▌ gait" in result.output
    assert "reducing --block-size 16 → 8" in result.output  # the TLDR claim
    assert "dry run, nothing was changed." in result.output
    assert launch.read_text() == before  # untouched


def test_plain_output_has_no_ansi_codes(runner, launch):
    result = _invoke(runner, launch, "--dry-run")
    assert not _ANSI.search(result.output), "plain mode must emit no ANSI escapes"


def test_verbose_narrates_each_step(runner, launch):
    result = _invoke(runner, launch, "--dry-run", "--verbose")
    assert result.exit_code == 0
    for marker in ("resolve", "propose", "predicted effect", "diagnosis:"):
        assert marker in result.output


def test_low_confidence_caveat_is_shown(runner, launch):
    result = _invoke(runner, launch, "--dry-run")
    assert "low-confidence (65%)" in result.output


def test_decline_changes_nothing(runner, launch):
    before = launch.read_text()
    result = _invoke(runner, launch, input="n\n")
    assert result.exit_code == 0
    assert "nothing changed" in result.output
    assert launch.read_text() == before


def test_approve_applies_then_verify_reports_no_change(runner, launch):
    # Re-collecting the same static dump shows no improvement → honest NO CHANGE,
    # then the rollback prompt restores the prior value (input: approve, then yes).
    result = _invoke(runner, launch, input="y\ny\n")
    assert result.exit_code == 0
    assert "applied" in result.output
    assert "verify: NO CHANGE" in result.output
    assert "rolled back" in result.output
    assert "--block-size 16" in launch.read_text()  # restored


def test_yes_flag_refused_for_low_confidence(runner, launch):
    before = launch.read_text()
    result = _invoke(runner, launch, "--yes")
    assert result.exit_code == 0
    assert "gait stopped" in result.output
    assert "auto-approval bar" in result.output
    assert launch.read_text() == before


def test_undo_cli_reverses_an_applied_change_cross_process(runner, launch):
    # The days-later path: apply in one invocation, undo in a separate one.
    # input="n\n" declines the inline NO_CHANGE rollback so the change stays applied.
    applied = _invoke(runner, launch, "--yes", "--threshold", "0.6", input="n\n")
    assert applied.exit_code == 0, applied.output
    assert "--block-size 8" in launch.read_text()

    result = runner.invoke(cli, ["undo", "--config", str(launch)])
    assert result.exit_code == 0, result.output
    assert "rolled back" in result.output
    assert "--block-size 16" in launch.read_text()  # restored

    # Nothing left to undo once the only applied change has been rolled back.
    again = runner.invoke(cli, ["undo", "--config", str(launch)])
    assert again.exit_code != 0
    assert "no applied change to undo" in again.output


def test_undo_cli_refuses_a_mismatched_config(runner, launch, tmp_path):
    # Undo must not restore a prior value into a file the change never touched.
    applied = _invoke(runner, launch, "--yes", "--threshold", "0.6", input="n\n")
    assert applied.exit_code == 0, applied.output

    other = tmp_path / "other.txt"
    other.write_text("python -m vllm --model X --block-size 32\n")
    result = runner.invoke(cli, ["undo", "--config", str(other)])

    assert result.exit_code != 0
    assert "was applied to" in result.output
    assert "--block-size 32" in other.read_text()        # the wrong file is untouched
    assert "--block-size 8" in launch.read_text()         # the real change stands


def test_ambiguous_config_abstains(runner, tmp_path):
    launch = tmp_path / "amb.txt"
    launch.write_text("python -m vllm --model X --block-size 16 --block-size 32\n")
    result = _invoke(runner, launch)
    assert result.exit_code == 0
    assert "gait stopped" in result.output
    assert "ambiguous" in result.output
    assert "16" in result.output and "32" in result.output


class TestVerdictColour:
    """The verdict table is built at import time, before any Styler exists.

    Three verdicts carry a fixed instrument colour; INSUFFICIENT_DATA carries a
    sentinel instead, because its dim tone depends on the terminal surface and is
    only known once a Styler has resolved one.
    """

    @staticmethod
    def _verified(verdict):
        from unittest.mock import Mock

        from gait.state import Verified

        return Verified(applied=Mock(), verdict=verdict, before={}, after={},
                        detail="a detail")

    @pytest.mark.parametrize("surface,expected", [("light", "#4D524F"), ("dark", "#727774")])
    def test_insufficient_data_dims_with_the_surface_tone(self, surface, expected):
        from cli import brand, gait_render
        from cli.style import Styler
        from gait import Verdict

        head = gait_render.verdict_block(
            Styler(True, surface=surface), self._verified(Verdict.INSUFFICIENT_DATA)
        ).splitlines()[0]
        r, g, b = brand.rgb(expected)
        assert f"\x1b[38;2;{r};{g};{b}m" in head
        assert "? verify: INSUFFICIENT DATA" in _ANSI.sub("", head)

    def test_other_verdicts_keep_their_instrument_colour(self):
        from cli import gait_render
        from cli.style import Styler
        from gait import Verdict

        st = Styler(True, surface="dark")
        for verdict, code in ((Verdict.CONFIRMED, "32"), (Verdict.NO_CHANGE, "33"),
                              (Verdict.INCONCLUSIVE, "36")):
            head = gait_render.verdict_block(st, self._verified(verdict)).splitlines()[0]
            assert f"\x1b[{code}m" in head

    def test_plain_mode_paints_no_verdict(self):
        from cli import gait_render
        from cli.style import Styler
        from gait import Verdict

        for verdict in Verdict:
            out = gait_render.verdict_block(Styler(False), self._verified(verdict))
            assert not _ANSI.search(out)
