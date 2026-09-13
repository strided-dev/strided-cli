"""The interactive workspace: binding a subject, and routing it to the right flag.

``home.py`` is the one place that adds behaviour on top of the click commands, so
the parts worth pinning are the ones that could silently do the wrong thing: what
kind of thing a bare ``use`` argument is, which flag reads it, and whether a bound
value is allowed to reach a given command. Everything else in there is I/O.

The routing tests are the important ones. A bound *file* must never be handed to
``watch --vllm``, which wants a URL, and a bound *URL* must never reach
``diagnose --vllm``, which wants a path on disk. That decision is derived from the
live click parameter rather than a table, so these tests are really asserting that
the derivation still holds.
"""

from __future__ import annotations

import click
import pytest

from cli import lastrun
from cli.home import (
    Session,
    Slot,
    _accepts,
    _classify,
    _Completer,
    _fill,
    _flag_for,
    _params,
    _suggest,
    render_home,
)
from cli.main import cli
from cli.style import Styler


@pytest.fixture
def ctx() -> click.Context:
    return click.Context(cli, info_name="strided")


@pytest.fixture
def dump(tmp_path):
    p = tmp_path / "metrics.prom"
    p.write_text("# empty\n")
    return p


# --------------------------------------------------------------------------- #
# What did the user just point at?
# --------------------------------------------------------------------------- #

class TestClassify:
    def test_file(self, dump) -> None:
        assert _classify(str(dump)) == Slot(str(dump), "file")

    def test_directory(self, tmp_path) -> None:
        assert _classify(str(tmp_path)) == Slot(str(tmp_path), "dir")

    def test_url(self) -> None:
        got = _classify("http://localhost:8000/metrics")
        assert got == Slot("http://localhost:8000/metrics", "url")

    def test_url_is_not_probed_on_disk(self) -> None:
        # A URL is a URL even though nothing by that name exists locally.
        assert _classify("https://example.invalid/metrics").kind == "url"

    def test_missing_path_is_not_classified(self, tmp_path) -> None:
        assert _classify(str(tmp_path / "nope.prom")) is None


class TestFlagInference:
    @pytest.mark.parametrize("name,expected", [
        ("m.prom", "--vllm"),
        ("m.txt", "--vllm"),
        ("d.json", "--dcgm"),
        ("n.csv", "--nsight"),
        ("n.ncu-rep", "--nsight"),
    ])
    def test_extension_picks_the_reader(self, tmp_path, name, expected) -> None:
        p = tmp_path / name
        p.write_text("x")
        assert _flag_for(_classify(str(p))) == expected

    def test_directory_is_a_replay(self, tmp_path) -> None:
        assert _flag_for(_classify(str(tmp_path))) == "--replay"

    def test_url_is_a_live_vllm_endpoint(self) -> None:
        assert _flag_for(_classify("http://localhost:8000/metrics")) == "--vllm"

    def test_unknown_extension_needs_an_explicit_flag(self, tmp_path) -> None:
        p = tmp_path / "capture.bin"
        p.write_text("x")
        assert _flag_for(_classify(str(p))) is None


# --------------------------------------------------------------------------- #
# Routing a binding to a command
# --------------------------------------------------------------------------- #

class TestAccepts:
    def test_file_binding_fits_diagnose_but_not_watch(self) -> None:
        diagnose_vllm = _params(cli.commands["diagnose"])["--vllm"]
        watch_vllm = _params(cli.commands["watch"])["--vllm"]
        assert _accepts(diagnose_vllm, "file") is True
        assert _accepts(watch_vllm, "file") is False

    def test_url_binding_fits_watch_but_not_diagnose(self) -> None:
        diagnose_vllm = _params(cli.commands["diagnose"])["--vllm"]
        watch_vllm = _params(cli.commands["watch"])["--vllm"]
        assert _accepts(watch_vllm, "url") is True
        assert _accepts(diagnose_vllm, "url") is False

    def test_directory_binding_fits_replay(self) -> None:
        replay = _params(cli.commands["watch"])["--replay"]
        assert _accepts(replay, "dir") is True
        assert _accepts(replay, "file") is False


class TestFill:
    def _session(self, dump, tmp_path):
        cfg = tmp_path / "launch.txt"
        cfg.write_text("python -m vllm --block-size 16\n")
        return (Session.empty()
                .bind("--vllm", Slot(str(dump), "file"))
                .bind("--gpu", Slot("H100-SXM", "text"))
                .bind("--config", Slot(str(cfg), "file")))

    def test_bound_flags_are_added(self, dump, tmp_path) -> None:
        session = self._session(dump, tmp_path)
        argv, added, notes = _fill(cli.commands["fix"], session, ["r03"])
        assert set(added) == {"--vllm", "--gpu", "--config"}
        assert argv[0] == "r03" and "--vllm" in argv and notes == []

    def test_a_typed_flag_wins_over_the_binding(self, dump, tmp_path) -> None:
        session = self._session(dump, tmp_path)
        argv, added, _ = _fill(cli.commands["fix"], session, ["r03", "--gpu", "A100-80G"])
        assert "--gpu" not in added
        assert argv.count("--gpu") == 1
        assert "H100-SXM" not in argv

    def test_a_flag_the_command_lacks_is_not_a_note(self, dump, tmp_path) -> None:
        # diagnose has no --config. That is ordinary, not a mismatch worth reporting.
        session = self._session(dump, tmp_path)
        _, added, notes = _fill(cli.commands["diagnose"], session, [])
        assert "--config" not in added
        assert notes == []

    def test_an_incompatible_binding_is_reported_not_dropped(self, dump, tmp_path) -> None:
        session = self._session(dump, tmp_path)
        argv, added, notes = _fill(cli.commands["watch"], session, [])
        assert "--vllm" not in added and str(dump) not in argv
        assert len(notes) == 1 and "--vllm" in notes[0] and "watch" in notes[0]

    def test_url_binding_reaches_watch(self, tmp_path) -> None:
        session = Session.empty().bind("--vllm", Slot("http://h:8000/metrics", "url"))
        argv, added, notes = _fill(cli.commands["watch"], session, [])
        assert added == ["--vllm"] and "http://h:8000/metrics" in argv and notes == []


# --------------------------------------------------------------------------- #
# The prompt's sense of place
# --------------------------------------------------------------------------- #

class TestLabel:
    def test_unbound(self) -> None:
        assert Session.empty().label == "~"

    def test_file_shows_its_stem(self, dump) -> None:
        assert Session.empty().bind("--vllm", Slot(str(dump), "file")).label == "metrics"

    def test_url_shows_its_host(self) -> None:
        s = Session.empty().bind("--vllm", Slot("http://localhost:8000/metrics", "url"))
        assert s.label == "localhost:8000"

    def test_replay_wins_over_other_bindings(self, dump, tmp_path) -> None:
        replay = tmp_path / "replay"
        replay.mkdir()
        s = (Session.empty()
             .bind("--vllm", Slot(str(dump), "file"))
             .bind("--replay", Slot(str(replay), "dir")))
        assert s.label == "replay"

    def test_a_long_name_is_truncated_visibly(self, tmp_path) -> None:
        long = tmp_path / ("a" * 60 + ".prom")
        long.write_text("x")
        label = Session.empty().bind("--vllm", Slot(str(long), "file")).label
        assert len(label) == 28 and label.endswith("…")

    def test_config_alone_is_not_a_subject(self, tmp_path) -> None:
        # A launch file is something you act *with*, not the thing you are looking at.
        cfg = tmp_path / "launch.txt"
        cfg.write_text("x")
        assert Session.empty().bind("--config", Slot(str(cfg), "file")).label == "~"


# --------------------------------------------------------------------------- #
# Completion and forgiveness
# --------------------------------------------------------------------------- #

class TestCompletion:
    def _hits(self, ctx, tokens, text):
        c = _Completer(ctx)
        return [x for x in c._candidates(tokens, text) if x.startswith(text)]

    def test_bare_line_offers_commands_and_session_verbs(self, ctx) -> None:
        hits = self._hits(ctx, [], "")
        assert {"diagnose", "watch", "fix", "undo"} <= set(hits)
        assert {"use", "unset", "guide"} <= set(hits)

    def test_flags_come_from_the_live_command(self, ctx) -> None:
        hits = self._hits(ctx, ["diagnose"], "--")
        assert "--vllm" in hits and "--nsys" in hits
        assert "--config" not in hits  # diagnose has no --config, and never claims one

    def test_paths_complete_after_a_path_flag(self, ctx, tmp_path, monkeypatch) -> None:
        (tmp_path / "capture.prom").write_text("x")
        monkeypatch.chdir(tmp_path)
        assert self._hits(ctx, ["diagnose", "--vllm"], "cap") == ["capture.prom"]

    def test_fix_completes_the_rules_that_actually_fired(self, ctx) -> None:
        lastrun.record(["r02", "r03"])
        try:
            assert self._hits(ctx, ["fix"], "") == ["r02", "r03"]
        finally:
            lastrun.record([])

    def test_fix_falls_back_to_the_registry_before_any_run(self, ctx) -> None:
        lastrun.record([])
        hits = self._hits(ctx, ["fix"], "r0")
        assert "r01" in hits and len(hits) > 2

    def test_unknown_command_offers_nothing(self, ctx) -> None:
        assert self._hits(ctx, ["nonsense"], "") == []


class TestSuggest:
    @pytest.mark.parametrize("typo,expected", [
        ("diagnos", "diagnose"), ("fxi", "fix"), ("wathc", "watch"), ("und", "undo"),
    ])
    def test_near_misses_are_named(self, typo, expected) -> None:
        assert _suggest(typo, ["diagnose", "watch", "fix", "undo", "use"]) == expected

    def test_nothing_close_stays_quiet(self) -> None:
        assert _suggest("qqqq", ["diagnose", "watch", "fix", "undo"]) is None


# --------------------------------------------------------------------------- #
# The non-interactive path
# --------------------------------------------------------------------------- #

class TestStaticScreen:
    def test_plain_screen_has_no_ansi(self) -> None:
        assert "\x1b[" not in render_home(Styler(False), 80)

    def test_screen_leads_with_how_to_bind_a_workload(self) -> None:
        out = render_home(Styler(False), 80)
        assert "use examples/vllm_kv_fragmentation.prom" in out
        assert "tab completes" in out
