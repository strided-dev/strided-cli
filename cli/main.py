"""strided CLI entry point.

Usage:
    python -m cli diagnose --vllm metrics.prom --dcgm dcgm.json --nsight report.csv

The CLI is plumbing only:

    raw files → parsers → [DiagnosisInput, ...] → merge → engine → format → stdout

It never invents data, never re-runs rules, never persists. The engine is the
source of truth for diagnoses; the formatter is the source of truth for layout.
Two halves, one direction, no cross-talk.
"""

from __future__ import annotations

import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from cli import CLI_VERSION, lastrun
from cli.render import logo_banner, render_live_event, render_report, render_status_line
from cli.style import Styler, should_color
from collect import tiers
from collect.session import WatchSession
from collect.sources import DcgmSource, ReplaySource, VllmSource
from engine import run_diagnosis
from engine.registry import ALL_RULES, rule_ids
from parsers.dcgm import parse_dcgm_json_file, parse_dcgm_prometheus_file
from parsers.nsight import parse_nsight_csv_file, parse_nsight_ncu_rep
from parsers.nsight_systems import parse_nsys_rep, parse_nsys_timeline_csv_file
from parsers.vllm import parse_vllm_metrics_file
from schema import DiagnosisInput
from schema.merge import merge_inputs

# NOTE: gait is imported lazily inside `fix`/`undo` (not at module scope), so the
# `strided` entry point, `diagnose`, `watch`, and the workspace never depend on the
# gait package being importable. Only the two commands that actually drive the agent
# pull it in. cli/gait_render imports gait too, so it is imported lazily for the same
# reason.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# \b keeps click from rewrapping the block, so the annotation column stays aligned.
_GROUP_EPILOG = """\b
Examples:
  strided diagnose --vllm metrics.prom --gpu H100-SXM   observe a capture
  strided watch --replay examples/replay                observe over time
  strided fix r03 --config launch.txt                   adjust, then verify
  strided undo --config launch.txt                      go back

Run `strided` with no command to open the workspace.
"""


@click.group(invoke_without_command=True, epilog=_GROUP_EPILOG)
@click.version_option(CLI_VERSION, prog_name="strided")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """strided: local model hosting, tuned live.

    strided hosts models on hardware you control and tunes them while they run:
    observe the workload, adjust within the limits you set, verify the result.
    This CLI is the tuning half, working against a model server you already run.

    Run with no command to open the interactive workspace.
    """
    if ctx.invoked_subcommand is None:
        from cli.home import run_home

        run_home(ctx)


@cli.command(epilog="""\b
Examples:
  strided diagnose --vllm examples/vllm_kv_fragmentation.prom --gpu H100-SXM
  strided diagnose --nsight examples/membound.csv --gpu H100-SXM
""")
@click.option("--vllm", "vllm_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a vLLM /metrics dump (.prom or .txt).")
@click.option("--vllm-baseline", "vllm_baseline_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional earlier vLLM /metrics scrape. When given, cumulative "
                   "metrics are differenced against it so pressure/fragmentation "
                   "signals reflect the current window, not the server's lifetime.")
@click.option("--dcgm", "dcgm_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a DCGM JSON dump.")
@click.option("--nsight", "nsight_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an Nsight Compute export (.csv or .ncu-rep).")
@click.option("--nsys", "nsys_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an Nsight Systems timeline export (.csv from "
                   "`nsys stats --report nvtx_pushpop_trace`, or .nsys-rep). Enables r08.")
@click.option("--model", "model_name", default=None,
              help="Model name override, e.g. 'meta-llama/Llama-3-70B'.")
@click.option("--gpu", "gpu_type", default=None,
              help="GPU type override, e.g. 'H100-SXM', 'A100-80G'.")
@click.option("--strict", is_flag=True, default=False,
              help="Re-raise parser errors, rule errors, and non-finite inputs "
                   "instead of degrading gracefully.")
@click.option("--color/--no-color", "color_flag", default=None,
              help="Force coloured output on or off. Default: auto-detect "
                   "(colour when stdout is an interactive terminal).")
def diagnose(
    vllm_path: Optional[Path],
    vllm_baseline_path: Optional[Path],
    dcgm_path: Optional[Path],
    nsight_path: Optional[Path],
    nsys_path: Optional[Path],
    model_name: Optional[str],
    gpu_type: Optional[str],
    strict: bool,
    color_flag: Optional[bool],
) -> None:
    """Observe a workload from one or more captured dumps, and say what is limiting it."""
    if not any([vllm_path, dcgm_path, nsight_path, nsys_path]):
        raise click.UsageError("Provide at least one of --vllm / --dcgm / --nsight / --nsys.")
    if vllm_baseline_path is not None and vllm_path is None:
        raise click.UsageError(
            "--vllm-baseline requires --vllm (it is the earlier of the two scrapes)."
        )

    t0 = time.perf_counter()
    inputs = _load_sources(
        vllm_path, vllm_baseline_path, dcgm_path, nsight_path,
        model_name, gpu_type, nsys_path=nsys_path, strict=strict,
    )
    merged = merge_inputs(inputs, model_name=model_name, gpu_type=gpu_type)
    report = run_diagnosis(merged, strict=strict)
    elapsed = time.perf_counter() - t0
    lastrun.record(r.diagnosis.rule_id for r in report.diagnoses)

    use_color = should_color(sys.stdout, color_flag)
    text = render_report(merged, report, color=use_color, elapsed_s=elapsed, one_shot=True)
    click.echo(text, nl=False, color=use_color)


# ---------------------------------------------------------------------------
# watch — continuous live diagnosis
# ---------------------------------------------------------------------------
#
# Same pipeline as diagnose, on a loop: poll endpoints (or replay captures) →
# parse → merge → window throughput → engine → render. The engine, parsers, and
# merge are reused verbatim; collect/ owns the loop-specific machinery. Output is
# an event log — a full block only when the diagnosis state changes — with a
# single rewritable status line between changes on an interactive terminal.

# Floor for live polling: interval 0 (or a sub-tick value) against a real endpoint
# is a busy loop hammering the server. Replay has no such risk — it ends on stream
# exhaustion, not the clock — so the floor applies only to the live path.
_MIN_LIVE_INTERVAL = 0.5


def _banner_width() -> int:
    """Width for the watch wordmark rule — capped so it never sprawls."""
    return min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)


@cli.command(epilog="""\b
Examples:
  strided watch --replay examples/replay --max-ticks 3 --interval 0
  strided watch --vllm http://localhost:8000/metrics --interval 5
""")
@click.option("--vllm", "vllm_url", default=None,
              help="vLLM /metrics endpoint URL to poll, e.g. http://localhost:8000/metrics.")
@click.option("--dcgm", "dcgm_url", default=None,
              help="dcgm-exporter /metrics endpoint URL to poll (enables r01 live).")
@click.option("--replay", "replay_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Replay captured scrapes from a directory (vllm/*.prom, optional dcgm/*). "
                   "Offline demo, no live server needed.")
@click.option("--interval", default=5.0, show_default=True, type=float,
              help="Seconds between polls.")
@click.option("--model", "model_name", default=None, help="Model name override.")
@click.option("--gpu", "gpu_type", default=None, help="GPU type override.")
@click.option("--max-ticks", type=int, default=None,
              help="Stop after N ticks. For non-interactive use and tests; a replay "
                   "stream also ends the loop when exhausted.")
@click.option("--color/--no-color", "color_flag", default=None,
              help="Force coloured output on or off. Default: auto-detect.")
def watch(
    vllm_url: Optional[str],
    dcgm_url: Optional[str],
    replay_dir: Optional[Path],
    interval: float,
    model_name: Optional[str],
    gpu_type: Optional[str],
    max_ticks: Optional[int],
    color_flag: Optional[bool],
) -> None:
    """Observe a running workload continuously by polling its metrics endpoints."""
    if not any([vllm_url, dcgm_url, replay_dir]):
        raise click.UsageError("Provide at least one of --vllm / --dcgm / --replay.")
    if replay_dir is not None and (vllm_url or dcgm_url):
        raise click.UsageError(
            "--replay is an offline mode and cannot be combined with the live "
            "--vllm / --dcgm endpoints."
        )
    if interval < 0:
        raise click.UsageError("--interval must be non-negative.")

    sources = _build_watch_sources(vllm_url, dcgm_url, replay_dir, model_name, gpu_type)
    use_color = should_color(sys.stdout, color_flag)
    interactive = sys.stdout.isatty()
    have_dcgm = dcgm_url is not None or (replay_dir is not None and (replay_dir / "dcgm").is_dir())

    st = Styler(use_color)
    for line in logo_banner(st, _banner_width()):
        click.echo(line, color=use_color)
    click.echo()

    is_live = vllm_url is not None or dcgm_url is not None
    if is_live and interval < _MIN_LIVE_INTERVAL:
        notice = (f"--interval {interval:g}s is below the live floor; "
                  f"polling every {_MIN_LIVE_INTERVAL:g}s instead.")
        click.echo("  " + st.warn("! ") + (st.dim(notice) if use_color else notice))
        interval = _MIN_LIVE_INTERVAL

    click.echo(tiers.startup_banner(have_dcgm, rule_ids()), color=use_color)
    click.echo()

    session = WatchSession(sources, model_name=model_name, gpu_type=gpu_type)
    _run_watch_loop(
        session, interval=interval, max_ticks=max_ticks,
        use_color=use_color, interactive=interactive,
    )


def _build_watch_sources(
    vllm_url: Optional[str],
    dcgm_url: Optional[str],
    replay_dir: Optional[Path],
    model_name: Optional[str],
    gpu_type: Optional[str],
) -> list:
    """Construct the live/replay sources for one watch session, in merge order."""
    model = model_name or "unknown"
    gpu = gpu_type or "unknown"
    sources: list = []
    if replay_dir is not None:
        sources.extend(_replay_sources(replay_dir))
    if vllm_url is not None:
        sources.append(VllmSource(vllm_url, model, gpu))
    if dcgm_url is not None:
        sources.append(DcgmSource(dcgm_url, model, gpu))
    return sources


def _dcgm_replay_parse(path_str: str) -> DiagnosisInput:
    """Parse a replayed DCGM capture, dispatching on extension (.json vs text)."""
    if path_str.endswith(".json"):
        return parse_dcgm_json_file(path_str)
    return parse_dcgm_prometheus_file(path_str)


def _natural_key(path: Path) -> list:
    """Sort key that orders numeric filename runs by value, not lexically.

    So a capture named ``2.prom`` precedes ``10.prom`` (and ISO-timestamped names
    still sort chronologically). Replay order is temporal: the wrong order would
    corrupt the windowed throughput deltas and the change-detection sequence.
    """
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", path.name)]


def _replay_sources(replay_dir: Path) -> list:
    """Build vLLM (and optional DCGM) replay streams from a captured directory.

    Convention: ``<dir>/vllm/*.prom`` is the vLLM stream and ``<dir>/dcgm/*`` the
    paired DCGM stream (so the offline demo can fire r01). When there is no
    ``vllm/`` subdirectory, any top-level ``*.prom`` is treated as the vLLM stream
    (a flat vLLM-only capture, or vLLM beside a ``dcgm/`` subdir).
    """
    vllm_dir = replay_dir / "vllm"
    dcgm_dir = replay_dir / "dcgm"
    sources: list = []

    if vllm_dir.is_dir():
        files = sorted(vllm_dir.glob("*.prom"), key=_natural_key)
        if files:
            sources.append(ReplaySource(files, parse_vllm_metrics_file, name="vllm"))
    else:
        files = sorted(replay_dir.glob("*.prom"), key=_natural_key)
        if files:
            sources.append(ReplaySource(files, parse_vllm_metrics_file, name="vllm"))

    if dcgm_dir.is_dir():
        files = sorted(
            (p for p in dcgm_dir.iterdir() if p.suffix in (".prom", ".json")),
            key=_natural_key,
        )
        if files:
            sources.append(ReplaySource(files, _dcgm_replay_parse, name="dcgm"))

    if not sources:
        raise click.UsageError(
            f"--replay {replay_dir}: no replay files found "
            "(expected vllm/*.prom and optionally dcgm/*)."
        )
    return sources


def _run_watch_loop(session, *, interval, max_ticks, use_color, interactive) -> None:
    """Drive the session on an interval, rendering events and a status heartbeat."""
    st = Styler(use_color)
    status_pending = False
    last_report = None
    last_warnings: set[str] = set()
    tick = 0

    def write_status(line: str) -> None:
        nonlocal status_pending
        sys.stdout.write("\r\033[K" + line)
        sys.stdout.flush()
        status_pending = True

    def clear_status() -> None:
        nonlocal status_pending
        if status_pending:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            status_pending = False

    try:
        while max_ticks is None or tick < max_ticks:
            tick += 1
            result = session.tick(time.monotonic())
            clock = datetime.now().strftime("%H:%M:%S")

            # Echo a warning only when it is new this tick. A persistently down
            # source re-records the same last_error every poll; without this it
            # would flood one identical line every interval forever.
            for warning in result.warnings:
                if warning in last_warnings:
                    continue
                clear_status()
                click.echo("  " + st.warn("! ") + st.dim(warning))
            last_warnings = set(result.warnings)

            if result.merged is not None:
                last_report = result.report

            if result.merged is None:
                if not result.exhausted and interactive:
                    write_status(render_status_line(
                        tick=tick, clock=clock, no_data=True, color=use_color))
            elif result.changed:
                clear_status()
                click.echo(
                    render_live_event(result.merged, result.report, clock=clock, color=use_color),
                    color=use_color,
                )
            elif interactive:
                write_status(render_status_line(
                    tick=tick, clock=clock,
                    active=len(result.report.diagnoses),
                    gen_tok_s=result.merged.token_throughput_gen,
                    color=use_color,
                ))

            if result.exhausted:
                break
            if max_ticks is None or tick < max_ticks:
                time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        clear_status()
        _echo_watch_summary(last_report)


def _echo_watch_summary(last_report) -> None:
    click.echo()
    if last_report is None:
        click.echo("watch stopped, no snapshots were diagnosed.")
        return
    fired = ", ".join(r.diagnosis.rule_id for r in last_report.diagnoses) or "none"
    click.echo(f"watch stopped. Last state: {fired} firing.")


# ---------------------------------------------------------------------------
# Stage 1 — load sources in fixed precedence order: vllm → dcgm → nsight
# ---------------------------------------------------------------------------
#
# model_name/gpu_type are passed *into* the parsers, not just applied after the
# merge: gpu_type is not a mere label — the Nsight parser uses it to pick the
# peak HBM bandwidth that turns observed byte counts into a [0,1] utilisation.
# Override it post-hoc and the computed hbm_bandwidth_util would already be
# wrong. `merge_inputs` re-asserts the overrides afterwards so they win the
# precedence regardless of source order.
#
# A parser that raises is the user's problem to see clearly, not a stack trace —
# so we wrap each call and surface a clean error unless `--strict` asks for the
# raw exception.

def _load_sources(
    vllm_path: Optional[Path],
    vllm_baseline_path: Optional[Path],
    dcgm_path: Optional[Path],
    nsight_path: Optional[Path],
    model_name: Optional[str],
    gpu_type: Optional[str],
    *,
    nsys_path: Optional[Path] = None,
    strict: bool,
) -> list[DiagnosisInput]:
    """Run each requested parser; return its DiagnosisInput in precedence order."""
    kwargs: dict[str, str] = {}
    if model_name is not None:
        kwargs["model_name"] = model_name
    if gpu_type is not None:
        kwargs["gpu_type"] = gpu_type

    inputs: list[DiagnosisInput] = []
    if vllm_path is not None:
        # baseline_path is vLLM-only; keep it out of the shared kwargs the other
        # parsers receive.
        vllm_kwargs = dict(kwargs)
        if vllm_baseline_path is not None:
            vllm_kwargs["baseline_path"] = str(vllm_baseline_path)
        inputs.append(_run_parser(parse_vllm_metrics_file, vllm_path, "vLLM", vllm_kwargs, strict=strict))
    if dcgm_path is not None:
        inputs.append(_run_parser(parse_dcgm_json_file, dcgm_path, "DCGM", kwargs, strict=strict))
    if nsight_path is not None:
        inputs.append(_dispatch_nsight(nsight_path, kwargs, strict=strict))
    if nsys_path is not None:
        inputs.append(_dispatch_nsys(nsys_path, kwargs, strict=strict))
    return inputs


def _run_parser(parser, path: Path, label: str, kwargs: dict[str, str], *, strict: bool) -> DiagnosisInput:
    """Invoke one parser, converting any failure into a clean CLI error.

    The engine half is hardened against bad rules; this hardens the ingestion
    half against bad files. ``--strict`` re-raises so CI and tests see the
    original traceback.
    """
    try:
        return parser(str(path), **kwargs)
    except Exception as exc:  # noqa: BLE001 — boundary; report, don't crash
        if strict:
            raise
        raise click.ClickException(
            f"Failed to parse {label} input {path.name!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _dispatch_nsight(path: Path, kwargs: dict[str, str], *, strict: bool) -> DiagnosisInput:
    """Pick the right Nsight parser based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _run_parser(parse_nsight_csv_file, path, "Nsight", kwargs, strict=strict)
    if suffix == ".ncu-rep":
        # The binary parser is a stub today. Surface its guidance as a clean
        # usage error (exit 2, no traceback) rather than letting
        # NotImplementedError escape; when it lands, this call just works.
        try:
            return parse_nsight_ncu_rep(str(path), **kwargs)
        except NotImplementedError as exc:
            raise click.UsageError(str(exc)) from exc
    raise click.UsageError(
        f"Unrecognised Nsight extension {suffix!r}. Expected .csv or .ncu-rep."
    )


def _dispatch_nsys(path: Path, kwargs: dict[str, str], *, strict: bool) -> DiagnosisInput:
    """Pick the right Nsight Systems parser based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _run_parser(parse_nsys_timeline_csv_file, path, "nsys", kwargs, strict=strict)
    if suffix in (".nsys-rep", ".sqlite"):
        # The binary/SQLite parser is a stub today; surface its guidance as a clean
        # usage error (exit 2, no traceback) rather than letting NotImplementedError
        # escape. When it lands, this call just works.
        try:
            return parse_nsys_rep(str(path), **kwargs)
        except NotImplementedError as exc:
            raise click.UsageError(str(exc)) from exc
    raise click.UsageError(
        f"Unrecognised nsys extension {suffix!r}. Expected .csv or .nsys-rep."
    )


# ---------------------------------------------------------------------------
# Stage 2 — merge multiple DiagnosisInputs into one: schema/merge.py owns the
# precedence rules, shared with the live collector so neither path forks them.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 3 — formatting lives in cli/render.py; the engine returns data only.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# fix — drive the gait state machine from a diagnosis to a verified change
# ---------------------------------------------------------------------------
#
# Thin wrapper: it builds the snapshot the same way `diagnose` does, picks the named
# diagnosis, and walks gait's read-only stages to `Proposed`. The only place it
# crosses from read-only to mutating is the approval gate — interactive by default,
# or `--yes` (refused below `--threshold`). After applying it re-collects and prints
# gait's honest four-way verdict.

@cli.command(epilog="""\b
Examples:
  strided fix r03 --vllm metrics.prom --config launch.txt
  strided fix r03 --vllm metrics.prom --config launch.txt --dry-run
""")
@click.argument("rule")
@click.option("--vllm", "vllm_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a vLLM /metrics dump (.prom or .txt).")
@click.option("--dcgm", "dcgm_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to a DCGM JSON dump.")
@click.option("--nsight", "nsight_path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to an Nsight Compute export (.csv or .ncu-rep).")
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="vLLM launch-command file gait may read and (after approval) edit.")
@click.option("--model", "model_name", default=None, help="Model name override.")
@click.option("--gpu", "gpu_type", default=None, help="GPU type override.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Stop at the proposal: state the plan and change nothing.")
@click.option("--yes", is_flag=True, default=False,
              help="Auto-approve IF confidence ≥ --threshold. Never for low confidence.")
@click.option("--threshold", default=0.80, show_default=True, type=float,
              help="Confidence bar for --yes auto-approval.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Narrate every step of the agent's process (resolve → verify).")
@click.option("--color/--no-color", "color_flag", default=None,
              help="Force coloured output on or off. Default: auto-detect.")
@click.option("--strict", is_flag=True, default=False, help="Re-raise parser/rule errors.")
def fix(
    rule: str,
    vllm_path: Optional[Path],
    dcgm_path: Optional[Path],
    nsight_path: Optional[Path],
    config_path: Path,
    model_name: Optional[str],
    gpu_type: Optional[str],
    dry_run: bool,
    yes: bool,
    threshold: float,
    verbose: bool,
    color_flag: Optional[bool],
    strict: bool,
) -> None:
    """Adjust one setting within your limits, then verify the result.

    gait is strided's human-in-the-loop fix agent. Adjustments are automatic;
    decisions are visible. By default it states a one-line plan (its TLDR) and asks
    you to approve it, records what it expects to happen, applies one change, and
    checks the result. Pass --verbose to watch every step of the state machine.
    When a change cannot be justified, it keeps the current settings and says so,
    and `strided undo` is always the way back.

    Note: verification needs a *fresh* capture. With a static --vllm <file> the
    after-snapshot equals the before-snapshot, so verify can only return NO CHANGE
    or INSUFFICIENT DATA, never CONFIRMED. Point at a live endpoint (or re-capture
    after restarting the server) for a real verdict.
    """
    if not any([vllm_path, dcgm_path, nsight_path]):
        raise click.UsageError("Provide at least one of --vllm / --dcgm / --nsight.")

    # Lazy: only `fix` (and `undo`) depend on gait, so diagnose/watch/home don't.
    from cli import gait_render as gr
    from gait import (
        Abstained, Diagnosed, HumanDecision, Journal, VllmArgsTarget, Verdict,
        apply, approve, approve_auto, propose, resolve, rollback, verify,
    )

    st = Styler(should_color(sys.stdout, color_flag))
    rule_id = rule.lower()

    def collect() -> DiagnosisInput:
        # gait fix has no --vllm-baseline flag (yet): no earlier scrape to difference.
        inputs = _load_sources(
            vllm_path, None, dcgm_path, nsight_path, model_name, gpu_type, strict=strict,
        )
        return merge_inputs(inputs, model_name=model_name, gpu_type=gpu_type)

    snapshot = collect()
    report = run_diagnosis(snapshot, strict=strict)

    ranked = next((r for r in report.diagnoses if r.diagnosis.rule_id == rule_id), None)
    if ranked is None:
        unmet = next((n for n in report.insufficient_data if n.rule_id == rule_id), None)
        if unmet is not None:
            missing = ", ".join(unmet.missing) if unmet.missing else (unmet.reason or "required data")
            raise click.ClickException(
                f"{rule_id} could not evaluate on this input (needs: {missing}). Nothing to fix."
            )
        fired = ", ".join(r.diagnosis.rule_id for r in report.diagnoses) or "none"
        raise click.ClickException(
            f"{rule_id} did not fire on this input (fired: {fired}). Nothing to fix."
        )

    title = next((c.title for c in ALL_RULES if c.rule_id == rule_id), rule_id)
    target = VllmArgsTarget.from_file(config_path)

    click.echo(gr.header(st, rule_id, title, ranked.diagnosis.confidence))
    click.echo("")

    # --- read-only walk to a proposal --------------------------------------- #
    resolved = resolve(Diagnosed(ranked.diagnosis, snapshot), target)
    if isinstance(resolved, Abstained):
        return click.echo(gr.abstained(st, resolved))
    if verbose:
        click.echo(gr.step(st, "resolve", f"found {st.accent(resolved.param)} = "
                                          f"{resolved.current_value!r} in {target.ref}"))

    proposed = propose(resolved)
    if isinstance(proposed, Abstained):
        return click.echo(gr.abstained(st, proposed))
    if verbose:
        click.echo(gr.step(st, "propose",
                           st.accent(f"{proposed.param} {proposed.current_value!r} → "
                                     f"{proposed.proposed_value!r}")))
        for line in gr.diagnosis_line(st, proposed):
            click.echo(line)
        for line in gr.predicted_effect(st, proposed):
            click.echo(line)
        click.echo("")

    # --- the TLDR: the agent states its plan -------------------------------- #
    click.echo(gr.tldr(st, proposed, confidence_threshold=threshold))
    click.echo("")

    if dry_run:
        return click.echo(_INDENT_NOTE(st, "dry run, nothing was changed."))

    # --- the hard gate ------------------------------------------------------ #
    if yes:
        approved = approve_auto(proposed, confidence_threshold=threshold)
        if isinstance(approved, Abstained):
            return click.echo(gr.abstained(st, approved))
    else:
        ok = click.confirm("  " + st.accent("apply this change?"), default=False)
        approved = approve(proposed, HumanDecision(approved=ok))
        if isinstance(approved, Abstained):
            return click.echo(gr.declined_line(st))
    if verbose:
        click.echo(gr.step(st, "approve", st.dim(f"approved · {approved.approval.mode}")))

    # --- the one mutation, then the honest verdict -------------------------- #
    applied = apply(approved, journal=Journal.default())
    click.echo("")
    click.echo(gr.applied_line(st, applied))

    if verbose:
        click.echo(gr.step(st, "verify", st.dim("re-collecting a fresh snapshot to check the prediction…")))
    verified = verify(applied, _safe_collector(collect), confidence_threshold=threshold)
    click.echo("")
    click.echo(gr.verdict_block(st, verified))

    if verified.verdict is Verdict.NO_CHANGE:
        click.echo("")
        if click.confirm("  roll back this change?", default=True):
            rolled = rollback(applied)
            click.echo(gr.rolled_back_line(st, rolled.change.param, rolled.restored_value))


def _INDENT_NOTE(st: Styler, text: str) -> str:
    return "  " + st.dim(text)


def _safe_collector(collect):
    """Wrap a collector so a re-collection failure becomes None (→ INSUFFICIENT_DATA)."""
    def _inner():
        try:
            return collect()
        except Exception:  # noqa: BLE001 — verify treats None as insufficient data
            return None
    return _inner


# ---------------------------------------------------------------------------
# undo — reverse an applied change, reconstructed from the journal
# ---------------------------------------------------------------------------

@cli.command(epilog="""\b
Examples:
  strided undo --config launch.txt
  strided undo <change-id> --config launch.txt
""")
@click.argument("change_id", required=False)
@click.option("--config", "config_path", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="vLLM launch-command file to restore the prior value into.")
@click.option("--color/--no-color", "color_flag", default=None,
              help="Force coloured output on or off. Default: auto-detect.")
def undo(change_id: Optional[str], config_path: Path, color_flag: Optional[bool]) -> None:
    """Go back: restore the prior value of an applied change (last one if id omitted)."""
    # Lazy: keep gait off the import path of diagnose/watch/home (see note up top).
    from cli import gait_render as gr
    from gait import Journal, VllmArgsTarget
    from gait.journal import STATUS_ROLLED_BACK

    st = Styler(should_color(sys.stdout, color_flag))
    journal = Journal.default()
    entry = journal.get(change_id) if change_id else journal.last_applied()
    if entry is None:
        raise click.ClickException(
            f"no applied change found for {change_id!r}." if change_id
            else "no applied change to undo."
        )
    if entry.status == STATUS_ROLLED_BACK:
        click.echo("  " + st.dim(f"change {entry.change_id} is already rolled back; nothing to do."))
        return

    target = VllmArgsTarget.from_file(config_path)

    # Guard: refuse to restore into a file the change never touched. The journal
    # records the target_ref the change was applied to; writing prior_value into a
    # different launch command would silently corrupt an unrelated file.
    if not _same_target(entry.target_ref, target.ref, config_path):
        raise click.ClickException(
            f"change {entry.change_id} was applied to {entry.target_ref!r}, but "
            f"--config points at {target.ref!r}. Re-run with "
            f"--config {entry.target_ref}."
        )

    target.write(entry.param, entry.prior_value)
    journal.mark_rolled_back(entry.change_id)
    click.echo(gr.rolled_back_line(st, entry.param, entry.prior_value)
               + st.dim(f"   ·   {target.ref}  (change {entry.change_id})"))


def _same_target(recorded_ref: str, current_ref: str, config_path: Path) -> bool:
    """Whether the undo target matches the change's recorded target.

    Exact ref match first; otherwise compare resolved filesystem paths so a
    relative/absolute spelling of the same file is not flagged as a mismatch.
    """
    if recorded_ref == current_ref:
        return True
    try:
        return Path(recorded_ref).resolve() == Path(config_path).resolve()
    except Exception:  # noqa: BLE001 — a non-path ref simply can't match a file
        return False


if __name__ == "__main__":
    cli()
