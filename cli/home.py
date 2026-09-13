"""The ``strided`` workspace: an interactive session bound to one workload.

Typing ``strided`` on its own opens this. It is not a menu. A menu would fence the
tool off to whatever someone thought to enumerate, and it would still make you type
the flags afterwards, so it costs a step and saves nothing. Instead this is a prompt
over a *subject*: bind a workload once with ``use``, then run the loop against it.

    ~ › use examples/vllm_kv_fragmentation.prom --gpu H100-SXM
    kv_fragmentation › observe
    kv_fragmentation › fix r02 --dry-run

Three properties hold the whole thing together:

1. **One string, two homes.** Whatever you type at the prompt is valid after
   ``strided `` in a shell. The session adds nothing to the grammar: it only fills
   in flags you already bound, and it prints the assembled command when it does.
   So using the workspace teaches the real CLI instead of a parallel one.
2. **Everything is derived from click.** Completions, flag injection, and the help
   map all read the live ``Command`` objects, so none of it can drift from
   ``--help``. There is still exactly one definition of every command.
3. **The subject is always visible.** ``use`` introduces session state, which is the
   usual way this pattern goes wrong. The prompt carries the bound workload, and
   ``use`` with no arguments prints the context in full, so it is never hidden.

When stdout/stdin is not an interactive terminal (piped, CI, a test harness), there
is nothing to interact with, so we print the screen once followed by the normal help
and return, never blocking on input that will never come.
"""

from __future__ import annotations

import glob
import os
import shlex
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import click

from cli import lastrun
from cli.render import logo_banner
from cli.style import Styler, should_color

try:  # readline gives history, emacs keys and completion for free, and is stdlib.
    import readline
except ImportError:  # pragma: no cover - Windows without pyreadline3
    readline = None  # type: ignore[assignment]


_HISTORY = Path.home() / ".strided" / "history"


# --------------------------------------------------------------------------- #
# The bound subject
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Slot:
    """One bound value, and what kind of thing it is.

    The kind is what lets a single ``--vllm`` binding be routed correctly: a *file*
    satisfies ``diagnose --vllm`` (a click Path) and a *url* satisfies
    ``watch --vllm`` (a plain string). Rather than hard-code which command wants
    which, :func:`_accepts` compares the kind against the live parameter's type.
    """

    value: str
    kind: str  # "file" | "dir" | "url" | "text"


@dataclass(frozen=True)
class Session:
    """The workload this session is pointed at. Empty until ``use`` says otherwise."""

    slots: dict[str, Slot]

    @classmethod
    def empty(cls) -> "Session":
        return cls(slots={})

    def bind(self, flag: str, slot: Slot) -> "Session":
        return replace(self, slots={**self.slots, flag: slot})

    def unbind(self, flag: str) -> "Session":
        return replace(self, slots={k: v for k, v in self.slots.items() if k != flag})

    @property
    def label(self) -> str:
        """Short name for the prompt: what am I pointed at?"""
        for flag in ("--replay", "--vllm", "--dcgm", "--nsight", "--nsys"):
            slot = self.slots.get(flag)
            if slot is None:
                continue
            if slot.kind == "url":
                return slot.value.split("//", 1)[-1].split("/", 1)[0]
            stem = Path(slot.value).stem or Path(slot.value).name
            return stem if len(stem) <= 28 else stem[:27] + "…"
        return "~"


# Flags `use` can bind, and the kind each one holds. Everything else stays a
# per-invocation flag: these are the ones you would otherwise retype every time.
_BINDABLE: dict[str, str] = {
    "--vllm": "file-or-url",
    "--dcgm": "file-or-url",
    "--nsight": "file",
    "--nsys": "file",
    "--replay": "dir",
    "--config": "file",
    "--model": "text",
    "--gpu": "text",
}

# Extension → the flag that reads it, so `use foo.prom` needs no flag at all.
_BY_SUFFIX: dict[str, str] = {
    ".prom": "--vllm", ".txt": "--vllm",
    ".json": "--dcgm",
    ".csv": "--nsight", ".ncu-rep": "--nsight",
    ".nsys-rep": "--nsys",
}

_SESSION_VERBS = ("use", "unset", "guide", "help", "clear", "exit", "quit")


# --------------------------------------------------------------------------- #
# Click introspection: the single source of truth for flags and completions
# --------------------------------------------------------------------------- #

def _params(cmd: click.Command) -> dict[str, click.Parameter]:
    """Every long flag the command accepts, keyed by the flag string."""
    out: dict[str, click.Parameter] = {}
    for p in cmd.params:
        for opt in getattr(p, "opts", ()):
            if opt.startswith("--"):
                out[opt] = p
    return out


def _accepts(param: click.Parameter, kind: str) -> bool:
    """Would ``param`` accept a bound value of this kind?

    A click ``Path`` wants a real file or directory on disk; anything else (a plain
    string) is where a URL belongs. Comparing against the live type is what keeps a
    file binding out of ``watch --vllm`` and a URL out of ``diagnose --vllm``,
    without a hand-maintained table that could fall out of step.
    """
    is_path = isinstance(param.type, click.Path)
    if kind == "file":
        return is_path and param.type.file_okay
    if kind == "dir":
        return is_path and param.type.dir_okay
    return not is_path  # "url" and "text" both want a plain string


def _path_flags(cmd: click.Command) -> set[str]:
    return {f for f, p in _params(cmd).items() if isinstance(p.type, click.Path)}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _context_lines(st: Styler, session: Session) -> list[str]:
    """The bound subject, spelled out. Never let `use` hide what it did."""
    if not session.slots:
        return ["  " + st.dim("nothing bound yet")]
    width = max(len(f) for f in session.slots)
    return [
        "  " + st.label(flag.ljust(width)) + "  " + slot.value
        for flag, slot in session.slots.items()
    ]


def _opening(st: Styler) -> str:
    """What to do first, when nothing is bound."""
    lines = [
        "  " + st.dim("nothing loaded. point strided at a workload:"),
        "",
        "    " + "use examples/vllm_kv_fragmentation.prom".ljust(44)
        + st.dim("a bundled capture"),
        "    " + "use http://localhost:8000/metrics".ljust(44)
        + st.dim("a running server"),
        "    " + "use examples/replay".ljust(44) + st.dim("captured scrapes, for watch"),
        "",
        "  " + st.dim("tab completes · ? lists everything · ctrl-d leaves"),
    ]
    return "\n".join(lines)


def render_home(st: Styler, width: int) -> str:
    """The static landing screen, used for non-interactive output."""
    return "\n".join(logo_banner(st, width)) + "\n\n" + _opening(st)


def _help_map(st: Styler, ctx: click.Context) -> str:
    """The full map, printed on `?`. Command summaries come from click itself."""
    cmds = ctx.command.commands  # type: ignore[attr-defined]
    steps = {"diagnose": "observe", "watch": "observe", "fix": "adjust, verify",
             "undo": "go back"}
    width = max(len(n) for n in cmds)
    step_w = max(len(v) for v in steps.values()) + 2
    lines = ["  " + st.dim("commands")]
    for name in sorted(cmds):
        summary = (cmds[name].get_short_help_str(limit=50) or "").strip()
        lines.append("  " + st.accent(name.ljust(width)) + "  "
                     + st.label(steps.get(name, "").ljust(step_w)) + st.dim(summary))
    lines += [
        "",
        "  " + st.dim("session"),
        "  " + st.accent("use".ljust(width)) + "  "
        + st.dim("bind a workload: use <path|url> [--gpu X] [--config launch.txt]"),
        "  " + st.accent("unset".ljust(width)) + "  " + st.dim("drop a binding: unset --config"),
        "  " + st.accent("guide".ljust(width)) + "  " + st.dim("walk one command's inputs step by step"),
        "  " + st.accent("?".ljust(width)) + "  " + st.dim("this map"),
        "",
        "  " + st.dim("Any command's own --help works here too, e.g. `fix --help`."),
        "  " + st.dim("Bound flags are filled in automatically; the full command is echoed."),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# `use` / `unset`
# --------------------------------------------------------------------------- #

def _classify(raw: str) -> Optional[Slot]:
    """Work out what kind of thing a bare `use <argument>` is pointing at."""
    if raw.startswith(("http://", "https://")):
        return Slot(raw, "url")
    p = Path(raw).expanduser()
    if p.is_dir():
        return Slot(str(p), "dir")
    if p.is_file():
        return Slot(str(p), "file")
    return None


def _flag_for(slot: Slot) -> Optional[str]:
    """Which flag reads this? Inferred from the extension, so `use x.prom` works."""
    if slot.kind == "dir":
        return "--replay"
    if slot.kind == "url":
        return "--vllm"
    suffix = "".join(Path(slot.value).suffixes[-1:]) or Path(slot.value).suffix
    return _BY_SUFFIX.get(suffix.lower())


def _do_use(st: Styler, session: Session, args: list[str]) -> Session:
    """Bind a workload. Bare arguments are classified; flags are taken as given."""
    if not args:
        click.echo("\n".join(_context_lines(st, session)))
        return session

    i = 0
    while i < len(args):
        token = args[i]
        if token in _BINDABLE:
            if i + 1 >= len(args):
                _warn(st, f"{token} needs a value.")
                return session
            value = args[i + 1]
            kind = _BINDABLE[token]
            if kind == "text":
                slot = Slot(value, "text")
            else:
                slot = _classify(value)
                if slot is None:
                    _warn(st, f"{value!r} is not a file, directory, or URL.")
                    return session
            session = session.bind(token, slot)
            i += 2
            continue
        if token.startswith("-"):
            _warn(st, f"use cannot bind {token}. Bindable: {', '.join(_BINDABLE)}.")
            return session
        slot = _classify(token)
        if slot is None:
            _warn(st, f"{token!r} is not a file, directory, or URL.")
            return session
        flag = _flag_for(slot)
        if flag is None:
            _warn(st, f"cannot tell what {token!r} is. Name the flag, e.g. use --nsight {token}.")
            return session
        session = session.bind(flag, slot)
        i += 1

    click.echo("\n".join(_context_lines(st, session)))
    return session


def _do_unset(st: Styler, session: Session, args: list[str]) -> Session:
    if not args:
        _warn(st, f"unset which? One of: {', '.join(sorted(session.slots)) or 'nothing bound'}.")
        return session
    for flag in args:
        if flag not in session.slots:
            _warn(st, f"{flag} is not bound.")
            continue
        session = session.unbind(flag)
    click.echo("\n".join(_context_lines(st, session)))
    return session


# --------------------------------------------------------------------------- #
# Dispatch: fill in bound flags, then re-enter the real parser
# --------------------------------------------------------------------------- #

def _wants(param: click.Parameter) -> str:
    """Plain-English description of what a parameter will accept."""
    if not isinstance(param.type, click.Path):
        return "a URL or plain value"
    if param.type.dir_okay and not param.type.file_okay:
        return "a directory"
    return "a file"


_KIND_NAMES = {"file": "a file", "dir": "a directory", "url": "a URL", "text": "a value"}


def _fill(
    cmd: click.Command, session: Session, typed: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Add bound flags the command accepts and the user did not already type.

    Returns the completed argv, the flags that were added, and notes about bindings
    this command has but cannot use, so nothing is skipped silently. `_accepts`
    decides, from the live parameter type, whether a binding fits: that is what
    keeps a bound *file* out of ``watch --vllm``, which wants a URL.

    A flag the command does not have at all is not a note. ``diagnose`` having no
    ``--config`` is ordinary, not a mismatch worth a line of output.
    """
    params = _params(cmd)
    added: list[str] = []
    notes: list[str] = []
    argv = list(typed)
    for flag, slot in session.slots.items():
        param = params.get(flag)
        if param is None or flag in typed:
            continue
        if not _accepts(param, slot.kind):
            notes.append(
                f"{cmd.name} cannot use the bound {flag} "
                f"({_KIND_NAMES.get(slot.kind, slot.kind)}); it wants {_wants(param)}."
            )
            continue
        argv += [flag, slot.value]
        added.append(flag)
    return argv, added, notes


def _run(ctx: click.Context, cmd_name: str, typed: list[str], st: Styler,
         session: Session) -> None:
    """Re-enter the real parser, so the workspace reuses every bit of real logic."""
    cmd = ctx.command.commands[cmd_name]  # type: ignore[attr-defined]
    argv, added, notes = _fill(cmd, session, typed)
    for note in notes:
        _warn(st, note)
    if added:
        # Never let the session change the command silently: show what it assembled.
        click.echo("  " + st.dim("$ strided " + " ".join([cmd_name, *argv])))
        click.echo("")
    try:
        ctx.command.main(args=[cmd_name, *argv], prog_name="strided",
                         standalone_mode=False)
    except click.exceptions.Abort:
        click.echo("  " + st.dim("aborted."))
    except click.exceptions.Exit:
        pass  # --help and friends: click already printed
    except click.ClickException as exc:
        _warn(st, exc.format_message())
    except KeyboardInterrupt:
        click.echo("  " + st.dim("stopped."))


def _suggest(word: str, options: list[str]) -> Optional[str]:
    """Nearest command name, for a typo. Forgiveness costs one stdlib import."""
    import difflib

    hit = difflib.get_close_matches(word, options, n=1, cutoff=0.6)
    return hit[0] if hit else None


# --------------------------------------------------------------------------- #
# Completion
# --------------------------------------------------------------------------- #

class _Completer:
    """Tab completion over the live click commands, the session, and the filesystem."""

    def __init__(self, ctx: click.Context) -> None:
        self.ctx = ctx
        self.session = Session.empty()
        self._matches: list[str] = []

    @property
    def _commands(self) -> dict[str, click.Command]:
        return self.ctx.command.commands  # type: ignore[attr-defined]

    def _candidates(self, tokens: list[str], text: str) -> list[str]:
        if not tokens:
            return sorted(self._commands) + list(_SESSION_VERBS)

        head, rest = tokens[0], tokens[1:]

        if head == "use":
            return list(_BINDABLE) if text.startswith("-") else self._paths(text)
        if head == "unset":
            return sorted(self.session.slots)
        if head == "guide":
            return sorted(self._commands)
        if head not in self._commands:
            return []

        cmd = self._commands[head]
        if text.startswith("-"):
            return sorted(_params(cmd)) + ["--help"]
        if rest and rest[-1] in _path_flags(cmd):
            return self._paths(text)
        if head == "fix" and not [t for t in rest if not t.startswith("-")]:
            return list(lastrun.FIRED) or self._rule_ids()
        return self._paths(text)

    @staticmethod
    def _rule_ids() -> list[str]:
        from engine.registry import rule_ids  # local: keeps startup light

        return list(rule_ids())

    @staticmethod
    def _paths(text: str) -> list[str]:
        out = []
        for hit in glob.glob(os.path.expanduser(text) + "*"):
            out.append(hit + "/" if os.path.isdir(hit) else hit)
        return sorted(out)

    def __call__(self, text: str, state: int) -> Optional[str]:
        if state == 0:
            line = readline.get_line_buffer()[: readline.get_begidx()]
            try:
                tokens = shlex.split(line)
            except ValueError:  # an unbalanced quote mid-type
                tokens = line.split()
            self._matches = [c for c in self._candidates(tokens, text)
                             if c.startswith(text)]
        return self._matches[state] if state < len(self._matches) else None


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

def run_home(ctx: click.Context) -> None:
    """Open the workspace and drive it until the user leaves."""
    color = should_color(sys.stdout, None)
    st = Styler(color)
    width = _width()

    if not (_isatty(sys.stdin) and _isatty(sys.stdout)):
        # Nothing to interact with: show the screen, then the real help, and stop.
        click.echo(render_home(st, width))
        click.echo()
        click.echo(ctx.get_help())
        return

    _play_intro(st, width)
    completer = _install_readline(ctx)
    session = Session.empty()
    click.echo(_opening(st))
    click.echo("")

    while True:
        if completer is not None:
            completer.session = session
        try:
            raw = _prompt(st, session)
        except EOFError:      # ctrl-d
            click.echo("")
            break
        except KeyboardInterrupt:  # ctrl-c clears the line, does not leave
            click.echo("")
            continue

        raw = raw.strip()
        if not raw:
            continue
        try:
            tokens = shlex.split(raw)
        except ValueError as exc:
            _warn(st, f"couldn't parse that: {exc}")
            continue

        head, rest = tokens[0], tokens[1:]
        if head in ("exit", "quit"):
            break
        if head in ("?", "help"):
            click.echo(_help_map(st, ctx))
            continue
        if head == "clear":
            click.echo("\033[2J\033[H", nl=False)
            continue
        if head == "use":
            session = _do_use(st, session, rest)
            continue
        if head == "unset":
            session = _do_unset(st, session, rest)
            continue
        if head == "guide":
            _do_guide(st, ctx, rest, session)
            continue
        if head in ctx.command.commands:  # type: ignore[attr-defined]
            click.echo("")
            _run(ctx, head, rest, st, session)
            click.echo("")
            continue

        near = _suggest(head, sorted(ctx.command.commands) + list(_SESSION_VERBS))  # type: ignore[attr-defined]
        _warn(st, f"no command {head!r}." + (f" Did you mean `{near}`?" if near else " Try ?"))

    _save_history()
    click.echo("  " + st.dim("bye."))


def _prompt(st: Styler, session: Session) -> str:
    """``<subject> › ``, and nothing else.

    An earlier draft prefilled the suggested next command into the line buffer so
    Enter would advance the loop. In practice that is hostile: to type anything
    other than the suggestion you must first clear a line you did not write. The
    report's own closing ``next`` line carries the suggestion instead, and readline
    history (up-arrow) is the real answer to re-running things quickly.
    """
    return input(st.accent(session.label) + st.dim(" › "))


def _do_guide(st: Styler, ctx: click.Context, args: list[str], session: Session) -> None:
    """Walk one command's inputs step by step, for anyone who wants to be led."""
    if not args or args[0] not in _GUIDES:
        _warn(st, f"guide which? One of: {', '.join(sorted(_GUIDES))}.")
        return
    name = args[0]
    built = _GUIDES[name](st)
    if built is None:
        return
    click.echo("")
    _run(ctx, name, built[1:], st, session)
    click.echo("")


def _play_intro(st: Styler, width: int) -> None:
    """Draw the banner, run a brief load meter beneath it, then open the prompt.

    Purely cosmetic and interactive-only: a short beat that phases the start-up
    banner into the workspace. The meter line rewrites itself in place and is
    cleared before the prompt appears, so nothing is left behind.
    """
    for line in logo_banner(st, width):
        click.echo(line)

    stages = ("rules", "engine", "gait", "ready")
    steps = 30
    for i in range(steps + 1):
        frac = i / steps
        stage = stages[min(len(stages) - 1, int(frac * len(stages)))]
        bar = st.meter(frac, 22)
        sys.stdout.write("\r  " + bar + "  " + st.dim(f"loading {stage}…"))
        sys.stdout.flush()
        time.sleep(0.039)
    sys.stdout.write("\r\033[K")  # wipe the meter line
    sys.stdout.flush()
    click.echo()


def _install_readline(ctx: click.Context) -> Optional["_Completer"]:
    """History, emacs keys and tab completion, or None where readline is absent."""
    if readline is None:
        return None
    completer = _Completer(ctx)
    readline.set_completer(completer)
    # The defaults split on "/" and "-", which would break path and flag completion.
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")
    try:
        readline.read_history_file(str(_HISTORY))
    except (OSError, ValueError):
        pass  # no history yet, or an unreadable one: not worth a word to the user
    return completer


def _save_history() -> None:
    if readline is None:
        return
    try:
        _HISTORY.parent.mkdir(parents=True, exist_ok=True)
        readline.set_history_length(1000)
        readline.write_history_file(str(_HISTORY))
    except OSError:
        pass  # a read-only home is not a reason to fail on the way out


# --------------------------------------------------------------------------- #
# Guided prompts, one per command, each returning argv or None to cancel
# --------------------------------------------------------------------------- #

def _guide_diagnose(st: Styler) -> Optional[list[str]]:
    _intro(st, "diagnose", "point me at one or more captured dumps and I will read them")
    vllm = _ask_path(st, "vLLM /metrics dump (.prom/.txt)")
    dcgm = _ask_path(st, "DCGM JSON dump")
    nsight = _ask_path(st, "Nsight export (.csv/.ncu-rep)")
    if not any([vllm, dcgm, nsight]):
        return _need(st, "at least one of vLLM / DCGM / Nsight")
    model = _ask_text(st, "model name override (optional)")
    gpu = _ask_text(st, "GPU type override (optional)")
    return _argv("diagnose", vllm=vllm, dcgm=dcgm, nsight=nsight, model=model, gpu=gpu)


def _guide_watch(st: Styler) -> Optional[list[str]]:
    _intro(st, "watch", "replay a capture offline, or poll a live server")
    replay = _ask_path(st, "replay directory (offline), blank for live URLs", is_dir=True)
    if replay:
        return _argv("watch", replay=replay)
    vllm = _ask_text(st, "vLLM /metrics URL")
    dcgm = _ask_text(st, "dcgm-exporter /metrics URL (optional)")
    if not vllm and not dcgm:
        return _need(st, "a replay directory, or at least one live URL")
    return _argv("watch", vllm=vllm, dcgm=dcgm)


def _guide_fix(st: Styler) -> Optional[list[str]]:
    _intro(st, "fix", "one change, within your limits, verified, with a way back")
    rule = _ask_text(st, "rule id to fix (e.g. r03)")
    if not rule:
        return _need(st, "a rule id")
    config = _ask_path(st, "launch-command file gait may edit")
    if not config:
        return _need(st, "a --config launch file")
    vllm = _ask_path(st, "vLLM /metrics dump (.prom/.txt)")
    dcgm = _ask_path(st, "DCGM JSON dump")
    nsight = _ask_path(st, "Nsight export (.csv/.ncu-rep)")
    if not any([vllm, dcgm, nsight]):
        return _need(st, "at least one of vLLM / DCGM / Nsight")
    extra = ["--verbose"] if _ask_yes(st, "narrate every step (--verbose)?") else []
    if _ask_yes(st, "dry run: state the plan and change nothing?"):
        extra.append("--dry-run")
    return _argv("fix", rule, vllm=vllm, dcgm=dcgm, nsight=nsight, config=config) + extra


def _guide_undo(st: Styler) -> Optional[list[str]]:
    _intro(st, "undo", "go back to the value that was there before")
    config = _ask_path(st, "launch-command file to restore into")
    if not config:
        return _need(st, "a --config launch file")
    return _argv("undo", config=config)


_GUIDES = {
    "diagnose": _guide_diagnose,
    "watch": _guide_watch,
    "fix": _guide_fix,
    "undo": _guide_undo,
}


# --------------------------------------------------------------------------- #
# Prompt helpers
# --------------------------------------------------------------------------- #

def _warn(st: Styler, message: str) -> None:
    click.echo("  " + st.warn("! ") + st.dim(message))


def _intro(st: Styler, name: str, blurb: str) -> None:
    click.echo("")
    click.echo("  " + st.accent(name) + st.dim(f": {blurb}"))


def _need(st: Styler, what: str) -> None:
    _warn(st, f"need {what}. Back to the prompt.")
    return None


def _ask(st: Styler, prompt: str, default: str = "") -> str:
    return click.prompt(
        "  " + st.dim("›") + " " + prompt,
        default=default,
        show_default=False,
        prompt_suffix=st.dim(" : "),
    )


def _ask_text(st: Styler, prompt: str) -> str:
    return _ask(st, prompt).strip()


def _ask_yes(st: Styler, prompt: str) -> bool:
    return click.confirm("  " + st.dim("›") + " " + prompt, default=False)


def _ask_path(st: Styler, prompt: str, *, is_dir: bool = False) -> str:
    """Prompt for a path, re-asking on a non-existent one. Blank means 'skip'."""
    while True:
        raw = _ask_text(st, prompt)
        if not raw:
            return ""
        p = Path(raw).expanduser()
        if not p.exists():
            click.echo("    " + st.warn("! ")
                       + st.dim(f"{raw!r} does not exist. Try again, or blank to skip."))
            continue
        if is_dir and not p.is_dir():
            click.echo("    " + st.warn("! ") + st.dim(f"{raw!r} is not a directory."))
            continue
        return str(p)


def _argv(command: str, *positionals: str, **options: str) -> list[str]:
    """Assemble an argv list, dropping options the user left blank."""
    args = [command, *(p for p in positionals if p)]
    for flag, value in options.items():
        if value:
            args += [f"--{flag}", value]
    return args


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:  # noqa: BLE001 — a stream without isatty is simply "not a tty"
        return False


def _width() -> int:
    import shutil

    return min(shutil.get_terminal_size(fallback=(80, 24)).columns, 80)


__all__ = ["render_home", "run_home", "Session"]
