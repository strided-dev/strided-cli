# Commands reference

strided has four commands plus an interactive workspace. Every example here runs
offline against the bundled [`examples/`](../examples/README.md).

- [`strided diagnose`](#strided-diagnose) — one-shot diagnosis from captured dumps
- [`strided watch`](#strided-watch) — continuous live diagnosis
- [`strided fix`](#strided-fix) — walk a diagnosis to a verified change (the agent)
- [`strided undo`](#strided-undo) — reverse the last applied change
- [`strided` (workspace)](#the-workspace) — interactive session

Global notes:
- Use `python -m cli <command>` if you didn't `pip install -e .`.
- `--color / --no-color` is available on every subcommand; the default auto-detects
  (colour only when stdout is an interactive terminal). `NO_COLOR=1` or
  `FORCE_COLOR=1` in the environment override the default.

---

## `strided diagnose`

One-shot diagnosis from one or more captured telemetry dumps. This is the core
command.

```bash
strided diagnose --vllm examples/vllm_kv_fragmentation.prom --gpu H100-SXM
```

You can pass several sources at once; they're merged into one picture of the
workload before the rules run (see [data-sources.md](data-sources.md) for merge
precedence):

```bash
strided diagnose \
  --vllm examples/vllm.prom \
  --nsight examples/membound.csv \
  --gpu H100-SXM --model meta-llama/Llama-3-8B
```

| Flag | Description |
|---|---|
| `--vllm PATH` | A vLLM `/metrics` dump (Prometheus text, `.prom` or `.txt`). |
| `--dcgm PATH` | A DCGM JSON dump. |
| `--nsight PATH` | An Nsight Compute export — `.csv` (supported) or `.ncu-rep` (not yet, see below). |
| `--model NAME` | Model name override, e.g. `meta-llama/Llama-3-70B`. Not present in most dumps. |
| `--gpu TYPE` | GPU type override, e.g. `H100-SXM`, `A100-80G`. **Affects results** — the Nsight parser uses it to pick peak HBM bandwidth, so set it for Nsight inputs. |
| `--strict` | Re-raise parser/rule errors and non-finite inputs instead of degrading gracefully. For CI and debugging. |
| `--color / --no-color` | Force colour on/off (default: auto). |

You must pass at least one of `--vllm` / `--dcgm` / `--nsight`.

> **`.ncu-rep` is not yet supported.** Binary Nsight reports need the full Nsight
> Compute install; export to CSV instead (`ncu --csv --page raw`). strided prints a
> clear message if you pass one. See [data-sources.md](data-sources.md).

---

## `strided watch`

The same pipeline as `diagnose`, on a loop: poll metrics endpoints (or replay
captured files), parse, diagnose, and render. It prints a full block only when the
**diagnosis state changes**, with a single rewritable status line in between.

Offline (replay bundled captures — no server needed):

```bash
strided watch --replay examples/replay --max-ticks 3 --interval 0
```

Live (point at a running server):

```bash
strided watch \
  --vllm http://localhost:8000/metrics \
  --dcgm http://localhost:9400/metrics \
  --interval 5
```

| Flag | Description |
|---|---|
| `--vllm URL` | vLLM `/metrics` endpoint to poll. |
| `--dcgm URL` | dcgm-exporter `/metrics` endpoint to poll (enables r01 live). |
| `--replay DIR` | Replay captured scrapes from a directory (offline). Cannot be combined with `--vllm`/`--dcgm`. |
| `--interval SECONDS` | Seconds between polls (default `5.0`). Live polling has a 0.5s floor; replay ignores it. |
| `--model NAME` / `--gpu TYPE` | Overrides, as in `diagnose`. |
| `--max-ticks N` | Stop after N ticks (for tests / non-interactive use). A replay stream also stops when exhausted. |
| `--color / --no-color` | Force colour on/off. |

At start-up `watch` prints an honest **tiering banner**: which rules each connected
source can feed live, and which need a captured dump. A replay directory follows
the convention `<dir>/vllm/*.prom` and (optionally) `<dir>/dcgm/*`. See
[rules.md](rules.md#which-source-feeds-which-rule-live) for the tiering, and
[interpreting-output.md](interpreting-output.md#watch-output) for how to read the
live output.

---

## `strided fix`

Walk a single diagnosis to a reviewed, reversible, **verified** config change. This
is the `gait` agent. A human approves every change; nothing is mutated without it.

```bash
# Preview only — states the plan, changes nothing:
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --dry-run
```

| Flag | Description |
|---|---|
| `RULE` (argument) | The rule id to act on, e.g. `r03`. |
| `--config PATH` | **Required.** The vLLM launch-command file gait may read and (after approval) edit. |
| `--vllm` / `--dcgm` / `--nsight PATH` | The dump(s) to diagnose (same as `diagnose`). At least one required. |
| `--dry-run` | Stop at the proposal: state the plan, change nothing. |
| `--yes` | Auto-approve **only if** confidence ≥ `--threshold`. Never auto-approves a low-confidence diagnosis. |
| `--threshold FLOAT` | Confidence bar for `--yes` (default `0.80`). |
| `--verbose` / `-v` | Narrate every step of the state machine (resolve → propose → verify). |
| `--model` / `--gpu` / `--strict` | As in `diagnose`. |
| `--color / --no-color` | Force colour on/off. |

> Only rules with a registered, machine-actionable fix can be `fix`ed. Today that's
> **r03** (reduce `--block-size`). For others, gait stops and says so.

The complete flow — propose, approve, apply, verify, roll back — is in **[the `fix`
walkthrough](fix-walkthrough.md)**.

---

## `strided undo`

Reverse an applied change, reconstructed from gait's journal.

```bash
strided undo --config examples/launch.txt              # undo the most recent change
strided undo <change-id> --config examples/launch.txt  # undo a specific change by id
```

| Flag | Description |
|---|---|
| `CHANGE_ID` (argument, optional) | The change to undo. Omit to undo the last applied change. |
| `--config PATH` | **Required.** The launch file to restore the prior value into. |
| `--color / --no-color` | Force colour on/off. |

A change id is printed when `fix` applies a change (`undo: strided undo <id>`).
Undoing an already-rolled-back change is a no-op and says so.

---

## The workspace

Run `strided` with no command to open an interactive session. It is not a menu:
you get a prompt, and you type the same commands you would type in a shell.

```bash
strided
```

The one thing the workspace adds is a **bound subject**. `use` points the session
at a workload once; every later command inherits the flags it can accept, so you
stop retyping `--vllm` and `--gpu`:

```
~ › use examples/vllm_kv_fragmentation.prom --gpu H100-SXM
    --vllm  examples/vllm_kv_fragmentation.prom
    --gpu   H100-SXM

vllm_kv_fragmentation › diagnose
    ...
    next  strided fix r02 --config <launch-file>

vllm_kv_fragmentation › use --config examples/launch.txt
vllm_kv_fragmentation › fix r02 --dry-run
    $ strided fix r02 --dry-run --vllm examples/… --gpu H100-SXM --config examples/launch.txt
```

The prompt always names what you are pointed at, and whenever the session fills in
flags it echoes the full assembled command first. Nothing the session knows is
hidden from you.

| At the prompt | What it does |
|---|---|
| `use <path\|url> [--gpu X] [--config f]` | Bind a workload. A bare path is classified by extension: `.prom`/`.txt` is vLLM, `.json` is DCGM, `.csv`/`.ncu-rep` is Nsight, a directory is `--replay`, an `http://` URL is a live vLLM endpoint. |
| `use` | Print the current bindings. |
| `unset --config` | Drop one binding. |
| `?` | The full map: every command, its place in the loop, and the session verbs. |
| `guide <command>` | Walk that command's inputs step by step. |
| `<command> --help` | Any command's real help, unchanged. |
| `clear`, `exit`, ctrl-d | Housekeeping. |

**Tab completes** command names, each command's real flags, filesystem paths after
a path flag, and rule ids after `fix` (the ones that actually fired, once you have
run `diagnose`). Up-arrow walks history, which persists in `~/.strided/history`.
A mistyped command suggests the nearest real one.

Bindings are routed by kind, so a bound *file* is never handed to `watch --vllm`
(which wants a URL) and a bound URL never reaches `diagnose --vllm` (which wants a
path). When a binding cannot be used the session says so rather than quietly
dropping it.

Completions and flag routing are both read from the live click commands, so the
workspace cannot drift from `--help`. Every command still re-enters the same
parser, so there is exactly one definition of each.

When stdin/stdout isn't an interactive terminal (piped, CI, a test), it prints the
screen once followed by the normal `--help` and exits, never blocking on input that
will not come. The workspace has no `--no-color` flag; use `NO_COLOR=1 strided` for
plain output. Without `readline` (a bare Windows Python), the prompt still works;
completion and history do not.
