# strided documentation

Runtime tuning for local models: observe, adjust, verify. These guides are usage-first:
every command can be run **offline** against the bundled `examples/`, so you can
learn the tool with no GPU and no running server.

## Read in this order

1. **[Getting started](getting-started.md)** — install, then your first offline
   diagnosis in under a minute. Start here.
2. **[Commands reference](commands.md)** — `diagnose`, `watch`, `undo`, and the
   interactive workspace, with every flag and a runnable example each.
3. **[The `fix` agent walkthrough](fix-walkthrough.md)** — the showcase feature:
   walk one diagnosis to a reviewed, reversible, verified config change.
4. **[Interpreting output](interpreting-output.md)** — how to read a diagnosis:
   the confidence meter, evidence, and the "could not evaluate" abstentions.
5. **[What the rules detect](rules.md)** — r01 / r02 / r03 in plain English, plus
   which data source feeds which rule when you run live.
6. **[Bringing your own data](data-sources.md)** — capture vLLM `/metrics`, DCGM,
   and Nsight from a real workload; what each source provides.
7. **[Limitations & feedback](limitations.md)** — what is provisional, why some
   confidences are capped, and how to send us useful findings.
8. **[Extending strided](extending.md)** — add a rule, a parser, or a fix.

## Reference

- **[Engine architecture](engine/ARCHITECTURE.md)** — how rules are fired, ranked,
  and reconciled into one report.
- **Per-rule details** — each rule module under [`rules/`](../rules/) documents
  its literature, thresholds, confidence model, and false-positive guards in its
  docstring.

## Conventions in these docs

- Commands are shown as `strided <command>` (the installed console script). The
  exact equivalent **without installing** is `python -m cli <command>`.
- Copy-paste blocks use `--no-color` for clean terminal output. By default strided
  colours output when stdout is an interactive terminal; set `NO_COLOR=1` to
  disable it everywhere (including the workspace, which has no `--no-color` flag).
- File paths in examples are relative to the repository root.
