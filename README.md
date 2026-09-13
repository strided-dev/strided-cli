# strided

> Runtime tuning for local models. Observe, adjust, verify. CLI prototype.

## What this is

strided runs a control loop for a model you host on your own hardware: **observe**
the workload, **adjust** one runtime setting within the limits you set, **verify**
the result. It is the same loop the product carries upward from a single local host,
here as a command line tool.

You feed it telemetry from a real workload (vLLM `/metrics`, DCGM, or Nsight output)
and it tells you **why** the GPU was slow and **what to change**, each diagnosis
carrying a confidence score and a literature citation. Then `strided fix` takes one
of those diagnoses, states what it expects to happen, asks for a yes, applies a
single change, and checks whether the prediction held. `strided undo` is the way back.

Existing tools tell you *"your GPU is slow."* strided tells you *"your GPU is slow
because **X**, here is the change, and here is how sure I am"*, then makes that change
reviewable, verifiable, and reversible.

Automatic adjustments, visible decisions. When a change cannot be justified, strided
keeps the current settings and says so.

**Not** a profiler. **Not** a dashboard. **Not** an always-on agent. It reads a
snapshot (a captured dump, or a live `/metrics` poll) and reasons about it with a
small set of deterministic, inspectable rules.

## Quickstart (60 seconds, no GPU required)

```bash
# Requires Python 3.11+ (check: python3 --version). On macOS the system python3
# is 3.9, so install a newer one first (e.g. `brew install python@3.12`, then use
# python3.12 below).
git clone https://github.com/strided-dev/strided-cli.git && cd strided-cli
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip   # old pip gives a misleading "setup.py not found" error
pip install -e .

# Diagnose a bundled example dump. No server, no GPU:
strided diagnose --vllm examples/vllm_kv_fragmentation.prom --gpu H100-SXM
```

You'll get a ranked diagnosis with two rules firing (colocation contention and KV
cache fragmentation), each with a cause, a fix, the evidence, and a confidence
meter. Try the others:

```bash
strided diagnose --nsight examples/membound.csv --gpu H100-SXM         # decode memory-bound
strided diagnose --nsys examples/nsys_prefill_interference.csv         # prefill stalling decode (timeline)
strided watch --replay examples/replay --max-ticks 3 --interval 0      # live loop, offline
strided                                                                # interactive workspace
```

New here? Start with **[docs/getting-started.md](docs/getting-started.md)**.

## What it diagnoses today

strided ships **9 rules** (the set is growing toward a broader catalogue). Each is
backed by published literature and abstains rather than guess:

| # | Rule | Fires when | Suggested fix |
|---|---|---|---|
| r01 | Decode memory-bound at low batch | high HBM bandwidth + low SM occupancy in decode | increase batch size |
| r02 | Colocation contention (prefill ↔ decode) | preemptions + TPOT tail + prefill token share, on a colocated deployment | tune chunked prefill, or (at scale) PD disaggregation |
| r03 | KV cache fragmentation | high cache fragmentation while the cache is near capacity | reduce `--block-size` / backend version / pressure relief |
| r04 | Tensor-parallel rank imbalance | one rank's SM clock a robust outlier ≥10% below its peers | investigate that rank: thermals/hardware first, sharding last |
| r05 | NCCL collective dominates step time | NCCL kernel share above the topology band (~20% single-node, ~45% multi-node) | network/topology triage: NVLink vs PCIe path, fabric, NCCL config |
| r06 | Throughput decay over a sustained run | tokens/sec trends down across a `watch` run with an attributable mechanism (memory pressure or thermal throttle) | relieve memory pressure / fix cooling; the mechanism names the knob |
| r07 | Attention bottleneck (unfused attention path) | standalone-softmax signature in the kernel trace with no fused attention kernel present | enable a fused attention backend (FlashAttention / SDPA) |
| r08 | Prefill↔decode interference (timeline) | long prefill steps in an Nsight Systems timeline stall the decode cadence | set `--max-num-batched-tokens` to the chunk budget computed from the trace |
| r12 | Queue growth | the waiting queue grows super-linearly across a `watch` run | shed or route load (replicas, admission control); don't raise the scheduler cap |

Plain-English explanations are in **[docs/rules.md](docs/rules.md)**; each rule's
full spec is the docstring of its module under [`rules/`](rules/).

## Commands

| Command | Loop step | What it does |
|---|---|---|
| `strided diagnose` | observe | One-shot read of one or more captured dumps. |
| `strided watch` | observe | Continuous read: poll a running server's metrics, or replay captures offline. |
| `strided fix` | adjust, verify | One change within your limits, reviewed and **verified** (the `gait` agent). |
| `strided undo` | go back | Restore the prior value of an applied change. |
| `strided` (no command) | | Interactive workspace: bind a workload with `use`, then run the loop against it. |

Full reference: **[docs/commands.md](docs/commands.md)** and the `fix` walkthrough
in **[docs/fix-walkthrough.md](docs/fix-walkthrough.md)**.

## Documentation

| Guide | For |
|---|---|
| [Getting started](docs/getting-started.md) | Install + your first offline run. **Start here.** |
| [Commands reference](docs/commands.md) | Every command and flag, with runnable examples. |
| [The `fix` agent walkthrough](docs/fix-walkthrough.md) | Diagnosis → reviewed → applied → verified → undo. |
| [Interpreting output](docs/interpreting-output.md) | Reading a diagnosis: confidence, evidence, abstentions. |
| [What the rules detect](docs/rules.md) | Every rule in plain English, and live source tiering. |
| [Bringing your own data](docs/data-sources.md) | Capture vLLM / DCGM / Nsight from a real workload. |
| [Limitations & feedback](docs/limitations.md) | What's provisional, and how to report findings. |
| [Extending strided](docs/extending.md) | Add a rule, a parser, or a fix. |
| [Engine architecture](docs/engine/ARCHITECTURE.md) | How rules are fired, ranked, and reconciled. |

## How it works

```
raw dumps ─▶ parsers ─▶ DiagnosisInput ─▶ merge ─▶ engine ─▶ report ─▶ CLI render
            (vllm /     (the canonical            (rules     (ranked,
             dcgm /      schema, the              fire,      annotated)
             nsight)     only interface)           rank)
```

The canonical schema (`DiagnosisInput`, in
[`schema/diagnosis_input.py`](schema/diagnosis_input.py)) is the **only** interface
between the two halves. Parsers write to it; rules read from it. Each rule sees one
input, returns a `Diagnosis` (with confidence + evidence) or abstains. The engine
ranks and annotates; it never invents a diagnosis or a confidence number.

## Stack

- Python 3.11+
- `pydantic` (schema validation at the boundary) and `click` (CLI). That's it.
- `pytest` for tests. No ML frameworks, no cloud SDKs, no telemetry. Dependencies stay minimal by design.

## Status & context

strided is an early prototype in a focused validation sprint: the bet is whether a
deterministic engine can match a senior engineer's read of real telemetry before any
always-on instrumentation is built. Local hosting is the starting point; the longer
term ambition is to extend the same observe/adjust/verify loop to shared model serving
and, eventually, coordinated control of data center systems. Nine rules are live; their
firing thresholds are **calibration seeds**, so most rules cap their confidence at
65% until validated on real dumps and field samples. Read
[docs/limitations.md](docs/limitations.md) before trusting a diagnosis on production
data, and please send us what you find.

## Design rules

- **Schema is sacred.** It's the contract between parsers and rules; don't change it unilaterally.
- **Rules are deterministic and inspectable.** No ML models. Same input → same output.
- **Every diagnosis carries confidence.** No false certainty.
- **Ambiguous case → no diagnosis.** Trust collapses on the first wrong answer; we'd rather say nothing.

## Tests

```bash
pip install pytest
python -m pytest -q    # ~630 tests, runs in about a second, no GPU needed
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
