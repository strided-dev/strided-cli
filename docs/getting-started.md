# Getting started

This gets you from a fresh clone to a real diagnosis in about a minute — **no GPU,
no running vLLM server**. strided ships example dumps that exercise every rule, so
you can learn the whole tool offline before you ever point it at your own workload.

## Prerequisites

- **Python 3.11 or newer** (`python3 --version` to check). On macOS the system
  `python3` is 3.9, so install a newer one first (e.g. `brew install python@3.12`)
  and use it to create the venv.
- That's it. The only runtime dependencies are `pydantic` and `click`, pulled in
  automatically on install.

## Install

```bash
git clone https://github.com/strided-dev/strided-cli.git && cd strided-cli
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip          # old pip fails with a misleading "setup.py not found"
pip install -e .
```

`pip install -e .` installs strided in *editable* mode and registers the `strided`
console script. Verify:

```bash
strided --version
# strided, version 0.1.0
```

> **Prefer not to install?** Every command also runs as `python -m cli <command>`
> from the repo root. Wherever this doc says `strided diagnose ...`, you can type
> `python -m cli diagnose ...` instead.

## Your first diagnosis (offline)

Point `diagnose` at a bundled vLLM metrics dump:

```bash
strided diagnose --vllm examples/vllm_kv_fragmentation.prom --gpu H100-SXM --no-color
```

```
strided v0.1.0
local model hosting, tuned live
────────────────────────────────────────────────────────────────────────────────

▪ loaded
  model        unknown
  engine       vllm
  gpu          H100-SXM
  batch_size   24

▪ phase breakdown
  prefill    100.0 ms

▪ diagnosis / 2 rules fired, ranked by confidence

▌ Colocation contention (prefill ↔ decode)   r02               ████████░░░░  65%
    cause     Colocated prefill and decode are contending on the same GPU pool...
    fix       Enable/tune chunked prefill to smooth interference within your...
    evidence  preemption_rate=0.05, tpot_tail_ratio=5.36, num_requests_running=24,
              prefill_share=0.16, ...

▌ KV cache fragmentation   r03                                 ████████░░░░  65%
    cause     KV cache fragmentation was 47% with utilization 86%: roughly 47%...
    fix       You already run PagedAttention, so the fix is not to enable it...
    evidence  kv_cache_fragmentation=0.47, kv_cache_util=0.86, kv_block_size=16, ...

▪ could not evaluate / missing data
  - r01 Decode memory-bound at low batch, needs decode.hbm_bandwidth_util, decode.sm_occupancy
  - r04 Tensor-parallel rank imbalance
  - r05 NCCL collective dominates step time
  - r07 Attention bottleneck (unfused attention path), needs layers
  - r08 Prefill↔decode interference (timeline), needs nsys_timeline.steps

  2 trend rules idle (r06, r12): they observe over a run, so use `strided watch`

────────────────────────────────────────────────────────────────────────────────
parsed in 0.00s · 9 rules evaluated

  next  strided fix r02 --config <launch-file>
```

### What you just saw

- **loaded / phase breakdown**: what strided parsed out of the dump. It only
  prints fields that were actually present; nothing is invented to fill the frame.
- **diagnosis**: rules that fired, ranked by confidence. The bar (`████████░░░░`)
  and the `65%` are the rule's confidence. Each card gives a **cause** (what's
  wrong, in past tense), a **fix** (what to do), and the **evidence** (the exact
  numbers that triggered it).
- `9 rules evaluated`: strided ran every registered rule; two fired. The others
  (r01, r04, r05, r07, r08) needed GPU-kernel or multi-rank fields this vLLM-only dump
  doesn't carry, so they abstained with the missing fields named, and the trend
  rules (r06, r12) are folded into one quiet line, because a single snapshot can
  never feed a rule that observes over a run (see
  [Interpreting output](interpreting-output.md)).

Full breakdown of every part of this output:
**[Interpreting output](interpreting-output.md)**.

## Try the other inputs

```bash
# Nsight Compute CSV → decode memory-bound (r01):
strided diagnose --nsight examples/membound.csv --gpu H100-SXM --no-color

# A continuous live loop, replayed from captured files (offline) → r01 + r02 + r03:
strided watch --replay examples/replay --max-ticks 3 --interval 0 --no-color

# The interactive workspace (just run it with no arguments):
strided
```

Every bundled file and the command that uses it is catalogued in
[`examples/README.md`](../examples/README.md).

## The `fix` agent (offline)

strided can also walk a diagnosis to an actual config change — proposing it,
asking your approval, applying it, then re-checking whether it worked. Preview the
plan without changing anything:

```bash
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --dry-run --no-color
```

The full flow (apply, verify, undo) is in
**[the `fix` walkthrough](fix-walkthrough.md)**.

## Run the tests (optional)

```bash
pip install pytest
python -m pytest -q
```

## Where to next

- **[Commands reference](commands.md)** — all flags for every command.
- **[What the rules detect](rules.md)** — understand r01 / r02 / r03.
- **[Bringing your own data](data-sources.md)** — when you're ready to point
  strided at a real workload.
- **[Limitations & feedback](limitations.md)** — read before trusting a diagnosis
  on production data.
