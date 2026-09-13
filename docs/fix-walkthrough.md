# The `fix` agent — a walkthrough

`strided fix` is **gait**: a human-in-the-loop agent that walks one diagnosis to a
reviewed, reversible, and *verified* config change. It is the part of strided that
doesn't just tell you what's wrong — it offers to fix it, carefully.

What "carefully" means, concretely:

1. **A human approves every change.** The approval gate is enforced by the type
   system, not by convention — the code that applies a change literally cannot run
   without an approval token that only the gate can mint.
2. **Every change is reversible.** The prior value and an undo are recorded before
   anything is written.
3. **Verification is allowed to fail.** After applying, gait re-collects and checks
   a prediction it recorded *beforehand*. It returns an honest four-way verdict —
   including "no change" and "I can't tell" — and offers to roll back.

The state machine is: `Diagnosed → Resolved → Proposed → Approved → Applied →
Verified`. Everything up to and including `Proposed` is **read-only** and safe to
run anywhere; the line between `Proposed` and `Approved` is the only place
read-only becomes mutating.

> **Scope today:** only rules with a registered, machine-actionable fix can be
> `fix`ed. That's **r03** (KV cache fragmentation → reduce vLLM `--block-size`). For
> any other rule, gait stops cleanly and tells you there's no fix mapping.

## Setup

We'll fix r03 using two bundled files:

- `examples/vllm_kv_fragmentation.prom` — a dump where r03 fires.
- `examples/launch.txt` — a stand-in for your vLLM launch command. gait reads the
  current `--block-size` from here and (after approval) edits it.

```
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B \
  --block-size 16 --max-num-seqs 256 --gpu-memory-utilization 0.90
```

## Step 1 — dry run: see the plan, change nothing

```bash
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --dry-run --no-color
```

```
▌ gait   KV cache fragmentation   r03                          ████████░░░░  65%

  reducing --block-size 16 → 8 should cut KV-cache fragmentation below 20% and
  relieve utilization below 80%.

  change  block-size 16 → 8
  target  examples/launch.txt

  this diagnosis is low-confidence (65%); I won't auto-apply it, and I'll be
  cautious about claiming it worked.

  dry run, nothing was changed.
```

The **TLDR** is gait stating, in one breath: what it will change (`block-size 16 →
8`), where (`examples/launch.txt`), and the checkable prediction it's committing to
(fragmentation below 20%, utilization below 80%). Because r03's confidence is capped
at 65% (it's not yet calibrated — see [limitations.md](limitations.md)), gait warns
that it won't auto-apply and will be conservative about declaring success.

## Step 2 — see every step (`--verbose`)

`--verbose` opens the curtain on the state machine — resolve, propose, the recorded
prediction, then (after approval) apply and verify:

```bash
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --verbose --dry-run --no-color
```

You'll see the `resolve` step locate `block-size` in the launch file, the `propose`
step size the new value, and the **predicted effect** rendered as a checklist gait
will verify against later:

```
predicted effect (recorded now, checked after apply):
  · kv_cache_fragmentation < 0.2  (fragmentation falls below the 20% firing floor)
  · kv_cache_util < 0.8  (utilization eases below 80%)
```

Recording the prediction *before* acting is the point: verification tests a
commitment made in advance, not a story told afterward.

## Step 3 — apply it

Interactively, gait asks before changing anything:

```bash
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt
#   apply this change? [y/N]:
```

Answer `y` and gait applies the edit, then verifies. For a **non-interactive** run,
use `--yes` — but note `--yes` only auto-approves when confidence clears
`--threshold` (default `0.80`). r03 caps at 65%, so you must lower the bar to
demonstrate it:

```bash
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --yes --threshold 0.6 --no-color
```

```
  ✓ applied  block-size 16 → 8   ·   undo: strided undo 5ef11f6132e9

  ✗ verify: NO CHANGE
  predicted effect did not materialize; consider rollback

  kv_cache_fragmentation  0.46875 → 0.46875
  kv_cache_util  0.86 → 0.86
  seq_len_distribution.mean  17.0 → 17.0

  roll back this change? [Y/n]:
```

### Why "NO CHANGE" here is correct, not a bug

You're fixing from a **static dump**. gait's verify step re-collects a fresh
snapshot — but with `--vllm <file>` the "fresh" snapshot is the *same file*, so the
metrics are identical before and after. gait sees no improvement and says so plainly
rather than pretending an edit to a text file changed a past measurement. This is the
honesty guarantee working as designed.

On a **live** system you'd point `fix` at the running endpoint (or re-capture after
restarting vLLM with the new block size), and verify would compare genuinely fresh
metrics. The four possible verdicts:

| Verdict | Meaning |
|---|---|
| `CONFIRMED` | The recorded prediction materialized. Shows before/after numbers. |
| `NO CHANGE` | No improvement, or a regression. Offers rollback. |
| `INCONCLUSIVE` | The signal moved but traffic shifted underneath, or the diagnosis was too low-confidence to attribute one before/after pair to the change. |
| `INSUFFICIENT DATA` | Couldn't collect a clean verifying snapshot. |

## Step 4 — undo

`fix` offers rollback inline on a `NO CHANGE` verdict. You can also undo any applied
change later from the journal:

```bash
strided undo --config examples/launch.txt --no-color
```

```
  ↩ rolled back  block-size → 16   ·   examples/launch.txt  (change 5ef11f6132e9)
```

`examples/launch.txt` is now back to `--block-size 16`. Undo without an id reverses
the most recent change; pass a change id (printed at apply time) to target a
specific one.

## When gait stops

gait abstains — cleanly, naming what was missing — rather than guess. You'll see a
`gait stopped: …` line when:

- the rule has no registered fix (`strided fix r01 ...` → no fix mapping);
- the rule didn't fire on this input (nothing to fix);
- the `--config` doesn't contain the param, or contains it twice (ambiguous);
- the snapshot isn't the shape the fix applies to (e.g. a non-vLLM engine for the
  block-size fix);
- `--yes` was given but confidence is below `--threshold`.

Each stop names the gap, so you know exactly what to provide.

## See also

- [Commands reference → `strided fix`](commands.md#strided-fix) for all flags.
- [Interpreting output](interpreting-output.md) for the confidence meter and evidence.
- [Limitations & feedback](limitations.md) for why r03 is confidence-capped.
