# Interpreting strided's output

strided's output is designed to be read top-to-bottom and to never overstate what
it knows. This guide walks through every part of a report.

## Anatomy of a diagnosis

```
strided v0.1.0                                    ← brand header
local model hosting, tuned live
────────────────────────────────────────────────────────────────────────────────

▪ loaded                                          ← what was parsed from the dump(s)
  model        unknown
  engine       vllm
  gpu          H100-SXM
  batch_size   24

▪ phase breakdown                                 ← per-phase timing + roofline class
  prefill    100.0 ms

▪ diagnosis / 2 rules fired, ranked by confidence

▌ Colocation contention (prefill ↔ decode)   r02               ████████░░░░  65%
    cause     ...                                 ← a diagnosis card (see below)
    fix       ...
    evidence  ...

▪ could not evaluate / missing data               ← rules that couldn't run
  - r01 Decode memory-bound at low batch, needs decode.hbm_bandwidth_util, decode.sm_occupancy
  - r04 Tensor-parallel rank imbalance
  - r05 NCCL collective dominates step time
  - r07 Attention bottleneck (unfused attention path), needs layers

  1 trend rule idle (r06): it observes over a run, so use `strided watch`   ← temporal rules, folded

────────────────────────────────────────────────────────────────────────────────
parsed in 0.00s · 7 rules evaluated               ← footer

  next  strided fix r02 --config <launch-file>    ← the next step in the loop
```

**Every visible number comes from your input or the engine.** A field that's absent
is simply not rendered — strided never invents a metric to fill the layout. So a
short `loaded` block just means the dump didn't carry those fields.

### phase breakdown and the roofline dot

Each phase shows its duration and, when known, a coloured **roofline class** dot:

- `● memory-bound` (yellow) — bottlenecked on HBM bandwidth, not compute.
- `● compute-bound` (cyan) — bottlenecked on the SMs.
- `● balanced` (green) — neither dominates.
- `● unknown` (grey) — not enough data to classify.

## Reading a diagnosis card

```
▌ KV cache fragmentation   r03                                 ████████░░░░  65%
    cause     KV cache fragmentation was 47% with utilization 86%: roughly 47% ...
    fix       You already run PagedAttention, so the fix is not to enable it ...
    evidence  kv_cache_fragmentation=0.47, kv_cache_util=0.86, kv_block_size=16, ...
```

| Part | Meaning |
|---|---|
| `▌` | The accent rail marking a diagnosis card. |
| title + `r03` | The rule's name and stable id (matches `rules/r03_*.py`). |
| `████████░░░░  65%` | The **confidence meter** and value. |
| `cause` | What went wrong, in past tense — kernel signals translated to model terms. |
| `fix` | What to do about it — concrete, actionable, often with exact flags. |
| `evidence` | The actual schema fields and values that fired the rule. This is your audit trail: every claim traces back to a number. |

A card may carry a small sub-line under the meter, e.g. `base 60% · corroborated by
r01`. That appears only when the engine adjusted or annotated the confidence (off by
default in v1 — see [Ranking](#ranking-and-the-engine)).

## Confidence: what the number means

- Confidence is a scalar in **[0.5, 1.0]**. A rule that would be less than 50%
  confident **abstains** instead of firing — strided would rather say nothing than
  be wrong. So every diagnosis you see is at least 50%.
- A single rule caps itself at **0.9**: one rule shouldn't claim near-certainty
  without corroborating evidence from others.
- **Every rule currently caps at 0.65.** Their firing thresholds are *calibration
  seeds*, not validated literature values, so they deliberately hold back
  confidence until tested on real customer dumps. A 65% from r03 means "the signal
  is strong, but the threshold behind it is still provisional." r01 was the last
  rule to join that cap — a real A100 measurement contradicted its 0.80 gate. See
  [limitations.md](limitations.md).

Treat confidence as **triage priority**, not probability of correctness. The
`evidence` line and the per-rule spec are what let you judge correctness.

## Abstentions: the two ways a rule says "no"

A rule that doesn't fire does one of two things, and strided treats them
differently on purpose:

1. **could not evaluate / missing data** — the rule needed schema fields the input
   didn't have. This is *surfaced*, because you should know strided couldn't even
   assess that possibility, and what to provide:
   ```
   ▪ could not evaluate / missing data
     - r01 Decode memory-bound at low batch, needs decode.hbm_bandwidth_util, decode.sm_occupancy
   ```
   The `, needs …` clause names the exact missing fields (or a reason, like "only
   40 successful requests; need ≥200"). [data-sources.md](data-sources.md) maps
   each field to the source that provides it.

2. **Below threshold** — the data was present but the signal didn't cross the firing
   line. This is **silent**: the rule simply doesn't appear. (`examples/membound2.csv`
   is a deliberate example — its HBM utilization sits just under r01's 80% bar, so
   r01 stays quiet and no rule fires.)

The footer's `N rules evaluated` always reflects the full rule set, so you can tell
how many ran versus how many you see.

Below it, when at least one rule fired, `strided diagnose` closes with the command
that acts on the top-ranked finding:

```
  next  strided fix r02 --config <launch-file>
```

Observing is half the loop. That line is the hand-off to the other half, and it names
the rule to adjust so you do not have to copy it out of the card yourself. `watch`
omits it: on a live stream it would repeat under every event.

## Other sections you may see

| Section | When it appears |
|---|---|
| `input warnings` | The input contained non-finite values (NaN/inf), which strided flags. |
| `parse warnings` | A parser noted a caveat, e.g. "Nsight CSV doesn't tag prefill/decode; all kernels aggregated into 'decode'." |
| `rule errors` | A rule raised an exception. Only the exception's **class name** is shown — never its message — so a pasted report can't leak your metric values. Use `--strict` locally to see the full traceback. |
| `suppressed by conflict resolution` | A diagnosis was removed because it conflicts with a higher-confidence one (only when conflict suppression is enabled; off by default). |

## Ranking and the engine

Diagnoses are sorted by `(confidence, then rule id)`, so the order is deterministic
— the same input always produces the same report, regardless of the order rules ran.
The engine **ranks and annotates; it does not invent**: it never raises a rule's
confidence or deletes a diagnosis unless explicitly enabled (both behaviours ship
off by default in v1). Details in
[engine/ARCHITECTURE.md](engine/ARCHITECTURE.md).

## `watch` output

`watch` reuses the same report layout, plus three live-specific elements:

1. **Tiering banner** (once, at start-up) — which rules each connected source can
   feed live, and which need a captured dump:
   ```
   watch · live rules by data source
     vLLM /metrics : r02 (Colocation contention), r03 (KV cache fragmentation)
     DCGM          : r01 (Decode memory-bound)   [connected]
   ```
2. **"state changed" blocks** — a full timestamped report is printed **only when the
   set of firing rules (or surfaced insufficiencies) changes**. A confidence wobble
   alone doesn't re-print; what changed is *which conclusions hold*.
   ```
   ▌ 12:21:22   state changed · firing: r01, r02, r03
   ```
3. **A status heartbeat** — between changes, a single rewritable line shows liveness
   (`tick 4 · 12:21:30 · 3 firing · 1,200 gen tok/s`), or `no data` when every source
   failed or ran dry that tick. On exit, `watch` prints the last diagnosis state.

Dump-only rules (those that can never fire from live polling) are explained once in
the banner and then kept out of the per-tick "could not evaluate" noise.

## See also

- [What the rules detect](rules.md) — the meaning behind each firing.
- [The `fix` walkthrough](fix-walkthrough.md) — the agent's verdicts.
- [Limitations & feedback](limitations.md) — what the confidence caps mean.
