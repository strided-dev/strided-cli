# Limitations & how to give feedback

You're among strided's first users, so this page is deliberately blunt about what's
provisional. Read it before trusting a diagnosis on production data — and then please
tell us where it was right and where it was wrong.

## The big picture

strided is an **early prototype in a validation sprint**. The whole bet is a narrow
one: *can a deterministic, inspectable diagnosis engine match a senior engineer's
read of real GPU telemetry?* Everything is built to test that bet honestly — which
is why the tool abstains so readily and caps its own confidence. It is **not** a
finished product, and a diagnosis is a starting point for your own analysis, not a
verdict to action blindly.

## What's provisional

### Thresholds are calibration seeds, not validated constants
The numbers that decide whether a rule fires (preemption rates, TPOT-tail ratios,
fragmentation fractions, HBM/occupancy cutoffs) are **seeds chosen from literature
and reasoning**, not yet validated against real customer dumps. Consequences:

- **Every rule caps its confidence at 0.65.** That cap is the tool saying "the
  signal is here, but the threshold behind it is unproven." When these thresholds
  are validated, the caps lift.
- **r01's thresholds (HBM > 0.80, SM < 0.30) are largely from A100-era literature.**
  H100 has a higher HBM peak, so the right cutoffs may shift — and a real A100
  measurement has already contradicted the 0.80 seed (see below). r01 is now under the
  same 0.65 cap; treat boundary firings with the same
  caution.

### Only 9 rules exist
r01 (decode memory-bound), r02 (colocation contention), r03 (KV fragmentation),
r04 (TP rank imbalance), r05 (NCCL dominance), r06 (throughput decay, live-only),
r07 (unfused attention path, dump-only), r08 (prefill↔decode interference from an
Nsight Systems timeline, dump-only), r12 (queue growth, live-only). Many real bottlenecks — quantization
opportunities, long-context costs, MoE routing, scheduler misconfiguration — are
**not yet diagnosed**. A clean report
means "none of these nine fired," not "your serving is healthy."

### Some signals are estimates, not measurements
r03's fragmentation is *estimated* from block size × mean sequence length when the
exporter doesn't expose block accounting (it usually doesn't). It points at the right
knob but is an expected value, not ground truth — hence the confidence cap. The
`evidence` line marks `fragmentation_source=estimate` vs `measured`.

**One confirmed false positive from this estimate (kept here on purpose).** On a
real instruction/chat workload (Alpaca, mean prompt ≈ 17 tokens) at block size 8,
the estimate `frag(mean L) = 0.278` clears r03's 20% gate while the ground truth —
per-request fragmentation averaged over the actual length distribution — is
`mean frag(L) = 0.187`, below it. The gap is a Jensen gap: fragmentation is a
sawtooth in sequence length, so evaluating it at the histogram mean is not the same
as averaging it per request. Across the workloads measured so far this is the one
cell where the shortcut flips the verdict (magnitude error elsewhere is real but
verdict-preserving). It is the first entry in the field-calibration record — if r03
fires on your short-prompt chat workload at a small block size, weigh it against
this known failure mode and tell us either way.

### r01's HBM gate does not reproduce on real A100 decode
*(Partly fixed — see the amendment at the end of this entry.)*

r01 fires on `hbm_bandwidth_util > 0.80 AND sm_occupancy < 0.30`. Measured on an
A100-PCIE-40GB running vLLM 0.10.2 (Qwen2.5-7B), batch-1
decode — the textbook memory-bound case — reads **0.595**, so r01 stays silent on
the pathology it exists to detect. That is a **false negative**: the rule misses
a real problem rather than inventing one. The mechanism itself is confirmed:
raising batch 1 → 256 moves HBM utilisation 0.595 → 0.295 and occupancy
0.124 → 0.170, exactly the predicted walk off the memory-bound corner.

The gap is aggregation, not physics. The dominant decode GEMM sits at **0.795**,
essentially at the seed; but `parsers/nsight.py` folds all kernels into one
`decode` phase, and ~17% of batch-1 time is elementwise/norm work at ~0.002
utilisation, pulling the duration-weighted mean down. That dilution is
asymmetric — at batch 256 those kernels do real work, so they are only ~1% of
time — which biases the aggregate *against* the memory-bound case specifically.
Note that the parser's mean is **already duration-weighted**: `0.720 × (1 −
0.174) = 0.5947` reproduces the measurement. Weighting by duration cannot remove
a kernel that moves no bandwidth from the *denominator* — only scoping the read
to the kernels that carry the work does.

**Amendment (kernel scoping) — the locus half is CLOSED, the threshold
half is OPEN.** r01 now declares a `KERNEL_SCOPE` and reads its bandwidth signal
off the dominant kernel in the per-kernel `layers` breakdown rather than the
collapsed phase average (see the [r01 module docstring](../rules/r01_decode_memory_bound.py)).
Four things you should know about what that did and did not change:

- On the field capture the scoped read recovers **0.795** — closing 0.200 of the
  0.205 shortfall, so **97.6% of the miss was locus, not physics**. That half of
  the finding is closed. The remaining 0.005 is the threshold's: 0.795 still does
  not clear a strict 0.80, so **that capture still abstains**.
- **We are not moving the threshold on this evidence, and it may not need to
  move at all.** `ncu` serialises kernel replay, which stretches measured kernel
  durations and so can depress an apparent bandwidth utilisation — the 0.795 may
  be an artifact of the *instrument* rather than of the workload. The required
  next step is an **`nsys` cross-check of the same arm pair, first**; only then
  does the question of a gate move arise. **If the un-serialised figure clears
  0.80, no threshold change is needed and the finding closes on the locus fix
  alone.** (This is a standing principle, not an r01 quirk: any threshold change
  resting on `ncu`-measured bandwidth needs a second instrument first.)
- **r01 is now capped at 0.65 like every other rule** (`THRESHOLDS_UNCALIBRATED
  = True`). It was the last rule holding the uncapped 0.9 ceiling while carrying
  an open finding against its own gate. Its status is **`carrying-findings`** —
  it is not a validated rule.
- **Every r01 diagnosis now names the locus it read.** `hbm_read_locus` is
  either `layers.dominant_by_total_ms` (scoped) or `decode.collapsed_aggregate`
  (the DCGM path, which has no kernels to scope and is therefore still subject to
  the dilution above). When scoped, the evidence also carries
  `flat_hbm_mean_all_kernels` — the number the old read would have gated on — so
  the gap is visible in the report rather than buried in a parser.

Until the threshold question is settled: **r01 silence is still not evidence that
decode is compute-bound**, and that goes double for a `decode.collapsed_aggregate`
reading. Read the `layers` breakdown directly if you need the roofline position
of the GEMMs.

### r01's occupancy term is a false-positive guard that has never been tested
r01 fires on high bandwidth **and** low SM occupancy. The occupancy half is not a
second detector — it exists to suppress one specific wrong answer: a *saturated
compute-bound* kernel, which moves a lot of bytes because it is doing a lot of
work, and which a bandwidth-only rule would misread as memory-bound and tell you
to raise a batch that is already large.

The field pass measured occupancy at 0.124 and 0.170 against a 0.30 gate — both
arms low, so the guard passed in both and **never had to do its job.** It is
therefore untested rather than validated, and we have said so in code
(`OCCUPANCY_FP_GUARD`) rather than quietly assuming it works. A third validation
arm — compute-bound, high occupancy, r01 must stay silent — is planned and
covered synthetically in the test suite pending hardware.

One known weakness while it stays untested: occupancy is read from the collapsed
phase aggregate while bandwidth is read from the dominant kernel. On a mixed
capture, idle-ish kernels can drag the aggregate under 0.30 even when the kernel
r01 actually read is busy — which is the direction that lets the false positive
through. **If r01 fires on a workload you know to be compute-saturated, that is
the failure mode, and it is exactly the report we want.**

### r02's long-context guard was inverted (fixed; confirming field run pending)
r02 documents a long-context workload as its **dominant false positive**: a fat
TPOT tail caused by attention cost, not by prefill↔decode contention. The guard
meant to suppress that case tested `prefill_share < 0.20`, encoding the assumption
that long-context work has a *low* prefill share.

That assumption is backwards. A long-context request is one whose prompt is large
relative to what it generates, so its prefill share is **high** — measured at
**0.941** on a 4096-token prompt generating 256 tokens (A100, vLLM
0.10.2). The guard's band was exactly where long-context workloads
never land, so its −0.40 confidence penalty was unreachable, and the rule fired at
full strength (score 0.541 against a 0.30 gate, confidence 0.65) on its own
disqualifying control.

Worse, the long-context arm scored **0.541** against the genuinely contended
server's **0.526** — the fingerprint ranks its own false positive above its true
positive, on all three signals it reads. So this is not a matter of flipping a
comparison: no reweighting of preemption rate, TPOT tail and prefill share can
separate the two cases.

**Fixed.** The discriminating signal — concurrent requests,
32.6 vs 5.5 in the same captures — was already being scraped and is now a hard
gate: r02 requires `num_requests_running > 16` before it scores anything, and
abstains with `InsufficientData` when that gauge is absent. It also replaced
`prefill_share` as the score's third term, which reverses the ranking as well as
the verdict: contend now scores 0.550 against longctx's 0.426. `prefill_share`
keeps its honest jobs (admission gate, reported evidence) and is no longer asked
to discriminate anywhere in the firing decision. The three field arms are pinned
as a regression test at their measured values.

**What is still provisional.** The fix is verified offline only; the confirming
field re-run has not happened. The floor is one session's measurement from one
site, one GPU, one 7B model, and r02 reads an *instantaneous* gauge while those
numbers are means over ~100 scrapes — 16 is also the long-context arm's observed
maximum, so its busiest single tick sits on the boundary. Until that run:
**treat an r02 fire on long-prompt / short-generation traffic as strong only if
the reported `num_requests_running` is comfortably above 16.**

**The same fix narrowed r02 deliberately, in a second place.** The TPOT tail
band is now an explicit gate (`tpot_p99/p50 >= 2.0`) rather than one term among
three, so **r02 no longer fires on preemptions alone**: a server preempting 5% or
10% of requests with a flat decode tail was diagnosed before and is silent now.
That is intended — r02 claims prefill is stalling *decode*, and a preemption
storm invisible in the decode tail does not support that claim; r12 (queue
growth) is the rule that owns pure scheduler pressure. But it is the one change
in this fix that costs coverage rather than adding it, and **no field capture
tests it**: all three field arms measured a preemption rate of 0.0, so
none of them lives in the region that was removed. If your server preempts
heavily and r02 is silent, check `num_preemptions_total` against r12 before
concluding there is no contention.

### Inputs strided can't fully use
- **`.ncu-rep` (binary Nsight) isn't parsed yet** — export to CSV. (Stub by design.)
- **DCGM and Nsight CSV can't separate prefill from decode** — their metrics are
  aggregated into the `decode` phase, with a `PARSE WARNING`.
- **A single `/metrics` scrape can't compute throughput rates** — those need two
  snapshots, so request/token-per-second fields stay unset from one dump.

### Engine "smart" features are off by default
The engine can boost a diagnosis's confidence when others corroborate it, and
suppress a diagnosis that conflicts with a higher-confidence one. **Both ship
disabled in v1** — an invented boost could promote a wrong answer, and the conflict
table isn't validated. So today the engine purely **ranks and annotates**; it never
invents a confidence number or deletes a diagnosis. The cross-rule relation tables
are intentionally empty.

### `fix` on a static dump always verifies as "NO CHANGE"
Verifying a fix re-collects metrics, but with a static `--vllm <file>` the "fresh"
snapshot is the same file — so gait honestly reports no change rather than pretending
a file edit altered a past measurement. This is correct behaviour; see the
[`fix` walkthrough](fix-walkthrough.md#why-no-change-here-is-correct-not-a-bug). Real
verification needs a live re-capture.

## What strided does NOT do (by design)

No always-on agent, no eBPF/CUPTI sampling, no dashboard or web UI, no database or
historical query, no multi-vendor (NVIDIA only) or multi-engine (vLLM-first)
support, and no network/telemetry of any kind. It's a local, single-shot CLI.

## Privacy (relevant when you share findings)

strided makes this safe to run on sensitive workloads:

- **No network, no disk persistence, no telemetry.** It reads your dump, prints a
  report, and exits. Nothing is sent anywhere or written behind your back.
- **Rule errors record only the exception's class name** — never its message or
  traceback — so a report you paste into an issue can't leak your metric values. (Run
  with `--strict` locally if you *want* the full traceback for debugging.)

## How to give useful feedback

You are the calibration data this sprint needs. The most valuable reports, roughly
in order:

1. **A diagnosis that disagreed with a human expert.** If a senior engineer looked at
   the same workload and concluded something different, that's gold — tell us the
   workload, what strided said, and what they said.
2. **False positives / false negatives.** A rule fired when it shouldn't have, or
   stayed silent when it clearly should have. Include the `evidence` line.
3. **A confidence that felt wrong.** Right diagnosis but the % seemed too high/low,
   or a 65%-capped rule that you'd trust at 90%.
4. **Parser failures or surprises.** A dump that wouldn't parse, parsed wrong, or
   produced a confusing `PARSE WARNING`. Re-run with `--strict` to capture the
   traceback.
5. **Missing rules.** A real bottleneck you hit that none of the shipped rules covers.

Open an issue at <https://github.com/strided-dev/strided-cli/issues>. When you can,
attach the (redacted) dump and the exact command. A dump + the
`evidence` line is far more actionable than a screenshot — and because the dump never
leaves your machine unless you choose to share it, redact whatever you need to first.

## See also

- [What the rules detect](rules.md) — the literature and reasoning behind each rule.
- [Rule modules](../rules/) — each rule's docstring states its thresholds and what
  still needs validation.
- [Extending strided](extending.md) — if you'd rather add the missing rule yourself.
