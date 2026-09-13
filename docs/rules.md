# What the rules detect

strided ships **9 diagnostic rules**. Each one looks at a parsed snapshot (or, for
the temporal rule, the live series `watch` builds), decides
whether a specific pathology is present, and either fires a `Diagnosis` (with a
cause, a fix, evidence, and confidence) or abstains. Rules are deterministic and
backed by published literature — no ML, no folk wisdom.

This page is the plain-English tour. For the full thresholds, confidence math, and
false-positive guards, read each rule's module docstring under [`rules/`](../rules/).

---

## r01 — Decode memory-bound at low batch

**What it means.** During token-by-token decoding, the GPU reloads the model
weights every step. If the batch is small, there aren't enough tokens to amortize
that memory traffic, so the GPU spends its time waiting on HBM instead of computing
— it's *memory-bound*. You're paying for compute you can't use.

**Fires when** decode HBM bandwidth utilization is high **and** SM occupancy is low
(`hbm_bandwidth_util > 0.80` and `sm_occupancy < 0.30`). The bandwidth term is the
detector; the occupancy term is a **false-positive guard** that keeps r01 off a
saturated compute-bound kernel — one that moves a lot of bytes because it is doing
a lot of work. The guard has never been exercised in the field; it is covered
synthetically in the test suite.

**Read on the dominant kernel, not the phase average.** When per-kernel data is
present, the bandwidth number comes from the single kernel holding the most time
(≥15% of profiled kernel time) — averaging it across every kernel in the trace
buried the signal on real hardware. Every diagnosis prints
`hbm_read_locus` so you can tell which reading you got.

**Fix.** Increase the decode batch size to raise arithmetic intensity (or, if KV
cache headroom is the limit, free capacity first via chunked prefill).

**Needs.** `layers[].hbm_bandwidth_util` (Nsight) or `decode.hbm_bandwidth_util`
(DCGM), plus `decode.sm_occupancy` — kernel/GPU counters that come from **Nsight
Compute** or **DCGM**, not a vLLM scrape.

**Confidence ceiling:** 0.65 (capped — thresholds uncalibrated).

---

## r02 — Colocation contention (prefill ↔ decode)

**What it means.** On a standard vLLM deployment, prefill (processing the prompt)
and decode (generating tokens) share one GPU pool. When a new prompt arrives, the
scheduler pauses in-flight decodes or fuses the long prefill into the decode batch —
stalling per-token latency. This rule detects that interference from a
serving-metrics fingerprint and quantifies the "interference tax."

**Fires when**, on a confirmed-colocated deployment, **at least 16 requests are
resident at once** and the fingerprint is strong: a non-trivial preemption rate, a
heavy TPOT tail (`p99/p50`), and prefill making up a real share of token work. The
concurrency floor is what guards the main false positive (long-context workloads,
whose TPOT tail comes from attention cost, not contention): contention needs
competition, a long tail does not. Field-measured 32.6 concurrent requests on a
contended server against 5.5 on a long-context one. An earlier `prefill_share`
guard could not do this job: long-context work has a *high* prefill share, not a
low one.

**Fix (one of two tiers).** Mild → tune **chunked prefill**
(`--enable-chunked-prefill`, `--max-num-batched-tokens`). Severe *and* at scale →
consider **PD disaggregation** (separate prefill/decode pools), but only if bound by
both a TTFT and a TPOT SLO.

**Needs.** The vLLM serving block (`num_preemptions_total`, `request_success_total`,
`prompt_tokens_total`, `generation_tokens_total`, `num_requests_running`) plus
`tpot_ms` percentiles — all from **vLLM `/metrics`** — and both `prefill`/`decode`
present to confirm a colocated topology. Without `num_requests_running` the rule
abstains with INSUFFICIENT_DATA rather than guessing: it is the field that tells
contention from long context.

**Confidence capped at 0.65** while thresholds are uncalibrated.

---

## r03 — KV cache fragmentation

**What it means.** KV cache is allocated in fixed-size **blocks**. When a sequence's
length isn't a multiple of the block size, its last block is partly empty — wasted
capacity that can't be given to another sequence. Across many sequences this
*internal fragmentation* caps the concurrent batch and forces premature
preemption. (Note: if you run vLLM/SGLang you already have PagedAttention; the fix
is **not** "enable PagedAttention" — it's tuning what you have.)

**Fires when** fragmentation is high **and** the cache is near capacity
(`kv_cache_fragmentation > 0.20` and `kv_cache_util > 0.80`). High utilization alone
is healthy demand, not fragmentation, and won't fire.

**Fix (engine-specific).** vLLM/SGLang → reduce `--block-size` toward your sequence
mix, ensure a v2/FlashAttention backend, or relieve pressure. TRT-LLM → enable paged
KV and tune `tokens_per_block`. Unknown/non-paged → adopt PagedAttention.

**Needs.** `kv_cache_fragmentation` and `kv_cache_util` from **vLLM `/metrics`**.
Fragmentation is *estimated* from block size × sequence length when block accounting
isn't exported (it usually isn't), which is why it's confidence-capped.

**Confidence capped at 0.65** while the estimate is uncalibrated.

---

## r04 — Tensor-parallel rank imbalance

**What it means.** In tensor parallelism every rank synchronises at an all-reduce
after every matmul — a barrier that can't complete until the *slowest* rank
arrives. One degraded GPU (thermal throttle, bad NVLink, noisy neighbour) doesn't
just slow itself; it drags all N GPUs to its pace. The rule looks for a single
rank standing apart as a robust outlier in per-rank **SM clocks** — a throttling
GPU down-clocks, and busy-wait can't mask that.

**Fires when** one rank's clock lags the median peer by ≥ 10% *and* it's a robust
statistical outlier (modified z-score > 3.5). Needs ≥ 3 ranks — with two numbers
you can't tell an outlier from a difference.

**Fix.** Deliberately *not* "rebalance the shards": it names the rank and orders
the investigation — hardware/thermals first, noisy neighbour second, sharding last.

**Needs.** `tp_rank_sm_clocks` from **DCGM**. Utilization-only DCGM isn't enough —
the rule abstains rather than guess from a maskable signal.

**Confidence capped at 0.65** while thresholds are uncalibrated.

---

## r05 — NCCL collective dominates step time

**What it means.** Ranks synchronise on collectives (all-reduce/all-gather) twice
per transformer layer — hundreds per forward pass. When those collectives, rather
than compute, consume a disproportionate share of step time, throughput is bound
by the *interconnect*, not the silicon.

**Fires when** `nccl_time_pct` exceeds a topology-relative band: ~20% on an
inferred single node (NVLink collectives should be cheap), ~45% multi-node (high
NCCL share is partly inherent there). Self-guards against r04: if one rank is a
straggler, high NCCL time is that rank *waiting at the barrier*, and r05 steps
aside.

**Fix.** A network/topology *triage*, not a verdict: confirm NVLink/NVSwitch vs a
PCIe fallback, the multi-node fabric (IB/RoCE, GPUDirect RDMA), and NCCL
algorithm/protocol settings.

**Needs.** `nccl_time_pct`, computed by the **Nsight** parser from kernel time —
a captured dump; this rule never fires from live polling.

**Confidence capped at 0.65** while the bands are uncalibrated.

---

## r06 — Throughput decay over a sustained run

**What it means.** The first *temporal* rule: generation throughput trends down
over a sustained `watch` run while a mechanism that explains it as work-limited
is building — memory pressure (rising preemptions or queue time) or a thermal
throttle (falling SM clock as temperature climbs). A downtrend alone never
fires: load that tapered is not a fault, and the rule refuses to invent a cause.

**Fires when** the throughput series trends significantly down (Mann–Kendall)
*and* an attributable mechanism leg opens. Only opened legs contribute to the
score.

**Fix (mechanism-specific).** Memory pressure → relieve it (vLLM:
`gpu_memory_utilization`, batch ceiling, capacity). Thermal → cooling/power, not
config.

**Needs.** `throughput_history` — the per-tick series only `strided watch`
builds. One-shot `diagnose` reports it as an idle trend rule, quietly.

**Confidence capped at 0.65** while thresholds are uncalibrated.

---

## r07 — Attention bottleneck (unfused attention path)

**What it means.** A kernel trace showing *standalone softmax* kernels eating a
material share of GPU time — with no fused attention kernel (FlashAttention /
SDPA / fMHA) anywhere in the trace — is the fingerprint of an unfused attention
path, which materialises the S×S score matrix and pays a separate memory-bound
softmax pass for it.

**Fires when** standalone softmax ≥ 8% of total kernel time (or generic
attention kernels dominate) *and* fused-attention share is absent/minor — the
fused check is a hard guard, so an already-optimised path can't fire it.

**Fix.** Enable a fused attention backend (FlashAttention / `sdpa` /
FlashInfer) — one of the best-documented single-change speedups in the
literature (Dao et al., NeurIPS 2022).

**Needs.** The per-kernel `layers` breakdown from an **Nsight** dump — like
r05, it never fires from live polling.

**Confidence capped at 0.65** while thresholds are uncalibrated.

---

## r08 — Prefill↔decode interference (timeline)

**What it means.** With continuous batching, scheduler steps run one at a time on
the GPU, so a long prefill step doesn't overlap decode; it *blocks* it, leaving a
gap in the token stream. r02 infers this from serving counters; r08 sees it
directly in an **Nsight Systems** timeline. When both fire they corroborate each
other.

**Fires when** the timeline's prefill steps run well past the decode-step baseline
and the excess costs a material share of busy time.

**Fix.** A *computed* chunked-prefill budget: `--max-num-batched-tokens` sized from
the trace's own prefill token rate and decode-step slack, not a generic range.

**Needs.** `nsys_timeline.steps` from `strided diagnose --nsys <export.csv>`
(try `examples/nsys_prefill_interference.csv`). Dump-only.

**Confidence capped at 0.65** while thresholds are uncalibrated.

---

## r12 — Queue growth

**What it means.** A waiting queue that grows *linearly* is a server running hot but
keeping pace. One whose growth *accelerates* is past saturation: by queueing
theory the backlog diverges as utilisation approaches 1, and latency follows it.

**Fires when** `num_requests_waiting` trends up over a sustained `watch` run *and*
the growth is super-linear.

**Fix.** Shed or route load: add replicas, use least-outstanding-requests routing,
or add admission control upstream. Raising the scheduler cap only deepens the
spiral on a GPU that is already full.

**Needs.** A live or replayed vLLM series (`strided watch --vllm ...`). Temporal:
it can't fire from a single snapshot.

**Confidence capped at 0.65** while thresholds are uncalibrated.

---

## How rules relate

Rules are evaluated independently and in isolation — no rule can see another. Any
cross-rule reasoning (corroboration, conflict) lives in the engine, not the rules.
Two relationships are documented but **not yet active** in v1 (they need a "cause-of"
mechanism the engine doesn't have, and validation data):

- r02 (colocation) is often the *upstream cause* of r01 and r03 symptoms on a
  colocated dump.
- r03 (fragmentation) is often the *upstream cause* of r01's memory-bound decode
  (fragmentation caps the batch, the small batch makes decode look memory-bound).

So when multiple rules fire together, read them as a stack, not a list — and check
the `evidence` lines. Details in [engine/ARCHITECTURE.md](engine/ARCHITECTURE.md).

## Which source feeds which rule live

When you run `strided watch`, not every rule can fire from every source. The
start-up banner spells this out; here's the mapping for the registered rules:

| Rule | Live source that feeds it | Notes |
|---|---|---|
| r01 | **DCGM** (`--dcgm`) | needs GPU occupancy + HBM counters |
| r02 | **vLLM** (`--vllm`) | needs the serving counters + TPOT |
| r03 | **vLLM** (`--vllm`) | needs KV cache usage + block size |
| r04 | **DCGM** (`--dcgm`) | needs per-rank SM clocks (multi-GPU) |
| r05 | — dump-only | needs an Nsight capture; never fires live |
| r06 | **vLLM** (`--vllm`), temporal | builds its series over the run; can't fire one-shot |
| r07 | — dump-only | needs an Nsight capture; never fires live |
| r08 | — dump-only | needs an Nsight Systems timeline (`--nsys`); never fires live |
| r12 | **vLLM** (`--vllm`), temporal | builds its queue series over the run; can't fire one-shot |

So a `watch --vllm <url>` run (no DCGM) can fire r02, r03, and — given a sustained
run — r06 and r12, but not r01/r04; add `--dcgm <url>` to enable those. Rules that need a
captured Nsight dump (r05, r07, r08) are flagged as "dump-only" in the banner and
pointed at `strided diagnose`. The tiering is defined in `collect/tiers.py` and
pinned to the rules' real behaviour by tests, so it can't silently drift.

## See also

- [Interpreting output](interpreting-output.md) — how a firing is rendered.
- [Bringing your own data](data-sources.md) — how to capture the inputs each rule needs.
- [Extending strided](extending.md) — how to add a fourth rule.
