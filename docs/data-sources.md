# Bringing your own data

Once you've explored the bundled examples, point strided at a real workload. This
page covers the three input formats, how to capture each, what fields they
populate, and how multiple sources combine.

strided reads **snapshots** — a captured dump, or a single live `/metrics` poll. It
is not a profiler or an always-on collector; you bring the telemetry, it does the
diagnosis.

## The three sources at a glance

| Source | Capture | Feeds rules | Strength |
|---|---|---|---|
| **vLLM `/metrics`** | `curl` the endpoint, or save it to a file | r02, r03 | Easiest. Serving counters, KV cache, latency. |
| **DCGM** | `dcgmi dmon --json`, or dcgm-exporter | r01 | Cluster GPU counters: occupancy, HBM, per-rank. |
| **Nsight Compute** | `ncu --csv` | r01 | Per-kernel detail; phase/roofline classification. |

You can pass any combination to `diagnose` / `fix`; they're merged (see
[Combining sources](#combining-sources)).

---

## vLLM `/metrics`

The easiest and richest source. vLLM exposes Prometheus-format text at `/metrics`.

**Capture it:**

```bash
# From a running vLLM server:
curl -s http://localhost:8000/metrics > mydump.prom

# Then diagnose:
strided diagnose --vllm mydump.prom --gpu H100-SXM --model meta-llama/Llama-3-8B
```

For a representative dump, capture **after** the server has served real traffic —
r02 needs at least ~200 successful requests before it trusts the histograms, and the
KV/latency signals are meaningless on an idle server.

**What the parser populates:** KV cache usage and block size (from
`kv_cache_usage_perc` and the `cache_config_info` label), the serving block
(preemptions, request/token totals, queue gauges), and TTFT/TPOT/e2e latency
distributions (from the histograms). It also derives a prefill phase from TTFT and a
KV-fragmentation **estimate** from block size × sequence-length mean.

**What it can't give you:** per-second throughput rates (those need two scrapes — a
single dump leaves them unset), and GPU-kernel counters like SM occupancy or HBM
utilization (those are r01's domain — use DCGM or Nsight).

---

## DCGM

NVIDIA Data Center GPU Manager — cluster-level GPU telemetry (occupancy, HBM
bandwidth, NVLink, per-rank). This is what feeds **r01** live.

**Capture it (two accepted shapes):**

```bash
# A) dcgmi dmon JSON  → for `strided diagnose --dcgm`:
dcgmi dmon -e 1007,1005,203 --json > dcgm.json
strided diagnose --dcgm dcgm.json --gpu H100-SXM

# B) dcgm-exporter Prometheus text → for `strided watch --dcgm <url>`:
strided watch --dcgm http://localhost:9400/metrics --vllm http://localhost:8000/metrics
```

> `strided diagnose --dcgm` expects **JSON** (the `dcgmi dmon --json` or
> dcgm-exporter snapshot form). The live `watch --dcgm <url>` path polls the
> dcgm-exporter **Prometheus text** endpoint. Same fields, two transports.

**What the parser populates:** aggregate SM occupancy and HBM bandwidth utilization
(preferring the profiling counters `DCGM_FI_PROF_SM_OCCUPANCY` /
`DCGM_FI_PROF_DRAM_ACTIVE`), a roofline classification, `num_gpus`, and a per-rank
timing proxy for future TP-imbalance detection.

**Caveat:** DCGM is a cluster snapshot and can't separate prefill from decode — its
aggregate metrics are placed in the `decode` phase, and the parser says so in a
`PARSE WARNING`. Use Nsight for phase-resolved data.

---

## Nsight Compute

Per-kernel GPU profiling. The richest source, and the one that can classify the
roofline position directly.

**Capture it (CSV):**

```bash
ncu --csv --page raw -o - <your-workload> > report.csv
# or export from the Nsight Compute UI: File → Export → CSV (raw or details page)
strided diagnose --nsight report.csv --gpu H100-SXM
```

> **Set `--gpu` for Nsight inputs.** The parser converts observed byte counts into a
> [0,1] HBM-bandwidth utilization using the GPU's *peak* bandwidth, so the GPU type
> directly affects the computed signal (and therefore whether r01 fires). It
> defaults to H100-SXM if unset.

**`.ncu-rep` is not yet supported.** The binary report format needs the full Nsight
Compute install and CUDA bindings; that parser is a documented stub. Export to CSV
instead — strided prints a clear message (not a traceback) if you pass a `.ncu-rep`.

**What the parser populates:** a `LayerMetrics` entry per kernel (occupancy, HBM
util, achieved FLOPs, a layer-type classification, roofline position) and a
duration-weighted aggregate phase. Note that Nsight CSV doesn't tag prefill vs
decode, so all kernels are aggregated into `decode` with a `PARSE WARNING`;
per-kernel detail stays in `layers`.

---

## Combining sources

Pass several at once and strided merges them into one `DiagnosisInput` before the
rules run — e.g. vLLM for the serving/KV signals and Nsight (or DCGM) for the
kernel signals, so r01, r02, and r03 can all evaluate:

```bash
strided diagnose --vllm dump.prom --nsight report.csv --gpu H100-SXM
```

**Merge precedence** (in `schema/merge.py`):

- Sources are merged in fixed order **vLLM → DCGM → Nsight**, field by field,
  **first-non-None wins**. Nested phase metrics recurse, so vLLM's TTFT-derived
  prefill phase gets *enriched* with Nsight's occupancy/HBM rather than overwritten.
- `--model` and `--gpu` overrides always win, regardless of source order.
- Provenance lists (`source_files`, `parse_warnings`) are concatenated.

This is why parsers are careful never to write the same field with differently-scoped
values: vLLM stores TPOT in `decode.latency_ms` (a per-token figure), *not*
`decode.duration_ms`, so Nsight's kernel-measured phase duration is the one that
survives the merge.

## The canonical schema

Everything above writes into one object: `DiagnosisInput`, defined in
[`schema/diagnosis_input.py`](../schema/diagnosis_input.py). It's the **only**
interface between parsers and rules.

- **Required:** `model_name`, `gpu_type` (with `inference_engine` defaulting to
  `unknown`). Everything else is optional — parsers fill what they can, rules guard
  for what's missing.
- If a dump carries no usable metrics, strided still validates it but appends a
  warning that rules will produce nothing. The `could not evaluate` section then
  tells you exactly which fields each rule wanted — cross-reference them with the
  tables above to know which source to capture next.

## Capturing for `watch --replay`

To build an offline replay (a demo, or a captured incident), drop sequential
scrapes into a directory:

```
mycapture/
  vllm/   00.prom  01.prom  02.prom   ← one vLLM /metrics scrape per file
  dcgm/   00.prom  01.prom  02.prom   ← optional paired DCGM scrapes
```

Files are replayed one per tick in natural-sorted order (so `2.prom` precedes
`10.prom`). Then: `strided watch --replay mycapture`. The bundled
[`examples/replay/`](../examples/) follows exactly this layout.

## See also

- [What the rules detect](rules.md) — which fields each rule needs.
- [Commands reference](commands.md) — all input flags.
- [Limitations & feedback](limitations.md) — known parser gaps.
