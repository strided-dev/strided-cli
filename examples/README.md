# Example data

Bundled telemetry dumps so you can run every `strided` command **offline** — no
GPU, no running vLLM server. Each file is a real input format the parsers accept;
the replay captures are synthetic but well-formed.

| File | Format | Run it | Fires |
|---|---|---|---|
| `vllm.prom` | vLLM `/metrics` (Prometheus text) | `strided diagnose --vllm examples/vllm.prom --gpu H100-SXM` | **r02** |
| `vllm_kv_fragmentation.prom` | vLLM `/metrics` | `strided diagnose --vllm examples/vllm_kv_fragmentation.prom --gpu H100-SXM` | **r02 + r03** |
| `membound.csv` | Nsight Compute CSV export | `strided diagnose --nsight examples/membound.csv --gpu H100-SXM` | **r01** |
| `membound2.csv` | Nsight Compute CSV export | `strided diagnose --nsight examples/membound2.csv --gpu H100-SXM` | _nothing_ — HBM util ~60% sits below r01's 80% bar, so the rule stays silent (a "below threshold" example) |
| `replay/` | a 3-scrape sequence (vLLM + DCGM) | `strided watch --replay examples/replay --max-ticks 3 --interval 0` | **r01 + r02 + r03** |
| `launch.txt` | a sample vLLM launch command | used by `strided fix` / `strided undo` (see below) | — |

Note the `--gpu` flag is part of the diagnosis, not just metadata: utilisation is
judged against that GPU's peak bandwidth, so the *same* Nsight file can fire on one
GPU and stay silent on another (`membound2.csv` is below r01's bar on H100-SXM but
fires on A100). If a diagnosis surprises you, check the `--gpu` you passed first.

## The `fix` / `undo` walkthrough

`launch.txt` is a stand-in for the command you'd use to start vLLM. `strided fix`
reads and (after you approve) edits it; `strided undo` restores it.

```bash
# 1. See the plan without changing anything:
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --dry-run

# 2. Apply it (r03 caps confidence at 65%, so non-interactive --yes needs a lower bar):
strided fix r03 --vllm examples/vllm_kv_fragmentation.prom --config examples/launch.txt --yes --threshold 0.6

# 3. Put it back:
strided undo --config examples/launch.txt
```

> The `fix` verify step re-reads the **same static dump**, so it honestly reports
> `NO CHANGE` — gait won't pretend a file edit changed a past measurement. On a
> live server you'd re-collect fresh metrics instead. See
> [`docs/fix-walkthrough.md`](../docs/fix-walkthrough.md).

`replay/` mirrors the watch test fixtures (`tests/fixtures/collect/replay/`):
`replay/vllm/*.prom` is the vLLM stream, `replay/dcgm/*.prom` the paired DCGM
stream, replayed one file per tick in natural-sorted order.
