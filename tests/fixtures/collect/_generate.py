#!/usr/bin/env python3
"""Generate (and self-verify) the replay fixtures for `strided watch` tests/demo.

Hand-authoring Prometheus histograms that interpolate to specific p50/p99 values
is error-prone, so we generate the ``.prom`` files here and assert — via the real
parser + engine — that each tick fires the intended rules. Re-run after changing a
rule threshold:  ``.venv/bin/python tests/fixtures/collect/_generate.py``

The replay stream tells a small story so the event log has something to react to:

    tick 00 : vLLM fires r02 + r03, DCGM fires r01      → first CHANGED event
    tick 01 : identical diagnosis, counters advanced     → quiet (status line)
    tick 02 : pressure subsides, r03 clears               → second CHANGED event

r03 now needs fragmentation AND a pressure signal (util OR preemption rate OR
queue), not util alone, so clearing it on tick 02 means dropping *every* pressure
signal: kv_cache_usage falls below 0.80 AND the lifetime preemption rate dilutes
below 1% (a burst of clean completions, preemptions nearly flat). r02 keeps firing:
`num_requests_running` stays above its concurrency floor (16), so the still-high
TPOT tail reads as colocation contention rather than long-context attention cost —
the only state delta is r03.

The concurrency values were once 8/9/7.
That is a server with no meaningful competition, and r02 fired on it anyway
because its fingerprint could not see concurrency at all; the story these fixtures
tell — "prefill and decode are contending" — needs a genuinely busy scheduler to
be true, so the gauges are now 24/26/22 (field-measured contended mean: 32.6).

The DCGM stream carries balanced per-rank SM clocks: r04 (TP imbalance) reads them,
finds no straggler, and stays below threshold — present as a live-capable rule (so
it never lands in INSUFFICIENT_DATA on a full snapshot) without polluting the firing
set. gpu_util is no longer a timings fallback, so clocks are what make r04 evaluable.

vLLM counters grow every tick (so windowed throughput is positive); the latency /
prompt-length histograms are held fixed (their percentiles must stay put).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from engine import run_diagnosis  # noqa: E402
from parsers.dcgm import parse_dcgm_prometheus  # noqa: E402
from parsers.vllm import parse_vllm_metrics  # noqa: E402

# Fixed histograms (shape chosen so the parser interpolates the percentiles we
# want): TPOT p50≈9ms / p99≈50ms → tail ≈5x; prompt-length mean ≈17 tokens so
# that, against block_size 16, internal fragmentation ≈0.47 (> the 0.20 bar).
_HISTOGRAMS = """\
vllm:time_to_first_token_seconds_bucket{le="0.05"} 60
vllm:time_to_first_token_seconds_bucket{le="0.1"} 250
vllm:time_to_first_token_seconds_bucket{le="0.5"} 300
vllm:time_to_first_token_seconds_bucket{le="+Inf"} 300
vllm:time_to_first_token_seconds_count 300
vllm:time_to_first_token_seconds_sum 30.0
vllm:time_per_output_token_seconds_bucket{le="0.005"} 20
vllm:time_per_output_token_seconds_bucket{le="0.01"} 170
vllm:time_per_output_token_seconds_bucket{le="0.025"} 270
vllm:time_per_output_token_seconds_bucket{le="0.05"} 297
vllm:time_per_output_token_seconds_bucket{le="0.1"} 300
vllm:time_per_output_token_seconds_bucket{le="+Inf"} 300
vllm:time_per_output_token_seconds_count 300
vllm:time_per_output_token_seconds_sum 3.6
vllm:request_prompt_tokens_bucket{le="16"} 180
vllm:request_prompt_tokens_bucket{le="32"} 300
vllm:request_prompt_tokens_bucket{le="+Inf"} 300
vllm:request_prompt_tokens_count 300
vllm:request_prompt_tokens_sum 5100
vllm:cache_config_info{block_size="16"} 1.0
"""


def vllm_text(*, preemptions: int, successes: int, prompt_tokens: int,
              gen_tokens: int, kv_util: float, running: int, waiting: int) -> str:
    return (
        f"# strided watch replay fixture (generated)\n"
        f"vllm:num_requests_running {running}\n"
        f"vllm:num_requests_waiting {waiting}\n"
        f"vllm:kv_cache_usage_perc {kv_util}\n"
        f'vllm:num_preemptions_total{{}} {preemptions}\n'
        f'vllm:request_success_total{{finished_reason="stop"}} {successes}\n'
        f"vllm:prompt_tokens_total {prompt_tokens}\n"
        f"vllm:generation_tokens_total {gen_tokens}\n"
        + _HISTOGRAMS
    )


def dcgm_text(*, sm_occ: float, dram_active: float, gpu_util: list[int],
              sm_clock: list[int]) -> str:
    lines = ["# strided watch replay fixture (generated)"]
    for i, util in enumerate(gpu_util):
        lines.append(f'DCGM_FI_PROF_SM_OCCUPANCY{{gpu="{i}"}} {sm_occ}')
        lines.append(f'DCGM_FI_PROF_DRAM_ACTIVE{{gpu="{i}"}} {dram_active}')
        lines.append(f'DCGM_FI_DEV_GPU_UTIL{{gpu="{i}"}} {util}')
        lines.append(f'DCGM_FI_DEV_SM_CLOCK{{gpu="{i}"}} {sm_clock[i]}')
    return "\n".join(lines) + "\n"


# Three ticks. Counters grow; on the last tick kv_util drops below 0.80 AND the
# lifetime preemption rate dilutes below 1% (preemptions nearly flat while a burst
# of completions lands), so every r03 pressure signal clears at once.
_VLLM_TICKS = [
    dict(preemptions=15, successes=300, prompt_tokens=5100, gen_tokens=27000,
         kv_util=0.86, running=24, waiting=3),   # preempt rate 5.0%
    dict(preemptions=22, successes=450, prompt_tokens=7650, gen_tokens=40500,
         kv_util=0.86, running=26, waiting=4),   # preempt rate 4.9%
    dict(preemptions=23, successes=2400, prompt_tokens=17000, gen_tokens=60000,
         kv_util=0.78, running=22, waiting=1),   # preempt 0.96% <1% bar; prefill 22%
]

# DCGM: a memory-bound decode (high HBM, low SM) so r01 fires; held steady. SM
# clocks are balanced (spread < 0.5%) so r04 sees no straggler and stays below
# threshold — evaluable (not INSUFFICIENT_DATA) but not firing.
_DCGM_TICKS = [
    dict(sm_occ=0.15, dram_active=0.88, gpu_util=[95, 80, 82, 81],
         sm_clock=[1400, 1398, 1402, 1399]),
    dict(sm_occ=0.16, dram_active=0.89, gpu_util=[94, 81, 83, 80],
         sm_clock=[1401, 1399, 1403, 1400]),
    dict(sm_occ=0.15, dram_active=0.90, gpu_util=[96, 80, 82, 81],
         sm_clock=[1399, 1397, 1401, 1398]),
]

_EXPECTED_VLLM = [{"r02", "r03"}, {"r02", "r03"}, {"r02"}]
_EXPECTED_DCGM = [{"r01"}, {"r01"}, {"r01"}]


def _fired(dx) -> set[str]:
    return {r.diagnosis.rule_id for r in run_diagnosis(dx).diagnoses}


def main() -> int:
    vllm_dir = HERE / "replay" / "vllm"
    dcgm_dir = HERE / "replay" / "dcgm"
    vllm_dir.mkdir(parents=True, exist_ok=True)
    dcgm_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for i, spec in enumerate(_VLLM_TICKS):
        text = vllm_text(**spec)
        (vllm_dir / f"{i:02d}.prom").write_text(text)
        fired = _fired(parse_vllm_metrics(text, model_name="meta-llama/Llama-3.1-8B", gpu_type="H100"))
        flag = "OK" if _EXPECTED_VLLM[i] <= fired else "MISMATCH"
        ok = ok and (_EXPECTED_VLLM[i] <= fired)
        print(f"vllm/{i:02d}.prom  fired={sorted(fired)}  expected⊇{sorted(_EXPECTED_VLLM[i])}  [{flag}]")

    for i, spec in enumerate(_DCGM_TICKS):
        text = dcgm_text(**spec)
        (dcgm_dir / f"{i:02d}.prom").write_text(text)
        fired = _fired(parse_dcgm_prometheus(text, model_name="meta-llama/Llama-3.1-8B", gpu_type="H100"))
        flag = "OK" if _EXPECTED_DCGM[i] <= fired else "MISMATCH"
        ok = ok and (_EXPECTED_DCGM[i] <= fired)
        print(f"dcgm/{i:02d}.prom  fired={sorted(fired)}  expected⊇{sorted(_EXPECTED_DCGM[i])}  [{flag}]")

    print("ALL OK" if ok else "FAILURES — fixtures do not fire as intended")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
