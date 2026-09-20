# KV-Cache-Aware LLM Serving on Constrained GPUs

Serving an LLM with vLLM on a memory-limited GPU, then measuring exactly where the KV cache
saturates and what it costs in tail latency. Runs on CU Boulder's Alpine supercomputer: an
OpenAI-compatible vLLM API on a 20 GB NVIDIA A100 MIG slice, scheduled with Slurm and
containerized with Apptainer.

Everything here is a reproducible harness: one command runs a workload at a given concurrency,
collects vLLM engine metrics and GPU telemetry alongside the load, and writes a self-contained
run directory of CSVs that the analysis scripts aggregate and plot.

## Results: the `heavy` sweep (2048 prompt / 1024 output tokens, 512 requests)

| Concurrency | Throughput (tok/s) | p50 latency (s) | p99 latency (s) | Peak KV usage | Preemptions |
|---|---|---|---|---|---|
| 32 | 2,186 | 15.0 | 15.3 | 47% | 0 |
| 64 | 3,577 | 18.3 | 19.6 | 90% | 0 |
| 96 | 4,208 | 21.5 | 24.2 | 100% | 1,660 |
| 112 | 4,602 | 24.0 | 30.1 | 100% | 2,164 |
| 128 | 4,578 | 24.8 | 33.5 | 100% | 1,035 |

The KV cache fills between concurrency 64 and 96. Past that point the engine keeps admitting
requests and pays for them by preempting and recomputing: throughput flattens (4.2k -> 4.6k ->
4.6k tok/s) while p99 latency keeps climbing and the waiting queue becomes permanently
non-empty. Concurrency 64 is the last operating point with zero preemptions; concurrency ~112
is where added load buys tail latency instead of tokens.

Earlier phase-1 runs on a tighter cache configuration hit the harder version of the same wall
(throughput halving, p99 in the hundreds of seconds, 2,075 preemptions); those CSVs and plots
are kept under `archive/phase1/`.

## KV cache, briefly

vLLM stores per-token key/value tensors in fixed-size blocks (PagedAttention). When the block
pool runs out, the scheduler preempts requests — evicting their blocks and recomputing them
later — so the GPU spends cycles re-doing prefill instead of generating tokens. The wall
condition is roughly `concurrency x tokens_per_request > KV cache capacity`, and the observable
signature is `gpu_cache_usage_perc` pinned at 100% with `num_preemptions_total` rising.

## Repo layout

```
benchmark/
  run_experiment.py   one run: starts collectors, drives load, writes results/<run_id>/
  run_sweep.py        loops run_experiment.py over the concurrency list in the config
  loadgen.py          concurrent streaming load generator (throughput, TTFT, p50/p95/p99)
  vllm_metrics.py     scrapes vLLM /metrics (KV usage, running/waiting, preemptions) to CSV
  gpu_metrics.py      nvidia-smi telemetry (utilization, memory, power, temperature) to CSV
analysis/
  collect_results.py  concatenates every run's request_summary.csv -> results/all_results.csv
  plot_results.py     throughput / latency / TTFT / failure-rate vs concurrency, plus frontier
  plot_vllm_metrics.py per-run time series of KV usage, queue depth, preemptions
scripts/
  env.sh              Alpine paths, container image, model, vLLM server settings
  gpu_testing_20gb.sh sinteractive request for an A100 3g.20gb MIG slice
  serve_vllm_qwen.sh  launches the vLLM OpenAI server inside Apptainer
  test_vllm.sh        smoke test against /v1/models and /v1/chat/completions
configs/workloads.yaml  workload token shapes and the sweep's concurrency list
results/                one directory per run (see below)
archive/phase1/         earlier harness and the original KV-cache-wall data
```

## Workloads

Defined in `configs/workloads.yaml` as prompt/output token targets:

| Workload | Prompt tokens | Output tokens |
|---|---|---|
| `short` | 128 | 128 |
| `medium` | 512 | 256 |
| `context_heavy` | 2048 | 256 |
| `generation_heavy` | 512 | 1024 |
| `heavy` | 2048 | 1024 |

`context_heavy` stresses prefill and KV footprint per request; `generation_heavy` stresses
decode steps and cache growth over time; `heavy` does both and is what reaches the wall fastest.

## Running an experiment

Allocate a GPU and start the server (from the Alpine login node):

```bash
./scripts/gpu_testing_20gb.sh          # interactive A100 3g.20gb slice
./scripts/serve_vllm_qwen.sh           # vLLM server in Apptainer, port 8000
./scripts/test_vllm.sh                 # smoke test
```

Then drive load from another shell on the same node:

```bash
# single operating point
python3 benchmark/run_experiment.py --workload heavy --concurrency 112

# full concurrency sweep from configs/workloads.yaml
python3 benchmark/run_sweep.py --workload heavy
```

`run_experiment.py` issues `max(512, concurrency * requests_per_worker)` requests so that every
concurrency level generates a comparable amount of work, starts the vLLM and GPU collectors
before the load and shuts them down 5 s after it, and names the run
`<UTC timestamp>_<workload>_c<concurrency>_<hash>`.

Aggregate and plot:

```bash
python3 analysis/collect_results.py      # -> results/all_results.csv
python3 analysis/plot_results.py         # -> plots/*.png
python3 analysis/plot_vllm_metrics.py    # -> plots/kv_cache_<run_id>.png, etc.
```

## What a run directory contains

```
results/20260828T220254Z_heavy_c112_dfd124/
  request_summary.csv   one row: throughput, p50/p95/p99 latency and TTFT, success/failure counts
  request_details.csv   one row per request: TTFT, latency, token counts, error
  vllm_metrics.csv      1 Hz: running, waiting, kv_usage_pct, cumulative preemptions, token totals
  gpu_metrics.csv       1 Hz: GPU/memory utilization, memory used, power draw, temperature
```

Token counts come from the server's own `usage` field in the final stream chunk
(`stream_options.include_usage`), not from character counts — `token_count_source` records which
was used so a run with missing usage data is never mistaken for exact throughput.

## Configuration

Server settings live in `scripts/env.sh`: the model (`Qwen/Qwen2.5-1.5B-Instruct`), the
container image, `VLLM_MAX_MODEL_LEN=8192`, and `VLLM_GPU_MEMORY_UTILIZATION=0.3` — the knob
that sets how much of the MIG slice the KV cache pool gets, and therefore where the wall sits.
The server runs with `--enforce-eager` (single-process execution), which avoids a
multiprocessing crash on engine init under MIG.

## Environment notes

Three things had to be worked around to get vLLM running on Alpine:

1. CUDA driver/runtime mismatch — the node ships 12.8 while recent vLLM builds want 12.9, so the
   project pins a known-good CUDA 12.4 vLLM 0.8.5 image.
2. MIG multiprocessing crash on engine init — fixed by eager, single-process execution.
3. Dependency conflicts on a shared cluster — avoided entirely by running the server inside
   Apptainer, with `HF_HOME` and the Apptainer cache on project storage.

## Roadmap

- [x] Phase 0 — benchmark harness, metrics pipeline, run/sweep automation
- [x] Phase 1 — reproduce the KV-cache wall (2,075 preemptions)
- [x] Phase 2 — streamlined experiment runner with per-run vLLM + GPU telemetry; heavy-workload
      sweep locating cache saturation between concurrency 64 and 96
- [ ] Phase 3 — admission control (`--max-num-seqs`), prefix caching, chunked prefill
- [ ] Phase 4 — KV-pressure-aware routing across replicas
- [ ] Phase 5 — modify vLLM's scheduler/preemption policy from source

## Tech stack

vLLM (PagedAttention), CUDA, PyTorch, Slurm/HPC, Apptainer, Prometheus metrics, pandas,
matplotlib, Python, NVIDIA A100 (MIG 3g.20gb)
