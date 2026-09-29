# KV-Cache-Aware LLM Serving on Constrained GPUs

Serving an LLM with vLLM on a memory-limited GPU, then measuring where the KV cache saturates
and what that costs in *useful* served load. Runs on CU Boulder's Alpine supercomputer: an
OpenAI-compatible vLLM 0.8.5 API on a 20 GB NVIDIA A100 MIG slice, scheduled with Slurm and
containerized with Apptainer.

The harness is built around three things that make the measurement honest:

- **Unique prompts by default**, so requests don't hit vLLM's prefix cache and the GPU does the
  prefill work a real workload would cause.
- **SLO-based goodput**, not raw token throughput — a server past its knee still emits plenty of
  tokens, just not ones anyone is waiting on happily.
- **Open-loop (Poisson) arrivals**, so a stuck request doesn't throttle the load generator and
  hide the overload it caused (coordinated omission).

## Headline result: throughput hides the knee, goodput shows it

`heavy` workload (2048 prompt / 1024 output tokens), unique prompts, closed loop, 3 repeats per
point, SLOs of TTFT < 2 s and TPOT < 50 ms. Values are mean [min-max] across repeats
(`results/knee_summary.txt`, produced by `analysis/knee_table.py`):

| Concurrency | Goodput (req/s) | SLO attainment | Throughput (tok/s) | TTFT p99 (s) | Preemptions |
|---|---|---|---|---|---|
| 16 | 1.21 | 100% | 1,237 | 1.32 | 0 |
| 20 | 1.44 | 100% | 1,471 | 1.61 | 0 |
| **24** | **1.64** | **100%** | 1,686 | 1.95 | 0 |
| 28 | 1.45 | 95% | 1,566 | 2.31 | 17 |
| 32 | 1.41 | 85% | 1,709 | 8.10 | 46 |
| 40 | 1.00 | 59% | 1,742 | 17.04 | 80 |
| 48 | 0.21 | 12% | 1,774 | 17.65 | 96 |

Raw throughput rises monotonically all the way to c48 (1,237 → 1,774 tok/s) and would suggest
the server is fine. Goodput peaks at **c24** and then falls off a cliff — at c48 the server is
producing its highest token rate while satisfying 12% of requests. The crossover lines up with
the cache: peak KV usage is 0.65 at c16, 0.98 at c24, and pinned at 1.00 from c28 on, which is
exactly where preemptions start and TTFT p99 breaks the 2 s SLO.

Also visible past the knee: `max_running` stops tracking concurrency (36 running at c48) while
`max_waiting` keeps growing — the scheduler is admitting a fixed working set and queueing the
rest.

## Prefix caching makes benchmarks lie (shared vs unique prompts)

Same 48 concurrent requests, same workload, only the prompt mode differs (`results/l4_*`):

| Prompt mode | Prefix-cache hit rate | Peak KV usage | Throughput (tok/s) | TTFT p99 (s) | SLO attainment |
|---|---|---|---|---|---|
| `shared` | 99.0% | 68% | 3,020 | 0.24 | 100% |
| `unique` | 46.1% | 100% | 1,622 | 18.33 | 14% |

With one identical prompt body reused by every request, vLLM serves nearly all prefill from
cached blocks: 1.9x the throughput, 77x lower tail TTFT, and the KV cache never even fills.
That is a measurement of the cache, not of the model serving load. `--prompt-mode unique`
(the default) prefixes each request with `[req=<i> nonce=<uuid4>]`; because prefix-cache block
hashes are chained, a unique *first* block makes every later block unique too, and the uuid
nonce keeps a rerun from hitting blocks left over from the previous run.

The residual 46% hit rate under `unique` is intra-request reuse (a request's own blocks across
its decode steps), not cross-request sharing.

Earlier sweeps in this repo (including the phase-2 numbers that reached ~4.6k tok/s at c112,
kept under `results/` and `archive/phase1/`) predate the prompt-mode flag and were effectively
shared-prompt runs, so their absolute throughput is prefix-cache inflated and not comparable to
the table above.

## Open-loop arrivals

Closed-loop load (N threads, each sending the next request only after the previous returns) is
self-throttling: when the server stalls, the client stops offering load, so the measured
arrival rate silently drops to whatever the server can handle. `--arrival-rate` instead
generates Poisson arrivals at a fixed mean rate, independent of completions, with a thread pool
of up to `--max-inflight` workers.

Latency is measured from the *scheduled* send time (`t_sched`), not from when a worker actually
got the socket, so client-side queueing is charged to the request rather than hidden —
`send_lag_max_s` in the summary is the audit of how well the client kept up (single-digit ms in
the runs here; if it grows, the client, not the server, is the bottleneck).

Example from `results/test_open/` (heavy, unique prompts):

| Arrival rate | Throughput (tok/s) | TTFT p50 / p99 (s) | SLO attainment | Goodput (req/s) |
|---|---|---|---|---|
| 0.8 req/s | 497 | 0.10 / 0.23 | 100% | 0.49 |
| 2.4 req/s | 1,591 | 2.98 / 12.24 | 42% | 0.66 |

Tripling offered load roughly triples tokens but barely moves goodput — the extra requests are
served too slowly to count.

## KV cache, briefly

vLLM stores per-token key/value tensors in fixed-size blocks (PagedAttention). When the block
pool runs out, the scheduler preempts requests — evicting their blocks and recomputing them
later — so the GPU spends cycles re-doing prefill instead of generating tokens. The wall
condition is roughly `concurrency x tokens_per_request > KV cache capacity`; the observable
signature is `vllm:gpu_cache_usage_perc` pinned at 1.0, `vllm:num_preemptions_total` rising, and
the waiting queue never draining.

## Repo layout

```
benchmark/
  run_experiment.py   one run: records provenance, starts collectors, drives load
  run_sweep.py        loops run_experiment.py over the concurrency list in the config
  loadgen.py          load generator: closed/open loop, prompt modes, SLO + goodput stats
  vllm_metrics.py     scrapes vLLM /metrics (KV usage, queue, preemptions, prefix cache) to CSV
  gpu_metrics.py      nvidia-smi telemetry (utilization, memory, power, temperature) to CSV
analysis/
  knee_table.py       goodput/SLO/throughput/preemptions per concurrency, mean [min-max] over repeats
  collect_results.py  concatenates every run's request_summary.csv -> results/all_results.csv
  plot_results.py     throughput / latency / TTFT / failure-rate vs concurrency, plus frontier
  plot_vllm_metrics.py per-run time series of KV usage, queue depth, preemptions
scripts/
  env.sh              Alpine paths, container image, model, vLLM server settings
  gpu_testing_20gb.sh sinteractive request for an A100 3g.20gb MIG slice
  serve_vllm_qwen.sh  launches the vLLM OpenAI server inside Apptainer
  test_vllm.sh        smoke test against /v1/models and /v1/chat/completions
configs/workloads.yaml  workload token shapes and the sweep's concurrency list
results/
  knee_heavy/, knee_summary.txt   the concurrency knee sweep (3 repeats x 7 points)
  l4_shared/, l4_unique/          prefix-cache shared-vs-unique comparison
  test_open/                      open-loop arrival-rate runs
archive/phase1/                   original harness and the first KV-cache-wall data
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
decode steps and cache growth over time; `heavy` does both and reaches the knee fastest.

## Running an experiment

Allocate a GPU and start the server (from the Alpine login node):

```bash
./scripts/gpu_testing_20gb.sh          # interactive A100 3g.20gb slice
./scripts/serve_vllm_qwen.sh           # vLLM server in Apptainer, port 8000
./scripts/test_vllm.sh                 # smoke test
```

Then drive load from another shell on the same node:

```bash
# closed loop: 32 concurrent workers, unique prompts (default)
python3 benchmark/run_experiment.py --workload heavy --concurrency 32

# open loop: Poisson arrivals at 2.4 req/s for 180 s (requests = ceil(rate x duration))
python3 benchmark/run_experiment.py --workload heavy --arrival-rate 2.4 --duration 180

# prefix-cache comparison run
python3 benchmark/run_experiment.py --workload heavy --concurrency 48 --prompt-mode shared

# knee sweep: repeat points into their own directory, then tabulate
for rep in 1 2 3; do
  for c in 16 20 24 28 32 40 48; do
    python3 benchmark/run_experiment.py --workload heavy --concurrency $c \
      --requests-per-worker 4 --min-requests 0 --results-dir results/knee_heavy
  done
done
python3 analysis/knee_table.py results/knee_heavy
```

Exactly one of `--concurrency` or `--arrival-rate` must be given. Request count is
`ceil(arrival_rate * duration)` in open loop, and `max(min_requests, concurrency *
requests_per_worker)` in closed loop — `--min-requests 0` turns off the floor so each
concurrency point does proportional work instead of a fixed 512 requests.

`run_sweep.py` covers the simple case (the concurrency list in `configs/workloads.yaml`, closed
loop, default prompt mode and results directory); anything else is a direct `run_experiment.py`
loop as above.

Aggregate and plot:

```bash
python3 analysis/collect_results.py      # -> results/all_results.csv
python3 analysis/plot_results.py         # -> plots/*.png
python3 analysis/plot_vllm_metrics.py    # -> plots/kv_cache_<run_id>.png, etc.
```

## What a run directory contains

Runs are named `<UTC timestamp>_<workload>_<c{concurrency}|r{rate}>_<hash>_<prompt_mode>`:

```
results/knee_heavy/20260927T231231Z_heavy_c32_6d44f2_unique/
  run_config.json         every CLI arg, workload spec, vLLM version, model list, hostname,
                          SLURM_JOB_ID, start/finish times, ok|failed status, and the
                          KV-cache-size / max-concurrency lines from the server log
  metrics_before.prom     raw /metrics snapshot before the load
  metrics_after.prom      raw /metrics snapshot after the load
  request_summary.csv     one row: throughput, latency/TTFT/TPOT percentiles, SLO attainment,
                          goodput, timeouts, failure rate, send lag
  request_details.csv     one row per request: TTFT, latency, TPOT, tokens, timed_out, error
  vllm_metrics.csv        1 Hz: running, waiting, kv_usage_pct, preemptions, token totals,
                          prefix-cache queries/hits
  vllm_metric_names.json  which metric alias actually matched on this server
  gpu_metrics.csv         1 Hz: GPU/memory utilization, memory used, power draw, temperature
```

Preemption counts are taken as `metrics_after - metrics_before` from the raw snapshots rather
than from the 1 Hz series, which can miss activity between samples or across a collector restart.

## Metric definitions

- **TTFT** — time from the request's *scheduled* send to its first streamed token.
- **TPOT** — `(latency - ttft) / (completion_tokens - 1)`, the steady-state per-token cost.
- **SLO attainment** — fraction of *all* requests (failures included) that succeeded with
  `ttft < --slo-ttft` (default 2 s) **and** `tpot < --slo-tpot` (default 50 ms).
- **Goodput** — SLO-satisfying requests per second of wall clock. The number to optimize.
- **`lat_p99_all_s`** — p99 latency with failed requests right-censored at the timeout, so
  dropping requests can't improve the tail.
- **`send_lag_max_s`** — worst gap between scheduled and actual send; the client's own honesty check.
- Percentiles are linearly interpolated and return `NaN` (never 0.0) when there is no data, so
  an empty bucket is never silently reported as a perfect score.
- Token counts come from the server's `usage` field in the final stream chunk
  (`stream_options.include_usage`); `token_count_source` records whether that was available.

`vllm_metrics.py` resolves each metric through an alias list (e.g. `vllm:gpu_cache_usage_perc`
or `vllm:kv_cache_usage_perc`), writes an empty cell — not a zero — when a server exposes
neither, records the resolved names once per run, and subtracts scrape time from its sleep to
hold a steady 1 Hz cadence.

## Configuration

Server settings live in `scripts/env.sh`: the model (`Qwen/Qwen2.5-1.5B-Instruct`), the
container image, `VLLM_MAX_MODEL_LEN=8192`, and `VLLM_GPU_MEMORY_UTILIZATION=0.3` — the knob
that sets how much of the MIG slice the KV cache pool gets, and therefore where the knee sits.
The server runs with `--enforce-eager` (single-process execution), which avoids a
multiprocessing crash on engine init under MIG.

## Environment notes

Three things had to be worked around to get vLLM running on Alpine:

1. CUDA driver/runtime mismatch — the node ships 12.8 while recent vLLM builds want 12.9, so the
   project pins a known-good CUDA 12.4 vLLM 0.8.5 image.
2. MIG multiprocessing crash on engine init — fixed by eager, single-process execution.
3. Dependency conflicts on a shared cluster — avoided by running the server inside Apptainer,
   with `HF_HOME` and the Apptainer cache on project storage.

## Roadmap

- [x] Phase 0 — benchmark harness, metrics pipeline, run/sweep automation
- [x] Phase 1 — reproduce the KV-cache wall (2,075 preemptions)
- [x] Phase 2 — per-run vLLM + GPU telemetry, heavy-workload concurrency sweep
- [x] Phase 3 — honest measurement: unique-prompt mode, SLO/goodput, open-loop arrivals,
      per-run provenance; knee located at c24 with goodput collapsing by c48
- [ ] Phase 4 — admission control (`--max-num-seqs`), prefix caching, chunked prefill, measured
      against goodput rather than throughput
- [ ] Phase 5 — KV-pressure-aware routing across replicas
- [ ] Phase 6 — modify vLLM's scheduler/preemption policy from source

## Tech stack

vLLM 0.8.5 (PagedAttention, prefix caching), CUDA, PyTorch, Slurm/HPC, Apptainer, Prometheus
metrics, pandas, matplotlib, Python, NVIDIA A100 (MIG 3g.20gb)
