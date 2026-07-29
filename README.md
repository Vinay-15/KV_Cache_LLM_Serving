# KV-Cache-Aware LLM Serving on Constrained GPUs

Serving LLMs with vLLM on a memory-limited GPU, then finding and fixing the point where the
KV cache saturates and throughput collapses under load. Run on CU Boulder's Alpine
supercomputer: an OpenAI-compatible vLLM API on a 20 GB NVIDIA A100 MIG slice, scheduled with
Slurm and containerized with Apptainer.

## TL;DR results

| Scenario | Concurrency | Throughput | p99 latency | KV preemptions |
|---|---|---|---|---|
| Healthy baseline | 64 | ~4,000 tok/s | ~4 s | 0 |
| KV-cache wall | 128 | ~1,700 tok/s | ~475 s | 2,075 |

Doubling concurrency on a memory-starved cache cut throughput nearly in half and pushed p99
latency past 7 minutes -- the engine spent its cycles preempting and recomputing evicted
requests (2,075 preemptions) instead of generating tokens.

## The story

1. CUDA driver/runtime mismatch (node had 12.8, newest vLLM needs 12.9) -> fixed by version-
   matching to a CUDA 12.4 vLLM build.
2. MIG multiprocessing crash on engine init -> fixed with single-process / eager execution.
3. Dependency hell -> solved by containerizing the runtime with Apptainer.

## KV cache, briefly

vLLM caches per-token key/value tensors in fixed-size blocks (PagedAttention). When the pool
runs out, it preempts (evicts + recomputes) requests, which thrashes and collapses throughput.
Wall condition: concurrency x tokens_per_request > KV cache capacity.

## Repo contents

- serve_vllm.sbatch - Slurm job to launch the vLLM server
- loadgen.py - concurrent load generator (throughput, TTFT, p50/p99 latency)
- metrics_probe.py - scrapes vLLM /metrics (KV usage, preemptions) to CSV
- run_sweep.sh - concurrency sweep
- results/ - benchmark CSVs

## Roadmap

- [x] Phase 0 - baseline benchmark harness + metrics pipeline
- [x] Phase 1 - reproduce the KV-cache wall (2,075 preemptions)
- [ ] Phase 2 - admission control (--max-num-seqs), prefix caching, chunked prefill
- [ ] Phase 3 - KV-pressure-aware router across replicas
- [ ] Phase 4 - modify vLLM's scheduler/preemption policy from source

## Tech stack

vLLM, PagedAttention, CUDA, PyTorch, Slurm/HPC, Apptainer, Prometheus metrics, Python, NVIDIA A100 (MIG)
