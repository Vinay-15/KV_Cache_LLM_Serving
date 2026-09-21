#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import subprocess
import time
import uuid
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    #ap.add_argument("--requests-per-worker", type=int, default=4)
    ap.add_argument("--requests-per-worker", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--prompt-mode", choices=["unique", "shared"], default="unique")
    args = ap.parse_args()

    with open(os.path.join(ROOT, "configs/workloads.yaml")) as f:
        cfg = yaml.safe_load(f)

    spec = cfg["workloads"][args.workload]
    #requests = args.concurrency * args.requests_per_worker
    requests = max(
    512,
    args.concurrency * args.requests_per_worker
)


    run_id = (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + args.workload
        + "_c"
        + str(args.concurrency)
        + "_"
        + uuid.uuid4().hex[:6]
        + "_" + args.prompt_mode
    )

    run_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Start collectors first so they capture queue/cache/GPU behavior.
    vllm_csv = os.path.join(run_dir, "vllm_metrics.csv")
    gpu_csv = os.path.join(run_dir, "gpu_metrics.csv")
    request_csv = os.path.join(run_dir, "request_summary.csv")

    details_csv = os.path.join(run_dir, "request_details.csv")

    vllm_proc = subprocess.Popen([
        "python3",
        os.path.join(ROOT, "benchmark/vllm_metrics.py"),
        "--url", args.url,
        "--interval", "1",
        "--csv", vllm_csv,
        "--run-id", run_id,
    ])

    gpu_proc = subprocess.Popen([
        "python3",
        os.path.join(ROOT, "benchmark/gpu_metrics.py"),
        "--interval", "1",
        "--csv", gpu_csv,
        "--run-id", run_id,
    ])

    time.sleep(2)

    try:
        run([
            "python3",
            os.path.join(ROOT, "benchmark/loadgen.py"),
            "--url", args.url,
            "--model", args.model,
            "--concurrency", str(args.concurrency),
            "--requests", str(requests),
            "--prompt-tokens", str(spec["prompt_tokens"]),
            "--output-tokens", str(spec["output_tokens"]),
            "--timeout", str(args.timeout),
            "--workload", args.workload,
            "--run-id", run_id,
            "--csv", request_csv,
            "--details-csv", details_csv,
            "--prompt-mode", args.prompt_mode,
        ])
    finally:
        print(">> Load finished. "
            "Collecting final metrics for 5 seconds...")
        time.sleep(5)
        vllm_proc.terminate()
        gpu_proc.terminate()
        try:
            vllm_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

        try:
            gpu_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gpu_proc.kill()

    print(f"\nDONE: {run_id}")
    print(f"Results: {run_dir}")

if __name__ == "__main__":
    main()

