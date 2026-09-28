#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import subprocess
import time
import uuid
import yaml
import math
import json
import socket
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get(url, path):
    """GET a server endpoint as text. None on failure: recording must never crash a run."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=10) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return None


def get_json(url, path):
    t = get(url, path)
    try:
        return json.loads(t) if t else None
    except ValueError:
        return t


def save(path, text):
    if text is not None:
        with open(path, "w") as f:
            f.write(text)


# Startup lines worth keeping from the vLLM log (it also logs every request).
LOG_KEYS = ("API server version", "args:", "KV cache size", "Maximum concurrency")


def server_log_lines(path):
    if not path or not os.path.exists(path):
        return []
    keep = []
    with open(path, errors="ignore") as f:
        for line in f:
            if any(k in line for k in LOG_KEYS):
                keep.append(line.strip())
    return keep[:20]


def run(cmd):
    print("\n$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--arrival-rate", type=float, help="open-loop requests/second")
    ap.add_argument("--duration", type=float, default=180, help="open-loop: seconds of arrivals; requests = rate x duration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests-per-worker", type=int, default=1)
    ap.add_argument("--min-requests", type=int, default=512, help="floor on total requests; 0 = exactly concurrency x requests-per-worker")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--prompt-mode", choices=["unique", "shared"], default="unique")
    ap.add_argument("--server-log", default=os.path.join(ROOT, "logs", "vllm_server.log"))
    args = ap.parse_args()

    with open(os.path.join(ROOT, "configs/workloads.yaml")) as f:
        cfg = yaml.safe_load(f)

    spec = cfg["workloads"][args.workload]
    #requests = args.concurrency * args.requests_per_worker
    
    if (args.concurrency is None) == (args.arrival_rate is None):
        ap.error("give exactly one of --concurrency or --arrival-rate")
    if args.arrival_rate is not None:
        requests = math.ceil(args.arrival_rate * args.duration)
        load_label = f"r{args.arrival_rate:g}"
    else:
        requests = max(args.min_requests, args.concurrency * args.requests_per_worker)
        load_label = f"c{args.concurrency}"

    run_id = (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + args.workload
        + "_" 
        + load_label
        + "_"
        + uuid.uuid4().hex[:6]
        + "_" + args.prompt_mode
    )

    run_dir = os.path.join(args.results_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    save(os.path.join(run_dir, "metrics_before.prom"), get(args.url, "/metrics"))
    config = {
        "run_id": run_id,
        "args": vars(args),
        "workload_spec": spec,
        "requests": requests,
        "vllm_version": get_json(args.url, "/version"),
        "models": get_json(args.url, "/v1/models"),
        "server_log": server_log_lines(args.server_log),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    status = "failed"

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

    load_args = (["--arrival-rate", str(args.arrival_rate), "--seed", str(args.seed)]
                 if args.arrival_rate is not None
                 else ["--concurrency", str(args.concurrency)])


    try:
        run([
            "python3",
            os.path.join(ROOT, "benchmark/loadgen.py"),
            "--url", args.url,
            "--model", args.model,
            *load_args,
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
        status = "ok"
    finally:
        print(">> Load finished. "
            "Collecting final metrics for 5 seconds...")
        time.sleep(5)
        vllm_proc.terminate()
        gpu_proc.terminate()
        
        save(os.path.join(run_dir, "metrics_after.prom"), get(args.url, "/metrics"))
        config["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        config["status"] = status
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(config, f, indent=2)
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

