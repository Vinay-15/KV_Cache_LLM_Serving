#!/usr/bin/env python3
import argparse
import os
import subprocess
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--workload", default="medium")
    ap.add_argument("--requests-per-worker", type=int, default=4)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "configs/workloads.yaml")) as f:
        cfg = yaml.safe_load(f)

    concurrencies = cfg["concurrency"]

    for c in concurrencies:
        cmd = [
            "python3",
            os.path.join(ROOT, "benchmark/run_experiment.py"),
            "--url", args.url,
            "--model", args.model,
            "--workload", args.workload,
            "--concurrency", str(c),
            "--requests-per-worker", str(args.requests_per_worker),
        ]

        print("\n" + "=" * 80)
        print(f"WORKLOAD={args.workload} CONCURRENCY={c}")
        print("=" * 80)

        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()

