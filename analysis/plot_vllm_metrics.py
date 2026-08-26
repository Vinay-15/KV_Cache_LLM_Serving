#!/usr/bin/env python3
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "plots")
os.makedirs(OUTDIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, name), dpi=220, bbox_inches="tight")
    plt.close()

def main():
    paths = glob.glob(os.path.join(ROOT, "results", "*", "vllm_metrics.csv"))

    if not paths:
        raise SystemExit("No vllm_metrics.csv files found.")

    frames = []

    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue

        # The run_id encodes workload/concurrency, but we keep parsing
        # intentionally simple and use the raw time series here.
        frames.append(df)

    # Plot every run separately so the behavior during a run is visible.
    for df in frames:
        run_id = str(df["run_id"].iloc[0])

        for metric, ylabel, filename in [
            ("kv_usage", "KV cache usage (%)", "kv_cache"),
            ("waiting", "Waiting requests", "waiting"),
            ("running", "Running requests", "running"),
            ("preemptions_total", "Cumulative preemptions", "preemptions"),
        ]:
            if metric not in df:
                continue

            plt.figure(figsize=(9, 5))
            plt.plot(df.index, df[metric], linewidth=2)
            plt.xlabel("Sample")
            plt.ylabel(ylabel)
            plt.title(f"{ylabel} — {run_id}")
            plt.grid(True, alpha=0.25)
            savefig(f"{filename}_{run_id}.png")

    print(f"vLLM plots written to {OUTDIR}")

if __name__ == "__main__":
    main()

