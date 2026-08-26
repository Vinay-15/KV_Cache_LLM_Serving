#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "results", "all_results.csv")
OUTDIR = os.path.join(ROOT, "plots")

os.makedirs(OUTDIR, exist_ok=True)

def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, name), dpi=220, bbox_inches="tight")
    plt.close()

def plot_metric(df, y, ylabel, filename, logy=False):
    plt.figure(figsize=(9, 6))

    for workload, g in df.groupby("workload"):
        g = g.sort_values("concurrency")
        plt.plot(
            g["concurrency"],
            g[y],
            marker="o",
            linewidth=2,
            label=workload,
        )

    plt.xlabel("Concurrency")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Concurrency")
    plt.xticks(sorted(df["concurrency"].unique()))
    plt.grid(True, alpha=0.25)
    plt.legend()

    if logy:
        plt.yscale("log")

    savefig(filename)

def main():
    if not os.path.exists(INPUT):
        raise SystemExit(f"Missing {INPUT}. Run collect_results.py first.")

    df = pd.read_csv(INPUT)

    # Core performance plots
    plot_metric(
        df,
        "throughput_tok_s",
        "Output throughput (tokens/s)",
        "throughput_vs_concurrency.png",
    )

    plot_metric(
        df,
        "lat_p50_s",
        "Latency p50 (s)",
        "latency_p50_vs_concurrency.png",
    )

    plot_metric(
        df,
        "lat_p99_s",
        "Latency p99 (s)",
        "latency_p99_vs_concurrency.png",
    )

    plot_metric(
        df,
        "ttft_p50_s",
        "TTFT p50 (s)",
        "ttft_p50_vs_concurrency.png",
    )

    plot_metric(
        df,
        "ttft_p99_s",
        "TTFT p99 (s)",
        "ttft_p99_vs_concurrency.png",
    )

    # Throughput / latency frontier
    plt.figure(figsize=(9, 6))
    for workload, g in df.groupby("workload"):
        plt.plot(
            g["lat_p99_s"],
            g["throughput_tok_s"],
            marker="o",
            linewidth=2,
            label=workload,
        )

    plt.xlabel("P99 latency (s)")
    plt.ylabel("Output throughput (tokens/s)")
    plt.title("Throughput–Tail-Latency Frontier")
    plt.grid(True, alpha=0.25)
    plt.legend()
    savefig("throughput_latency_frontier.png")

    # Failure rate
    df["failure_rate"] = df["failures"] / df["requests"]

    plot_metric(
        df,
        "failure_rate",
        "Failure rate",
        "failure_rate_vs_concurrency.png",
    )

    print(f"Plots written to {OUTDIR}")

if __name__ == "__main__":
    main()

