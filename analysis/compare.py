#!/usr/bin/env python3
"""Before/after comparison: one line per policy on the same axes.
Usage: python3 analysis/compare.py figures/compare_open \
         vLLM=results/open_heavy_gpu20_test lookahead=results/open_heavy_lookahead [peak=...]
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots import load, mean, col, CAPACITY   # same folder, reuses plots.py

def summary_mean(by_x, key):
    """Mean of one request_summary.csv column at each load level."""
    out = []
    for runs in by_x.values():
        vals = [float(next(csv.DictReader(open(os.path.join(r["run"], "request_summary.csv"))))[key])
                for r in runs]
        out.append(sum(vals) / len(vals))
    return out

def timeline(run):
    rows = list(csv.DictReader(open(os.path.join(run, "vllm_metrics.csv"))))
    h = list(rows[0].keys()); c_run = col(h, "running")
    t0, ts, ys = None, [], []
    for i, r in enumerate(rows):
        if not r[c_run]:
            continue
        try:
            t = float(r[h[0]]); t0 = t if t0 is None else t0; t -= t0
        except ValueError:
            t = i
        ts.append(t); ys.append(float(r[c_run]))
    return ts, ys

def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    series = []
    for arg in sys.argv[2:]:
        label, folder = arg.split("=", 1)
        by_x, mode = load(folder)
        series.append((label, by_x, mode))
    closed = series[0][2] == "closed"
    xlabel = "concurrent clients" if closed else "arrival rate (requests/s)"

    # Four metrics side by side, one line per policy
    panels = [
        ("preemptions per run", lambda b: mean(b, "preempt")),
        ("p99 longest token gap (s)", lambda b: summary_mean(b, "max_gap_p99_s")),
        ("p99 time per token (ms)", lambda b: [1000 * v for v in summary_mean(b, "tpot_p99_s")]),
        ("goodput (req/s)", lambda b: mean(b, "goodput")),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (ylabel, fn) in zip(axes, panels):
        for label, by_x, _ in series:
            ax.plot(list(by_x), fn(by_x), "o-", label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"vLLM default vs admission control ({'closed' if closed else 'open'}-loop, same load)")
    fig.tight_layout(); fig.savefig(f"{out}/compare_metrics.png", dpi=200)

    # Running requests over time at the heaviest load, one line per policy
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, by_x, _ in series:
        ts, ys = timeline(by_x[list(by_x)[-1]][0]["run"])
        ax.plot(ts, ys, lw=1.2, label=label)
    ax.axhline(CAPACITY, ls="--", color="k", lw=1,
               label=f"fits if all at full size (≈{CAPACITY:.0f})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("running requests")
    ax.set_title("Requests admitted at the heaviest load")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/compare_timeline.png", dpi=200)
    print(f"wrote 2 figures to {out}/")

if __name__ == "__main__":
    main()
