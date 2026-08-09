#!/usr/bin/env python3
import csv, statistics, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_SWEEP = "baseline_sweep.csv"    # baseline concurrency sweep
LOADGEN        = "loadgen_results.csv"   # all loadgen runs (has the 128-conc wall rows)
WALL_METRICS   = "wall3_metrics.csv"     # metrics probe during the wall run

def read(p):
    with open(p) as f: return list(csv.DictReader(f))

# ---- Plot 1: baseline scaling ----
if os.path.exists(BASELINE_SWEEP):
    rows = sorted(read(BASELINE_SWEEP), key=lambda r: int(r["concurrency"]))
    conc = [int(r["concurrency"]) for r in rows]
    thr  = [float(r["throughput_tok_s"]) for r in rows]
    p99  = [float(r["lat_p99"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(conc, thr, "o-", color="#1f77b4", lw=2, ms=7)
    ax1.set_xscale("log", base=2); ax1.set_xticks(conc); ax1.set_xticklabels(conc)
    ax1.set_xlabel("Concurrent requests")
    ax1.set_ylabel("Throughput (tokens/sec)", color="#1f77b4")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(conc, p99, "s--", color="#d62728", lw=2, ms=6)
    ax2.set_ylabel("p99 latency (s)", color="#d62728")
    plt.title("Baseline: throughput scales with load (0 preemptions)", fontweight="bold")
    fig.tight_layout(); fig.savefig("baseline_scaling.png", dpi=140)
    print("saved baseline_scaling.png")
else:
    print("skip baseline_scaling:", BASELINE_SWEEP, "not found")

# ---- Plot 2: the wall ----
base_thr = wall_thr = None
if os.path.exists(BASELINE_SWEEP):
    for r in read(BASELINE_SWEEP):
        if int(r["concurrency"]) == 64: base_thr = float(r["throughput_tok_s"])
if os.path.exists(LOADGEN):
    wr = [float(r["throughput_tok_s"]) for r in read(LOADGEN) if int(r["concurrency"]) == 128]
    if wr: wall_thr = statistics.mean(wr)
wall_pre = max((float(r["preemptions"]) for r in read(WALL_METRICS)), default=0) if os.path.exists(WALL_METRICS) else 0

if base_thr and wall_thr:
    fig, (axa, axb) = plt.subplots(1,2, figsize=(10,5))
    labels = ["Healthy\n(64 conc)", "Wall\n(128 conc,\nstarved cache)"]
    colors = ["#2ca02c", "#d62728"]
    for ax,vals,ylab,title in [(axa,[base_thr,wall_thr],"Throughput (tokens/sec)","Throughput"),
                               (axb,[0,wall_pre],"KV-cache preemptions","KV-cache preemptions")]:
        bars = ax.bar(labels, vals, color=colors)
        ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold"); ax.grid(True, axis="y", alpha=0.3)
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:,.0f}", ha="center", va="bottom", fontweight="bold")
    fig.suptitle("The KV-cache wall: more load, less throughput", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig("the_wall.png", dpi=140)
    print("saved the_wall.png")
else:
    print("skip the_wall: need baseline 64-conc throughput and 128-conc wall rows")
