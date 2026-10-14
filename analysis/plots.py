#!/usr/bin/env python3
"""Result figures for one sweep, closed-loop or open-loop.
Usage: python3 analysis/plots.py results/knee_heavy_a100_80_graphs figures/closed
       python3 analysis/plots.py results/open_heavy_gpu20_test figures/open
"""
import csv, glob, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")                      # no display on Alpine; write PNGs
import matplotlib.pyplot as plt

CAPACITY = 4735 / 194                      # blocks / blocks per heavy request = 24.4

def preemptions(run):
    """Counter after minus counter before = preemptions during this run."""
    v = []
    for name in ("metrics_before.prom", "metrics_after.prom"):
        val = None
        p = os.path.join(run, name)
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("vllm:num_preemptions_total"):
                    val = float(line.split()[-1])
        v.append(val)
    return v[1] - v[0] if None not in v else float("nan")

def load(sweep):
    by_x, mode = defaultdict(list), "closed"
    for run in sorted(glob.glob(os.path.join(sweep, "*/"))):
        p = os.path.join(run, "request_summary.csv")
        if not os.path.exists(p):
            continue
        r = next(csv.DictReader(open(p)))
        if r.get("arrival_rate") not in (None, ""):
            mode, x = "open", float(r["arrival_rate"])
        else:
            x = int(float(r["concurrency"]))
        by_x[x].append({
            "goodput": float(r["goodput_rps"]),
            "toks": float(r["throughput_tok_s"]),
            "slo": float(r["slo_attainment"]),
            "frozen": float(r["stalled_1s"]) / float(r["requests"]),
            "preempt": preemptions(run),
            "run": run,
        })
    return dict(sorted(by_x.items())), mode

def mean(by_x, key):
    return [sum(x[key] for x in runs) / len(runs) for runs in by_x.values()]

def col(header, *keys):
    for h in header:
        if any(k in h.lower() for k in keys):
            return h
    sys.exit(f"no column like {keys} in vllm_metrics.csv; header = {header}")

def main():
    sweep, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    by_x, mode = load(sweep)
    xs = list(by_x)
    closed = mode == "closed"
    xlabel = "concurrent clients" if closed else "arrival rate (requests/s)"

    # Fig 1: throughput flat, goodput collapses
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, mean(by_x, "toks"), "o-", color="tab:gray", label="throughput (tok/s)")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("throughput (tok/s)"); ax1.set_ylim(bottom=0)
    ax2 = ax1.twinx()
    ax2.plot(xs, mean(by_x, "goodput"), "s-", color="tab:red", label="goodput (req/s meeting SLO)")
    ax2.set_ylabel("goodput (req/s)"); ax2.set_ylim(bottom=0)
    if closed:
        ax1.axvline(CAPACITY, ls="--", color="k", lw=1)
        ax1.text(CAPACITY + 0.5, 500, f"KV capacity\n≈{CAPACITY:.0f} requests", fontsize=8)
    fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.15), fontsize=8)
    ax1.set_title(f"Throughput hides the goodput collapse ({mode}-loop)")
    fig.tight_layout(); fig.savefig(f"{out}/fig1_throughput_vs_goodput.png", dpi=200)

    # Fig 2: preemptions rise past capacity; SLO misses the freezes
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.bar(xs, mean(by_x, "preempt"), width=2.5 if closed else 0.12, color="tab:orange")
    if closed:
        a.axvline(CAPACITY, ls="--", color="k", lw=1)
    a.set_xlabel(xlabel); a.set_ylabel("preemptions per run")
    a.set_title("Preemptions rise past capacity")
    b.plot(xs, [100 * x for x in mean(by_x, "slo")], "o-", label="SLO attainment (%)")
    b.plot(xs, [100 * x for x in mean(by_x, "frozen")], "s-", color="tab:red",
           label="responses frozen > 1 s (%)")
    b.set_xlabel(xlabel); b.set_ylabel("% of requests"); b.set_ylim(0, 105)
    b.legend(fontsize=8); b.set_title("SLO metrics miss mid-stream freezes")
    fig.tight_layout(); fig.savefig(f"{out}/fig2_preemption_and_hidden_stalls.png", dpi=200)

    # Fig 3: over-admission timeline from the heaviest-load run
    run = by_x[xs[-1]][0]["run"]
    rows = list(csv.DictReader(open(os.path.join(run, "vllm_metrics.csv"))))
    h = list(rows[0].keys())
    c_run, c_kv = col(h, "running"), col(h, "kv", "cache")
    t0 = None; ts, running, kv = [], [], []
    for i, r in enumerate(rows):
        if not r[c_run]:
            continue
        try:
            t = float(r[h[0]]); t0 = t if t0 is None else t0; t -= t0
        except ValueError:
            t = i                                   # fall back to sample number
        ts.append(t); running.append(float(r[c_run])); kv.append(float(r[c_kv]))
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a.plot(ts, running, color="tab:blue", label="requests vLLM is running")
    a.axhline(CAPACITY, ls="--", color="k", lw=1,
              label=f"fits if all at full size (≈{CAPACITY:.0f})")
    a.set_ylabel("running requests"); a.legend(fontsize=8)
    load_label = f"c={xs[-1]}" if closed else f"rate={xs[-1]}/s"
    a.set_title(f"vLLM over-admits under memory pressure ({load_label})")
    b.plot(ts, [100 * x for x in kv], color="tab:purple")
    b.set_ylabel("KV cache used (%)"); b.set_xlabel("time (s)"); b.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(f"{out}/fig3_overadmission_timeline.png", dpi=200)

    print(f"{mode}-loop: wrote 3 figures to {out}/  (timeline from {run})")

if __name__ == "__main__":
    main()
