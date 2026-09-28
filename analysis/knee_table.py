#!/usr/bin/env python3
"""One line per concurrency: mean [min-max] over repeats."""
import csv
import glob
import os
import re
import statistics
import sys

sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "results/knee_heavy"


def preemptions(run_dir):
    """after - before, from the raw /metrics snapshots (Lesson 6). None if missing."""
    vals = []
    for name in ("metrics_before.prom", "metrics_after.prom"):
        try:
            with open(os.path.join(run_dir, name)) as f:
                m = re.search(r"^vllm:num_preemptions_total\S* (\S+)$", f.read(), re.M)
            vals.append(float(m.group(1)) if m else None)
        except OSError:
            vals.append(None)
    return None if None in vals else vals[1] - vals[0]


rows = {}
for path in glob.glob(os.path.join(sweep_dir, "*", "request_summary.csv")):
    with open(path) as f:
        r = next(csv.DictReader(f))
    rows.setdefault(int(r["concurrency"]), []).append({
        "goodput": float(r["goodput_rps"]),
        "slo": float(r["slo_attainment"]),
        "tput": float(r["throughput_tok_s"]),
        "ttft99": float(r["ttft_p99_s"]),
        "preempt": preemptions(os.path.dirname(path)),
    })


def fmt(xs, f="{:.2f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    return f"{f.format(statistics.mean(xs))} [{f.format(min(xs))}-{f.format(max(xs))}]"


print(f"{'c':>3} {'n':>2}  {'goodput req/s':<20} {'SLO attain':<20} "
      f"{'tok/s':<22} {'TTFT p99 s':<22} {'preemptions':<16}")
for c in sorted(rows):
    rs = rows[c]
    print(f"{c:>3} {len(rs):>2}  "
          f"{fmt([r['goodput'] for r in rs]):<20} "
          f"{fmt([r['slo'] for r in rs]):<20} "
          f"{fmt([r['tput'] for r in rs], '{:.0f}'):<22} "
          f"{fmt([r['ttft99'] for r in rs]):<22} "
          f"{fmt([r['preempt'] for r in rs], '{:.0f}'):<16}")
