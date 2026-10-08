#!/usr/bin/env python3
"""One table per results folder, grouped by load (c = concurrency, r = arrival rate)."""
import csv, glob, json, math, os, re, sys
from collections import defaultdict

COLS = [("goodput", "goodput_rps", "{:.2f}"), ("SLO", "slo_attainment", "{:.2f}"),
        ("tok/s", "throughput_tok_s", "{:.0f}"), ("TTFTp99", "ttft_p99_s", "{:.2f}"),
        ("TPOTp99ms", "tpot_p99_ms", "{:.1f}"), ("gap_p99", "max_gap_p99_s", "{:.2f}"),
        ("stall>1s", "stalled_1s", "{:.0f}"), ("preempt", "preemptions", "{:.0f}"),
        ("run_max", "max_running", "{:.0f}"), ("wait_max", "max_waiting", "{:.0f}"),
        ("kv_peak", "peak_kv", "{:.3f}")]

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")

def prom(path, name):
    if os.path.exists(path):
        for line in open(path):
            if line.startswith(name): return fnum(line.split()[-1])
    return float("nan")

def run_summary(d):
    r = next(csv.DictReader(open(os.path.join(d, "request_summary.csv"))))
    out = {k: fnum(r.get(k)) for _, k, _ in COLS}
    out["tpot_p99_ms"] = fnum(r.get("tpot_p99_s")) * 1000
    out["preemptions"] = (prom(os.path.join(d, "metrics_after.prom"), "vllm:num_preemptions_total")
                          - prom(os.path.join(d, "metrics_before.prom"), "vllm:num_preemptions_total"))
    vm = os.path.join(d, "vllm_metrics.csv")
    rows = [x for x in csv.DictReader(open(vm)) if x.get("scrape_ok") == "1"] if os.path.exists(vm) else []
    if rows:
        out["max_running"] = max(fnum(x["running"]) for x in rows)
        out["max_waiting"] = max(fnum(x["waiting"]) for x in rows)
        out["peak_kv"] = max(fnum(x["kv_usage"]) for x in rows)
    if (r.get("mode") or "closed") == "open":
        return ("r", fnum(r.get("arrival_rate"))), out
    return ("c", fnum(r.get("concurrency"))), out

def config_note(d):
    note = ""
    p = os.path.join(d, "run_config.json")
    if os.path.exists(p):
        c = json.load(open(p))
        kv = [l for l in c.get("server_log", []) if "KV cache size" in l]
        note = f"host={c.get('host')}  " + (re.sub(r".*(KV cache size)", r"\1", kv[0]) if kv else "KV size: not in log")
    pb = os.path.join(d, "metrics_before.prom")
    if os.path.exists(pb):
        for line in open(pb):
            if line.startswith("vllm:cache_config_info"):
                u = re.search(r'gpu_memory_utilization="([^"]*)"', line)
                o = re.search(r'num_gpu_blocks_override="([^"]*)"', line)
                note += f"  gpu_util={u.group(1) if u else '?'} blocks_override={o.group(1) if o else '?'}"
    return note

def mean(vals):
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")

root = sys.argv[1]
groups, notes = defaultdict(list), set()
for d in sorted(glob.glob(os.path.join(root, "*", ""))):
    if not os.path.exists(os.path.join(d, "request_summary.csv")): continue
    try:
        key, out = run_summary(d)
    except Exception as e:
        print(f"  skipped {d}: {e}"); continue
    groups[key].append(out); notes.add(config_note(d))

print(f"\n##### {root}")
for n in sorted(notes): print("  " + n)
print(f"{'load':>7} {'n':>2} " + " ".join(f"{h:>9}" for h, _, _ in COLS) + "  SLO[min-max]")
for key in sorted(groups):
    runs = groups[key]
    cells = []
    for _, k, fmt in COLS:
        m = mean([o[k] for o in runs])
        cells.append(f"{'-' if math.isnan(m) else fmt.format(m):>9}")
    slo = [o["slo_attainment"] for o in runs if not math.isnan(o["slo_attainment"])]
    rng = f"[{min(slo):.2f}-{max(slo):.2f}]" if slo else "-"
    print(f"{key[0] + format(key[1], 'g'):>7} {len(runs):>2} " + " ".join(cells) + "  " + rng)
