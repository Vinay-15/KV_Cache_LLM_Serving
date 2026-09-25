#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import re
import json
import os
import time
import urllib.request

# These names are commonly exposed by vLLM, but metric names can vary
# across vLLM releases. Unknown metrics are simply ignored.
ALIASES = {
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "kv_usage": ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc"],
    "preemptions_total": ["vllm:num_preemptions_total"],
    "prompt_tokens_total": ["vllm:prompt_tokens_total"],
    "generation_tokens_total": ["vllm:generation_tokens_total"],
    "prefix_queries_total": ["vllm:gpu_prefix_cache_queries_total",
                             "vllm:prefix_cache_queries_total"],
    "prefix_hits_total": ["vllm:gpu_prefix_cache_hits_total",
                          "vllm:prefix_cache_hits_total"],
}

def fetch(url):
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as r:
        return r.read().decode("utf-8", "ignore")

def parse(body):
    """Every metric line -> {name without labels: sum across label sets}."""
    raw = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue
        raw[name] = raw.get(name, 0.0) + value
    return raw

def scrape(url):
    """Returns (values, found).
    values[col] is None when no alias exists on this server -- never 0.
    found[col] is the metric name that matched, or None."""
    raw = parse(fetch(url))
    values, found = {}, {}
    for col, names in ALIASES.items():
        values[col], found[col] = None, None
        for n in names:
            if n in raw:
                values[col], found[col] = raw[n], n
                break
    return values, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    names_path = os.path.join(os.path.dirname(args.csv) or ".",
                              "vllm_metric_names.json")
    def cell(v):
        return "" if v is None else round(v, 6)


    cols = [
        "timestamp",
        "run_id",
        "scrape_ok",
        "running",
        "waiting",
        "kv_usage",
        "kv_usage_pct",
        "preemptions_total",
        "prompt_tokens_total",
        "generation_tokens_total",
        "prefix_queries_total",
        "prefix_hits_total",
        "error",
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        print(
            f">> vLLM metrics: " 
            f"{args.url}/metrics")

        print(f">> writing {args.csv}")

        names_written = False
        try:
            while True:
                tick = time.monotonic()
                ts = dt.datetime.now(dt.timezone.utc).isoformat()
                try:
                    m, found = scrape(args.url)

                    kv = m["kv_usage"]
                    kv_pct = None if kv is None else kv * 100.0   # 0.8.5 reports a 0-1 fraction

                    row = [ts, args.run_id, 1,
                           cell(m["running"]), cell(m["waiting"]),
                           cell(kv), cell(kv_pct),
                           cell(m["preemptions_total"]),
                           cell(m["prompt_tokens_total"]),
                           cell(m["generation_tokens_total"]),
                           cell(m["prefix_queries_total"]),
                           cell(m["prefix_hits_total"]),
                           ""]

                    # Names don't change during a run: record them once.
                    if not names_written:
                        with open(names_path, "w") as nf:
                            json.dump(found, nf, indent=2)
                        missing = [c for c, n in found.items() if n is None]
                        if missing:
                            print(f"   WARNING: not exposed by this server: {missing}")
                        names_written = True

                except Exception as e:
                    row = [ts, args.run_id, 0] + [""] * 9 + [str(e)[:200]]

                writer.writerow(row)
                f.flush()
                # Steady cadence: subtract the time the scrape itself took.
                time.sleep(max(0.0, args.interval - (time.monotonic() - tick)))

        except KeyboardInterrupt:
            print("\n>> metrics collector stopped")

if __name__ == "__main__":
    main()

