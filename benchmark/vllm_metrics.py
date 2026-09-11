#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import re
import time
import urllib.request

# These names are commonly exposed by vLLM, but metric names can vary
# across vLLM releases. Unknown metrics are simply ignored.
METRICS = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:gpu_cache_usage_perc": "kv_usage",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
}

def scrape(url):
    out = {v: 0.0 for v in METRICS.values()}

    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=10) as r:
        body = r.read().decode("utf-8", "ignore")

    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue

        name = line.split("{", 1)[0].split(" ", 1)[0]

        # Support metrics that have labels by stripping labels from the
        # metric name. The base name is enough for this experiment.
        if name not in METRICS:
            continue

        try:
            value = float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            continue

        out[METRICS[name]] += value

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

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
        "error",
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        print(
            f">> vLLM metrics: " 
            f"{args.url}/metrics")

        print(f">> writing {args.csv}")

        try:
            while True:
                ts = dt.datetime.now(dt.timezone.utc).isoformat()
                try:
                    m = scrape(args.url)

                    scrape_ok = 1
                    error = ""

                    # vLLM 0.8.5 reports this
                    # as a fraction:
                    #
                    # 0.50 = 50%
                    # 1.00 = 100%
                    kv_usage_pct = (
                        m["kv_usage"] * 100.0
                    )

                    row = [
                        ts,
                        args.run_id,
                        scrape_ok,
                        m["running"],
                        m["waiting"],
                        m["kv_usage"],
                        kv_usage_pct,
                        m["preemptions_total"],
                        m["prompt_tokens_total"],
                        m["generation_tokens_total"],
                        error,
                    ]

                except Exception as e:

                    # IMPORTANT:
                    # failed scrapes are still written.
                    row = [
                        ts,
                        args.run_id,
                        0,      # scrape_ok
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        str(e),
                    ]

                writer.writerow(row)
                f.flush()

                time.sleep(args.interval)

        except KeyboardInterrupt:

            print(
                "\n>> metrics collector stopped"
            )



'''def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    # cols = ["timestamp", "run_id"] + list(METRICS.values())

    cols = [
    "time",
    "scrape_ok",
    "running",
    "waiting",
    "kv_usage",
    "kv_usage_pct",
    "preemptions",
    "prompt_tokens",
    "gen_tokens",
    "error",
    ]

    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)

        print(f">> vLLM metrics: {args.url}/metrics")
        print(f">> writing {args.csv}")

    #    try:
    #        while True:
    #               m = scrape(args.url)
    #            except Exception as e:
    #                print(f"   scrape failed: {e}")
    #                time.sleep(args.interval)
    #                 continue

        try:
            m = scrape(args.url)
            scrape_ok = 1
            error = ""

            kv_usage_pct = m["kv_usage"] * 100.0
        except Exception as e:
            scrape_ok = 0
            error = str(e)
            m = {
                "running": "",
                "waiting": "",
                "kv_usage": "",
                "preemptions": "",
                "prompt_tokens": "",
                "gen_tokens": "",
            }
            kv_usage_pct = ""



            ts = dt.datetime.now(dt.timezone.utc).isoformat()

            row = [ts, args.run_id] + [
                round(m[c], 4) for c in METRICS.values()
            ]
            writer.writerow(row)
            f.flush()

            print(
                "   "
                + " ".join(
                    f"{k}={m[v]:.2f}"
                    for k, v in METRICS.items()
                )
            )

            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n>> metrics collector stopped")'''

if __name__ == "__main__":
    main()

