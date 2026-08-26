#!/usr/bin/env python3
import argparse, time, urllib.request, csv, datetime
WANT = {
    "vllm:num_requests_running":"running", "vllm:num_requests_waiting":"waiting",
    "vllm:gpu_cache_usage_perc":"kv_usage", "vllm:num_preemptions_total":"preemptions",
    "vllm:prompt_tokens_total":"prompt_tokens", "vllm:generation_tokens_total":"gen_tokens",
}
def scrape(url):
    out = {v: 0.0 for v in WANT.values()}
    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
        for line in r.read().decode("utf-8","ignore").splitlines():
            if not line or line.startswith("#"): continue
            name = line.split("{")[0].split(" ")[0]
            if name in WANT:
                try: val = float(line.rsplit(" ",1)[1])
                except ValueError: continue
                out[WANT[name]] += val
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--csv", default="metrics.csv")
    args = ap.parse_args()
    cols = ["time"] + list(WANT.values())
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        print(f">> Probing {args.url}/metrics every {args.interval}s -> {args.csv} (Ctrl-C to stop)")
        print("   " + "  ".join(f"{c:>12}" for c in cols))
        try:
            while True:
                try: m = scrape(args.url)
                except Exception as e:
                    print(f"   (scrape failed: {e})"); time.sleep(args.interval); continue
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                row = [ts] + [round(m[c],3) for c in WANT.values()]
                w.writerow(row); f.flush()
                print("   " + "  ".join(f"{str(x):>12}" for x in row))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n>> Stopped. Saved {args.csv}")
if __name__ == "__main__":
    main()
