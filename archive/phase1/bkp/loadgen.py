#!/usr/bin/env python3
import argparse, json, time, urllib.request, csv, os
from concurrent.futures import ThreadPoolExecutor

def build_prompt(n_tokens: int) -> str:
    sentence = "The quick brown fox jumps over the lazy dog. "
    return sentence * max(1, n_tokens // 10)

def one_request(url, model, prompt, out_tokens, timeout):
    payload = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": out_tokens,
        "ignore_eos": True, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(url + "/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft, n_tok = None, 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                txt = json.loads(data)["choices"][0].get("text", "")
                if txt:
                    if ttft is None: ttft = time.perf_counter() - t0
                    n_tok += 1
        return {"ok": True, "ttft": ttft, "latency": time.perf_counter() - t0, "tokens": n_tok}
    except Exception as e:
        return {"ok": False, "error": str(e), "latency": time.perf_counter() - t0}

def pct(values, p):
    if not values: return 0.0
    values = sorted(values)
    k = min(len(values)-1, int(round((p/100)*(len(values)-1))))
    return values[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--csv", default="loadgen_results.csv")
    args = ap.parse_args()
    prompt = build_prompt(args.prompt_tokens)
    print(f">> {args.requests} requests, concurrency={args.concurrency}, "
          f"prompt~{args.prompt_tokens}tok, output={args.output_tokens}tok")
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(
            lambda _: one_request(args.url, args.model, prompt, args.output_tokens, args.timeout),
            range(args.requests)))
    wall = time.perf_counter() - wall0
    ok = [r for r in results if r["ok"]]; fail = [r for r in results if not r["ok"]]
    total_tokens = sum(r["tokens"] for r in ok)
    lat = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    throughput = total_tokens / wall if wall > 0 else 0
    print(f"   success={len(ok)}/{len(results)}  failures={len(fail)}")
    print(f"   wall={wall:.1f}s  system throughput={throughput:.1f} tok/s")
    print(f"   latency  p50={pct(lat,50):.2f}s  p99={pct(lat,99):.2f}s")
    print(f"   TTFT     p50={pct(ttfts,50):.2f}s  p99={pct(ttfts,99):.2f}s")
    if fail: print(f"   first error: {fail[0].get('error')}")
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["concurrency","requests","prompt_tokens","output_tokens","success",
                        "failures","wall_s","throughput_tok_s","lat_p50","lat_p99","ttft_p50","ttft_p99"])
        w.writerow([args.concurrency,args.requests,args.prompt_tokens,args.output_tokens,len(ok),
                    len(fail),round(wall,2),round(throughput,1),round(pct(lat,50),3),
                    round(pct(lat,99),3),round(pct(ttfts,50),3),round(pct(ttfts,99),3)])
    print(f"   appended to {args.csv}")

if __name__ == "__main__":
    main()
