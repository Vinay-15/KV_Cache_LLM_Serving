#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def build_prompt(n_tokens: int) -> str:
    # Deliberately repetitive prompt so workload length is controlled.
    sentence = "The quick brown fox jumps over the lazy dog. "
    return sentence * max(1, n_tokens // 10)

def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    k = min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
    return float(values[k])

def one_request(request_id, url, model, prompt, out_tokens, timeout):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": out_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        # vLLM/OpenAI-compatible servers that support this return usage
        # in the final stream chunk.
        "stream_options": {"include_usage": True},
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft = None
    text_parts = []
    usage = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                chunk = json.loads(data_str)

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                txt = choices[0].get("text", "")
                if txt:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(txt)

        latency = time.perf_counter() - t0

        completion_tokens = None
        prompt_tokens = None
        total_tokens = None

        if usage:
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

        # Fallback: if the server did not return usage, count generated
        # characters only as a fallback indicator. Do NOT label this
        # fallback as exact token throughput in analysis.
        if completion_tokens is None:
            completion_tokens = 0
            token_count_source = "server_usage_missing"
        else:
            token_count_source = "server_usage"

        return {
            "request_id": request_id,
            "ok": True,
            "ttft": ttft,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "token_count_source": token_count_source,
            "output_chars": len("".join(text_parts)),
        }

    except Exception as e:
        return {
            "request_id": request_id,
            "ok": False,
            "error": repr(e),
            "latency": time.perf_counter() - t0,
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--output-tokens", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workload", default="unknown")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--details-csv", default="")
    args = ap.parse_args()

    prompt = build_prompt(args.prompt_tokens)

    print(
        f">> workload={args.workload} "
        f"requests={args.requests} concurrency={args.concurrency} "
        f"prompt~{args.prompt_tokens} output={args.output_tokens}"
    )

    wall0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(
            pool.map(
                lambda i: one_request(
                    i,
                    args.url,
                    args.model,
                    prompt,
                    args.output_tokens,
                    args.timeout,
                ),
                range(args.requests),
            )
        )

    wall = time.perf_counter() - wall0

    if args.details_csv:
        fields = [
            "request_id",
            "ok",
            "ttft",
            "latency",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "token_count_source",
            "output_chars",
            "error",
        ]

        with open(args.details_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fields,
                extrasaction="ignore",
            )

            writer.writeheader()

            for r in results:
                writer.writerow(r)

        print(f"   request details -> {args.details_csv}")


    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    lat = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]

    completion_tokens = [
        r["completion_tokens"]
        for r in ok
        if r.get("completion_tokens") is not None
    ]

    total_completion_tokens = sum(completion_tokens)

    throughput = total_completion_tokens / wall if wall > 0 else 0.0

    row = {
        "run_id": args.run_id,
        "workload": args.workload,
        "model": args.model,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "prompt_tokens_target": args.prompt_tokens,
        "output_tokens_target": args.output_tokens,
        "success": len(ok),
        "failures": len(fail),
        "wall_s": wall,
        "throughput_tok_s": throughput,
        "lat_p50_s": percentile(lat, 50),
        "lat_p95_s": percentile(lat, 95),
        "lat_p99_s": percentile(lat, 99),
        "ttft_p50_s": percentile(ttfts, 50),
        "ttft_p95_s": percentile(ttfts, 95),
        "ttft_p99_s": percentile(ttfts, 99),
        "completion_tokens": total_completion_tokens,
        "token_count_source": (
            ok[0].get("token_count_source", "unknown") if ok else "unknown"
        ),
    }

    print(f"   success={len(ok)}/{len(results)} failures={len(fail)}")
    print(f"   wall={wall:.2f}s throughput={throughput:.2f} tok/s")
    print(
        f"   latency p50={row['lat_p50_s']:.3f}s "
        f"p95={row['lat_p95_s']:.3f}s "
        f"p99={row['lat_p99_s']:.3f}s"
    )
    print(
        f"   TTFT p50={row['ttft_p50_s']:.3f}s "
        f"p95={row['ttft_p95_s']:.3f}s "
        f"p99={row['ttft_p99_s']:.3f}s"
    )

    if fail:
        print(f"   first error: {fail[0].get('error')}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    new = not os.path.exists(args.csv)

    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()

