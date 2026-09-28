#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import uuid
import math
import random

def build_prompt(n_tokens: int, request_id: int, prompt_mode: str) -> str:
    sentence = "The quick brown fox jumps over the lazy dog. "
    body = sentence * max(1, n_tokens // 10)
    if prompt_mode == "shared":
        return body
    # Unique tag at the START: prefix-cache block hashes are chained,
    # so a unique first block makes every later block unique too.
    # The uuid nonce prevents cache hits from previous runs.
    return f"[req={request_id} nonce={uuid.uuid4().hex}]\n" + body

def percentile(values, p):
    """Linear-interpolated percentile. NaN (not 0.0) when there's no data."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return float(vals[lo] + (vals[hi] - vals[lo]) * (k - lo))

def one_request(request_id, url, model, prompt, out_tokens, timeout, t_sched=None):
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

    t_start = time.perf_counter()
    # Open-loop: latency counts from when the request was SCHEDULED to be sent,
    # not when a thread got around to sending it (avoids coordinated omission).
    t0 = t_sched if t_sched is not None else t_start
    send_lag = t_start - t0
    deadline = t0 + timeout

    ttft = None
    text_parts = []
    usage = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"total deadline of {timeout}s exceeded")
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

        tpot = None
        if ttft is not None and completion_tokens and completion_tokens > 1:
            tpot = (latency - ttft) / (completion_tokens - 1)

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
            "tpot": tpot,
            "timed_out": False,
            "send_lag": send_lag,
        }

    except Exception as e:
        reason = getattr(e, "reason", None)
        timed_out = (isinstance(e, TimeoutError)
                     or isinstance(reason, TimeoutError)
                     or "timed out" in str(e).lower())
        return {
            "request_id": request_id,
            "ok": False,
            "timed_out": timed_out,
            "error": repr(e),
            "latency": time.perf_counter() - t0,
            "send_lag": send_lag,
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--arrival-rate", type=float, help="open-loop: mean requests/second, Poisson arrivals")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random arrival gaps")
    ap.add_argument("--max-inflight", type=int, default=1024, help="open-loop: max client threads")
    ap.add_argument("--requests", type=int, required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--output-tokens", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--workload", default="unknown")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--details-csv", default="")
    ap.add_argument("--prompt-mode", choices=["unique", "shared"], default="unique")
    ap.add_argument("--slo-ttft", type=float, default=2.0, help="SLO: max time to first token, seconds")
    ap.add_argument("--slo-tpot", type=float, default=0.05, help="SLO: max time per output token, seconds")
    args = ap.parse_args()

    if (args.concurrency is None) == (args.arrival_rate is None):
        ap.error("give exactly one of --concurrency (closed-loop) or --arrival-rate (open-loop)")
    mode = "open" if args.arrival_rate is not None else "closed"

    load = (f"rate={args.arrival_rate}/s seed={args.seed}" if mode == "open"
            else f"concurrency={args.concurrency}")
    print(f">> workload={args.workload} mode={mode} requests={args.requests} {load} "
          f"prompt~{args.prompt_tokens} output={args.output_tokens}")

    wall0 = time.perf_counter()

    if mode == "closed":
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(
                pool.map(
                    lambda i: one_request(
                        i, args.url, args.model,
                        build_prompt(args.prompt_tokens, i, args.prompt_mode),
                        args.output_tokens, args.timeout,
                    ),
                    range(args.requests),
                )
            )
    else:
        rng = random.Random(args.seed)
        futures = []
        t_next = wall0
        with ThreadPoolExecutor(max_workers=args.max_inflight) as pool:
            for i in range(args.requests):
                prompt = build_prompt(args.prompt_tokens, i, args.prompt_mode)
                # Random gap with mean 1/rate. Added to a running clock, so small
                # sleep errors don't accumulate and drag the real rate down.
                t_next += rng.expovariate(args.arrival_rate)
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                futures.append(pool.submit(
                    one_request, i, args.url, args.model, prompt,
                    args.output_tokens, args.timeout, t_next))
            results = [f.result() for f in futures]

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
            "tpot",
            "timed_out",
            "send_lag",
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
    tpots = [r["tpot"] for r in ok if r.get("tpot") is not None]
    # Right-censored: a failed request took at least the timeout.
    lat_all = lat + [max(r["latency"], args.timeout) for r in fail]
    # "Good" = succeeded AND met both SLO limits. Failures are never good.
    good = [r for r in ok
            if r["ttft"] is not None and r["ttft"] < args.slo_ttft
            and r.get("tpot") is not None and r["tpot"] < args.slo_tpot]

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
        "prompt_mode": args.prompt_mode,
        "model": args.model,
        "concurrency": args.concurrency,
        "mode": mode,
        "arrival_rate": args.arrival_rate,
        "seed": args.seed,
        "send_lag_max_s": max((r["send_lag"] for r in results), default=float("nan")),
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
        "timeouts": sum(1 for r in fail if r.get("timed_out")),
        "failure_rate": len(fail) / len(results) if results else float("nan"),
        "lat_p99_all_s": percentile(lat_all, 99),
        "tpot_p50_s": percentile(tpots, 50),
        "tpot_p95_s": percentile(tpots, 95),
        "tpot_p99_s": percentile(tpots, 99),
        "slo_ttft_s": args.slo_ttft,
        "slo_tpot_s": args.slo_tpot,
        "slo_attainment": len(good) / len(results) if results else float("nan"),
        "goodput_rps": len(good) / wall if wall > 0 else 0.0,
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
    print(
        f"   TPOT p50={row['tpot_p50_s'] * 1000:.1f}ms "
        f"p95={row['tpot_p95_s'] * 1000:.1f}ms "
        f"p99={row['tpot_p99_s'] * 1000:.1f}ms"
    )
    print(f"   timeouts={row['timeouts']} p99(incl. failures)={row['lat_p99_all_s']:.3f}s")
    print(f"   SLO attainment={row['slo_attainment']:.1%} goodput={row['goodput_rps']:.3f} req/s")
    print(f"   client send lag max={row['send_lag_max_s'] * 1000:.1f}ms")

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

