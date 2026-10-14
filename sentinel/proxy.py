#!/usr/bin/env python3
"""Sentinel: KV-aware admission controller, a proxy in front of vLLM.

client -> this proxy (:8001) -> vLLM (:8000)
Policies:
  none      forward everything immediately (measures proxy overhead)
  peak      admit if every request's FULL size fits at once (static reservation)
  lookahead admit if projected KV memory never overflows, counting when
            running requests finish and free their blocks
"""
import argparse, asyncio, json, time
from collections import deque
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector

BLOCK = 16                                   # vLLM block size, in tokens


def blocks(tokens):
    return -(-tokens // BLOCK)               # ceiling division


class Req:
    def __init__(self, prompt, max_tokens):
        self.prompt = prompt                 # prompt tokens (exact, from /tokenize)
        self.max_tokens = max_tokens         # upper bound on output tokens
        self.generated = 0                   # output tokens streamed so far
        self.admitted = asyncio.Event()


class Controller:
    def __init__(self, policy, budget):
        self.policy, self.budget = policy, budget
        self.queue, self.running = deque(), set()
        self.wake = asyncio.Event()

    def fits(self, new):
        if self.policy == "none":
            return True
        reqs = list(self.running) + [new]
        if self.policy == "peak":
            return sum(blocks(r.prompt + r.max_tokens) for r in reqs) <= self.budget
        # lookahead: every running request gains one token per step. Memory only
        # drops when a request finishes, so the peaks are just before finishes.
        # s = steps from now at which some request finishes.
        for s in {r.max_tokens - r.generated for r in reqs}:
            total = sum(blocks(r.prompt + r.generated + s)
                        for r in reqs if r.max_tokens - r.generated >= s)
            if total > self.budget:
                return False
        return True

    async def scheduler(self):
        while True:
            # Strict first-come-first-served: only the head of the line may enter.
            while self.queue and self.fits(self.queue[0]):
                r = self.queue.popleft()
                self.running.add(r)
                r.admitted.set()
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=0.02)
            except asyncio.TimeoutError:
                pass  # re-check anyway: running requests progressed


async def completions(request):
    app = request.app
    ctl, http, vllm = app["ctl"], app["http"], app["vllm"]
    body = await request.json()

    async with http.post(f"{vllm}/tokenize",
                         json={"model": body.get("model"), "prompt": body["prompt"]}) as t:
        n_prompt = (await t.json())["count"]
    req = Req(n_prompt, body.get("max_tokens", 16))
    if blocks(req.prompt + req.max_tokens) > ctl.budget:
        return web.json_response({"error": "request can never fit in KV cache"}, status=400)

    ctl.queue.append(req)
    ctl.wake.set()
    try:
        await req.admitted.wait()
    finally:
        if not req.admitted.is_set():        # client gave up while waiting
            ctl.queue.remove(req)

    try:
        async with http.post(f"{vllm}/v1/completions", json=body) as up:
            resp = web.StreamResponse(status=up.status, headers={
                "Content-Type": up.headers.get("Content-Type", "text/event-stream")})
            await resp.prepare(request)
            async for line in up.content:
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    try:
                        if json.loads(line[5:]).get("choices"):
                            req.generated += 1
                    except ValueError:
                        pass
                await resp.write(line)
            await resp.write_eof()
            return resp
    finally:
        ctl.running.discard(req)             # free the reservation
        ctl.wake.set()


async def passthrough(request):
    http, vllm = request.app["http"], request.app["vllm"]
    async with http.request(request.method, f"{vllm}{request.path_qs}",
                            data=await request.read(),
                            headers={"Content-Type": request.headers.get(
                                "Content-Type", "application/json")}) as up:
        return web.Response(body=await up.read(), status=up.status,
                            content_type=up.content_type)


async def stats(ctl):
    while True:
        await asyncio.sleep(5)
        print(f"[{time.strftime('%H:%M:%S')}] running={len(ctl.running)} "
              f"queued={len(ctl.queue)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--policy", choices=["none", "peak", "lookahead"], default="lookahead")
    ap.add_argument("--kv-blocks", type=int, default=4735)
    ap.add_argument("--margin", type=int, default=32, help="safety cushion, in blocks")
    args = ap.parse_args()
    budget = args.kv_blocks - args.margin

    async def on_start(app):
        app["http"] = ClientSession(connector=TCPConnector(limit=0),
                                    timeout=ClientTimeout(total=None))
        app["ctl"] = Controller(args.policy, budget)
        app["tasks"] = [asyncio.create_task(app["ctl"].scheduler()),
                        asyncio.create_task(stats(app["ctl"]))]

    async def on_stop(app):
        for t in app["tasks"]:
            t.cancel()
        await app["http"].close()

    app = web.Application(client_max_size=64 * 2**20)
    app["vllm"] = args.vllm.rstrip("/")
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_post("/v1/completions", completions)
    app.router.add_route("*", "/{tail:.*}", passthrough)
    print(f"Sentinel proxy :{args.port} -> {args.vllm} policy={args.policy} "
          f"budget={budget} blocks", flush=True)
    web.run_app(app, port=args.port, print=None)


if __name__ == "__main__":
    main()
