#!/usr/bin/env python
"""First attempts of the current model at the problems of the agent pool (data_prep/agent_pool.py), for the multi-step
training data (data_prep/agent_traj.py): the model answers the plain question, as in its SFT data, once greedily and
once sampled, and every reply is kept with its thinking. Whether an attempt passes is decided later, by running it.

    python runs/mac_eval/agent_gen.py --export runs/mac_eval/sft_0002862 --device cuda --n 14000 --minutes 75
        -> data/agent/attempts.jsonl   {"id", "pass": "greedy" | "sampled", "think", "content", "closed", "tokens"}
The greedy pass runs over all --n problems first, then the sampled pass (temperature --temperature) for as long as
--minutes allows. Re-running continues: attempts already in the file are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import agent_eval as A  # noqa: E402
import chat_eval as CE  # noqa: E402
from moe_llm import chat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", default=os.path.join(ROOT, "data", "agent", "pool.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "agent", "attempts.jsonl"))
    ap.add_argument("--n", type=int, default=14000)
    ap.add_argument("--minutes", type=float, default=75)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=448)
    ap.add_argument("--max-rows", type=int, default=224)
    ap.add_argument("--chunk", type=int, default=1792)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", default="greedy,sampled", help="names of the passes in order; \"greedy\" decodes greedily, "
                    "any other name samples at --temperature with its own seed")
    ap.add_argument("--ids", default=None, help="a file of problem ids: only these problems")
    args = ap.parse_args()
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    if args.ids:
        keep = set(open(args.ids).read().split())
        pool = [p for p in pool if p["id"] in keep]
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            x = json.loads(line)
            done.add((x["id"], x["pass"]))
    import mac_lm
    model, tok, _ = mac_lm.load(args.export, args.device)
    CE.set_moe_impl(model, "bmm")
    chat.check_tokenizer(tok)
    t0, wrote = time.time(), 0
    with open(args.out, "a", encoding="utf-8") as f:
        for k, name in enumerate(args.passes.split(",")):
            temp = 0.0 if name == "greedy" else args.temperature
            todo = [p for p in pool if (p["id"], name) not in done]
            for i in range(0, len(todo), args.chunk):
                if time.time() - t0 > args.minutes * 60:
                    break
                part = todo[i:i + args.chunk]
                convs = [{"messages": [{"role": "user", "content": p["prompt"]}]} for p in part]
                try:
                    replies = A.generate(model, tok, convs, max_new=args.max_new, temperature=temp, seed=args.seed + 1000 * k + i,
                                         max_rows=args.max_rows, label=f"{name} {i + len(part)}/{len(todo)}")
                except Exception as e:  # noqa: BLE001  the attempts written so far are enough to go on with
                    print(f"{name}: generation stopped with {e!r}", flush=True)
                    break
                for p, r in zip(part, replies):
                    f.write(json.dumps({"id": p["id"], "pass": name, "think": r["think"], "content": r["content"],
                                        "closed": r["closed"], "tokens": r["tokens"]}, ensure_ascii=False) + "\n")
                f.flush()
                wrote += len(part)
                print(f"{name}: {i + len(part)}/{len(todo)} attempts, {wrote} written, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"AGENT_GEN_DONE {args.out}: {wrote} attempts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
