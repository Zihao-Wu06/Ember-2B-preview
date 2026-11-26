#!/usr/bin/env python
"""The data mix of an agent fine-tuning as compact JSON for train.py --mix (2026-11-24): every agent source that exists
in data/bins_sft (at least --min-tokens) gets the weight that reads it --passes times in --steps steps, the agent
sources together at most --cap; the rest is the SFT mix sft_v3_64k scaled down (replay).
    python runs/posttrain/agent_mix.py --steps 350 sft_agent_tool sft_agent_fix sft_agent_env sft_agent_self"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def mix(sources, steps, passes=2.0, cap=0.40, global_tokens=524288, min_tokens=200_000, bins=None):
    import train
    bins = bins or os.path.join(ROOT, "data", "bins_sft")
    total = steps * global_tokens
    w = {}
    for s in sources:
        meta = os.path.join(bins, s, "meta.json")
        if os.path.exists(meta) and json.load(open(meta))["tokens"] >= min_tokens:
            w[s] = passes * json.load(open(meta))["tokens"] / total
    share = sum(w.values())
    if share > cap:
        w = {s: v * cap / share for s, v in w.items()}
        share = cap
    return {**w, **{n: v * (1 - share) for n, v in train.MIXES["sft_v3_64k"].items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--passes", type=float, default=2.0)
    ap.add_argument("--cap", type=float, default=0.40)
    a = ap.parse_args()
    m = mix(a.sources, a.steps, a.passes, a.cap)
    print(json.dumps({k: round(v, 6) for k, v in m.items()}, separators=(",", ":")))
