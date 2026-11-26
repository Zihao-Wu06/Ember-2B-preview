#!/usr/bin/env python
"""Held-out evaluation sets built from the converted public data.

    python data_prep/sft_eval_sets.py            # data/sft/eval/tools_test.jsonl

tools_test.jsonl: function-calling decisions. From the glaive, hermes and ToolACE conversations that the teacher
did NOT get for its thoughts (data/distill/backfill_in/tools.jsonl is the training selection, so none of these is in
any training set), per source 70 conversations whose first assistant turn calls functions and 30 whose first assistant
turn does not (it asks for a missing value, answers or refuses). An item is the conversation up to the first assistant
turn, with its function list, and the reference turn:

    {"id", "source", "messages": [system?, user], "tools": [...], "ref_calls": [{"name", "arguments"}, ...], "ref_text"}

The evaluation (runs/mac_eval/chat_eval.py --tasks tools) lets the model write that turn and compares its calls.
The longdoc test file is written by distill_convert.py once the teacher has asked its questions about the held-out
documents.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONV = os.path.join(ROOT, "data", "sft", "conv")
EVAL = os.path.join(ROOT, "data", "sft", "eval")
TRAIN_SELECTION = os.path.join(ROOT, "data", "distill", "backfill_in", "tools.jsonl")
SOURCES = ("tools_glaive", "tools_hermes", "tools_toolace")


def first_turn(conv):
    """(messages before the first assistant turn, that turn) or None."""
    for i, m in enumerate(conv["messages"]):
        if m["role"] == "assistant":
            return (conv["messages"][:i], m) if i and conv["messages"][i - 1]["role"] == "user" else None
    return None


def build(n_call, n_plain, max_chars, seed):
    rng = random.Random(seed)
    train = set()
    if os.path.exists(TRAIN_SELECTION):
        with open(TRAIN_SELECTION, encoding="utf-8") as f:
            train = {json.loads(line)["id"] for line in f}
    items, report = [], {}
    for src in SOURCES:
        calls, plain = [], []
        with open(os.path.join(CONV, src + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                ft = first_turn(c)
                if c["id"] in train or ft is None or not c.get("tools") or len(line) > max_chars:
                    continue
                prefix, turn = ft
                item = {"id": c["id"], "source": src, "messages": prefix, "tools": c["tools"],
                        "ref_calls": [{"name": t["name"], "arguments": t.get("arguments") or {}}
                                      for t in turn.get("tool_calls") or ()],
                        "ref_text": (turn.get("content") or "").strip()}
                (calls if item["ref_calls"] else plain).append(item)
        pick = rng.sample(calls, min(n_call, len(calls))) + rng.sample(plain, min(n_plain, len(plain)))
        report[src] = {"with_call": len(calls), "without_call": len(plain), "chosen": len(pick)}
        items += pick
    return items, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-call", type=int, default=70)
    ap.add_argument("--n-plain", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    items, report = build(a.n_call, a.n_plain, a.max_chars, a.seed)
    os.makedirs(EVAL, exist_ok=True)
    path = os.path.join(EVAL, "tools_test.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(i, ensure_ascii=False) + "\n" for i in items)
    os.replace(path + ".tmp", path)
    print(json.dumps(report), f"-> {len(items)} items in {path}")
    print("SFT_EVAL_SETS_DONE")


if __name__ == "__main__":
    main()
