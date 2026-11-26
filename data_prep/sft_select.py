#!/usr/bin/env python
"""Which public conversations without reasoning are used, and how:

  chat    about 80,000 SmolTalk conversations, a fixed number per subset (CHAT). A quarter of each subset goes to the
          teacher, which writes one short thought per reply (data/distill/backfill_in/chat.jsonl -> the set chat_api);
          the rest is used as it is, with an empty thinking block (data/sft/conv/chat.jsonl).
  tools   about 35,000 function-calling conversations (TOOLS), all of them to the teacher
          (data/distill/backfill_in/tools.jsonl -> the set tools_api): choosing a function and its arguments is where
          a thought helps, so none is used with an empty block.

    python data_prep/sft_select.py

The choice is a seeded random sample, so running it again gives the same files. Conversations of more than
--max-chars characters are not chosen (they cost more to send and the stream cuts off at 3,500 tokens anyway).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_eval_sets import first_turn  # noqa: E402
EVAL_MAX_CHARS = 6000  # sft_eval_sets.py --max-chars
CONV = os.path.join(ROOT, "data", "sft", "conv")
BACKFILL = os.path.join(ROOT, "data", "distill", "backfill_in")
# SmolTalk subset -> conversations taken (everyday-conversations has about 2,000 in all)
CHAT = {"everyday-conversations": 5000, "systemchats-30k": 12000, "smol-magpie-ultra": 30000,
        "openhermes-100k": 15000, "smol-constraints": 6000, "smol-rewrite": 6000, "smol-summarize": 6000,
        "explore-instruct-rewriting": 4000}
CHAT_TEACHER_SHARE = 0.25
# set -> (conversations taken, share of them that contain no function call: the assistant asks, refuses or chats)
TOOLS = {"tools_glaive": (18000, 0.2), "tools_hermes": (8000, 0.2), "tools_toolace": (9000, 0.13)}
# hermes has only 8,769 conversations and the selection above would take nearly all of them: these are kept back
# (conversations with a call, without one) for the held-out function-calling evaluation (sft_eval_sets.py)
EVAL_RESERVE = {"tools_hermes": (100, 50)}


def read(name, max_chars):
    with open(os.path.join(CONV, name + ".jsonl"), encoding="utf-8") as f:
        for line in f:
            if len(line) <= max_chars:
                yield line


def take(lines, n, rng):
    lines = sorted(lines)  # the sample does not depend on the order of the file
    return rng.sample(lines, n) if len(lines) > n else lines


def write(path, lines, rng):
    lines = list(lines)
    rng.shuffle(lines)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.replace(path + ".tmp", path)
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng, report = random.Random(a.seed), {"chat": {}, "tools": {}}
    by_src = collections.defaultdict(list)
    for line in read("smoltalk", a.max_chars):
        i = line.find('"source": "')  # meta is the last field of the line
        src = line[i + 11:line.find('"', i + 11)] if i >= 0 else None
        if src in CHAT:
            by_src[src].append(line)
    plain, teacher = [], []
    for src, n in CHAT.items():
        got = take(by_src[src], n, rng)
        k = round(len(got) * CHAT_TEACHER_SHARE)
        teacher += got[:k]
        plain += got[k:]
        report["chat"][src] = {"available": len(by_src[src]), "empty_thinking": len(got) - k, "to_teacher": k}
    report["chat"]["total"] = {"empty_thinking": write(os.path.join(CONV, "chat.jsonl"), plain, rng),
                               "to_teacher": write(os.path.join(BACKFILL, "chat.jsonl"), teacher, rng)}
    tools = []
    for name, (n, no_call) in TOOLS.items():
        with_call, without = [], []
        for line in read(name, a.max_chars):
            (with_call if '"tool_calls": [{' in line else without).append(line)
        if name in EVAL_RESERVE:  # conversations sft_eval_sets.py can use: short, a user turn first, a function list
            def usable(line):
                c = json.loads(line)
                return len(line) <= EVAL_MAX_CHARS and first_turn(c) is not None and bool(c.get("tools"))
            held = take([x for x in with_call if usable(x)], EVAL_RESERVE[name][0], rng)
            held += take([x for x in without if usable(x)], EVAL_RESERVE[name][1], rng)
            kept_back = set(held)
            with_call, without = [x for x in with_call if x not in kept_back], [x for x in without if x not in kept_back]
            report["tools"][name + "_kept_back_for_evaluation"] = len(held)
        got = take(without, round(n * no_call), rng)
        got += take(with_call, n - len(got), rng)
        tools += got
        report["tools"][name] = {"with_call": len(with_call), "without_call": len(without), "to_teacher": len(got)}
    report["tools"]["total"] = {"to_teacher": write(os.path.join(BACKFILL, "tools.jsonl"), tools, rng)}
    with open(os.path.join(BACKFILL, "SELECT.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("SFT_SELECT_DONE")


if __name__ == "__main__":
    main()
