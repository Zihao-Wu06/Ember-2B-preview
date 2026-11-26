#!/usr/bin/env python
"""The fine-tuned model solving pool problems in the python-tool loop, to harvest the conversations it gets right by
itself (second round of the agent data): its own fixes after a real error replace
the teacher's solution as the model for the "fix" step. A conversation is kept only if
  - it ends with a code block that passes EVERY assert of the problem (the task shows only some of them, so the rest
    are tests the model never saw),
  - the last run before the answer printed "all tests passed",
  - every assert in every script it ran is one the task shows (the first model invented tests; those are dropped).
Runs that failed are marked {"no_loss": true}: they stay in the conversation as context and are not training targets.

    python runs/mac_eval/agent_gen2.py --export runs/mac_eval/agent_v1 --device cuda --minutes 110
        -> data/agent/self_traj.jsonl   conversations with meta.kind "self_fix" (a failed run before the answer) or
           "self_pass"; re-running continues after the problems already tried (data/agent/self_tried.txt)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import agent_eval as A  # noqa: E402
import agent_traj as T  # noqa: E402
import chat_eval as CE  # noqa: E402
import chat_score as S  # noqa: E402
from moe_llm import chat  # noqa: E402

AGENT = os.path.join(ROOT, "data", "agent")


def run_chunk(problems, gen, rounds, seed):
    """Tool loop on a list of pool problems: (kept conversations, counters)."""
    stats = collections.Counter()
    tasks = [T.make_task(p, random.Random(f"{seed}/{p['id']}")) for p in problems]
    convs = [{"messages": [{"role": "user", "content": t[0] + A.TOOL_NOTE}], "tools": A.RUN_TOOL} for t in tasks]
    outs = [[] for _ in problems]  # tool results per problem, in order
    final = [None] * len(problems)
    active = list(range(len(problems)))
    for r in range(rounds):
        if not active:
            break
        replies = gen([convs[i] for i in active], f"turn {r + 1}")
        nxt = []
        for i, p in zip(active, replies):
            calls = [c for c in p["tool_calls"] if c.get("name") == A.TOOL_NAME and isinstance(c.get("arguments"), dict)
                     and isinstance(c["arguments"].get("code"), str)]
            if len(calls) != len(p["tool_calls"]) or len(calls) > 1 or not p["closed"]:
                stats["dropped: malformed turn"] += 1
                continue
            convs[i]["messages"].append(A.assistant_msg({**p, "tool_calls": calls}))
            if calls and r + 1 < rounds:
                res = A.tool_output(calls[0]["arguments"]["code"])
                outs[i].append(res)
                convs[i]["messages"].append({"role": "tool", "content": res})
                nxt.append(i)
            elif calls:
                stats["dropped: out of turns"] += 1
            else:
                final[i] = p["content"]
        active = nxt
    kept = []
    for i, p in enumerate(problems):
        if final[i] is None:
            continue
        code = S.code_of(final[i])
        shown = set(tasks[i][1]) | {A.with_actual(a) for a in tasks[i][1]}
        scripts = [m["tool_calls"][0]["arguments"]["code"] for m in convs[i]["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        if code is None or not outs[i] or outs[i][-1].strip() != T.OK_LINE:
            stats["dropped: answer without a passing run"] += 1
            continue
        if any(line.strip() not in shown for s in scripts for line in s.splitlines() if line.startswith("assert ")):
            stats["dropped: a test the task does not show"] += 1
            continue
        if A.tool_output(T.script(code, p["asserts"])).strip() != T.OK_LINE:
            stats["dropped: fails an assert it was not shown"] += 1
            continue
        k = 0
        for m in convs[i]["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                if outs[i][k].strip() != T.OK_LINE:
                    m["no_loss"] = True
                k += 1
        kind = "self_fix" if any(o.strip() != T.OK_LINE for o in outs[i]) else "self_pass"
        stats[f"kept: {kind}"] += 1
        kept.append({"id": f"agent_tool/{p['id']}/{kind}", "set": "agent_tool", "licence": T.LICENCE, "messages": convs[i]["messages"],
                     "tools": A.RUN_TOOL, "meta": {"kind": kind, "form": tasks[i][2], "attempt": "own", "source": p["source"],
                                                    "runs": len(outs[i])}})
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pool", default=os.path.join(AGENT, "pool_v2.jsonl"))
    ap.add_argument("--skip", default=os.path.join(AGENT, "attempts.jsonl"), help="problems of this attempts file are left out")
    ap.add_argument("--out", default=os.path.join(AGENT, "self_traj.jsonl"))
    ap.add_argument("--minutes", type=float, default=110)
    ap.add_argument("--chunk", type=int, default=384)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=448)
    ap.add_argument("--max-rows", type=int, default=96)
    ap.add_argument("--rows-at-32k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    skip = set()
    if args.skip and os.path.exists(args.skip):
        skip = {json.loads(line)["id"] for line in open(args.skip, encoding="utf-8")}
    tried_path = os.path.join(os.path.dirname(args.out), "self_tried.txt")
    if os.path.exists(tried_path):
        skip |= set(open(tried_path).read().split())
    pool = [p for p in pool if p["id"] not in skip and len(p["asserts"]) >= 2]
    random.Random(args.seed).shuffle(pool)
    import mac_lm
    model, tok, _ = mac_lm.load(args.export, args.device)
    CE.set_moe_impl(model, "bmm")
    chat.check_tokenizer(tok)
    gen = lambda convs, label: A.generate(model, tok, convs, max_new=args.max_new, rows_at_32k=args.rows_at_32k,  # noqa: E731
                                         max_rows=args.max_rows, label=label)
    t0, total = time.time(), collections.Counter()
    for i in range(0, len(pool), args.chunk):
        if time.time() - t0 > args.minutes * 60:
            break
        part = pool[i:i + args.chunk]
        try:
            kept, stats = run_chunk(part, gen, args.rounds, args.seed)
        except Exception as e:  # noqa: BLE001  what is written so far stays usable
            print(f"chunk at {i} stopped with {e!r}", flush=True)
            break
        with open(args.out, "a", encoding="utf-8") as f:
            for c in kept:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with open(tried_path, "a") as f:
            f.write("\n".join(p["id"] for p in part) + "\n")
        total.update(stats)
        total["problems"] += len(part)
        print(f"{total['problems']} problems, {(time.time() - t0) / 60:.1f} min: {dict(total)}", flush=True)
    print(f"AGENT_GEN2_DONE {args.out}: {dict(total)}", flush=True)


if __name__ == "__main__":
    main()
