#!/usr/bin/env python
"""Agent evaluation in a small project (data_prep/agent_env.py): the model
gets solution.py and test_solution.py, the tools run_tests / read_file / edit_file / write_file, and up to --turns
assistant turns. A task counts as done when the tests pass on the files as the model left them (implement tasks also
report the hidden tests: all of MBPP's asserts, of which the model saw one). Tasks: data/agent/env_eval.jsonl
(HumanEval and MBPP's test split; nothing of them is in the training data).

    python runs/mac_eval/agent_env_eval.py --export runs/mac_eval/agent --device cuda --out runs/eval/env.json
Every conversation, the files at the end and the counters are kept in <out>_samples.jsonl.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import agent_env as E  # noqa: E402
import agent_eval as A  # noqa: E402
import chat_eval as CE  # noqa: E402
from moe_llm import chat  # noqa: E402


def tool_results(convs):
    """What the tools answered so far, by kind: the early sign of a model that cannot work the tools (in one
    fine-tune 533 of 887 edits did not find their text; nothing of it was visible before the run ended)."""
    n = collections.Counter()
    for c in convs:
        ms = c["messages"]
        for i, m in enumerate(ms):
            if m["role"] != "assistant":
                continue
            for call in m.get("tool_calls") or []:
                a = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                if call["name"] == "edit_file":
                    n["edits"] += 1
                    n["edits with old == new"] += isinstance(a.get("old"), str) and a.get("old") == a.get("new")
                    n["edits quoting 4+ lines"] += isinstance(a.get("old"), str) and a["old"].count("\n") >= 3
            if i + 1 < len(ms) and ms[i + 1]["role"] == "tool":
                out = ms[i + 1]["content"]
                n["tool results"] += 1
                if out.startswith("Error:"):
                    n["errors"] += 1
                    n["error: " + ("text not found" if "was not found" in out else "call not readable" if "could not be read" in out
                                   else "text occurs several times" if "occurs" in out else "other")] += 1
                n["passing test runs"] += out.strip() == E.PASS_LINE
    return dict(n)


def run_env(tasks, gen, turns, retry_temperature=0.0, progress=None, after_fail_temperature=0.0):
    """rows per task (and the conversations): {"done", "hidden", "turns", "answered", "calls", "bad_calls",
    "test_writes", "claimed_done"}. progress(turn, active, convs, st) is called after every turn.
    after_fail_temperature: a task's turns are greedy until one of its test runs fails after a change that was
    applied, and are sampled at this temperature from the next turn on ("sampled_from" in its row)."""
    envs = [E.Env(t["files"]) for t in tasks]
    convs = [{"messages": [{"role": "user", "content": t["prompt"]}], "tools": E.TOOLS} for t in tasks]
    st = [{"id": t["id"], "kind": t["kind"], "source": t["source"], "turns": 0, "answered": False,
           "calls": collections.Counter(), "bad_calls": 0, "sampled_from": None} for t in tasks]
    changed = [False] * len(tasks)
    active = list(range(len(tasks)))
    for r in range(turns):
        if not active:
            break
        if after_fail_temperature:
            groups = [([i for i in active if st[i]["sampled_from"] is None], 0.0, ""),
                      ([i for i in active if st[i]["sampled_from"] is not None], after_fail_temperature, " sampled")]
        else:
            groups = [(active, retry_temperature if r else 0.0, "")]
        replies = {}
        for idx, temp, tag in groups:
            if idx:
                replies.update(zip(idx, gen([convs[i] for i in idx], f"turn {r + 1}{tag}", temperature=temp, seed=r)))
        nxt = []
        for i in active:
            p = replies[i]
            s = st[i]
            s["turns"] = r + 1
            good = [c for c in p["tool_calls"] if "name" in c]
            msg = {"role": "assistant", "think": p["think"], "content": p["content"]}
            if good:
                msg["tool_calls"] = good
            convs[i]["messages"].append(msg)
            if not p["tool_calls"]:
                s["answered"] = True
                continue
            if r + 1 == turns:
                continue
            bad = len(p["tool_calls"]) - len(good)
            s["bad_calls"] += bad
            if bad and not good:
                convs[i]["messages"].append({"role": "tool", "content": "Error: the tool call could not be read. Write it as "
                                             "JSON with \"name\" and \"arguments\"."})
            for c in good:
                s["calls"][c["name"]] += 1
                out = envs[i].call(c["name"], c.get("arguments"))
                convs[i]["messages"].append({"role": "tool", "content": out})
                if c["name"] in ("edit_file", "write_file") and not out.startswith("Error:"):
                    changed[i] = True
                elif c["name"] == "run_tests" and changed[i] and out.strip() != E.PASS_LINE and s["sampled_from"] is None:
                    s["sampled_from"] = r + 2  # the next turn, counted from 1
            nxt.append(i)
        active = nxt
        if progress:
            try:
                progress(r + 1, active, convs, st)
            except Exception as e:  # noqa: BLE001  the progress file must never end the run
                print(f"progress failed: {e!r}", flush=True)
    for t, env, s in zip(tasks, envs, st):
        s["done"] = env.passed()
        s["hidden"] = E.run_files({**env.files, E.TEST: t["hidden"]})[0] if t.get("hidden") else None
        s["test_writes"] = env.test_writes
        s["calls"] = dict(s["calls"])
        s["files"] = env.files
    return st, convs


def summary(rows):
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[f"{r['kind']}/{r['source']}"].append(r)
    groups["all"] = rows
    for g, rs in groups.items():
        n = len(rs)
        hid = [r["hidden"] for r in rs if r["hidden"] is not None]
        calls = collections.Counter()
        for r in rs:
            calls.update(r["calls"])
        out[g] = {"n": n, "done": round(sum(r["done"] for r in rs) / n, 4),
                  "hidden_pass": round(sum(hid) / len(hid), 4) if hid else None,
                  "answered": sum(r["answered"] for r in rs), "claimed_but_failing": sum(r["answered"] and not r["done"] for r in rs),
                  "mean_turns": round(sum(r["turns"] for r in rs) / n, 2), "bad_calls": sum(r["bad_calls"] for r in rs),
                  "test_writes": sum(r["test_writes"] for r in rs), "calls": dict(calls),
                  "made_no_call": sum(not r["calls"] for r in rs),
                  "failed_after_change": sum(r.get("sampled_from") is not None for r in rs),
                  "done_after_failing": sum(r.get("sampled_from") is not None and r["done"] for r in rs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", default=os.path.join(ROOT, "data", "agent", "env_eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="at most this many tasks of each kind/source")
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=640)
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--rows-at-32k", type=int, default=5)
    ap.add_argument("--retry-temperature", type=float, default=0.0, help="sample every turn after the first at this temperature")
    ap.add_argument("--after-fail-temperature", type=float, default=0.0,
                    help="greedy until a task's test run fails after an applied change, then sample that task at this temperature")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.retry_temperature and args.after_fail_temperature:
        ap.error("--retry-temperature and --after-fail-temperature are two ways to sample; give one")
    tasks = [json.loads(line) for line in open(args.tasks, encoding="utf-8")]
    if args.limit:
        seen = collections.Counter()
        keep = []
        for t in tasks:
            g = (t["kind"], t["source"])
            if seen[g] < args.limit:
                seen[g] += 1
                keep.append(t)
        tasks = keep
    import mac_lm
    model, tok, meta = mac_lm.load(args.export, args.device)
    CE.set_moe_impl(model, "bmm")
    chat.check_tokenizer(tok)
    gen = lambda convs, label, temperature=0.0, seed=0: A.generate(  # noqa: E731
        model, tok, convs, max_new=args.max_new, temperature=temperature, seed=seed, rows_at_32k=args.rows_at_32k,
        max_rows=args.max_rows, label=label)
    t0 = time.time()
    partial = args.out.replace(".json", "_partial.jsonl")

    def progress(turn, active, convs, st):
        res = tool_results(convs)
        print(f"after turn {turn}: {len(active)} still working, {sum(s['answered'] for s in st)} answered, "
              f"{sum(s['sampled_from'] is not None for s in st)} failed a test run after a change; {json.dumps(res)}", flush=True)
        with open(partial + ".tmp", "w", encoding="utf-8") as f:
            for s, c in zip(st, convs):
                f.write(json.dumps({"id": s["id"], "kind": s["kind"], "source": s["source"], "turns": s["turns"],
                                    "answered": s["answered"], "messages": c["messages"]}, ensure_ascii=False) + "\n")
        os.replace(partial + ".tmp", partial)
    rows, convs = run_env(tasks, gen, args.turns, args.retry_temperature, progress, args.after_fail_temperature)
    res = {"step": meta["step"], "args": vars(args), "seconds": round(time.time() - t0), "summary": summary(rows)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    with open(args.out.replace(".json", "_samples.jsonl"), "w", encoding="utf-8") as f:
        for r, c in zip(rows, convs):
            f.write(json.dumps({**r, "messages": c["messages"]}, ensure_ascii=False) + "\n")
    print(json.dumps(res["summary"]), flush=True)
    print("AGENT_ENV_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
