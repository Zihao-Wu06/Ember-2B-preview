#!/usr/bin/env python
"""Multi-step training conversations for the small agent: write
code, run it, read the result, fix it, answer. Built without the API from three things that exist:
  - the problem pool (agent_pool.py): statement, example asserts, the teacher's verified solution and its explanation
  - the current model's own first attempts at those problems (runs/mac_eval/agent_gen.py), right or wrong
  - real executions: every tool result and every error in the data is what the sandbox actually returned

Two conversation sets (moe_llm/chat.py format, tokenized by sft_tokenize.py):
  agent_tool  the python tool of the SFT data. "pass": the code is run against the examples, the tool prints
              "all tests passed", the final answer is that code. "fix": the model's own failed attempt is run, the
              tool returns the real error, the next turn says what went wrong (only what the tool output shows) and
              runs the verified solution, then answers with it.
  agent_fix   no tool: the attempt as a reply, the user's message with the failing test and the error (the wording of
              agent_eval.py's repair loop), the corrected reply.
Every problem with a failed attempt also gets a "pass" conversation with the verified solution as its first code.

Second version (2026-11-24, after the first fine-tuning; --v1 gives the first version's conversations):
  - the model's failed attempt is marked {"no_loss": true}: it is shown as context and is not a training target (the
    first version trained on it; MBPP's first-reply score went from 69 to 59 of 200)
  - a script holds exactly the tests the task shows, never more: the first model, trained on scripts with 2-5 asserts,
    invented asserts when a task gave one example, and its correct code then failed its own wrong test. Three forms:
    "plain" (the statement with all its examples), "docstring" (signature, statement and 1-3 >>> examples: HumanEval's
    form), "single" (the statement without its examples and one assert: MBPP's form)
  - conversations the fine-tuned model produced itself in the tool loop and that end in code passing every assert of
    the problem are added from --self (runs/mac_eval/agent_gen2.py): its own fixes, not the teacher's

    python data_prep/agent_traj.py     -> data/sft/conv/agent_tool.jsonl, agent_fix.jsonl, data/agent/traj_stats.json
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
import agent_eval as A  # noqa: E402
import agent_pool  # noqa: E402
import code_check  # noqa: E402

CONV = os.path.join(ROOT, "data", "sft", "conv")
AGENT = os.path.join(ROOT, "data", "agent")
LICENCE = "problems and solutions as in code_api (DeepSeek API / self-oss-instruct), attempts by the model, real runs"
OK_LINE = "all tests passed"
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
PASS_THINK = ("The run printed \"all tests passed\", so the code does what the examples ask.",
              "All tests passed, so this is the final code.",
              "The examples all pass with this code, so I can give it as the answer.",
              "The tool printed \"all tests passed\": the code is correct on the examples.")
RUN_FIRST = (" I will run it against the examples first.", " Let me check it against the examples.",
             " I'll test it on the examples before answering.", "")


def script(code: str, asserts: list) -> str:
    return code.rstrip() + "\n\n" + "\n".join(A.with_actual(a) for a in asserts) + f'\nprint("{OK_LINE}")\n'


def block(code: str) -> str:
    return "```python\n" + code.rstrip() + "\n```"


def simple(asserts: list):
    """[(call, value)] if every assert is a plain "assert CALL == VALUE", else None."""
    out = []
    for a in asserts:
        if ";" in a or not a.startswith("assert "):
            return None
        eq = agent_pool.split_eq(a[len("assert "):])
        if not eq:
            return None
        out.append(eq)
    return out


def docstring_prompt(p: dict, pairs=None):
    """The task as a signature with the statement as docstring and >>> examples (HumanEval's form), or None."""
    pairs = simple(p["asserts"]) if pairs is None else pairs
    if not p.get("signature") or not pairs or '"""' in p["prompt"] or "\\" in p["prompt"] or "```" in p["prompt"]:
        return None
    body = "\n".join(("    " + line).rstrip() for line in strip_examples(p["prompt"]).strip().splitlines())
    ex = "\n".join(f"    >>> {c}\n    {v}" for c, v in pairs)
    return f"Complete the following Python function.\n\n```python\n{p['signature']}\n    \"\"\"\n{body}\n{ex}\n    \"\"\"\n```"


_EXAMPLES = re.compile(r"\s*(?:Examples?|For example|E\.g\.)\s*[:,].*$", re.S)
_GIVEN_BLOCK = re.compile(r"\s*(?:Your code should pass the following (?:assertions?|test cases?)[^\n]*\n+```(?:python|py)?[ \t]*\n.*?```)", re.S)


def strip_examples(statement: str) -> str:
    """The statement without its trailing examples (the sentence that starts with "Examples:" or the given assert
    block); unchanged if no such part is found or if examples remain elsewhere in the text."""
    out = _GIVEN_BLOCK.sub("", statement)
    out = _EXAMPLES.sub("", out).rstrip()
    if len(out) < 40 or agent_pool.candidate_asserts(out):
        return statement
    return out


def make_task(p: dict, rng, v1: bool = False):
    """(task text, the asserts the task shows, form). The script of a conversation holds exactly these asserts."""
    asserts = p["asserts"]
    if v1:
        doc = docstring_prompt_v1(p)
        return (doc, asserts, "docstring") if doc and rng.random() < 0.33 else (p["prompt"], asserts, "plain")
    pairs = simple(asserts)
    bare = strip_examples(p["prompt"])
    r = rng.random()
    if pairs and bare != p["prompt"]:
        if r < 0.30:  # MBPP's form: no examples in the text, one test
            k = rng.randrange(len(asserts))
            return f"{bare}\nYour code should pass this test: {asserts[k]}", [asserts[k]], "single"
        if r < 0.55 and p.get("signature"):
            idx = sorted(rng.sample(range(len(asserts)), rng.randint(1, min(3, len(asserts)))))
            doc = docstring_prompt(p, [pairs[i] for i in idx])
            if doc:
                return doc, [asserts[i] for i in idx], "docstring"
    return p["prompt"], asserts, "plain"


def docstring_prompt_v1(p: dict):
    pairs = simple(p["asserts"])
    if not p.get("signature") or not pairs or '"""' in p["prompt"] or "\\" in p["prompt"] or "```" in p["prompt"]:
        return None
    body = "\n".join(("    " + line).rstrip() for line in p["prompt"].strip().splitlines())
    ex = "\n".join(f"    >>> {c}\n    {v}" for c, v in pairs)
    return f"Complete the following Python function.\n\n```python\n{p['signature']}\n    \"\"\"\n{body}\n{ex}\n    \"\"\"\n```"


def diagnosis(output: str, test=None) -> str:
    """One or two sentences on a failed run that use only what the tool output (or the user's message) shows."""
    lines = [line for line in output.strip().splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    if "did not finish" in output or last == "timeout":
        return "The script did not finish in time, so the code loops for too long or never stops."
    if last.startswith("AssertionError"):
        got = last[len("AssertionError"):].lstrip(": ").strip()
        failing = test or next((line.strip() for line in reversed(lines) if line.strip().startswith("assert ")), None)
        if failing:
            failing = re.sub(r", repr\(.*\)$", "", failing)
        if got and failing:
            return f"The test `{failing}` fails: the call returned {got}."
        if failing:
            return f"The test `{failing}` fails, so the result is wrong for this example."
        return "One of the examples fails, so the result is wrong."
    if last:
        return f"The run stops with `{last[:160]}`, so the code has an error before it reaches a result."
    return "The run failed."


def work(job):
    """One problem: run its attempts and the solution, return the conversations (as dicts) and counters."""
    p, attempts, seed, v1 = job
    rng = random.Random(seed)
    stats = collections.Counter()
    out = {"agent_tool": [], "agent_fix": []}
    sol = p["solution"]
    if code_check.UNSAFE.search(sol):
        stats["solution not run (blocked pattern)"] += 1
        return out, stats
    if A.tool_output(script(sol, p["asserts"])).strip() != OK_LINE:
        stats["solution does not print the pass line"] += 1
        return out, stats
    task, asserts, form = make_task(p, rng, v1)  # asserts: the tests the task shows; the scripts hold exactly these
    sol_out = OK_LINE
    skip = {} if v1 else {"no_loss": True}  # the failed attempt: context, not a training target
    tools_user = {"role": "user", "content": task + A.TOOL_NOTE}

    def call(code):
        return [{"name": A.TOOL_NAME, "arguments": {"code": script(code, asserts)}}]

    def final(code):
        return {"role": "assistant", "think": rng.choice(PASS_THINK), "content": block(code)}

    def conv(name, kind, attempt, msgs, tools):
        out[name].append({"id": f"{name}/{p['id']}/{kind}", "set": name, "licence": LICENCE, "messages": msgs,
                          "tools": tools, "meta": {"kind": kind, "form": form, "attempt": attempt, "source": p["source"]}})

    failed = None
    for a in attempts:
        m = _BLOCK.search(a["content"] or "")
        if not a.get("closed") or not m or not (a.get("think") or "").strip():
            stats[f"{a['pass']}: no usable reply"] += 1
            continue
        code = m.group(1).rstrip()
        if code_check.UNSAFE.search(code):
            stats[f"{a['pass']}: not run (blocked pattern)"] += 1
            continue
        res = A.tool_output(script(code, asserts))
        if res.strip() == OK_LINE:
            stats[f"{a['pass']}: passes the examples"] += 1
            continue
        stats[f"{a['pass']}: fails the examples"] += 1
        if failed is None:
            failed = (a, code, res)
    # pass: the verified solution run once, then given as the answer
    conv("agent_tool", "pass", "teacher",
         [tools_user, {"role": "assistant", "think": p["explanation"] + rng.choice(RUN_FIRST), "content": "", "tool_calls": call(sol)},
          {"role": "tool", "content": sol_out}, final(sol)], A.RUN_TOOL)
    if failed:
        a, code, res = failed
        fix_think = diagnosis(res) + " " + p["explanation"]
        conv("agent_tool", "fix", a["pass"],
             [tools_user, {"role": "assistant", "think": a["think"].strip(), "content": "", "tool_calls": call(code), **skip},
              {"role": "tool", "content": res},
              {"role": "assistant", "think": fix_think, "content": "", "tool_calls": call(sol)},
              {"role": "tool", "content": sol_out}, final(sol)], A.RUN_TOOL)
        fail = A.first_failure(code, [(t, A.with_actual(t)) for t in asserts])
        if fail:
            test, err = fail
            conv("agent_fix", "fix", a["pass"],
                 [{"role": "user", "content": task}, {"role": "assistant", "think": a["think"].strip(), "content": block(code), **skip},
                  {"role": "user", "content": A.feedback_text(test, err)},
                  {"role": "assistant", "think": diagnosis(err, test) + " " + p["explanation"], "content": block(sol)}], None)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(AGENT, "pool.jsonl"))
    ap.add_argument("--attempts", default=os.path.join(AGENT, "attempts.jsonl"))
    ap.add_argument("--conv", default=CONV)
    ap.add_argument("--extra-pass", type=int, default=0, help="also this many pass conversations for problems without attempts")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v1", action="store_true", help="the first version's conversations (no masks, all asserts in every script)")
    ap.add_argument("--self", dest="self_traj", default=None, help="conversations the model produced itself (agent_gen2.py)")
    args = ap.parse_args()
    pool = {}
    for line in open(args.pool, encoding="utf-8"):
        p = json.loads(line)
        pool[p["id"]] = p
    attempts = collections.defaultdict(list)
    for line in open(args.attempts, encoding="utf-8"):
        a = json.loads(line)
        if a["id"] in pool:
            attempts[a["id"]].append(a)
    ids = sorted(attempts)
    rng = random.Random(args.seed)
    rest = [i for i in sorted(pool) if i not in attempts]
    rng.shuffle(rest)
    ids += rest[:args.extra_pass]
    if args.limit:
        ids = ids[:args.limit]
    jobs = [(pool[i], sorted(attempts.get(i, []), key=lambda a: a["pass"]), f"{args.seed}/{i}", args.v1) for i in ids]
    print(f"{len(jobs)} problems ({len(attempts)} with attempts)", flush=True)
    sets, stats = {"agent_tool": [], "agent_fix": []}, collections.Counter()
    with multiprocessing.Pool(args.workers) as mp:
        for k, (out, st) in enumerate(mp.imap_unordered(work, jobs, chunksize=8)):
            stats.update(st)
            for name, convs in out.items():
                sets[name] += convs
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(jobs)} problems", flush=True)
    summary = {"problems": len(jobs), "runs": dict(stats)}
    if args.self_traj and os.path.exists(args.self_traj):
        own = [json.loads(line) for line in open(args.self_traj, encoding="utf-8")]
        sets["agent_tool"] += own
        summary["own conversations added"] = dict(collections.Counter(c["meta"]["kind"] for c in own))
    for name, convs in sets.items():
        convs.sort(key=lambda c: c["id"])
        rng.shuffle(convs)
        path = os.path.join(args.conv, name + ".jsonl")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
        kinds = collections.Counter((c["meta"]["kind"], c["meta"]["form"]) for c in convs)
        chars = sum(len(json.dumps(c["messages"], ensure_ascii=False)) for c in convs)
        summary[name] = {"conversations": len(convs), "by kind and form": {f"{k}/{f}": n for (k, f), n in sorted(kinds.items())},
                         "characters": chars}
    with open(os.path.join(AGENT, "traj_stats.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("AGENT_TRAJ_DONE " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
