#!/usr/bin/env python
"""Small-agent evaluation of a post-trained checkpoint (the goal is a
model that completes a task over several steps, writing code, running it, reading the error and fixing it). Three
measurements on HumanEval and MBPP, all scored at the end with the official tests, which the model never sees:

  repair   the question of chat_eval.py, then a loop: the reply's code runs against the tests the model CAN see (the
           docstring examples of HumanEval, the one assert in MBPP's prompt); if one fails, the failing test and the
           error go back as the next user turn and the model answers again, up to --rounds replies in all.
           --feedback oracle shows the official tests' error instead: not a fair score, an upper bound on what
           reading errors could buy.
  tool     the same task with the python tool of the SFT data (function-calling format): the model may run code
           any number of times (up to --rounds assistant turns) and then gives its final code.
  resample no feedback at all: if the reply's code fails a test the model can see, the question is simply asked again and
           the reply sampled (temperature --temperature), up to --rounds replies; the first one that passes the visible
           tests is the answer, else the first reply. It uses "writes it right sometimes" directly: the model need not
           know how to fix, the run tells which draw is right.
  repair / tool with --retry-temperature T: the first reply greedy as before, every later one sampled, so that a model
           that would send the same wrong code again tries something else.
  samples  --k independent sampled replies (temperature --temperature) to the single question: the share of replies
           that pass (pass@1) against the share of problems where at least one of the k passes (pass@k). The gap is
           what "writes it right sometimes" is worth before any feedback.

    python runs/mac_eval/agent_eval.py --export runs/mac_eval/sft_0002862 --device cuda --mode repair --tasks humaneval,mbpp \
        --out runs/eval/agent_repair.json
Every conversation is kept in <out>_samples.jsonl (the successful ones are the raw material for multi-step training data).
Code runs only through data_prep/code_check.py (its sandbox; WSL only).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import chat_eval as CE  # noqa: E402
import chat_score as S  # noqa: E402
import code_check  # noqa: E402
from moe_llm import chat  # noqa: E402

MAX_NEW = 1024
CLOSE = ("def _close(a, b):\n    if a == b:\n        return True\n"
         "    return isinstance(a, float) and isinstance(b, (int, float)) and abs(a - b) < 1e-6\n")
# the tool of the SFT data's gsm8k_py conversations (6,997 of them): same name, same schema, the result is plain text
RUN_TOOL = [{"type": "function", "function": {
    "name": "python", "description": "Run Python code and return what it prints.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to run."}},
                   "required": ["code"]}}}]
TOOL_NAME = "python"
TOOL_NOTE = ("\n\nYou can run code with the python tool to test your solution before you answer. When the tests pass, "
             "reply with the final code in one Python code block.")
FIX = "Fix the code and reply with the complete corrected code in one Python code block."


# ---- the tests the model may see ------------------------------------------------------------------------------------
def doctest_examples(prompt: str, entry: str) -> list:
    """The ">>> call / expected value" examples of a HumanEval docstring as (call, value) pairs (literal values only)."""
    lines, out, i = prompt.splitlines(), [], 0
    while i < len(lines):
        s = lines[i].strip()
        if not s.startswith(">>>"):
            i += 1
            continue
        expr, want, j = s[3:].strip(), [], i + 1
        while j < len(lines):
            t = lines[j].strip()
            if not t or t.startswith(">>>") or t.startswith('"""') or t.startswith("'''"):
                break
            want.append(t)
            j += 1
        i = j
        if not want or entry + "(" not in expr:
            continue
        value = " ".join(want)
        try:
            ast.literal_eval(value)
            ast.parse(expr, mode="eval")
        except (ValueError, SyntaxError):
            continue
        out.append((expr, value))
    return out


def visible_tests(task: str, d: dict) -> list:
    """[(the test as shown to the model, the code that is run)]; a failed run reports the value the call returned."""
    if task == "humaneval":
        return [(f"assert {e} == {v}", CLOSE + f"assert _close({e}, {v}), repr({e})")
                for e, v in doctest_examples(d["prompt"], d["entry_point"])]
    return [(d["test_list"][0], (d.get("test_setup_code") or "") + "\n" + with_actual(d["test_list"][0]))]


def first_failure(code, tests):
    """(test as shown, error text) of the first visible test that fails, else None."""
    for show, run in tests:
        ok, err = code_check.check(code, {"type": "asserts", "code": run})
        if not ok:
            return show, "timeout" if err == "timeout" else tidy(err)
    return None


def official(task: str, d: dict, content: str):
    return (S.humaneval_check if task == "humaneval" else S.mbpp_check)(d, content)


def tidy(err: str) -> str:
    """The end of a traceback as a reader wants it: no half line at the cut, no lines of ^^^ markers."""
    lines = (err or "").strip().splitlines()
    if len(lines) > 1 and not lines[0].startswith(("Traceback", "  File")):
        lines = lines[1:]  # code_check keeps the last 300 characters: the first line is usually cut in the middle
    return "\n".join(line for line in lines if line.strip().strip("^~ ")) or "failed"


def tool_output(code: str) -> str:
    """What the python tool returns for a script (run in code_check's sandbox): what it printed, then the error."""
    if code_check.UNSAFE.search(code):
        return "Error: the script uses a module or call that is not allowed here"
    ok, stdout, err = code_check.run(code_check.PRELUDE + code)
    out = stdout[-500:].rstrip()
    if ok:
        return out
    err = "the script did not finish within 10 seconds" if err == "timeout" else tidy(err)
    return (out + "\n" if out else "") + err


def with_actual(stmt: str) -> str:
    """"assert CALL == VALUE" -> "assert CALL == VALUE, repr(CALL)": a failure then shows the value the call returned."""
    head, sep, last = stmt.rpartition(";")
    last = last.strip()
    if not last.startswith("assert ") or "==" not in last:
        return stmt
    import agent_pool
    eq = agent_pool.split_eq(last[len("assert "):])
    if not eq:
        return stmt
    return (head + sep + " " if sep else "") + f"assert {eq[0]} == {eq[1]}, repr({eq[0]})"


def feedback_text(test, err):
    if err == "timeout":
        return f"I ran your code with this test:\n```python\n{test}\n```\nIt did not finish within 10 seconds.\n{FIX}"
    return f"I ran your code with this test:\n```python\n{test}\n```\nIt failed:\n```\n{err}\n```\n{FIX}"


# ---- generation -------------------------------------------------------------------------------------------------------
def generate(model, tok, convs, max_new=MAX_NEW, temperature=0.0, seed=0, rows_at_32k=10, max_rows=128, label=""):
    """chat_eval.generate() with optional sampling: parsed replies ({"think", "content", "tool_calls", "closed"} plus
    "tokens") in the order of convs."""
    import torch
    import decode
    if temperature <= 0:
        outs = CE.generate(model, tok, convs, max_new, rows_at_32k, max_rows, label)
        return [{**chat.parse_assistant(o["text"]), "tokens": o["tokens"]} for o in outs]
    im_end, eos = chat.SPECIAL_IDS[chat.IM_END], chat.SPECIAL_IDS[chat.EOS]
    ids = [[eos] + tok.encode(chat.render(c, generation_prompt=True)).ids for c in convs]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    res, i, t0 = [None] * len(ids), 0, time.time()
    with torch.inference_mode():
        while i < len(order):
            n = min(max_rows, len(order) - i)
            while n > 1 and n > decode.batch_rows(len(ids[order[i + n - 1]]) + max_new, rows_at_32k):
                n -= 1
            chunk = order[i:i + n]
            caches = [decode.prefill(model, ids[j][:-1]) for j in chunk]
            outs, fin = [[] for _ in chunk], [False] * len(chunk)
            with decode.session(decode.merge_caches(caches)):
                h, _ = model.hidden(torch.tensor([[ids[j][-1]] for j in chunk], device=dev))
                for _ in range(max_new):
                    p = (model.lm_head(h[:, -1]).float() / temperature).softmax(-1)
                    nxt = torch.multinomial(p, 1, generator=g)[:, 0].tolist()
                    for b, t in enumerate(nxt):
                        if fin[b]:
                            continue
                        if t in (im_end, eos):
                            fin[b] = True
                        else:
                            outs[b].append(t)
                    if all(fin):
                        break
                    h, _ = model.hidden(torch.tensor([[t] for t in nxt], device=dev))
            for j, o in zip(chunk, outs):
                res[j] = {**chat.parse_assistant(tok.decode(o, skip_special_tokens=False)), "tokens": len(o)}
            i += n
            print(f"  {label}: {i}/{len(order)} replies [{time.time() - t0:.0f}s]", flush=True)
    return res


def assistant_msg(p):
    m = {"role": "assistant", "think": p["think"], "content": p["content"]}
    calls = [c for c in p["tool_calls"] if "name" in c]
    if calls:
        m["tool_calls"] = calls
    return m


# ---- the three measurements ------------------------------------------------------------------------------------------
def run_repair(task, items, gen, rounds, feedback, retry_temperature=0.0):
    """rows: per item {"first": passed the official tests after the first reply, "final", "rounds", "stopped": why}."""
    convs = [{"messages": list(it["conv"]["messages"])} for it in items]
    state = [{"id": it["id"], "rounds": 0, "stopped": None, "first": None, "final": False, "visible_ok": None,
              "n_visible": len(visible_tests(task, it["data"]))} for it in items]
    active = list(range(len(items)))
    for r in range(rounds):
        if not active:
            break
        replies = gen([convs[i] for i in active], f"{task} repair round {r + 1}",
                      temperature=retry_temperature if r else 0.0, seed=r)
        nxt = []
        for i, p in zip(active, replies):
            it, st = items[i], state[i]
            convs[i]["messages"].append(assistant_msg(p))
            st["rounds"] = r + 1
            ok, why = official(task, it["data"], p["content"])
            if r == 0:
                st["first"] = ok
            st["final"] = ok
            code = S.code_of(p["content"])
            if code is None:
                fb = "Your reply had no Python code block. Reply with the complete code in one Python code block."
            elif feedback == "oracle":
                fb = None if ok else feedback_text("the hidden tests of the task", "timeout" if why == "timeout" else tidy(why))
            else:
                fail = first_failure(code, visible_tests(task, it["data"]))
                st["visible_ok"] = fail is None
                fb = None if fail is None else feedback_text(*fail)
            if fb is None:
                st["stopped"] = "tests it can see pass" if feedback != "oracle" else "official tests pass"
            elif r + 1 < rounds:
                convs[i]["messages"].append({"role": "user", "content": fb})
                nxt.append(i)
            else:
                st["stopped"] = "out of rounds"
        active = nxt
    return state, convs


def run_tool(task, items, gen, rounds, retry_temperature=0.0):
    convs = []
    for it in items:
        msgs = list(it["conv"]["messages"])
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + TOOL_NOTE}
        convs.append({"messages": msgs, "tools": RUN_TOOL})
    state = [{"id": it["id"], "turns": 0, "calls": 0, "bad_calls": 0, "final": False, "stopped": None} for it in items]
    active = list(range(len(items)))
    for r in range(rounds):
        if not active:
            break
        replies = gen([convs[i] for i in active], f"{task} tool turn {r + 1}",
                      temperature=retry_temperature if r else 0.0, seed=r)
        nxt = []
        for i, p in zip(active, replies):
            st = state[i]
            st["turns"] = r + 1
            calls = [c for c in p["tool_calls"] if c.get("name") == TOOL_NAME and isinstance(c.get("arguments"), dict)
                     and isinstance(c["arguments"].get("code"), str)]
            st["bad_calls"] += len(p["tool_calls"]) - len(calls)
            convs[i]["messages"].append(assistant_msg({**p, "tool_calls": calls}))
            if calls and r + 1 < rounds:
                for c in calls:
                    st["calls"] += 1
                    convs[i]["messages"].append({"role": "tool", "content": tool_output(c["arguments"]["code"])})
                nxt.append(i)
            else:
                st["stopped"] = "answered" if not calls else "out of turns"
                st["final"] = official(task, items[i]["data"], p["content"])[0]
        active = nxt
    return state, convs


def run_resample(task, items, gen, rounds, temperature):
    """rows: {"first", "final", "rounds", "stopped", "n_visible"}; the conversation kept is the question and the reply
    that was chosen (every draw is in "draws")."""
    state = [{"id": it["id"], "rounds": 0, "stopped": None, "first": None, "final": False,
              "n_visible": len(visible_tests(task, it["data"])), "draws": []} for it in items]
    chosen = [None] * len(items)
    active = list(range(len(items)))
    for r in range(rounds):
        if not active:
            break
        replies = gen([items[i]["conv"] for i in active], f"{task} draw {r + 1}", temperature=temperature if r else 0.0, seed=r)
        nxt = []
        for i, p in zip(active, replies):
            it, st = items[i], state[i]
            st["rounds"] = r + 1
            ok = official(task, it["data"], p["content"])[0]
            if r == 0:
                st["first"], chosen[i] = ok, (p, ok)
            code = S.code_of(p["content"])
            tests = visible_tests(task, it["data"])
            passes = code is not None and bool(tests) and first_failure(code, tests) is None
            st["draws"].append({"official": ok, "visible": passes})
            if passes:
                chosen[i], st["stopped"] = (p, ok), "a draw passes the tests it can see"
            elif not tests:
                st["stopped"] = "no tests to see"   # nothing tells the draws apart: the first reply stands
            elif r + 1 < rounds:
                nxt.append(i)
            else:
                st["stopped"] = "out of rounds"
        active = nxt
    convs = []
    for it, st, (p, ok) in zip(items, state, chosen):
        st["final"] = ok
        convs.append({"messages": list(it["conv"]["messages"]) + [assistant_msg(p)]})
    return state, convs


def run_samples(task, items, gen, k):
    state = [{"id": it["id"], "passes": 0, "k": k} for it in items]
    convs = []
    for s in range(k):
        replies = gen([it["conv"] for it in items], f"{task} sample {s + 1}/{k}", seed=s)
        for it, st, p in zip(items, state, replies):
            ok = official(task, it["data"], p["content"])[0]
            st["passes"] += ok
            convs.append({"id": it["id"], "sample": s, "passed": ok, "think": p["think"], "content": p["content"]})
    return state, convs


def summary(mode, state):
    n = len(state)
    if mode == "resample":
        return {"n": n, "pass_first_reply": round(sum(s["first"] for s in state) / n, 4),
                "pass_final": round(sum(s["final"] for s in state) / n, 4),
                "with_visible_tests": sum(s["n_visible"] > 0 for s in state),
                "drew_again": sum(s["rounds"] > 1 for s in state),
                "gained": sum(s["final"] and not s["first"] for s in state),
                "lost": sum(s["first"] and not s["final"] for s in state),
                "a_draw_passed_official": sum(any(d["official"] for d in s["draws"]) for s in state),
                "mean_rounds": round(sum(s["rounds"] for s in state) / n, 2),
                "out_of_rounds": sum(s["stopped"] == "out of rounds" for s in state)}
    if mode == "repair":
        went = [s for s in state if s["rounds"] > 1]
        return {"n": n, "pass_first_reply": round(sum(s["first"] for s in state) / n, 4),
                "pass_final": round(sum(s["final"] for s in state) / n, 4),
                "with_visible_tests": sum(s["n_visible"] > 0 for s in state),
                "got_feedback": len(went), "fixed_by_feedback": sum(s["final"] and not s["first"] for s in state),
                "broken_by_feedback": sum(s["first"] and not s["final"] for s in state),
                "visible_pass_but_official_fail": sum(bool(s["visible_ok"]) and not s["final"] for s in state),
                "mean_rounds": round(sum(s["rounds"] for s in state) / n, 2),
                "out_of_rounds": sum(s["stopped"] == "out of rounds" for s in state)}
    if mode == "tool":
        return {"n": n, "pass_final": round(sum(s["final"] for s in state) / n, 4),
                "made_a_call": sum(s["calls"] > 0 for s in state), "calls": sum(s["calls"] for s in state),
                "bad_calls": sum(s["bad_calls"] for s in state), "mean_turns": round(sum(s["turns"] for s in state) / n, 2),
                "out_of_turns": sum(s["stopped"] == "out of turns" for s in state),
                "pass_among_callers": sum(s["final"] for s in state if s["calls"] > 0)}
    k = state[0]["k"]
    return {"n": n, "k": k, "pass_at_1": round(sum(s["passes"] for s in state) / (n * k), 4),
            "pass_at_k": round(sum(s["passes"] > 0 for s in state) / n, 4),
            "pass_all_k": round(sum(s["passes"] == k for s in state) / n, 4),
            "problems_by_passes": {str(c): sum(s["passes"] == c for s in state) for c in range(k + 1)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--moe-impl", default="bmm")
    ap.add_argument("--mode", choices=("repair", "tool", "samples", "resample"), required=True)
    ap.add_argument("--tasks", default="humaneval,mbpp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5, help="repair: replies in all; tool: assistant turns in all")
    ap.add_argument("--feedback", choices=("visible", "oracle"), default="visible")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--retry-temperature", type=float, default=0.0, help="repair, tool: sample every reply after the first")
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--rows-at-32k", type=int, default=10, help="fewer rows per batch for long conversations (GPU memory)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import mac_lm
    model, tok, meta = mac_lm.load(args.export, args.device)
    CE.set_moe_impl(model, args.moe_impl)
    chat.check_tokenizer(tok)
    res = {"step": meta["step"], "args": vars(args), "tasks": {}}
    samples = open(args.out.replace(".json", "_samples.jsonl"), "w", encoding="utf-8")
    for task in args.tasks.split(","):
        items = (CE.humaneval_items if task == "humaneval" else CE.mbpp_items)(args.limit)
        t0 = time.time()
        if args.mode == "samples":
            gen = lambda convs, label, seed=0: generate(model, tok, convs, temperature=args.temperature, seed=seed,  # noqa: E731
                                                        rows_at_32k=args.rows_at_32k, max_rows=args.max_rows, label=label)
            state, convs = run_samples(task, items, gen, args.k)
            for c in convs:
                samples.write(json.dumps({"task": task, **c}, ensure_ascii=False) + "\n")
        else:
            gen = lambda convs, label, temperature=0.0, seed=0: generate(  # noqa: E731
                model, tok, convs, temperature=temperature, seed=seed, rows_at_32k=args.rows_at_32k, max_rows=args.max_rows, label=label)
            if args.mode == "repair":
                state, convs = run_repair(task, items, gen, args.rounds, args.feedback, args.retry_temperature)
            elif args.mode == "tool":
                state, convs = run_tool(task, items, gen, args.rounds, args.retry_temperature)
            else:
                state, convs = run_resample(task, items, gen, args.rounds, args.temperature)
            for st, c in zip(state, convs):
                samples.write(json.dumps({"task": task, **st, "messages": c["messages"]}, ensure_ascii=False) + "\n")
        samples.flush()
        res["tasks"][task] = {**summary(args.mode, state), "seconds": round(time.time() - t0)}
        print(json.dumps({task: res["tasks"][task]}), flush=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("AGENT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
