#!/usr/bin/env python
"""Pilot for teacher distillation through the DeepSeek API: about 160 calls, well
under $1. Answers what the plan's cost and quality estimates rest on: is the thinking returned, how long is it at
each effort, how many answers survive checking, and how often the context cache hits.

    export DEEPSEEK_API_KEY=...          # set by the user in their own shell; this script only reads the variable
    python data_prep/distill_pilot.py              # run
    python data_prep/distill_pilot.py --dry-run    # build every request, call nothing

Three parts (everything sent is public data: MBPP, GSM8K, two public-domain PG19 books):
  code   50 MBPP train problems, effort low; kept if the answer's code passes the problem's asserts
  math   50 GSM8K problems, effort low, the first 20 again at effort high; kept if the number after #### is right.
         These are GSM8K *test* problems (the train split is not on disk): pilot_only, never used for training
  long   2 PG19 training books cut to ~100K tokens: one call writes 20 questions with a verbatim evidence quote
         each, then 20 calls answer one question each. Every answer call repeats the first request and its output as
         the prefix, so only the first call per book pays the cache-miss price
API facts checked on 2026-11-22 (api-docs.deepseek.com): model deepseek-flash (DeepSeek-V4.1-Flash-1017), thinking on
by default and returned as message.reasoning_content, reasoning_effort low/high/max, usage.prompt_cache_hit_tokens /
prompt_cache_miss_tokens; per 1M tokens off-peak $0.15 input (miss), $0.003 (hit), $0.60 output, twice that in peak
hours (01:00-04:00 and 06:00-10:00 UTC, Monday to Friday). The run stops sending once the spend, counted at the price
of the hour, reaches --max-usd. Writes data/distill/pilot/{code,math,long,think}.jsonl and summary.json.

It also tries the two jobs that write reasoning for existing data (distill_think.py), 20 conversations each: the
thought behind every reply of a function-calling conversation (teacher thinking off, and at effort low) and of a chat
conversation (thinking off), and a short rewrite of an OpenThoughts reasoning trace (effort low)."""
import argparse
import collections
import concurrent.futures as cf
import datetime
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_think as DT  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
PRICE = {"miss": 0.15, "hit": 0.003, "out": 0.60}  # $ per 1M tokens, off-peak; peak is twice this

STYLE = DT.SOLVE_SYS
Q_GEN = ("\n\n---\nThe text above is one document. Write 20 questions about it for a reading test. Spread them over "
         "the whole document, from the beginning to the end. Each must be answerable only by someone who has read "
         "the relevant passage, and must not be answerable from general knowledge. At least 5 must need two "
         "passages that are far apart. Reply with a JSON list of 20 objects with the keys \"question\" and "
         "\"evidence\", where evidence is a quote of at most 25 words copied exactly from the document.")


def is_peak(now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


class Budget:
    def __init__(self, max_usd):
        self.max, self.spent, self.lock = max_usd, 0.0, threading.Lock()
        self.tok = {"hit": 0, "miss": 0, "out": 0}

    def add(self, usage):
        hit = usage.get("prompt_cache_hit_tokens", 0)
        miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
        out = usage.get("completion_tokens", 0)
        k = 2 if is_peak() else 1
        with self.lock:
            self.spent += k * (hit * PRICE["hit"] + miss * PRICE["miss"] + out * PRICE["out"]) / 1e6
            for name, n in (("hit", hit), ("miss", miss), ("out", out)):
                self.tok[name] += n

    def left(self):
        with self.lock:
            return self.spent < self.max


def call(key, messages, effort, budget, max_tokens=8192, tries=5):
    """One chat completion; returns {"content", "reasoning", "usage"} or {"error"}."""
    if not budget.left():
        return {"error": "budget reached, not sent"}
    # effort "none": the teacher answers without thinking (the short rewriting jobs do not need it)
    think = {"thinking": {"type": "disabled"}} if effort == "none" else {"reasoning_effort": effort,
                                                                         "thinking": {"type": "enabled"}}
    body = json.dumps({"model": MODEL, "messages": messages, **think, "max_tokens": max_tokens}).encode()
    # a connection that goes quiet (the network dropped it, or the server queues the request) is given up after this
    # many seconds and the request is sent again; it used to be 900 s for every call, which stalled whole jobs
    wait = 240 if max_tokens <= 4096 else 360 if max_tokens <= 8192 else 900
    for t in range(tries):
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": "distill-pilot/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=wait) as r:
                d = json.load(r)
            budget.add(d.get("usage", {}))
            m = d["choices"][0]["message"]
            return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content"),
                    "usage": d.get("usage", {}), "finish": d["choices"][0].get("finish_reason")}
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (401, 402, 403):  # bad key or no balance: retrying cannot help
                return {"error": f"HTTP {e.code}: {msg}"}
            err = f"HTTP {e.code}: {msg}"
        except Exception as e:  # noqa: BLE001  network errors and timeouts: retry
            err = repr(e)[:300]
        time.sleep(min(60, 3 * 2 ** t))
    return {"error": err}


def cost(usage, peak=False):
    """Dollars for one call's usage at the listed price."""
    hit = usage.get("prompt_cache_hit_tokens", 0)
    miss = usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit)
    return (hit * PRICE["hit"] + miss * PRICE["miss"] + usage.get("completion_tokens", 0) * PRICE["out"]) / 1e6 * (
        2 if peak else 1)


def jsonl(*parts):
    with open(os.path.join(ROOT, "data", "raw", "bench", *parts)) as f:
        return [json.loads(line) for line in f]


def code_items(n):
    rows = jsonl("mbpp", "full", "train-00000-of-00001.jsonl")[:n]
    return [{"id": f"mbpp_train/{r['task_id']}", "kind": "code", "tests": r["test_list"],
             "setup": r.get("test_setup_code") or "",
             "prompt": f"Problem: {r['text']}\nYour code should pass this test: {r['test_list'][0]}\nReply:"}
            for r in rows]


def math_items(n):
    rows = jsonl("gsm8k", "test.jsonl")[700:700 + n]
    return [{"id": f"gsm8k_test/{700 + i}", "kind": "math", "pilot_only": True,
             "gold": r["answer"].split("####")[-1].strip().replace(",", ""),
             "prompt": f"Problem: {r['question']}\nReply:"} for i, r in enumerate(rows)]


def long_docs(n, tokens=100_000):
    """The first n PG19 training books of at least `tokens` tokens, cut to that length at a paragraph break."""
    import numpy as np
    from tokenizers import Tokenizer
    from moe_llm.data import TokenStream
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    ts = TokenStream(os.path.join(ROOT, "data", "bins_main", "pg19"))
    arr = ts.read(0, min(ts.total, 30_000_000))
    eos = np.flatnonzero(arr == 0)
    out = []
    for a, b in zip(eos[:-1], eos[1:]):
        if b - a - 1 >= tokens:
            text = tok.decode(arr[a + 1:a + 1 + tokens].tolist())
            text = text[:text.rfind("\n\n")] if "\n\n" in text[-4000:] else text
            out.append({"id": f"pg19_train@{int(a)}", "text": text})
            if len(out) == n + 1:
                break
    return out[1:]  # the first long book is Shakespeare's complete works: answerable from general knowledge


def think_items(n_tools, n_chat, n_condense):
    """Samples for the two jobs that write reasoning for existing data (distill_think.py): the first conversations of
    the files sft_select.py wrote, and the first non-math OpenThoughts conversations."""
    out = []
    for kind, path, n in (("backfill tools", "data/distill/backfill_in/tools.jsonl", n_tools),
                          ("backfill chat", "data/distill/backfill_in/chat.jsonl", n_chat),
                          ("condense", "data/sft/conv/openthoughts.jsonl", n_condense)):
        path = os.path.join(ROOT, path)
        if not n or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                if kind == "condense" and c["meta"].get("kind") == "math":
                    continue
                out.append({"id": c["id"], "kind": kind, "conv": c})
                if sum(x["kind"] == kind for x in out) >= n:
                    break
    return out


def check_code(item, content):
    import gen_eval
    m = re.search(r"```(?:python)?\n(.*?)```", content, re.S)
    if not m:
        return False, "no code block"
    code = m.group(1)
    if gen_eval.UNSAFE.search(code):
        return False, "not run (unsafe pattern)"
    return gen_eval.run_program(code + "\n" + item["setup"] + "\n" + "\n".join(item["tests"]) + "\n")


def check_math(item, content):
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", content or "")
    if not boxes:
        return False, "no boxed answer"
    try:
        return abs(float(boxes[-1].replace(",", "").strip().lstrip("$")) - float(item["gold"])) < 1e-6, boxes[-1]
    except ValueError:
        return False, boxes[-1]


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def stats(xs):
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(sum(xs) / len(xs)), "median": xs[len(xs) // 2], "max": xs[-1]} if xs else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-code", type=int, default=50)
    ap.add_argument("--n-math", type=int, default=50)
    ap.add_argument("--n-high", type=int, default=20, help="math problems repeated at effort high")
    ap.add_argument("--n-docs", type=int, default=2)
    ap.add_argument("--n-think", type=int, default=20, help="conversations per reasoning-rewrite group")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-usd", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "distill", "pilot"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    short = [(it, "low") for it in code_items(args.n_code) + math_items(args.n_math)]
    short += [(it, "high") for it in math_items(args.n_high)]
    docs = long_docs(args.n_docs)
    # reasoning for existing data: tool calls without and with teacher thinking, chat without, condensing with
    tk = think_items(args.n_think, args.n_think, args.n_think)
    think = [(it, e) for it in tk for e in (("none", "low") if it["kind"] == "backfill tools" else
                                            ("low",) if it["kind"] == "condense" else ("none",))]
    est = lambda s: len(s) // 4  # noqa: E731  rough token count
    doc_tok = sum(est(d["text"]) for d in docs)
    print(f"{len(think)} reasoning-rewrite calls ({collections.Counter(it['kind'] for it, _ in think)})")
    print(f"{len(short)} short calls (system prefix ~{est(STYLE)} tokens), {len(docs)} documents "
          f"(~{doc_tok} tokens) with 1 + 20 calls each; {'PEAK' if is_peak() else 'off-peak'} prices now")
    worst = ((len(short) * (est(STYLE) + 200) + doc_tok * 3) * PRICE["miss"] + doc_tok * 19 * PRICE["hit"]
             + (len(short) * 1500 + len(docs) * 21 * 1500) * PRICE["out"]
             + len(think) * (2500 * PRICE["miss"] + 800 * PRICE["out"])) / 1e6 * (2 if is_peak() else 1)
    print(f"estimate with a poor cache hit rate: ${worst:.2f}; hard stop at ${args.max_usd:.2f}")
    if args.dry_run:
        print("first short request:\n" + short[0][0]["prompt"][:400])
        print("first document starts:\n" + docs[0]["text"][:300] if docs else "no documents")
        return
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set in this shell; export it yourself and run again")
    os.makedirs(args.out, exist_ok=True)
    budget = Budget(args.max_usd)
    sysmsg = {"role": "system", "content": STYLE}

    def do_short(pair):
        it, effort = pair
        r = call(key, [sysmsg, {"role": "user", "content": it["prompt"]}], effort, budget)
        if "error" not in r:
            r.update(DT.parts(it["kind"], r["content"]))
            r["ok"], r["detail"] = (check_code if it["kind"] == "code" else check_math)(it, r["content"])
        return {**{k: v for k, v in it.items() if k not in ("tests", "setup")}, "effort": effort, **r}

    def do_think(pair):
        it, effort = pair
        back = it["kind"] != "condense"
        msgs = [{"role": "system", "content": DT.backfill_sys(it["conv"]) if back else DT.CONDENSE_SYS},
                {"role": "user", "content": (DT.backfill_prompt if back else DT.condense_prompt)(it["conv"])}]
        r = call(key, msgs, effort, budget, max_tokens=4096)
        if "error" not in r:
            r["ok"], r["detail"], r["new"] = (DT.check_backfill if back else DT.check_condense)(it["conv"], r["content"])
        return {"id": it["id"], "kind": f"{it['kind']} {effort}", "effort": effort, "conv": it["conv"], **r}

    def do_doc(d):
        first = [sysmsg, {"role": "user", "content": d["text"] + Q_GEN}]
        g = call(key, first, "low", budget, max_tokens=16384)
        rows = [{"id": d["id"], "step": "questions", **g}]
        if "error" in g:
            return rows
        try:
            qs = json.loads(re.search(r"\[.*\]", g["content"], re.S).group(0))
        except Exception:  # noqa: BLE001
            rows[0]["error"] = "questions are not a JSON list"
            return rows
        text = norm(d["text"])
        prefix = first + [{"role": "assistant", "content": g["content"]}]
        time.sleep(15)  # the cache takes seconds to build

        def ans(i):
            q = qs[i]
            r = call(key, prefix + [{"role": "user", "content": DT.Q_ANS.format(i=i + 1, q=q.get("question"))}],
                     "low", budget)
            return {"id": f"{d['id']}/q{i + 1}", "step": "answer", "question": q.get("question"),
                    "evidence": q.get("evidence"), "evidence_in_doc": norm(str(q.get("evidence"))) in text, **r,
                    **(DT.parts("reading", r.get("content")) if "error" not in r else {})}
        with cf.ThreadPoolExecutor(4) as ex:
            rows += list(ex.map(ans, range(len(qs))))
        return rows

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        fut_docs = [ex.submit(do_doc, d) for d in docs]
        res_short = list(ex.map(do_short, short))
        res_think = list(ex.map(do_think, think))
        res_long = [row for f in fut_docs for row in f.result()]
    for name, rows in (("code", [r for r in res_short if r["kind"] == "code"]),
                       ("math", [r for r in res_short if r["kind"] == "math"]), ("long", res_long),
                       ("think", res_think)):
        with open(os.path.join(args.out, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"seconds": round(time.time() - t0), "usd": round(budget.spent, 4), "tokens": budget.tok,
            "errors": [r["error"] for r in res_short + res_long + res_think if "error" in r][:5], "groups": {}}
    groups = {"code low": [r for r in res_short if r["kind"] == "code"],
              "math low": [r for r in res_short if r["kind"] == "math" and r["effort"] == "low"],
              "math high": [r for r in res_short if r["kind"] == "math" and r["effort"] == "high"],
              "long answers": [r for r in res_long if r["step"] == "answer"]}
    for r in res_think:
        groups.setdefault(r["kind"], []).append(r)
    for name, rows in groups.items():
        good = [r for r in rows if "error" not in r]
        if not good:
            summ["groups"][name] = {"calls": len(rows), "answered": 0}
            continue
        u = [r["usage"] for r in good]
        hit, inp = sum(x.get("prompt_cache_hit_tokens", 0) for x in u), sum(x.get("prompt_tokens", 0) for x in u)
        summ["groups"][name] = {
            "calls": len(rows), "answered": len(good),
            "thinking_returned": sum(bool(r["reasoning"]) for r in good),
            "kept": sum(bool(r.get("ok", r.get("evidence_in_doc"))) for r in good),
            "output_tokens": stats([x.get("completion_tokens", 0) for x in u]),
            "thinking_chars": stats([len(r["reasoning"] or "") for r in good]),
            "answer_chars": stats([len(r["content"]) for r in good]),
            "input_tokens": stats([x.get("prompt_tokens", 0) for x in u]),
            "usd_per_1000": round(sum(cost(x) for x in u) / len(u) * 1000, 3),
            "cache_hit_share": round(hit / inp, 3) if inp else None}
        if name in ("code low", "math low", "math high", "long answers"):
            summ["groups"][name]["style_clean"] = sum(r.get("style") is None for r in good)
            summ["groups"][name]["working_chars"] = stats([len(r.get("working") or "") for r in good])
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1, ensure_ascii=False))
    print("DISTILL_PILOT_DONE")


if __name__ == "__main__":
    main()
