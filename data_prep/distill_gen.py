#!/usr/bin/env python
"""Teacher data from the DeepSeek API for post-training: code, a math sample for
the comparison run, and questions on real long documents. Resumable: results are appended to
data/distill/<job>.jsonl one line per problem, and a problem that already has a line is skipped.

    export DEEPSEEK_API_KEY=...        # set by the user in their own shell; only read here
    python data_prep/distill_gen.py code --n 100000 --max-usd 35       # 1st pass, teacher not thinking
    python data_prep/distill_gen.py code --verify --max-usd 0                        # WSL: run the code
    python data_prep/distill_gen.py code --retry-failed --max-usd 25  # 2nd pass, thinking, failures only
    python data_prep/distill_gen.py code --retry-failed --verify --max-usd 0         # WSL: run the new code
    python data_prep/distill_gen.py math --n 20000 --max-usd 15
    python data_prep/distill_gen.py longdoc --max-usd 35       # documents from data/distill/longdocs/
    python data_prep/distill_gen.py backfill_tools --effort none --max-usd 12   # thoughts for the
    python data_prep/distill_gen.py backfill_chat --effort none --max-usd 8     # chosen public data
    python data_prep/distill_gen.py condense --max-usd 6       # short rewrite of OpenThoughts traces
    add --dry-run to print the plan and one request without calling the API

The last three jobs write reasoning for existing data (distill_think.py; conversations chosen by sft_select.py): the
replies stay as they are, the teacher only writes the thought behind each reply, or a short version of a long trace.

Every reply is checked before it counts: code must pass the problem's tests (code_check.py; for a problem without
tests the teacher writes asserts, which must pass on its own solution), a math answer must equal the reference, a
reading question's evidence quote must occur in the document. Failed replies are kept in the file with ok = false
(they were paid for and show what the teacher gets wrong) and are left out of the training data later.

Spending: --max-usd is a hard stop for this run, counted from the API's usage numbers at the price of the hour. By
default nothing is sent during peak hours (01:00-04:00 and 06:00-10:00 UTC on weekdays, double price): the run waits.
The long-document job stops if the share of cached input falls below --min-cache after the first documents, because
without the cache it costs about 50 times more.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import code_check  # noqa: E402
import distill_codeprob as DP  # noqa: E402
import distill_think as DT  # noqa: E402
from distill_pilot import Budget, call, is_peak, norm  # noqa: E402

THINK_JOBS = ("backfill_tools", "backfill_chat", "condense")
# the teacher thinks at this effort unless --effort says otherwise; none = thinking off (the job needs no reasoning)
# code: chosen after the 300-problem comparison: first pass without thinking (77% passed, $0.27
# per 1000), then the failures again at effort low (--retry-failed): 87% in all at $0.58 per 1000, against 82% at
# $0.92 with thinking for every problem
DEFAULT_EFFORT = {"code": "none", "codeprob": "none", "backfill_tools": "none", "backfill_chat": "none"}
OUT = os.path.join(ROOT, "data", "distill")
CONV = os.path.join(ROOT, "data", "sft", "conv")
STYLE = DT.SOLVE_SYS
Q_GEN = ("\n\n---\nThe text above is one document. Write 20 questions about it for a reading test. Spread them over "
         "the whole document, from the beginning to the end. Each must be answerable only by someone who has read "
         "the relevant passage, and must not be answerable from general knowledge. At least 5 must need two "
         "passages that are far apart. Reply with a JSON list of 20 objects with the keys \"question\" and "
         "\"evidence\", where evidence is a quote of at most 25 words copied exactly from the document.")
Q_ANS = DT.Q_ANS


def wait_off_peak(allow_peak):
    while not allow_peak and is_peak():
        print(time.strftime("[%H:%M] peak hours (double price): waiting"), flush=True)
        time.sleep(600)


class Sink:
    """Append-only JSONL with the set of ids already written."""

    def __init__(self, path):
        self.path, self.lock, self.done = path, threading.Lock(), set()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.done = {json.loads(line)["id"] for line in f if line.strip()}
        self.f = None  # opened by the first write: a dry run must not leave an empty file behind

    def write(self, row):
        with self.lock:
            if self.f is None:
                self.f = open(self.path, "a", encoding="utf-8")
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.f.flush()
            self.done.add(row["id"])


# ---- code ----------------------------------------------------------------------------------------------------------
def read_pool():
    """data/sft/code_pool.jsonl (datasets) and code_pool_gen.jsonl (written by the teacher), when they exist."""
    pool = []
    for f in ("code_pool.jsonl", "code_pool_gen.jsonl"):
        path = os.path.join(ROOT, "data", "sft", f)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                pool += [json.loads(line) for line in fh]
    return pool


def code_items(n, seed=0):
    pool = read_pool()
    tested = [p for p in pool if p["check"]]
    rest = [p for p in pool if not p["check"]]
    random.Random(seed).shuffle(rest)
    # every problem with real tests first, then the problems without tests (self-oss-instruct's instructions and the
    # ones the teacher wrote from code snippets, mixed) in a fixed random order
    return (tested + rest)[:n or None]


def code_sample(n, seed=0):
    """A random sample for measuring the price: a fifth of it from the problems with tests, the rest from the others."""
    pool = read_pool()
    tested = [p for p in pool if p["check"]]
    rest = [p for p in pool if not p["check"]]
    rng, k = random.Random(seed), round(n * 0.2)
    return rng.sample(tested, k) + rng.sample(rest, n - k)


def code_failed_items(include_unsafe=False):
    """The problems whose first answer did not pass (code_checked.jsonl): asked again, with thinking, by --retry-failed.
    Answers that were not run because they use files, processes or the network ("unsafe pattern": 4.8% of the first
    pass) are left out unless include_unsafe: thinking does not change what the problem needs, so they mostly fail again."""
    path = os.path.join(OUT, "code_checked.jsonl")
    if not os.path.exists(path):
        sys.exit(f"{path} is missing: run the first pass and its --verify (in WSL) first")
    pool = {p["id"]: p for p in read_pool()}
    with open(path, encoding="utf-8") as f:
        failed = [r["id"] for r in map(json.loads, f)
                  if not r["ok"] and (include_unsafe or r.get("detail") != "not run (unsafe pattern)")]
    return [pool[i] for i in failed if i in pool]


def code_prompt(p):
    return "Problem: " + p["prompt"] + ("" if p["check"] else "\nNo tests are given.") + "\nReply:"


_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def split_code_reply(content):
    """(solution code, tests code or None, the reply without the tests block)."""
    blocks = list(_BLOCK.finditer(content))
    if not blocks:
        return None, None, content
    tests = next((b for b in blocks[1:] if b.group(1).lstrip().startswith("# tests")), None)
    visible = (content[:tests.start()] + content[tests.end():]).strip() if tests else content.strip()
    return blocks[0].group(1), (tests.group(1) if tests else None), visible


def code_check_reply(p, content):
    sol, tests, visible = split_code_reply(content)
    if sol is None:
        return False, "no code block", visible
    if p["check"]:
        ok, detail = code_check.check(sol, p["check"])
    elif not tests or tests.count("assert") < 2:
        ok, detail = False, "no tests block"
    else:
        ok, detail = code_check.check(sol, {"type": "asserts", "code": tests})
    return ok, detail, visible


# ---- math ----------------------------------------------------------------------------------------------------------
_BOX = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")


def last_boxed(text):
    m = _BOX.findall(text or "")
    return m[-1] if m else None


def norm_answer(a):
    a = re.sub(r"\\(?:left|right|,|!|;|text\{[^}]*\}|mathrm\{[^}]*\})|\s|\$", "", str(a))
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("\\%", "")
    return a.rstrip(".")


def same_answer(got, gold):
    if got is None:
        return False
    g, w = norm_answer(got), norm_answer(gold)
    if g == w:
        return True
    try:
        return abs(float(g.replace(",", "")) - float(w.replace(",", ""))) < 1e-6
    except ValueError:
        return False


def math_items(n, seed=0):
    """All GSM8K train problems, then a fixed random sample of OpenMathInstruct-2 problems (read as a stream)."""
    items = []
    with open(os.path.join(CONV, "gsm8k.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            items.append({"id": c["id"], "prompt": c["messages"][0]["content"],
                          "gold": last_boxed(c["messages"][1]["content"])})
    rng, pick = random.Random(seed), []
    with open(os.path.join(CONV, "math.jsonl"), encoding="utf-8") as f:
        for k, line in enumerate(f):  # reservoir sample
            if len(pick) < max(0, n - len(items)):
                pick.append(line)
            elif (j := rng.randrange(k + 1)) < len(pick):
                pick[j] = line
    for line in pick:
        c = json.loads(line)
        items.append({"id": c["id"], "prompt": c["messages"][0]["content"],
                      "gold": last_boxed(c["messages"][1]["content"])})
    return items[:n]


# ---- long documents ------------------------------------------------------------------------------------------------
def think_items(job, n=0):
    """backfill_tools / backfill_chat: the conversations sft_select.py chose; condense: OpenThoughts without its math
    part. Each item is {"id", "conv"}."""
    path = (os.path.join(CONV, "openthoughts.jsonl") if job == "condense" else
            os.path.join(OUT, "backfill_in", job.split("_")[1] + ".jsonl"))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if job == "condense" and c["meta"].get("kind") == "math":
                continue
            out.append({"id": c["id"], "conv": c})
            if n and len(out) >= n:
                break
    return out


def do_think(key, job, it, a, budget):
    back = job != "condense"
    msgs = [{"role": "system", "content": DT.backfill_sys(it["conv"]) if back else DT.CONDENSE_SYS},
            {"role": "user", "content": (DT.backfill_prompt if back else DT.condense_prompt)(it["conv"])}]
    r = call(key, msgs, a.effort, budget, max_tokens=4096)
    if "error" in r:
        if not r["error"].startswith("budget"):
            print("API error: " + r["error"][:200], flush=True)
        return None  # not written: tried again on the next run
    ok, detail, new = (DT.check_backfill if back else DT.check_condense)(it["conv"], r["content"])
    return {"id": it["id"], "ok": ok, "detail": detail, "thoughts" if back else "think": new,
            "raw": None if ok else r["content"], "usage": r["usage"], "effort": a.effort}


def codeseed_items(n=0):
    """data/distill/codeseeds.jsonl (data_prep/sft_code_seeds.py): snippets to write problems from."""
    with open(os.path.join(OUT, "codeseeds.jsonl"), encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    return items[:n] if n else items


def do_codeprob(key, it, a, budget):
    msgs = [{"role": "system", "content": DP.PROBLEM_SYS}, {"role": "user", "content": DP.problem_prompt(it)}]
    r = call(key, msgs, a.effort, budget, max_tokens=1024)
    if "error" in r:
        if not r["error"].startswith("budget"):
            print("API error: " + r["error"][:200], flush=True)
        return None  # not written: tried again on the next run
    ok, detail, problem = DP.check_problem(r["content"])
    return {"id": it["id"], "ok": ok, "detail": detail, "problem": problem, "difficulty": it["difficulty"],
            "raw": None if ok else r["content"], "usage": r["usage"], "effort": a.effort}


def longdoc_items(split=""):
    """data/distill/longdocs/*.txt (written by data_prep/sft_longdocs.py), id = file name without .txt; split "test":
    the held-out documents in longdocs_test/."""
    d = os.path.join(OUT, "longdocs" + ("_" + split if split else ""))
    return [{"id": f[:-4], "path": os.path.join(d, f)} for f in sorted(os.listdir(d)) if f.endswith(".txt")]


def do_longdoc(key, it, a, budget):
    with open(it["path"], encoding="utf-8") as f:
        doc = f.read()
    sysmsg = {"role": "system", "content": STYLE}
    first = [sysmsg, {"role": "user", "content": doc + Q_GEN}]
    g = call(key, first, a.effort, budget, max_tokens=16384)
    row = {"id": it["id"], "doc_chars": len(doc), "questions_usage": g.get("usage"), "qa": []}
    if "error" in g:
        return None  # not written: the document is tried again on the next run
    try:
        qs = [q for q in json.loads(re.search(r"\[.*\]", g["content"], re.S).group(0)) if q.get("question")]
    except Exception:  # noqa: BLE001
        return {**row, "ok": False, "detail": "questions are not a JSON list"}
    text = norm(doc)
    prefix = first + [{"role": "assistant", "content": g["content"]}]
    time.sleep(a.cache_wait)  # the cache takes seconds to build

    def ans(i):
        q = qs[i]
        r = call(key, prefix + [{"role": "user", "content": Q_ANS.format(i=i + 1, q=q["question"])}], a.effort, budget)
        return {"question": q["question"], "evidence": q.get("evidence"),
                "evidence_in_doc": norm(str(q.get("evidence"))) in text, "reasoning": r.get("reasoning"),
                "content": r.get("content"), "usage": r.get("usage"), "error": r.get("error")}
    with cf.ThreadPoolExecutor(4) as ex:
        row["qa"] = list(ex.map(ans, range(len(qs))))
    if any((x["error"] or "").startswith(("budget", "HTTP 402")) for x in row["qa"]):
        return None  # cut off by the budget (or by an empty account) in the middle: redo the document later
    good = [x for x in row["qa"] if x["evidence_in_doc"] and x["content"] and not x["error"]]
    return {**row, "ok": len(good) >= 10, "detail": f"{len(good)}/{len(qs)} usable"}


def _verify_one(args):
    p, row = args
    ok, detail, visible = code_check_reply(p, row.get("raw") or row["content"])
    return {**row, "ok": ok, "detail": detail, "content": visible, "raw": None}


def verify_code(name="code"):
    """Check every stored code reply in <name>.jsonl (16 processes) -> <name>_checked.jsonl; prints the pass rate by
    source."""
    import multiprocessing as mp
    pool = {p["id"]: p for p in read_pool()}
    with open(os.path.join(OUT, name + ".jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    stats = {}
    with mp.get_context("fork").Pool(16) as mpool, open(os.path.join(OUT, name + "_checked.jsonl.tmp"), "w", encoding="utf-8") as f:
        for row in mpool.imap(_verify_one, [(pool[r["id"]], r) for r in rows], chunksize=16):
            src = pool[row["id"]]["source"]
            st = stats.setdefault(src, [0, 0])
            st[0] += bool(row["ok"])
            st[1] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(os.path.join(OUT, name + "_checked.jsonl.tmp"), os.path.join(OUT, name + "_checked.jsonl"))
    print({k: f"{a}/{b} passed" for k, (a, b) in stats.items()})
    print("DISTILL_VERIFY_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("code", "math", "longdoc", "codeprob") + THINK_JOBS)
    ap.add_argument("--retry-failed", action="store_true",
                    help="code: ask again, at effort low, every problem whose first answer failed (code_checked.jsonl); "
                         "results in code_retry.jsonl, checked with --retry-failed --verify")
    ap.add_argument("--include-unsafe", action="store_true",
                    help="code --retry-failed: also ask again the problems whose first answer was not run (unsafe pattern)")
    ap.add_argument("--sample", type=int, default=0,
                    help="code: a random sample of this many problems (a fifth with tests) to measure the price")
    ap.add_argument("--out", default=None, help="folder for the results (default data/distill)")
    ap.add_argument("--docs", default="", choices=("", "test"),
                    help="longdoc: test = the 50 held-out documents (data/distill/longdocs_test, longdoc_test.jsonl)")
    ap.add_argument("--n", type=int, default=0, help="problems (code, math) or documents (longdoc); 0 = all")
    ap.add_argument("--effort", default=None, choices=("none", "low", "high", "max"),
                    help="teacher thinking; none = off (default for codeprob and the backfill jobs, low otherwise)")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--allow-peak", action="store_true")
    ap.add_argument("--min-cache", type=float, default=0.8, help="longdoc: stop below this share of cached input")
    ap.add_argument("--cache-wait", type=float, default=15)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-here", action="store_true",
                    help="code: run the teacher's code on this machine right away (default: store the replies, check "
                         "them later with --verify in WSL, where the checker drops to the user nobody)")
    ap.add_argument("--verify", action="store_true",
                    help="code: no API calls; run the stored replies' code and write data/distill/code_checked.jsonl")
    a = ap.parse_args()
    global OUT
    OUT = a.out or OUT
    a.effort = a.effort or ("low" if a.retry_failed else DEFAULT_EFFORT.get(a.job, "low"))
    if a.verify:
        return verify_code("code_retry" if a.retry_failed else "code")
    if a.job in THINK_JOBS:
        items = think_items(a.job, a.n)
    elif a.job == "longdoc":
        items = longdoc_items(a.docs)[:a.n or None]
    elif a.job == "codeprob":
        items = codeseed_items(a.n)
    elif a.job == "code" and a.retry_failed:
        items = code_failed_items(a.include_unsafe)
    elif a.job == "code" and a.sample:
        items = code_sample(a.sample)
    else:
        items = {"code": code_items, "math": math_items}[a.job](a.n)
    name = a.job + ("_test" if a.docs == "test" else "") + ("_retry" if a.retry_failed else "")
    sink = Sink(os.path.join(OUT, name + ".jsonl"))
    todo = [it for it in items if it["id"] not in sink.done]
    print(f"{a.job}: {len(items)} items, {len(items) - len(todo)} already done, {len(todo)} to do; effort {a.effort}, "
          f"{a.workers} at a time, stop at ${a.max_usd:.2f}; {'PEAK' if is_peak() else 'off-peak'} now", flush=True)
    if a.dry_run:
        if todo:
            it = todo[0]
            if a.job == "codeprob":
                print("--- first request ---\n" + DP.problem_prompt(it))
                return
            if a.job in THINK_JOBS:
                print("--- first request ---\n" + (DT.backfill_prompt if a.job != "condense" else
                                                   DT.condense_prompt)(it["conv"])[:1500])
                return
            print("--- first request ---\n" + (code_prompt(it) if a.job == "code" else
                                               "Problem: " + it["prompt"] + "\nReply:" if a.job == "math" else it["path"])[:1500])
        return
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY is not set in this shell; export it yourself and run again")
    budget, sysmsg = Budget(a.max_usd), {"role": "system", "content": STYLE}
    n_ok, n_done, t0, lock = 0, 0, time.time(), threading.Lock()

    def one(it):
        wait_off_peak(a.allow_peak)
        if a.job == "longdoc":
            row = do_longdoc(key, it, a, budget)
        elif a.job == "codeprob":
            row = do_codeprob(key, it, a, budget)
        elif a.job in THINK_JOBS:
            row = do_think(key, a.job, it, a, budget)
        else:
            prompt = code_prompt(it) if a.job == "code" else "Problem: " + it["prompt"] + "\nReply:"
            r = call(key, [sysmsg, {"role": "user", "content": prompt}], a.effort, budget, max_tokens=a.max_tokens)
            if "error" in r:  # budget reached or the API failed after retries: not written, tried again next run
                if not r["error"].startswith("budget"):
                    print("API error: " + r["error"][:200], flush=True)
                return None
            if a.job == "code" and not a.check_here:
                ok, detail, visible = None, "not checked yet (run --verify in WSL)", r["content"].strip()
            elif a.job == "code":
                ok, detail, visible = code_check_reply(it, r["content"])
            else:
                visible = r["content"].strip()
                ok, detail = same_answer(last_boxed(visible), it["gold"]), f"gold {it['gold']}"
            row = {"id": it["id"], "ok": ok, "detail": detail, "reasoning": r["reasoning"], "content": visible,
                   "raw": r["content"] if visible != r["content"].strip() else None, "usage": r["usage"],
                   "effort": a.effort, "finish": r.get("finish")}
        return row

    with cf.ThreadPoolExecutor(a.workers) as ex:
        # written as soon as each is finished (ex.map hands them over in order, so one slow request held back the rows
        # behind it, and a stuck connection kept them in memory)
        for fut in cf.as_completed([ex.submit(one, it) for it in todo]):
            row = fut.result()
            if row is None:  # budget reached before this one was sent
                continue
            sink.write(row)
            with lock:
                n_done += 1
                n_ok += bool(row["ok"])
            if n_done % 200 == 0 or (a.job == "longdoc" and n_done % 5 == 0):
                tok = budget.tok
                share = tok["hit"] / max(1, tok["hit"] + tok["miss"])
                print(time.strftime("[%H:%M] ") + f"{n_done}/{len(todo)} done, {n_ok} passed, ${budget.spent:.2f}, "
                      f"cached input {share:.0%}, {n_done / max(1, time.time() - t0) * 3600:.0f}/h", flush=True)
                if a.job == "longdoc" and n_done >= 5 and share < a.min_cache:
                    print(f"cache hit share {share:.0%} is below {a.min_cache:.0%}: stopping", flush=True)
                    budget.max = 0
    tok = budget.tok
    print(json.dumps({"job": a.job, "done_now": n_done, "passed_now": n_ok, "usd": round(budget.spent, 2),
                      "tokens": tok, "total_in_file": len(sink.done), "seconds": round(time.time() - t0)}))
    print("DISTILL_GEN_DONE" if n_done == len(todo) else "DISTILL_GEN_STOPPED (budget or cache); run again to resume")


if __name__ == "__main__":
    main()
