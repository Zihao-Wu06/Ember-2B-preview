#!/usr/bin/env python
"""Teacher results (data/distill/*.jsonl, written by distill_gen.py) -> conversation sets in the format of
data_prep/sft_convert.py (data/sft/conv/<set>.jsonl), which sft_tokenize.py and sft_long_packs.py read
.

    python data_prep/distill_convert.py                  # every set whose results exist
    python data_prep/distill_convert.py --only code_api longdoc_api

Sets, and what becomes of a result:

  code_api          code_checked.jsonl (first pass) and code_retry_checked.jsonl (second pass, thinking): per problem the
                    first answer that passed its tests (the checker ran in WSL). The student's prompt is the problem
                    text of the pool; thinking = the teacher's working (plain sentences before the code), answer = the
                    code block and what follows it. Answers whose working is shorthand, too long or empty are dropped.
  math_api          math.jsonl: answers whose boxed value equals the reference; prompt from the GSM8K / OpenMathInstruct-2
                    sets, thinking = working, answer = "The answer is $\\boxed{...}$." (for the comparison run)
  longdoc_api       longdoc.jsonl: one conversation per document, the document and the first question in the first user
                    turn, then one turn per question (shuffled, so that the order does not follow the document); thinking
                    = the "Where" sentence, answer = the "Answer" sentences. Only questions whose evidence quote is in the
                    document and whose reply has the form; a document needs 10 of them.
  longdoc_test      the held-out documents (longdoc_test.jsonl) -> data/sft/eval/longdoc_test.jsonl, never a training set
  tools_api         backfill_tools.jsonl: the chosen function-calling conversations with the teacher's thought on every
                    assistant turn (the replies are unchanged)
  chat_api          backfill_chat.jsonl: the same for the chat conversations
  openthoughts_short  condense.jsonl: OpenThoughts (without its math) with the reasoning rewritten short. Not used by
                    sft_tokenize.py until its sft_general entry says so: the conversations are about a third of the
                    tokens, so at the same token share they would be read about three times as often.

Every conversation must render (moe_llm/chat.py) and, except for the long documents and function calling, shares no
13-gram with HumanEval, MBPP, GSM8K test or MATH-500 (MBPP's own train problems are exempt, as in sft_convert.py).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import decontam  # noqa: E402
import distill_think as DT  # noqa: E402
import sft_convert as SC  # noqa: E402

DISTILL = os.path.join(ROOT, "data", "distill")
SFT = os.path.join(ROOT, "data", "sft")
CONV = os.path.join(SFT, "conv")
EVAL = os.path.join(SFT, "eval")
SETS = ("code_api", "math_api", "longdoc_api", "longdoc_test", "tools_api", "chat_api", "openthoughts_short")
MAX_WORKING, MAX_ANSWER = 2500, 8000
MIN_QUESTIONS = 10
SEED = 0


def jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unique(rows):
    """One row per id: a job that was started twice writes some ids twice; the first usable row wins."""
    best = {}
    for r in rows:
        if r["id"] not in best or (not best[r["id"]].get("ok") and r.get("ok")):
            best[r["id"]] = r
    return list(best.values())


def decon_data():
    return (frozenset(g for t in SC.eval_texts() for g in decontam.ngrams(decontam.words(t))),
            decontam.benchmark_short_statements())


def student_parts(kind, content, why):
    """(working, answer) of a teacher reply fit for the student, or (None, None) with the reason counted in why."""
    p = DT.parts(kind, content)
    reason = p["style"] and f"working: {p['style']}"
    if not reason and len(p["working"]) > MAX_WORKING:
        reason = "working too long"
    if not reason and len(p["answer"]) > MAX_ANSWER:
        reason = "answer too long"
    if not reason and kind == "code" and "```" not in p["answer"]:
        reason = "no code block"
    if reason:
        why[reason] += 1
        return None, None
    return p["working"], p["answer"]


def qa(user, working, answer):
    return [{"role": "user", "content": user}, {"role": "assistant", "think": working, "content": answer}]


def write_set(name, convs, why, rows):
    """convs: [(conversation dict with id / messages / tools / meta, licence, decontaminate)] -> conv/<name>.jsonl."""
    grams, short = decon_data() if any(c[2] for c in convs) else (frozenset(), {})
    SC._init(grams, short)
    groups = collections.defaultdict(list)
    for conv, licence, decon in convs:
        groups[(licence, decon)].append(conv)
    lines, chars = [], []
    for (licence, decon), batch in groups.items():
        for i in range(0, len(batch), 2000):
            ls, drops, ch = SC._finish((name, licence, decon, batch[i:i + 2000]))
            lines += ls
            chars += ch
            why.update(drops)
    os.makedirs(CONV, exist_ok=True)
    path = os.path.join(CONV, name + ".jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(line + "\n" for line in lines)
    os.replace(path + ".tmp", path)
    chars.sort()
    q = (lambda p: chars[int(p * (len(chars) - 1))]) if chars else (lambda p: 0)
    stats_path = os.path.join(CONV, "STATS.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    stats[name] = {"rows": rows, "kept": len(lines), "dropped": dict(why),
                   "chars": {"total": sum(chars), "median": q(.5), "p90": q(.9), "max": q(1)}, "source": "distill_convert.py"}
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(f"{name}: {len(lines)} conversations from {rows} results, {sum(chars) / 1e6:.1f}M characters, "
          f"dropped {dict(why)}", flush=True)


def read_pool():
    pool = {}
    for f in ("code_pool.jsonl", "code_pool_gen.jsonl"):
        for p in jsonl(os.path.join(SFT, f)):
            pool[p["id"]] = p
    return pool


def code_api():
    first = {r["id"]: r for r in jsonl(os.path.join(DISTILL, "code_checked.jsonl"))}
    if not first:
        return None
    retry = {r["id"]: r for r in jsonl(os.path.join(DISTILL, "code_retry_checked.jsonl"))}
    why, convs = collections.Counter(), []
    for pid, p in read_pool().items():
        if pid not in first:
            continue
        how, r = ("first", first[pid]) if first[pid]["ok"] else ("retry", retry.get(pid))
        if r is None or not r["ok"]:
            why["did not pass"] += 1
            continue
        working, answer = student_parts("code", r["content"], why)
        if working:
            convs.append(({"id": "code_api/" + pid, "messages": qa(p["prompt"], working, answer),
                           "meta": {"source": p["source"], "pass": how, "effort": r.get("effort")}},
                          p.get("licence", ""), not pid.startswith("mbpp_train/")))
    write_set("code_api", convs, why, len(first))
    return True


def math_prompts(ids):
    """The problem texts of GSM8K / OpenMathInstruct-2 conversations (by id) from the converted sets."""
    want, out = set(ids), {}
    for f in ("gsm8k.jsonl", "math.jsonl"):
        path = os.path.join(CONV, f)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r'\{"id": "([^"]*)"', line)
                if m and m.group(1) in want:
                    out[m.group(1)] = json.loads(line)["messages"][0]["content"]
    return out


def math_api():
    rows = unique(jsonl(os.path.join(DISTILL, "math.jsonl")))
    if not rows:
        return None
    why, convs = collections.Counter(), []
    prompts = math_prompts([r["id"] for r in rows if r["ok"]])
    for r in rows:
        if not r["ok"]:
            why["wrong answer"] += 1
        elif r["id"] not in prompts:
            why["problem text not found"] += 1
        else:
            working, answer = student_parts("math", r["content"], why)
            if working:
                convs.append(({"id": "math_api/" + r["id"], "messages": qa(prompts[r["id"]], working, answer),
                               "meta": {"effort": r.get("effort")}}, "cc-by-4.0 / mit, reasoning by the DeepSeek API", True))
    write_set("math_api", convs, why, len(rows))
    return True


def longdoc_pairs(row, why):
    """The usable (question, evidence, where, answer) of a document's results, in the order they were asked."""
    pairs = []
    for x in row["qa"]:
        if x.get("error") or not x.get("evidence_in_doc"):
            why["question without evidence in the document"] += 1
            continue
        p = DT.parts("reading", x.get("content"))
        if p["style"]:
            why[f"reading answer: {p['style']}"] += 1
        else:
            pairs.append((x["question"].strip(), x.get("evidence"), p["working"], p["answer"]))
    return pairs


def longdoc_api():
    rows = unique(jsonl(os.path.join(DISTILL, "longdoc.jsonl")))
    if not rows:
        return None
    why, convs = collections.Counter(), []
    for r in rows:
        pairs = longdoc_pairs(r, why)
        path = os.path.join(DISTILL, "longdocs", r["id"] + ".txt")
        if len(pairs) < MIN_QUESTIONS or not os.path.exists(path):
            why["document with too few usable questions"] += 1
            continue
        random.Random(f"{SEED}/{r['id']}").shuffle(pairs)
        doc = open(path, encoding="utf-8").read()
        msgs = []
        for k, (q, _, working, answer) in enumerate(pairs):
            msgs += qa(doc + "\n\n" + q if k == 0 else q, working, answer)
        convs.append(({"id": "longdoc_api/" + r["id"], "messages": msgs, "meta": {"document": r["id"], "questions": len(pairs)}},
                      "teacher answers by the DeepSeek API; documents from PG19 and en_web", False))
    write_set("longdoc_api", convs, why, len(rows))
    return True


def longdoc_test():
    rows = unique(jsonl(os.path.join(DISTILL, "longdoc_test.jsonl")))
    if not rows:
        return None
    why, out = collections.Counter(), []
    for r in rows:
        pairs = longdoc_pairs(r, why)
        if len(pairs) < MIN_QUESTIONS:
            why["document with too few usable questions"] += 1
            continue
        out.append({"id": r["id"], "document": os.path.join("data", "distill", "longdocs_test", r["id"] + ".txt"),
                    "qa": [{"question": q, "evidence": e, "where": w, "answer": a} for q, e, w, a in pairs]})
    os.makedirs(EVAL, exist_ok=True)
    path = os.path.join(EVAL, "longdoc_test.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out)
    os.replace(path + ".tmp", path)
    print(f"longdoc_test: {len(out)} documents, {sum(len(o['qa']) for o in out)} questions -> {path}, dropped {dict(why)}")
    return True


def backfilled(name, job, src, tag, decon):
    rows = unique(jsonl(os.path.join(DISTILL, job + ".jsonl")))
    if not rows:
        return None
    source = {c["id"]: c for c in jsonl(os.path.join(DISTILL, "backfill_in", src + ".jsonl"))}
    why, convs = collections.Counter(), []
    for r in rows:
        if not r["ok"]:
            why["teacher reply not usable"] += 1
        elif r["id"] not in source:
            why["conversation not found"] += 1
        else:
            c = source[r["id"]]
            conv = DT.with_thoughts(c, r["thoughts"])
            convs.append(({"id": f"{tag}/{c['id']}", "messages": conv["messages"], "tools": c.get("tools"),
                           "meta": {**(c.get("meta") or {}), "from": c.get("set")}}, c.get("licence", ""), decon))
    write_set(name, convs, why, len(rows))
    return True


def openthoughts_short():
    rows = unique(jsonl(os.path.join(DISTILL, "condense.jsonl")))
    if not rows:
        return None
    source = {c["id"]: c for c in jsonl(os.path.join(CONV, "openthoughts.jsonl"))}
    why, convs = collections.Counter(), []
    for r in rows:
        if not r["ok"]:
            why["teacher reply not usable"] += 1
        elif r["id"] not in source:
            why["conversation not found"] += 1
        else:
            c = source[r["id"]]
            msgs = [dict(m) for m in c["messages"]]
            msgs[-1]["think"] = r["think"]
            convs.append(({"id": "openthoughts_short/" + c["id"].split("/", 1)[1], "messages": msgs, "tools": None,
                           "meta": {**(c.get("meta") or {}), "from": "openthoughts"}}, c.get("licence", ""), True))
    write_set("openthoughts_short", convs, why, len(rows))
    return True


def main():
    global DISTILL, CONV, EVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SETS), choices=SETS)
    ap.add_argument("--distill", default=DISTILL, help="folder with the teacher results")
    ap.add_argument("--conv", default=CONV, help="folder for the conversation sets")
    ap.add_argument("--eval", default=EVAL, help="folder for the held-out evaluation sets")
    a = ap.parse_args()
    DISTILL, CONV, EVAL = a.distill, a.conv, a.eval
    jobs = {"code_api": code_api, "math_api": math_api, "longdoc_api": longdoc_api, "longdoc_test": longdoc_test,
            "tools_api": lambda: backfilled("tools_api", "backfill_tools", "tools", "tools_api", False),
            "chat_api": lambda: backfilled("chat_api", "backfill_chat", "chat", "chat_api", True),
            "openthoughts_short": openthoughts_short}
    for name in a.only:
        if not jobs[name]():
            print(f"{name}: no results yet, skipped")
    print("DISTILL_CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
