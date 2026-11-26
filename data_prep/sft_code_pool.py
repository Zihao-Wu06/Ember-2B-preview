#!/usr/bin/env python
"""The programming problems the teacher (DeepSeek API) will be asked to solve ->
data/sft/code_pool.jsonl, one problem per line:

    {"id", "source", "licence", "prompt", "check", "ref"}

check is a code_check.py spec the teacher's code must pass, or None when the dataset has no tests (the teacher then
writes its own tests, which must pass on its own solution). Problems that share a 13-gram with HumanEval or MBPP's
test split are dropped; a problem with tests is kept only if a reference solution passes them here, so every check
is known to be runnable.

    python data_prep/sft_code_pool.py            # WSL; ~10 min (runs ~15k reference solutions)
    python data_prep/sft_code_pool.py --gen      # the problems the teacher wrote -> data/sft/code_pool_gen.jsonl

Sources: APPS train (function-call and stdin problems, at most 5 small test cases each; "competition" level left
out), MBPP train split (task ids 601-974; the test split 11-510 is the benchmark), self-oss-instruct's instructions
(no tests).
"""
from __future__ import annotations

import collections
import glob
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import code_check  # noqa: E402
import decontam  # noqa: E402

sys.set_int_max_str_digits(0)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "sft", "code_pool.jsonl")
MAX_CASES, MAX_CASE_CHARS = 5, 2000


def apps():
    with open(os.path.join(RAW, "sft", "apps", "train.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["difficulty"] == "competition" or not r["input_output"] or not r["solutions"]:
                continue
            io = json.loads(r["input_output"])
            cases = [(i, o) for i, o in zip(io.get("inputs", []), io.get("outputs", []))
                     if len(json.dumps(i)) + len(json.dumps(o)) <= MAX_CASE_CHARS][:MAX_CASES]
            if not cases:
                continue
            prompt = r["question"].strip()
            if "fn_name" in io:
                check = {"type": "call", "fn": io["fn_name"], "cases": [[i, o] for i, o in cases]}
                if r["starter_code"].strip():
                    prompt += "\n\nUse this starter code:\n```python\n" + r["starter_code"].strip("\n") + "\n```"
            else:
                def text(x):
                    return x if isinstance(x, str) else "\n".join(map(str, x)) + "\n"
                check = {"type": "stdin", "cases": [[text(i), text(o)] for i, o in cases]}
                prompt += "\n\nRead the input from standard input and print the answer to standard output."
            yield {"id": f"apps_train/{r['id']}", "source": "apps", "licence": "mit (codeparrot/apps)", "prompt": prompt,
                   "check": check, "refs": json.loads(r["solutions"])[:3], "meta": {"difficulty": r["difficulty"]}}


def mbpp():
    with open(os.path.join(RAW, "bench", "mbpp", "full", "train-00000-of-00001.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            yield {"id": f"mbpp_train/{r['task_id']}", "source": "mbpp_train", "licence": "cc-by-4.0 (mbpp)",
                   "prompt": f"{r['text']}\nYour code should pass this test: {r['test_list'][0]}",
                   "check": {"type": "asserts", "code": "\n".join(r["test_list"]), "setup": r.get("test_setup_code") or ""},
                   "refs": [r["code"].replace("\r\n", "\n")], "meta": {}}


def selfoss():
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(RAW, "sft", "self_oss_instruct_50k", "data", "*.parquet"))):
        for r in pq.read_table(f, columns=["id", "instruction", "concepts"]).to_pylist():
            yield {"id": f"selfoss/{r['id']}", "source": "selfoss",
                   "licence": "odc-by (bigcode/self-oss-instruct-sc2-exec-filter-50k)", "prompt": r["instruction"].strip(),
                   "check": None, "refs": [], "meta": {"concepts": r["concepts"]}}


_GRAMS = _SHORT = None


def _init(grams, short):
    global _GRAMS, _SHORT
    _GRAMS, _SHORT = grams, short


def _vet(p):
    """(problem without refs, or None; reason)."""
    # MBPP's own train split is training data here (the benchmark is the test split, task ids 11-510), but
    # decontam's n-grams cover every MBPP split, so it is exempt
    if p["source"] != "mbpp_train" and decontam.contaminated(p["prompt"] + "\n" + "\n".join(p["refs"]), _GRAMS, _SHORT):
        return None, "shares a 13-gram with HumanEval / MBPP"
    ref = None
    if p["check"] is not None:
        for cand in p["refs"]:
            if code_check.check(cand, p["check"], timeout=10)[0]:
                ref = cand
                break
        else:
            return None, "no reference solution passes the tests here"
    out = {k: v for k, v in p.items() if k != "refs"}
    out["ref"] = ref
    return out, None


def main():
    grams, short = decontam.benchmark_ngrams(), decontam.benchmark_short_statements()
    stats = collections.defaultdict(collections.Counter)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with mp.get_context("fork").Pool(16, initializer=_init, initargs=(grams, short)) as pool, \
            open(OUT + ".tmp", "w", encoding="utf-8") as f:
        for src in (mbpp, apps, selfoss):
            items = list(src())
            for p, (out, why) in zip(items, pool.imap(_vet, items, chunksize=8)):
                stats[p["source"]]["kept" if out else why] += 1
                if out:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            name = items[0]["source"]
            print(f"{name}: {dict(stats[name])}", flush=True)
    os.replace(OUT + ".tmp", OUT)
    with open(OUT[:-6] + "_stats.json", "w") as f:
        json.dump({k: dict(v) for k, v in stats.items()}, f, indent=1)
    print("CODE_POOL_DONE", flush=True)


GEN_IN = os.path.join(ROOT, "data", "distill", "codeprob.jsonl")
GEN_OUT = os.path.join(ROOT, "data", "sft", "code_pool_gen.jsonl")


def gen_pool():
    """data/sft/code_pool_gen.jsonl from the problems the teacher wrote (data/distill/codeprob.jsonl, job codeprob of
    distill_gen.py): the valid ones, without duplicates, minus those that share a 13-gram with HumanEval or MBPP. No
    tests (the code job's teacher writes them), so the entries look like self-oss-instruct's."""
    import re
    grams, short = decontam.benchmark_ngrams(), decontam.benchmark_short_statements()
    seen, kept, why = set(), 0, collections.Counter()
    with open(GEN_IN, encoding="utf-8") as f, open(GEN_OUT + ".tmp", "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            if not r.get("ok") or not r.get("problem"):
                why["not a valid problem"] += 1
                continue
            text = r["problem"]["problem"]
            key = re.sub(r"\W+", " ", text).lower().strip()
            if key in seen:
                why["duplicate"] += 1
            elif decontam.contaminated(text, grams, short):
                why["shares a 13-gram with HumanEval / MBPP"] += 1
            else:
                seen.add(key)
                kept += 1
                out.write(json.dumps({"id": "oss_gen/" + r["id"].split("/", 1)[1], "source": "oss_gen",
                                      "licence": "written by the DeepSeek API from Stack-Edu snippets", "prompt": text,
                                      "check": None, "meta": {"kind": r["problem"]["kind"], "seed": r["id"],
                                                              "difficulty": r.get("difficulty")}, "ref": None},
                                     ensure_ascii=False) + "\n")
    os.replace(GEN_OUT + ".tmp", GEN_OUT)
    print(f"oss_gen: {kept} problems kept, dropped {dict(why)}")
    print("CODE_POOL_GEN_DONE", flush=True)


if __name__ == "__main__":
    gen_pool() if "--gen" in sys.argv else main()
