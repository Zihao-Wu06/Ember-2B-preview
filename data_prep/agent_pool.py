#!/usr/bin/env python
"""Problem pool for the small-agent data: function problems of the
teacher's code set (data/sft/conv/code_api.jsonl) together with tests that can be shown to the model and run.

The teacher's own tests were not kept, but the problems carry their examples in the statement:
  - problems the teacher wrote (oss_gen): "two or three concrete examples written as `name(arguments) == result`"
  - self-oss-instruct problems: often "Your code should pass the following assertion ..." with a code block
Every example becomes an assert and is kept only if the teacher's verified solution passes it (code_check's sandbox),
so a wrongly parsed example or one the solution does not satisfy never reaches the data. A problem needs two asserts
(--min-asserts) to enter the pool. MBPP's train split is left out (its test split has near-duplicates of it).

    python data_prep/agent_pool.py            -> data/agent/pool.jsonl
        {"id", "source", "prompt", "asserts", "solution", "explanation", "fn", "signature"}
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import multiprocessing
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import code_check  # noqa: E402

CONV = os.path.join(ROOT, "data", "sft", "conv", "code_api.jsonl")
OUT = os.path.join(ROOT, "data", "agent", "pool.jsonl")
_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
# every `code span` of the statement, also the one-letter ones: skipping `n` made the next opening backtick pair with
# its closing one, so that the text BETWEEN two spans was read as a span and later examples were lost (found 2026-11-24)
_TICK = re.compile(r"`([^`\n]{1,300})`")
_GIVEN = re.compile(r"pass the following (?:assertions?|test cases?)[^\n]*\n+```(?:python|py)?[ \t]*\n(.*?)```", re.S)
HARNESS = '''
import json as _json
_res = []
for _t in _TESTS:
    try:
        exec(_t, globals())
        _res.append(True)
    except BaseException:
        _res.append(False)
print("@@" + _json.dumps(_res))
'''


def split_eq(span: str):
    """"call == value" -> (call, value) at the last top-level ==, else None."""
    depth, cut, quote, i = 0, None, None, 0
    while i < len(span):
        ch = span[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif span.startswith("==", i) and depth == 0:
            cut = i
            i += 1
        i += 1
    return (span[:cut].strip(), span[cut + 2:].strip()) if cut is not None else None


def candidate_asserts(prompt: str) -> list:
    """Assert statements from the examples of a problem statement (not yet checked against a solution)."""
    out = []
    m = _GIVEN.search(prompt)
    if m:
        out += [line.strip() for line in m.group(1).splitlines() if line.strip().startswith("assert ")]
    for span in _TICK.findall(prompt):
        span = span.strip().rstrip(".,;")
        if "==" not in span or "(" not in span:
            continue
        *setup, last = [s.strip() for s in span.split(";")]
        eq = split_eq(last)
        if not eq or "(" not in eq[0]:
            continue
        stmt = "; ".join(setup + [f"assert {eq[0]} == {eq[1]}"])
        try:
            ast.parse(stmt)
        except SyntaxError:
            continue
        out.append(stmt)
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq[:8]


def signature(code: str, asserts: list):
    """(function name, "def name(args):") of the function the asserts call, from the solution's source."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    text = " ".join(asserts)
    for name, node in defs.items():
        if re.search(rf"\b{re.escape(name)}\(", text):
            return name, f"def {name}({ast.unparse(node.args)}):"
    return None, None


def check_one(row):
    """row with "candidates" -> row with "asserts" (those the solution passes)."""
    code, cands = row["solution"], row.pop("candidates")
    if code_check.UNSAFE.search(code) or any(code_check.UNSAFE.search(c) for c in cands):
        row["asserts"] = []
        return row
    ok, out, _ = code_check.run(code_check.PRELUDE + code + "\n_TESTS = " + repr(cands) + "\n" + HARNESS, timeout=10)
    res = []
    for line in out.splitlines():
        if line.startswith("@@"):
            try:
                res = json.loads(line[2:])
            except ValueError:
                res = []
    row["asserts"] = [c for c, r in zip(cands, res) if r] if len(res) == len(cands) else []
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default=CONV)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-asserts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    rows, why = [], collections.Counter()
    for line in open(args.conv, encoding="utf-8"):
        x = json.loads(line)
        src = x["meta"]["source"]
        if src not in ("oss_gen", "selfoss"):
            why[f"source {src}"] += 1
            continue
        prompt, a = x["messages"][0]["content"], x["messages"][-1]
        m = _BLOCK.search(a["content"])
        cands = candidate_asserts(prompt)
        if not m or len(cands) < args.min_asserts:
            why["fewer examples than needed in the statement"] += 1
            continue
        rows.append({"id": x["id"][len("code_api/"):], "source": src, "prompt": prompt, "solution": m.group(1).rstrip(),
                     "explanation": a["think"].strip(), "candidates": cands})
        if args.limit and len(rows) >= args.limit:
            break
    print(f"{len(rows)} problems with examples; checking them against the teacher's solutions", flush=True)
    kept = []
    with multiprocessing.Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(check_one, rows, chunksize=16)):
            if len(row["asserts"]) < args.min_asserts:
                why["fewer than needed pass with the solution"] += 1
                continue
            row["fn"], row["signature"] = signature(row["solution"], row["asserts"])
            kept.append(row)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(rows)} checked, {len(kept)} kept", flush=True)
    kept.sort(key=lambda r: r["id"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".tmp", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(args.out + ".tmp", args.out)
    n = collections.Counter(len(r["asserts"]) for r in kept)
    print(f"AGENT_POOL_DONE {args.out}: {len(kept)} problems ({dict(collections.Counter(r['source'] for r in kept))}), "
          f"asserts per problem {dict(sorted(n.items()))}, with a signature {sum(bool(r['signature']) for r in kept)}; "
          f"left out {dict(why)}", flush=True)


if __name__ == "__main__":
    main()
