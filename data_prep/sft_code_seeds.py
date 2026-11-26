#!/usr/bin/env python
"""Snippets of real Python code that the teacher turns into new programming problems
(about 45,000 new code problems on top of the 54,658 that datasets provide).

    python data_prep/sft_code_seeds.py            # WSL (20 cores): a few minutes

reads the Stack-Edu Python files (data/raw/edu_python, edu_python2: files an educational-quality classifier scored 3 or
more), takes a random sample of them, and from each file at most two top-level functions of 5 to 40 lines that use only
the standard library (no import of a third-party module, no `self`). Functions that share a 13-gram with HumanEval or
MBPP are dropped (the problem the teacher writes would be too close to a benchmark problem), and so are duplicates.

writes data/distill/codeseeds.jsonl, shuffled, one snippet per line:
    {"id": "seed/<source>/<hash>", "source", "code", "difficulty": "easy" | "medium" | "hard"}
The difficulty is drawn here (35% easy, 40% medium, 25% harder) and handed to the teacher with the snippet, so that
the problems it writes are not all of one kind.
"""
from __future__ import annotations

import argparse
import ast
import glob
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "distill", "codeseeds.jsonl")
SOURCES = ("edu_python", "edu_python2")
STDLIB = sys.stdlib_module_names
# entry points and command-line plumbing teach nothing about a computation
SKIP_NAMES = frozenset(("main", "parse_args", "get_args", "get_arguments", "parse_arguments", "usage", "run", "setup",
                        "cli", "init", "start", "menu"))
CLI_MODULES = frozenset(("argparse", "optparse", "getopt"))


def non_stdlib_names(tree) -> set:
    """Names that the file's imports bind to modules outside the standard library (relative imports included)."""
    bad = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in STDLIB:
                    bad.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or node.module.split(".")[0] not in STDLIB:
                bad.update(a.asname or a.name for a in node.names)
    return bad


def functions(src: str, rng: random.Random, per_file: int = 2) -> list[str]:
    """Up to per_file suitable top-level functions of a source file, as source text."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    bad, out = non_stdlib_names(tree), []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith(("test", "__")) or node.name in SKIP_NAMES:
            continue
        if not 5 <= node.end_lineno - node.lineno + 1 <= 40 or node.args.args and node.args.args[0].arg in ("self", "cls"):
            continue
        seg = ast.get_source_segment(src, node)
        if not seg or len(seg) > 2500 or sum(ord(c) > 127 for c in seg) > 20:
            continue
        used = {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}
        if used & bad or used & CLI_MODULES or "sys.argv" in seg:
            continue
        if sum(1 for x in seg.split("\n")[1:] if x.strip() and not x.strip().startswith("#")) < 4:
            continue
        out.append(seg)
    rng.shuffle(out)
    return out[:per_file]


def read_file(job):
    """Worker: the suitable functions of a random sample of one Stack-Edu file."""
    import warnings
    warnings.simplefilter("ignore")  # the files are real code: invalid escape sequences and the like are everywhere
    path, keep, min_score, seed = job
    rng, out = random.Random(f"{seed}/{path}"), []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if rng.random() >= keep:
                continue
            r = json.loads(line)
            if r.get("int_score", 0) >= min_score and len(r["text"]) <= 60_000:
                out += functions(r["text"], rng)
    return os.path.basename(os.path.dirname(path)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70_000, help="snippets to keep")
    ap.add_argument("--keep", type=float, default=0.06, help="share of the files that is looked at")
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    import decontam
    t0 = time.time()
    files = [f for s in SOURCES for f in sorted(glob.glob(os.path.join(RAW, s, "*.json.gz")))]
    seen, seeds, dropped = set(), [], {"duplicate": 0, "benchmark": 0}
    grams, short = decontam.benchmark_ngrams(), decontam.benchmark_short_statements()
    with mp.get_context("fork").Pool(a.workers) as pool:
        for src, funcs in pool.imap_unordered(read_file, [(f, a.keep, a.min_score, a.seed) for f in files]):
            for code in funcs:
                key = hashlib.sha1(re.sub(r"\s+", " ", code).encode()).hexdigest()
                if key in seen:
                    dropped["duplicate"] += 1
                elif decontam.contaminated(code, grams, short):
                    dropped["benchmark"] += 1
                else:
                    seen.add(key)
                    seeds.append({"id": f"seed/{src}/{key[:10]}", "source": src, "code": code})
            print(f"  {len(seeds)} snippets so far [{time.time() - t0:.0f}s]", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(seeds)
    seeds = seeds[:a.n]
    for s in seeds:
        s["difficulty"] = rng.choices(("easy", "medium", "hard"), weights=(35, 40, 25))[0]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".tmp", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds)
    os.replace(a.out + ".tmp", a.out)
    n = sorted(len(s["code"]) for s in seeds)
    print(f"{len(seeds)} snippets written to {a.out}; characters median {n[len(n) // 2]}, max {n[-1]}; dropped {dropped}")
    print("SFT_CODE_SEEDS_DONE", flush=True)


if __name__ == "__main__":
    main()
