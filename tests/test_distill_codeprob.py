"""Teacher-written code problems and the long-document selection (data_prep/distill_codeprob.py, sft_code_seeds.py,
sft_code_pool.py --gen, sft_longdocs.py, and the codeprob job of distill_gen.py). Nothing here calls the API."""
import json
import os
import random
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
sys.path.insert(0, ROOT)
import distill_codeprob as DP  # noqa: E402
import sft_code_seeds as SEEDS  # noqa: E402
import sft_longdocs as LD  # noqa: E402

GOOD_FUNCTION = ("Write a function `longest_run(items)` that takes a list and returns the length of the longest block of "
                 "equal neighbouring elements. An empty list gives 0. Examples: `longest_run([1, 1, 2, 2, 2, 3]) == 3`, "
                 "`longest_run([]) == 0`, `longest_run(['a']) == 1`.")


def test_check_problem():
    ok, detail, p = DP.check_problem('Here:\n```json\n' + json.dumps({"kind": "function", "problem": GOOD_FUNCTION}) + "\n```")
    assert ok and p == {"kind": "function", "problem": GOOD_FUNCTION} and detail.startswith("function")
    prog = ("Write a program that reads a line of space-separated integers from standard input and prints the sum of "
            "the even ones, or 0 if there are none. Example: the input `1 2 3 4` gives the output `6`. " * 2)
    assert DP.check_problem(json.dumps({"kind": "program", "problem": prog}))[0]
    bad = [("not json", "no braces"), ("keys", json.dumps({"kind": "other", "problem": GOOD_FUNCTION})),
           ("short", json.dumps({"kind": "function", "problem": "Write `f(x)`."})),
           ("mentions snippet", json.dumps({"kind": "function", "problem": GOOD_FUNCTION + " Use the idea of the snippet."})),
           ("random", json.dumps({"kind": "function", "problem": GOOD_FUNCTION + " Pick the items at random."})),
           ("no signature", json.dumps({"kind": "function", "problem": GOOD_FUNCTION.replace("`", "")})),
           ("program without input", json.dumps({"kind": "program", "problem": GOOD_FUNCTION}))]
    for name, content in bad:
        assert not DP.check_problem(content)[0], name
    prompt = DP.problem_prompt({"difficulty": "hard", "code": "def f(x):\n    return x"})
    assert prompt.startswith("Difficulty: hard\n\nSnippet:\n```python\ndef f(x)") and prompt.endswith("Reply:")


SOURCE = '''
import os
import numpy as np
from collections import Counter

def top_words(text, k):
    counts = Counter(text.split())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    result = []
    for word, n in ranked[:k]:
        result.append((word, n))
    return result

def uses_numpy(a):
    b = np.array(a)
    c = b * 2
    d = c + 1
    return d.tolist()

def tiny(x):
    return x

class A:
    def method(self, a, b):
        x = a + b
        y = x * 2
        z = y - 1
        return z

def main():
    text = input()
    words = text.split()
    total = len(words)
    print(total)

def read_args(a):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--x")
    args = p.parse_args()
    return args

def test_something(a):
    x = 1
    y = 2
    z = 3
    return x + y + z
'''


def test_seed_functions():
    got = SEEDS.functions(SOURCE, random.Random(0))
    assert len(got) == 1 and got[0].startswith("def top_words(text, k):")   # numpy one, too short one, method, test_ out
    assert SEEDS.functions("def broken(:\n", random.Random(0)) == []


def test_longdocs_helpers():
    prose = ("The old man walked down to the sea and he said that it was not the same as it had been. " * 3000)
    assert LD.judge(prose) is None
    assert LD.judge("short") == "too short"
    assert LD.judge(("| a | b |\n" * 40000)) in ("few letters", "not running English prose", "repeated lines", "tables")
    assert LD.judge("Project Gutenberg\n" + prose) == "reference work or licence text"
    assert LD.judge(("1 2 3 4 5 6 7 8 9 " * 20000)) == "few letters"
    para = "First paragraph.\n\n" + "word " * 500 + "\n\n" + "tail " * 100
    cut = LD.paragraph_cut(para, True)
    assert cut.endswith("word") and "tail" not in cut
    assert LD.paragraph_cut(para, False).endswith("tail")
    a, b = LD.shingles("one two three four five six seven eight nine ten", 1), LD.shingles("x y z", 1)
    assert len(a) == 3 and len(b) == 0
    other = LD.shingles("zero one two three four five six seven eight nine ten eleven", 1)
    assert np.isin(a, other).all() and not np.isin(a, LD.shingles("a b c d e f g h i j", 1)).any()


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "data", "raw", "bench")),
                    reason="needs the benchmark files in data/raw/bench (decontamination)")
def test_codeprob_job_and_pool(tmp_path, monkeypatch):
    import distill_gen as G
    import sft_code_pool as P
    out = tmp_path / "distill"
    out.mkdir()
    seeds = [{"id": f"seed/edu_python/{i:03d}", "source": "edu_python", "code": f"def f{i}(x):\n    return x + {i}",
              "difficulty": "easy"} for i in range(4)]
    (out / "codeseeds.jsonl").write_text("".join(json.dumps(s) + "\n" for s in seeds))
    replies = iter([json.dumps({"kind": "function", "problem": GOOD_FUNCTION}), "not json at all",
                    json.dumps({"kind": "function", "problem": GOOD_FUNCTION}),   # a duplicate of the first
                    json.dumps({"kind": "function", "problem": GOOD_FUNCTION.replace("longest_run", "block_len")})])
    usage = {"prompt_tokens": 500, "completion_tokens": 100, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 500}

    def fake_call(key, messages, effort, budget, max_tokens=8192, tries=5):
        assert effort == "none" and messages[0]["content"] == DP.PROBLEM_SYS   # thinking is off by default for this job
        budget.add(usage)
        return {"content": next(replies), "reasoning": None, "usage": usage, "finish": "stop"}

    monkeypatch.setattr(G, "OUT", str(out))
    monkeypatch.setattr(G, "call", fake_call)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "stub")
    monkeypatch.setattr(sys, "argv", ["distill_gen.py", "codeprob", "--max-usd", "1", "--allow-peak", "--workers", "1"])
    G.main()
    rows = [json.loads(line) for line in open(out / "codeprob.jsonl")]
    assert [r["ok"] for r in rows] == [True, False, True, True] and rows[1]["raw"] == "not json at all"
    assert rows[0]["problem"]["kind"] == "function" and rows[0]["difficulty"] == "easy"
    monkeypatch.setattr(sys, "argv", ["distill_gen.py", "codeprob", "--max-usd", "1", "--allow-peak"])
    G.main()  # a second run finds everything done and sends nothing
    assert len(open(out / "codeprob.jsonl").readlines()) == 4
    # the pool of generated problems: valid ones, no duplicates
    monkeypatch.setattr(P, "GEN_IN", str(out / "codeprob.jsonl"))
    monkeypatch.setattr(P, "GEN_OUT", str(tmp_path / "code_pool_gen.jsonl"))
    P.gen_pool()
    pool = [json.loads(line) for line in open(tmp_path / "code_pool_gen.jsonl")]
    assert len(pool) == 2 and pool[0]["id"] == "oss_gen/edu_python/000" and pool[0]["source"] == "oss_gen"
    assert pool[0]["check"] is None and pool[0]["prompt"] == GOOD_FUNCTION and pool[0]["meta"]["kind"] == "function"
    # the code job reads both pools; problems with tests come first
    sft = tmp_path / "root" / "data" / "sft"
    sft.mkdir(parents=True)
    tested = {"id": "mbpp_train/1", "source": "mbpp_train", "prompt": "p", "check": {"type": "asserts", "code": "x"}}
    plain = {"id": "selfoss/1", "source": "selfoss", "prompt": "q", "check": None}
    (sft / "code_pool.jsonl").write_text(json.dumps(tested) + "\n" + json.dumps(plain) + "\n")
    (sft / "code_pool_gen.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pool))
    monkeypatch.setattr(G, "ROOT", str(tmp_path / "root"))
    items = G.code_items(100)
    assert items[0]["id"] == "mbpp_train/1" and {i["id"] for i in items} == {"mbpp_train/1", "selfoss/1"} | {p["id"] for p in pool}
    assert len(G.code_items(2)) == 2


def test_longdoc_test_split(tmp_path, monkeypatch):
    import distill_gen as G
    monkeypatch.setattr(G, "OUT", str(tmp_path))
    for d, names in (("longdocs", ("a", "b")), ("longdocs_test", ("t",))):
        (tmp_path / d).mkdir()
        for n in names:
            (tmp_path / d / f"{n}.txt").write_text("text")
    assert [i["id"] for i in G.longdoc_items()] == ["a", "b"] and [i["id"] for i in G.longdoc_items("test")] == ["t"]


def test_code_two_passes(tmp_path, monkeypatch):
    """First pass without thinking, the real checker runs the code, the failures go again with thinking."""
    import distill_gen as G
    root, out = tmp_path / "root", tmp_path / "distill"
    (root / "data" / "sft").mkdir(parents=True)
    pool = [{"id": "mbpp_train/1", "source": "mbpp_train", "prompt": "Add two numbers.",
             "check": {"type": "asserts", "code": "assert add(1, 2) == 3"}},
            {"id": "mbpp_train/2", "source": "mbpp_train", "prompt": "Multiply two numbers.",
             "check": {"type": "asserts", "code": "assert mul(2, 3) == 6"}}]
    (root / "data" / "sft" / "code_pool.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pool))
    good = {"Add": "Adding the two arguments gives the sum that is asked for.\n```python\ndef add(a, b):\n    return a + b\n```",
            "Multiply": "Multiplying the two arguments gives the product that is asked for.\n```python\ndef mul(a, b):\n    return a * b\n```"}
    wrong = "Adding the two arguments gives the product that is asked for.\n```python\ndef mul(a, b):\n    return a + b\n```"
    seen = []
    usage = {"prompt_tokens": 400, "completion_tokens": 100, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 400}

    def fake_call(key, messages, effort, budget, max_tokens=8192, tries=5):
        task = messages[-1]["content"].split()[1]
        seen.append((task, effort))
        budget.add(usage)
        reply = wrong if task == "Multiply" and effort == "none" else good[task]
        return {"content": reply, "reasoning": None, "usage": usage, "finish": "stop"}

    monkeypatch.setattr(G, "ROOT", str(root))
    monkeypatch.setattr(G, "OUT", str(out))
    monkeypatch.setattr(G, "call", fake_call)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "stub")
    base = ["distill_gen.py", "code", "--max-usd", "1", "--allow-peak", "--workers", "1"]
    for extra in ([], ["--verify"], ["--retry-failed"], ["--retry-failed", "--verify"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        G.main()
    assert sorted(seen) == [("Add", "none"), ("Multiply", "low"), ("Multiply", "none")]   # only the failure went again
    first = {r["id"]: r["ok"] for r in map(json.loads, open(out / "code_checked.jsonl"))}
    second = {r["id"]: r["ok"] for r in map(json.loads, open(out / "code_retry_checked.jsonl"))}
    assert first == {"mbpp_train/1": True, "mbpp_train/2": False} and second == {"mbpp_train/2": True}
