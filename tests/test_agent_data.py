"""data_prep/agent_pool.py and agent_traj.py: examples of a statement as asserts, and the multi-step conversations
built from a problem, the model's attempts and real runs (code_check's sandbox: WSL)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
import agent_eval as A  # noqa: E402
import agent_pool as P  # noqa: E402
import agent_traj as T  # noqa: E402
import code_check  # noqa: E402
from moe_llm import chat  # noqa: E402

pytestmark = pytest.mark.skipif(not code_check.check("x = 1", {"type": "asserts", "code": "assert x == 1"})[0],
                                reason="code_check's sandbox does not run here")

STATEMENT = ("Write a function `dbl(x)` that returns twice its argument. Examples: `dbl(2) == 4`, `dbl(-1) == -2`, "
             "`g = dbl; g('a') == 'aa'`, and `x == y` is not an example.")
SOL = "def dbl(x):\n    return x * 2"
PROB = {"id": "oss_gen/t/1", "source": "oss_gen", "prompt": STATEMENT, "solution": SOL,
        "explanation": "Doubling is one multiplication by two.", "asserts": ["assert dbl(2) == 4", "assert dbl(-1) == -2"],
        "fn": "dbl", "signature": "def dbl(x):"}


def test_candidate_asserts_from_ticked_examples_and_assert_blocks():
    assert P.split_eq("f(a == b, [1, 2]) == {'k': '=='}") == ("f(a == b, [1, 2])", "{'k': '=='}")
    assert P.split_eq("f(1)") is None
    assert P.candidate_asserts(STATEMENT) == ["assert dbl(2) == 4", "assert dbl(-1) == -2", "g = dbl; assert g('a') == 'aa'"]
    short = ("Write `rng(n)` for a non-negative integer `n`, up to but not including `n`. If `n` is 0, return an empty list. "
             "Examples: `rng(3) == [0, 1, 2]`, `rng(1) == [0]`, `rng(0) == []`.")
    assert P.candidate_asserts(short) == ["assert rng(3) == [0, 1, 2]", "assert rng(1) == [0]", "assert rng(0) == []"]
    given = "Do it.\n\nYour code should pass the following assertions:\n```python\nassert f(1) == 2\nassert f(2) == 3\n```"
    assert P.candidate_asserts(given) == ["assert f(1) == 2", "assert f(2) == 3"]
    assert P.signature("import math\n\ndef helper(a):\n    return a\n\ndef dbl(x, k=2):\n    return x * k\n", ["assert dbl(2) == 4"]) \
        == ("dbl", "def dbl(x, k=2):")


def test_check_one_keeps_only_asserts_the_solution_passes():
    row = P.check_one({"solution": SOL, "candidates": ["assert dbl(2) == 4", "assert dbl(2) == 5", "g = dbl; assert g('a') == 'aa'",
                                                      "assert nothere(1) == 1"]})
    assert row["asserts"] == ["assert dbl(2) == 4", "g = dbl; assert g('a') == 'aa'"]


def test_docstring_prompt_and_diagnosis():
    doc = T.docstring_prompt_v1(PROB)
    assert doc.startswith("Complete the following Python function.\n\n```python\ndef dbl(x):\n    \"\"\"\n    Write a function")
    assert "    >>> dbl(2)\n    4\n    >>> dbl(-1)\n    -2\n    \"\"\"\n```" in doc
    assert T.docstring_prompt_v1({**PROB, "asserts": ["g = dbl; assert g(1) == 2"]}) is None
    out = A.tool_output(T.script("def dbl(x):\n    return x + 2", PROB["asserts"]))
    assert T.diagnosis(out) == "The test `assert dbl(-1) == -2` fails: the call returned 1."
    assert "TypeError" in T.diagnosis(A.tool_output(T.script("def dbl(x):\n    return x + None", PROB["asserts"])))
    assert "did not finish" in T.diagnosis("the script did not finish within 10 seconds")


def test_work_builds_pass_and_fix_conversations_that_render():
    attempts = [{"id": PROB["id"], "pass": "greedy", "think": "Add two.", "content": "```python\ndef dbl(x):\n    return x + 2\n```", "closed": True},
                {"id": PROB["id"], "pass": "sampled", "think": "Multiply.", "content": "```python\ndef dbl(x):\n    return 2 * x\n```", "closed": True}]
    out, stats = T.work((PROB, attempts, "0/x", True))
    assert stats["greedy: fails the examples"] == 1 and stats["sampled: passes the examples"] == 1
    kinds = {c["meta"]["kind"]: c for c in out["agent_tool"]}
    assert set(kinds) == {"pass", "fix"} and len(out["agent_fix"]) == 1
    p, f = kinds["pass"]["messages"], kinds["fix"]["messages"]
    assert [m["role"] for m in p] == ["user", "assistant", "tool", "assistant"] and p[2]["content"] == "all tests passed"
    assert p[1]["tool_calls"][0]["name"] == "python" and SOL in p[1]["tool_calls"][0]["arguments"]["code"]
    assert p[3]["content"] == "```python\n" + SOL + "\n```" and p[0]["content"].endswith(A.TOOL_NOTE)
    assert [m["role"] for m in f] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert f[1]["think"] == "Add two." and "return x + 2" in f[1]["tool_calls"][0]["arguments"]["code"]
    assert f[2]["content"].endswith("AssertionError: 1")  # the real error of the model's attempt
    assert f[3]["think"].startswith("The test `assert dbl(-1) == -2` fails: the call returned 1. Doubling is")
    assert f[4]["content"] == "all tests passed" and f[5]["content"] == "```python\n" + SOL + "\n```"
    x = out["agent_fix"][0]["messages"]
    assert [m["role"] for m in x] == ["user", "assistant", "user", "assistant"]
    assert "assert dbl(-1) == -2" in x[2]["content"] and "AssertionError: 1" in x[2]["content"] and A.FIX in x[2]["content"]
    assert x[3]["content"] == "```python\n" + SOL + "\n```"
    for c in out["agent_tool"] + out["agent_fix"]:  # the chat template accepts them, and the reply parses back
        text = chat.render(c)
        assert text.count("<|im_start|>assistant") == sum(m["role"] == "assistant" for m in c["messages"])
        json.dumps(c)
    assert chat.render(kinds["pass"]).startswith("<|im_start|>system\n<tools>")


def test_work_without_a_failed_attempt_gives_only_the_pass_conversation():
    out, stats = T.work((PROB, [], "0/y", True))
    assert [c["meta"]["kind"] for c in out["agent_tool"]] == ["pass"] and out["agent_fix"] == []
    out, stats = T.work(({**PROB, "solution": "def dbl(x):\n    return x"}, [], "0/z", True))  # a solution that fails: nothing
    assert out["agent_tool"] == [] and stats["solution does not print the pass line"] == 1


def test_v2_tasks_show_exactly_the_tests_the_scripts_use():
    import random
    bare = T.strip_examples(STATEMENT)
    assert bare == "Write a function `dbl(x)` that returns twice its argument." and "==" not in bare
    assert T.strip_examples("Short. Examples: `f(1) == 2`.") == "Short. Examples: `f(1) == 2`."  # too little would be left
    given = "Write `f(x)` that adds one to its argument and returns it.\n\nYour code should pass the following assertions:\n```python\nassert f(1) == 2\n```"
    assert T.strip_examples(given) == "Write `f(x)` that adds one to its argument and returns it."
    seen = {}
    for seed in range(200):
        task, shown, form = T.make_task(PROB, random.Random(seed))
        seen.setdefault(form, (task, shown))
        assert set(shown) <= set(PROB["asserts"]) and shown
        if form == "single":
            assert len(shown) == 1 and task.endswith("Your code should pass this test: " + shown[0]) and "Examples" not in task
        elif form == "docstring":
            assert task.count(">>>") == len(shown) and "Examples:" not in task
        else:
            assert task == STATEMENT and shown == PROB["asserts"]
    assert set(seen) == {"plain", "docstring", "single"}


def test_v2_marks_the_failed_attempt_as_no_loss_and_never_runs_tests_the_task_does_not_show():
    attempts = [{"id": PROB["id"], "pass": "greedy", "think": "Add two.", "content": "```python\ndef dbl(x):\n    return x + 2\n```", "closed": True}]
    forms = set()
    for seed in range(40):
        out, stats = T.work((PROB, attempts, f"{seed}/x", False))
        for c in out["agent_tool"] + out["agent_fix"]:
            forms.add(c["meta"]["form"])
            flagged = [m for m in c["messages"] if m.get("no_loss")]
            assert len(flagged) == (1 if c["meta"]["kind"] == "fix" else 0)
            if flagged:
                assert flagged[0]["think"] == "Add two." and c["messages"][-1].get("no_loss") is None
            user = c["messages"][0]["content"]
            for m in c["messages"]:
                for call in m.get("tool_calls") or ():
                    tests = [line for line in call["arguments"]["code"].splitlines() if line.startswith("assert ")]
                    assert tests
                    for t in tests:  # every assert of a script is one the task text shows
                        call_text = t[len("assert "):].split(" == ")[0]
                        assert call_text in user, (t, user)
            chat.render(c)
        # an attempt that fails the examples the task shows gets a fix conversation; with one shown test it may pass
        assert len(out["agent_tool"]) in (1, 2)
    assert forms == {"plain", "docstring", "single"}


def test_gen2_keeps_only_conversations_the_model_got_right_by_itself():
    import random
    import agent_gen2 as G
    good, bad = "def dbl(x):\n    return x * 2", "def dbl(x):\n    return x + 2"
    cheat = "def dbl(x):\n    return {2: 4}.get(x, 0)"   # right on dbl(2) only

    def reply(code=None, content="", think="t"):
        calls = [{"name": "python", "arguments": {"code": code}}] if code else []
        return {"think": think, "content": content, "tool_calls": calls, "closed": True, "tokens": 9}

    probs = [{**PROB, "id": f"oss_gen/t/{i}"} for i in range(5)]
    shown = [T.make_task(p, random.Random(f"0/{p['id']}"))[1] for p in probs]
    block = lambda c: "```python\n" + c + "\n```"  # noqa: E731
    script = [
        # 0: fails, fixes it, answers -> self_fix;  1: passes at once -> self_pass;  2: invents a test -> dropped
        # 3: answers without a passing run -> dropped;  4: never stops calling -> dropped
        [reply(T.script(bad, shown[0])), reply(T.script(good, shown[1])),
         reply(T.script(good, shown[2] + ["assert dbl(10) == 20"])), reply(T.script(bad, shown[3])), reply(T.script(bad, shown[4]))],
        [reply(T.script(good, shown[0])), reply(content=block(good)), reply(content=block(good)), reply(content=block(bad)),
         reply(T.script(bad, shown[4]))],
        [reply(content=block(good)), reply(T.script(bad, shown[4]))],
    ]
    calls = []

    def gen(convs, label):
        calls.append(len(convs))
        return script[len(calls) - 1]

    kept, stats = G.run_chunk(probs, gen, rounds=3, seed=0)
    assert calls == [5, 5, 2]
    kinds = {c["id"].split("/")[-2] + ":" + c["meta"]["kind"] for c in kept}
    assert kinds == {"0:self_fix", "1:self_pass"}
    fix = next(c for c in kept if c["meta"]["kind"] == "self_fix")["messages"]
    assert [m["role"] for m in fix] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    assert fix[1].get("no_loss") is True and "no_loss" not in fix[3] and "no_loss" not in fix[5]
    assert fix[4]["content"] == "all tests passed" and "AssertionError" in fix[2]["content"]
    assert stats["dropped: a test the task does not show"] == 1 and stats["dropped: answer without a passing run"] == 1
    assert stats["dropped: out of turns"] == 1
    for c in kept:
        assert len(chat.no_loss_spans(c)) == (1 if c["meta"]["kind"] == "self_fix" else 0)
    # code that passes the one test it was shown and fails another assert of the problem is not kept
    one = [p for p, s in zip(probs, shown) if len(s) == 1 and "dbl(2)" in s[0]]
    if one:
        s1 = T.make_task(one[0], random.Random(f"0/{one[0]['id']}"))[1]
        seq = iter([[reply(T.script(cheat, s1))], [reply(content=block(cheat))]])
        kept, stats = G.run_chunk(one[:1], lambda convs, label: next(seq), rounds=3, seed=0)
        assert kept == [] and stats["dropped: fails an assert it was not shown"] == 1
