"""data_prep/agent_env.py and runs/mac_eval/agent_env_eval.py: the small-project environment, the injected bugs, the
training conversations and the evaluation loop (code_check's sandbox: WSL)."""
import json
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
import agent_env as E  # noqa: E402
import agent_env_eval as V  # noqa: E402
import code_check  # noqa: E402
from moe_llm import chat  # noqa: E402

pytestmark = pytest.mark.skipif(not code_check.check("x = 1", {"type": "asserts", "code": "assert x == 1"})[0],
                                reason="code_check's sandbox does not run here")

SOL = "def total(xs):\n    s = 0\n    for x in xs:\n        if x > 0:\n            s += x\n    return s\n"
ASSERTS = ["assert total([1, 2, 3]) == 6", "assert total([-1, 4]) == 4", "assert total([]) == 0"]
PROB = {"id": "oss_gen/t/1", "source": "oss_gen", "prompt": "Write a function `total(xs)` that adds the positive numbers of "
        "the list `xs`. Examples: `total([1, 2, 3]) == 6`, `total([-1, 4]) == 4`, `total([]) == 0`.", "solution": SOL,
        "explanation": "Add up the numbers that are greater than zero.", "asserts": ASSERTS, "fn": "total",
        "signature": "def total(xs):"}
OTHER = {**PROB, "id": "oss_gen/t/2", "solution": "def twice(x):\n    return x * 2\n", "asserts": ["assert twice(2) == 4", "assert twice(0) == 0"],
         "fn": "twice", "signature": "def twice(x):", "prompt": "Write `twice(x)`. Examples: `twice(2) == 4`, `twice(0) == 0`."}


def files(src=SOL):
    return {E.SOL: src, E.TEST: E.test_file(ASSERTS)}


def test_tools():
    env = E.Env(files(SOL.replace("x > 0", "x >= 0").replace("s += x", "s -= x")))
    assert env.call("read_file", {"path": "solution.py"}).splitlines()[0] == "  1 | def total(xs):"
    out = env.call("run_tests", {})
    assert out.endswith("AssertionError: -6") and 'File "test_solution.py", line 3' in out and "/tmp" not in out and "^^^" not in out
    assert env.call("edit_file", {"path": "solution.py", "old": "s -= x", "new": "s += x"}) == "Edited solution.py."
    assert "not found" in env.call("edit_file", {"path": "solution.py", "old": "nothing like this", "new": "x"})
    assert "occurs 2 times" in env.call("edit_file", {"path": "solution.py", "old": "s ", "new": "t "})
    assert "read-only" in env.call("write_file", {"path": "test_solution.py", "content": "print('All tests passed.')"})
    assert "read-only" in env.call("edit_file", {"path": "./test_solution.py", "old": "assert", "new": "#"}) and env.test_writes == 2
    assert env.call("run_tests", {}) == E.PASS_LINE and env.passed()
    assert "no tool" in env.call("delete_file", {}) and "no file" in env.call("read_file", {"path": "nope.py"})
    assert env.call("write_file", {"path": "solution.py", "content": "def total(xs):\n    return sum(x for x in xs if x > 0)"}) \
        == "Wrote solution.py (2 lines)." and env.passed()
    env.call("write_file", {"path": "solution.py", "content": "import os\ndef total(xs):\n    return 0\n"})
    assert "not allowed" in env.call("run_tests", {})
    crash = E.Env(files(SOL.replace("s += x", "s += x[0]")))
    out = crash.call("run_tests", {})
    assert 'File "solution.py", line 5, in total' in out and out.endswith("not subscriptable")
    assert E.diagnosis(out).endswith("in line 5 of solution.py.")


def test_mutants_change_one_token_outside_header_lines():
    ms = E.mutants(SOL)
    assert ms and all(m["src"].count("\n") == SOL.count("\n") for m in ms)
    assert all(not m["orig_line"].lstrip().startswith("def ") for m in ms)
    pairs = {(m["orig_tok"], m["new_tok"]) for m in ms}
    assert {(">", ">="), (">", "<"), ("+=", "-="), ("0", "1")} <= pairs
    assert any(m["new_tok"] == "" for m in E.mutants("def f(x):\n    return not x\n"))  # a missing not
    assert all("total" not in m["new_line"] or m["new_line"] != m["orig_line"] for m in ms)
    two = [m for m in E.mutants("def f(rows, i, j):\n    return rows[i][j - 1] + rows[i][j]\n") if m["orig_tok"] == "-"]
    assert two[0]["ctx_orig"] == "j - 1" and two[0]["ctx_new"] == "j + 1"
    assert E.locate(two[0]).endswith("is the bug: `j + 1` should be `j - 1`.")
    nots = [m for m in E.mutants("def f(x, y):\n    return x not in y\n") if m["new_tok"] == ""]
    assert nots[0]["ctx_orig"] == "x not in" and nots[0]["ctx_new"] == "x in" and "a `not` is missing" in E.locate(nots[0])


def test_bugfix_conversation_finds_and_undoes_the_bug():
    rng = random.Random(0)
    m = E.find_bug(SOL, E.test_file(ASSERTS), rng)
    assert m is not None and not E.run_files({E.SOL: m["src"], E.TEST: E.test_file(ASSERTS)})[0]
    msgs = E.bugfix_conv({E.TEST: E.test_file(ASSERTS)}, m, E.BUGFIX_PROMPTS[0], random.Random(1))
    roles = [x["role"] for x in msgs]
    assert roles[0] == "user" and roles[-1] == "assistant" and roles.count("tool") == 4
    calls = [x["tool_calls"][0]["name"] for x in msgs if x.get("tool_calls")]
    assert sorted(calls) == ["edit_file", "read_file", "run_tests", "run_tests"]
    assert msgs[-2]["content"] == E.PASS_LINE and f"line {m['line']}" in msgs[-1]["content"]
    edit = next(x for x in msgs if x.get("tool_calls") and x["tool_calls"][0]["name"] == "edit_file")
    assert f"Line {m['line']}" in edit["think"]
    env = E.Env({E.SOL: m["src"], E.TEST: E.test_file(ASSERTS)})
    env.call("edit_file", edit["tool_calls"][0]["arguments"])
    assert env.files[E.SOL] == SOL
    text = chat.render({"messages": msgs, "tools": E.TOOLS})
    assert text.startswith("<|im_start|>system\n<tools>") and '"name": "edit_file"' in text


def test_work_train_builds_every_kind():
    out, st = E.work_train(("bugfix", PROB, None, "s/1"))
    assert len(out) == 1 and out[0]["set"] == "agent_env" and out[0]["tools"] == E.TOOLS and st["bugfix: built"] == 1
    out, st = E.work_train(("bugfix2", PROB, OTHER, "s/2"))
    if out:  # both functions are in solution.py, the broken one is PROB's
        user_file = out[0]["messages"]
        read = next(x for x in user_file if x["role"] == "tool" and x["content"].lstrip().startswith("1 |"))
        assert "def total" in read["content"] and "def twice" in read["content"]
    bad = {"id": PROB["id"], "pass": "greedy", "think": "Sum them all.", "content": "```python\ndef total(xs):\n    return sum(xs)\n```", "closed": True}
    good = {"id": PROB["id"], "pass": "s1", "think": "Only the positive ones.",
            "content": "```python\ndef total(xs):\n    return sum(x for x in xs if x > 0)\n```", "closed": True}
    out, st = E.work_train(("implement", PROB, [bad, good], "s/3"))
    env_c = next(c for c in out if c["set"] == "agent_env")
    assert env_c["meta"]["writes"] == "own fail, own pass"
    writes = [x for x in env_c["messages"] if x.get("tool_calls") and x["tool_calls"][0]["name"] == "write_file"]
    assert len(writes) == 2 and writes[0].get("no_loss") is True and "no_loss" not in writes[1]
    assert writes[1]["think"].endswith("Only the positive ones.") and "fails" in writes[1]["think"]
    assert len(chat.no_loss_spans({"messages": env_c["messages"], "tools": E.TOOLS})) == 1
    self_c = [c for c in out if c["set"] == "agent_self"]
    if self_c:  # the failed attempt must fail the tests the task shows; with one shown test it may pass them
        m = self_c[0]["messages"]
        assert m[1].get("no_loss") is True and m[1]["think"] == "Sum them all." and m[4]["content"] == "all tests passed"
        assert self_c[0]["tools"] == E.A.RUN_TOOL
    out, st = E.work_train(("implement", PROB, [good], "s/4"))
    assert [c["meta"].get("writes") for c in out if c["set"] == "agent_env"] == ["own pass"]
    plain = next(c for c in out if c["set"] == "agent_plain")
    assert plain["tools"] is None and plain["meta"]["answer"] == "own" and [m["role"] for m in plain["messages"]] == ["user", "assistant"]
    assert plain["messages"][1]["content"].startswith("```python\n") and "tool_calls" not in plain["messages"][1]
    assert not chat.render(plain).startswith("<|im_start|>system")  # no tools block
    out, st = E.work_train(("implement", PROB, [], "s/5"))
    assert [c["meta"].get("writes") for c in out if c["set"] == "agent_env"] == ["teacher"]
    assert [c["meta"]["answer"] for c in out if c["set"] == "agent_plain"] == ["teacher"]


def test_stub():
    s = E.stub("def total(xs):", "Add the positive numbers.\nReturn 0 for an empty list.")
    assert s == 'def total(xs):\n    """\n    Add the positive numbers.\n    Return 0 for an empty list.\n    """\n    raise NotImplementedError\n'
    assert E.stub("def f():", 'has """ quotes') is None and E.stub(None, "x") is None


def _reply(think="t", content="", calls=()):
    return {"think": think, "content": content, "tool_calls": list(calls), "closed": True, "tokens": 9}


def test_run_env_scores_the_files_as_left():
    broken = SOL.replace("s += x", "s -= x")
    bug = {"id": "b", "kind": "bugfix", "source": "x", "prompt": E.BUGFIX_PROMPTS[0], "files": files(broken), "hidden": None}
    impl = {"id": "i", "kind": "implement", "source": "x", "prompt": E.IMPLEMENT_PROMPTS[0],
            "files": {E.SOL: E.stub("def total(xs):", "Add."), E.TEST: E.test_file(ASSERTS[:1])}, "hidden": E.test_file(ASSERTS)}
    script = [
        [_reply(calls=[{"name": "run_tests", "arguments": {}}]),
         _reply(calls=[{"name": "write_file", "arguments": {"path": "solution.py", "content": "def total(xs):\n    return sum(xs)\n"}}])],
        [_reply(calls=[{"name": "edit_file", "arguments": {"path": "solution.py", "old": "s -= x", "new": "s += x"}}]),
         _reply(calls=[{"raw": "{broken"}])],
        [_reply(calls=[{"name": "run_tests", "arguments": {}}]), _reply(content="done")],
        [_reply(content="Fixed it.")],
    ]
    k = []

    def gen(convs, label, temperature=0.0, seed=0):
        k.append(len(convs))
        return script[len(k) - 1]
    seen = []
    rows, convs = V.run_env([bug, impl], gen, turns=6, progress=lambda turn, active, cs, st: seen.append((turn, list(active), V.tool_results(cs))))
    assert k == [2, 2, 2, 1]
    assert [(t, a) for t, a, _ in seen] == [(1, [0, 1]), (2, [0, 1]), (3, [0]), (4, [])]  # after every turn, who still works
    assert seen[1][2] == {"edits": 1, "edits with old == new": 0, "edits quoting 4+ lines": 0, "tool results": 4, "errors": 1,
                          "error: call not readable": 1, "passing test runs": 0}
    assert seen[3][2]["passing test runs"] == 1
    V.run_env([bug], lambda convs, label, temperature=0.0, seed=0: [_reply(content="done")], turns=2,
              progress=lambda *a: 1 / 0)  # a failing progress call does not end the run
    b, i = rows
    assert b["done"] and b["answered"] and b["calls"] == {"run_tests": 2, "edit_file": 1} and b["turns"] == 4
    assert i["done"] and i["hidden"] is False and i["bad_calls"] == 1 and i["turns"] == 3  # sum(xs) passes the visible test only
    assert "could not be read" in convs[1]["messages"][4]["content"]
    s = V.summary(rows)
    assert s["all"]["done"] == 1.0 and s["implement/x"]["hidden_pass"] == 0.0 and s["bugfix/x"]["mean_turns"] == 4
    for c in convs:
        chat.render(c)


def test_run_env_samples_a_task_only_after_its_test_run_failed_after_a_change():
    broken = SOL.replace("s += x", "s -= x")
    task = lambda i: {"id": i, "kind": "bugfix", "source": "x", "prompt": E.BUGFIX_PROMPTS[0], "files": files(broken), "hidden": None}  # noqa: E731
    run, right = _reply(calls=[{"name": "run_tests", "arguments": {}}]), {"path": "solution.py", "old": "s -= x", "new": "s += x"}
    wrong = {"path": "solution.py", "old": "s = 0", "new": "s = 1"}
    plan = {"a": [run, _reply(calls=[{"name": "edit_file", "arguments": right}]), run, _reply(content="Fixed.")],  # right at once
            "b": [run, _reply(calls=[{"name": "edit_file", "arguments": wrong}]), run,
                  _reply(calls=[{"name": "edit_file", "arguments": right}]), _reply(calls=[{"name": "edit_file", "arguments": {**wrong, "old": "s = 1", "new": "s = 0"}}]),
                  run, _reply(content="Fixed.")],
            "c": [run, _reply(calls=[{"name": "edit_file", "arguments": {**wrong, "old": "no such text"}}]), run, _reply(content="I give up.")]}
    seen, by_id = [], {}

    def gen(convs, label, temperature=0.0, seed=0):
        if not by_id:                                          # the first call gets every conversation, in task order
            by_id.update(zip("abc", convs))
        ids = [next(i for i in "abc" if c is by_id[i]) for c in convs]
        seen.append((label, temperature, ids))
        return [plan[i][sum(m["role"] == "assistant" for m in by_id[i]["messages"])] for i in ids]

    def run(**kw):
        seen.clear()
        by_id.clear()
        return V.run_env([task("a"), task("b"), task("c")], gen, turns=8, **kw)
    rows, _ = run(after_fail_temperature=0.7)
    a, b, c = rows
    assert a["done"] and a["sampled_from"] is None            # the test run after its change passed
    assert b["done"] and b["sampled_from"] == 4               # run, change, failing run: sampled from turn 4 on
    assert not c["done"] and c["sampled_from"] is None        # a refused edit is no change, so the failing run does not count
    assert seen[:3] == [("turn 1", 0.0, ["a", "b", "c"]), ("turn 2", 0.0, ["a", "b", "c"]), ("turn 3", 0.0, ["a", "b", "c"])]
    assert seen[3:5] == [("turn 4", 0.0, ["a", "c"]), ("turn 4 sampled", 0.7, ["b"])]
    assert all(t == 0.7 and ids == ["b"] and lab.endswith(" sampled") for lab, t, ids in seen[5:])
    s = V.summary(rows)["all"]
    assert s["failed_after_change"] == 1 and s["done_after_failing"] == 1 and s["done"] == round(2 / 3, 4)
    rows, _ = run()                                            # without the option: one greedy call a turn, the row still says where it failed
    assert all(t == 0.0 and not lab.endswith("sampled") for lab, t, _ in seen) and rows[1]["sampled_from"] == 4
    rows, _ = run(retry_temperature=0.5)                       # the older option: every turn after the first
    assert [t for _, t, _ in seen] == [0.0] + [0.5] * (len(seen) - 1)


def test_test_styles():
    plain = E.test_file(ASSERTS, style="plain")
    assert "repr(" not in plain and "assert total([1, 2, 3]) == 6\n" in plain
    check = E.test_file(ASSERTS, style="check", fn="total")
    assert "def check(candidate):\n    assert candidate([1, 2, 3]) == 6" in check and "check(total)" in check
    assert E.test_file(["g = total; assert g([1]) == 1"], style="check", fn="total") == E.test_file(["g = total; assert g([1]) == 1"], style="plain")
    broken = SOL.replace("s += x", "s -= x")
    for t in (plain, check):
        assert E.run_files({E.SOL: SOL, E.TEST: t})[0]
        ok, out = E.run_files({E.SOL: broken, E.TEST: t})
        assert not ok and out.rstrip().endswith("AssertionError")
    d = E.diagnosis(E.run_files({E.SOL: broken, E.TEST: check})[1], "total")
    assert "candidate([1, 2, 3])" in d and d.endswith("`candidate` is the function `total`.")


def test_retry_and_editfail_conversations():
    tests = E.test_file(ASSERTS)
    for seed in range(20):
        rng = random.Random(seed)
        m = E.find_bug(SOL, tests, rng)
        d = E.decoy_for(m, tests, rng)
        if d is None:
            continue
        msgs = E.retry_conv({E.TEST: tests}, m, d, E.BUGFIX_PROMPTS[0], random.Random(seed))
        if msgs is None:
            continue
        edits = [x for x in msgs if x.get("tool_calls") and x["tool_calls"][0]["name"] == "edit_file"]
        assert len(edits) == 3 and edits[0].get("no_loss") is True and "no_loss" not in edits[1] and "no_loss" not in edits[2]
        assert "did not fix it, so I undo it" in edits[1]["think"] and f"Line {m['line']}" in edits[2]["think"]
        env = E.Env({E.SOL: m["src"], E.TEST: tests})
        for x in edits:
            env.call("edit_file", x["tool_calls"][0]["arguments"])
        assert env.files[E.SOL] == SOL and msgs[-2]["content"] == E.PASS_LINE
        assert len(chat.no_loss_spans({"messages": msgs, "tools": E.TOOLS})) == 1
        break
    else:
        raise AssertionError("no retry conversation in 20 seeds")
    m = E.find_bug(SOL, tests, random.Random(3))
    msgs = E.editfail_conv({E.TEST: tests}, m, E.BUGFIX_PROMPTS[0], random.Random(3))
    calls = [(x["tool_calls"][0]["name"], x.get("no_loss")) for x in msgs if x.get("tool_calls")]
    assert calls == [("run_tests", None), ("read_file", None), ("edit_file", True), ("read_file", None), ("edit_file", None), ("run_tests", None)]
    errs = [x["content"] for x in msgs if x["role"] == "tool"]
    assert errs[2].startswith("Error: the text to replace was not found") and errs[-1] == E.PASS_LINE


def test_work_train_new_kinds():
    seen = set()
    for seed in range(6):
        for kind in ("bugfix", "bugfix_retry", "bugfix_editfail"):
            out, st = E.work_train((kind, PROB, None, f"s/{kind}/{seed}"))
            for c in out:
                seen.add((kind, c["meta"]["tests"]))
                chat.render(c)
    assert {k for k, _ in seen} == {"bugfix", "bugfix_retry", "bugfix_editfail"}
    assert {t for _, t in seen} >= {"repr", "plain"}
