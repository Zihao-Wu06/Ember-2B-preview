"""runs/mac_eval/agent_eval.py: the visible tests, the repair loop, the tool loop and the sampled pass@k, with a
scripted stand-in for the model (code runs in code_check's sandbox, so these run in WSL)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import agent_eval as A  # noqa: E402
import code_check  # noqa: E402

pytestmark = pytest.mark.skipif(not code_check.check("x = 1", {"type": "asserts", "code": "assert x == 1"})[0],
                                reason="code_check's sandbox does not run here")

PROMPT = '''def add(a: int, b: int) -> int:
    """ Add two numbers.
    >>> add(1, 2)
    3
    >>> add(0.5, 0.25)
    0.75
    >>> print("no value")
    >>> add([1], [2])
    [1, 2]
    """
'''
HE = {"task_id": "HumanEval/x", "prompt": PROMPT, "entry_point": "add",
      "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(1, 2) == 3\n"}
MB = {"task_id": 11, "text": "Write a function to double a number.", "test_list": ["assert dbl(2) == 4", "assert dbl(5) == 10"],
      "test_setup_code": ""}
GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"


def reply(content, calls=()):
    return {"think": "a thought", "content": content, "tool_calls": list(calls), "closed": True, "tokens": 5}


def item(task="humaneval"):
    d = HE if task == "humaneval" else MB
    return {"id": str(d["task_id"]), "conv": {"messages": [{"role": "user", "content": "the task"}]}, "data": d}


def scripted(script):
    """gen(convs, label) that answers the i-th call with script[i] (one reply per conversation)."""
    calls = []

    def gen(convs, label, seed=0, temperature=0.0):
        calls.append([json.loads(json.dumps(c)) for c in convs])
        gen.temps.append(temperature)
        out = script[len(calls) - 1]
        assert len(out) == len(convs), (label, len(out), len(convs))
        return out
    gen.calls, gen.temps = calls, []
    return gen


def test_doctest_examples_and_visible_tests():
    assert A.doctest_examples(PROMPT, "add") == [("add(1, 2)", "3"), ("add(0.5, 0.25)", "0.75"), ("add([1], [2])", "[1, 2]")]
    tests = A.visible_tests("humaneval", HE)
    assert [t[0] for t in tests] == ["assert add(1, 2) == 3", "assert add(0.5, 0.25) == 0.75", "assert add([1], [2]) == [1, 2]"]
    assert A.first_failure("def add(a, b):\n    return a + b\n", tests) is None
    test, err = A.first_failure("def add(a, b):\n    return a - b\n", tests)
    assert test == "assert add(1, 2) == 3" and err.endswith("AssertionError: -1")  # the value the call returned
    assert A.visible_tests("mbpp", MB)[0][0] == "assert dbl(2) == 4"
    assert A.with_actual("assert dbl(2) == 4") == "assert dbl(2) == 4, repr(dbl(2))"
    assert A.with_actual("f = mk(1); assert f('a,b') == [1, 2]") == "f = mk(1); assert f('a,b') == [1, 2], repr(f('a,b'))"
    assert A.with_actual("assert ok(3)") == "assert ok(3)"
    test, err = A.first_failure("def dbl(x):\n    return x + 1\n", A.visible_tests("mbpp", MB))
    assert err.endswith("AssertionError: 3")


def test_tool_output_is_what_the_script_printed_then_the_error():
    assert A.tool_output("print(1 + 2)") == "3"
    out = A.tool_output("print('a')\nassert 1 == 2, repr(1)")
    assert out.startswith("a\n") and out.endswith("AssertionError: 1")
    assert "not allowed" in A.tool_output("import os\nprint(os.getcwd())")
    cut = 'in.py", line 3, in <module>\n    assert f(1) == 2, repr(f(1))\n           ^^^^^^^^^\nAssertionError: 5'
    assert A.tidy(cut) == "    assert f(1) == 2, repr(f(1))\nAssertionError: 5"
    assert A.tidy("Traceback (most recent call last):\n  File \"x\", line 1\nValueError: no").startswith("Traceback")
    assert A.tidy("") == "failed" and A.tidy("NameError: x") == "NameError: x"


def test_repair_feeds_the_failing_test_back_and_stops_when_the_visible_tests_pass():
    gen = scripted([[reply(BAD), reply(GOOD)], [reply(GOOD)]])
    state, convs = A.run_repair("humaneval", [item(), item()], gen, rounds=5, feedback="visible")
    assert [s["first"] for s in state] == [False, True] and [s["final"] for s in state] == [True, True]
    assert [s["rounds"] for s in state] == [2, 1] and state[1]["stopped"] == "tests it can see pass"
    assert len(gen.calls) == 2 and len(gen.calls[1]) == 1  # only the failed item gets a second round
    second = gen.calls[1][0]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "assert add(1, 2) == 3" in second[2]["content"] and "AssertionError" in second[2]["content"]
    assert second[1]["think"] == "a thought" and second[1]["content"] == BAD
    s = A.summary("repair", state)
    assert s["pass_first_reply"] == 0.5 and s["pass_final"] == 1.0 and s["fixed_by_feedback"] == 1 and s["got_feedback"] == 1


def test_repair_runs_out_of_rounds_and_handles_a_reply_without_code():
    gen = scripted([[reply("no code here")], [reply(BAD)], [reply(BAD)]])
    state, convs = A.run_repair("humaneval", [item()], gen, rounds=3, feedback="visible")
    assert state[0]["rounds"] == 3 and state[0]["stopped"] == "out of rounds" and not state[0]["final"]
    assert "no Python code block" in gen.calls[1][0]["messages"][2]["content"]
    assert len(convs[0]["messages"]) == 6  # user, (assistant, user) x 2, assistant: no feedback after the last reply


def test_repair_oracle_feedback_uses_the_official_tests():
    # passes the docstring examples' first case only by accident; the official test fails
    tricky = "```python\ndef add(a, b):\n    return 3 if (a, b) == (1, 2) else a + b if isinstance(a, (float, list)) else 0\n```"
    gen = scripted([[reply(tricky)]])
    state, _ = A.run_repair("humaneval", [item()], gen, rounds=5, feedback="visible")
    assert state[0]["visible_ok"] and not state[0]["final"] and state[0]["rounds"] == 1
    assert A.summary("repair", state)["visible_pass_but_official_fail"] == 1
    gen = scripted([[reply(tricky)], [reply(GOOD)]])
    state, _ = A.run_repair("humaneval", [item()], gen, rounds=5, feedback="oracle")
    assert state[0]["final"] and state[0]["rounds"] == 2 and "hidden tests" in gen.calls[1][0]["messages"][2]["content"]


def test_tool_loop_runs_the_code_and_scores_the_final_answer():
    call = {"name": "python", "arguments": {"code": "print(1 + 2)"}}
    gen = scripted([[reply("", [call]), reply(GOOD)], [reply(GOOD)]])
    state, convs = A.run_tool("humaneval", [item(), item()], gen, rounds=4)
    assert [s["calls"] for s in state] == [1, 0] and [s["final"] for s in state] == [True, True]
    first = gen.calls[0][0]
    assert first["tools"] == A.RUN_TOOL and first["messages"][0]["content"].endswith(A.TOOL_NOTE)
    msgs = gen.calls[1][0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"] and msgs[1]["tool_calls"] == [call]
    assert msgs[2]["content"] == "3"
    s = A.summary("tool", state)
    assert s["made_a_call"] == 1 and s["pass_final"] == 1.0 and s["bad_calls"] == 0


def test_tool_loop_counts_bad_calls_and_blocked_scripts():
    bad = {"raw": "{not json"}
    blocked = {"name": "python", "arguments": {"code": "import os\nprint(os.listdir('/'))"}}
    gen = scripted([[reply("", [bad, blocked])], [reply(BAD)]])
    state, convs = A.run_tool("humaneval", [item()], gen, rounds=4)
    assert state[0]["bad_calls"] == 1 and state[0]["calls"] == 1 and not state[0]["final"]
    assert "not allowed" in gen.calls[1][0]["messages"][2]["content"]


def test_samples_pass_at_k():
    gen = scripted([[reply(GOOD), reply(BAD)], [reply(BAD), reply(BAD)], [reply(GOOD), reply(BAD)], [reply(GOOD), reply(GOOD)]])
    state, rows = A.run_samples("humaneval", [item(), item()], gen, k=4)
    assert [s["passes"] for s in state] == [3, 1] and len(rows) == 8
    s = A.summary("samples", state)
    assert s["pass_at_1"] == 0.5 and s["pass_at_k"] == 1.0 and s["pass_all_k"] == 0.0
    assert s["problems_by_passes"]["3"] == 1 and s["problems_by_passes"]["1"] == 1


def test_resample_takes_the_first_draw_that_passes_the_visible_tests():
    # item 0: wrong, wrong, right -> gained at draw 3; item 1: right at once; item 2 (no visible tests): first reply stands
    no_tests = {"id": "n", "conv": {"messages": [{"role": "user", "content": "the task"}]},
                "data": {**HE, "prompt": "def add(a, b):\n    pass\n"}}
    gen = scripted([[reply(BAD), reply(GOOD), reply(BAD)], [reply(BAD)], [reply(GOOD)]])
    state, convs = A.run_resample("humaneval", [item(), item(), no_tests], gen, rounds=5, temperature=0.8)
    assert gen.temps == [0.0, 0.8, 0.8] and [len(c) for c in gen.calls] == [3, 1, 1]
    assert [s["rounds"] for s in state] == [3, 1, 1] and [s["final"] for s in state] == [True, True, False]
    assert state[2]["stopped"] == "no tests to see" and convs[0]["messages"][-1]["content"] == GOOD
    assert len(gen.calls[1][0]["messages"]) == 1  # the question again, without the failed reply or any feedback
    s = A.summary("resample", state)
    assert s["pass_first_reply"] == round(1 / 3, 4) and s["pass_final"] == round(2 / 3, 4) and s["gained"] == 1 and s["lost"] == 0
    # nothing passes: the first reply is the answer
    gen = scripted([[reply(BAD)], [reply(BAD)]])
    state, convs = A.run_resample("humaneval", [item()], gen, rounds=2, temperature=0.8)
    assert state[0]["stopped"] == "out of rounds" and not state[0]["final"] and len(state[0]["draws"]) == 2


def test_retry_temperature_samples_every_reply_after_the_first():
    gen = scripted([[reply(BAD)], [reply(BAD)], [reply(GOOD)]])
    state, _ = A.run_repair("humaneval", [item()], gen, rounds=5, feedback="visible", retry_temperature=0.7)
    assert gen.temps == [0.0, 0.7, 0.7] and state[0]["final"] and state[0]["rounds"] == 3
    call = {"name": "python", "arguments": {"code": "print(1)"}}
    gen = scripted([[reply("", [call])], [reply(GOOD)]])
    A.run_tool("humaneval", [item()], gen, rounds=4, retry_temperature=0.7)
    assert gen.temps == [0.0, 0.7]
