#!/usr/bin/env python
"""A small project as the agent's environment: solution.py and test_solution.py in the sandbox, four tools, two kinds of task, the training conversations
and the evaluation tasks.

Tools (function-calling format of the SFT data): run_tests (python test_solution.py), read_file (with line numbers),
edit_file (replace one exact piece of text), write_file. test_solution.py is read-only.

  bugfix     a verified solution with one injected bug: a single token changed (a comparison, an arithmetic
             operator, a number, and/or, True/False, min/max, a missing not), so that at least one test fails. The
             model runs the tests, reads the file, finds the line, edits it, runs the tests again and reports.
             Training: the fix is known (the changed token); a fifth of the files hold two functions, one of them broken.
  implement  the signature, a docstring and `raise NotImplementedError`; the model writes the code and runs the tests.
             Training: the model's own attempts (data/agent/attempts*.jsonl), one that passes at once or a failed one
             (marked no_loss: context, not a training target) followed by one of its own that passes; else the teacher.
Also (set agent_self): the same "failed, then a different attempt of its own that passes" with the python tool of the
earlier rounds, where the model used to send the same wrong code again; and (set agent_plain) the same tasks with no
tools offered, answered with the code itself, so that the model calls a tool only when one is offered.

Evaluation tasks come from HumanEval and MBPP's test split, which no training data comes from: bugfix with the official
tests (HumanEval, MBPP) and implement with one visible test and all of them hidden (MBPP).

    python data_prep/agent_env.py failing   -> data/agent/failing_ids.txt (greedy attempts that fail their problem)
    python data_prep/agent_env.py train     -> data/sft/conv/agent_env.jsonl, agent_self.jsonl, agent_plain.jsonl
    python data_prep/agent_env.py eval      -> data/agent/env_eval.jsonl
"""
from __future__ import annotations

import argparse
import ast
import collections
import io
import json
import multiprocessing
import os
import random
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
import agent_eval as A  # noqa: E402
import agent_traj as T  # noqa: E402
import code_check  # noqa: E402

AGENT = os.path.join(ROOT, "data", "agent")
CONV = os.path.join(ROOT, "data", "sft", "conv")
BENCH = os.path.join(ROOT, "data", "raw", "bench")
SOL, TEST = "solution.py", "test_solution.py"
PASS_LINE = "All tests passed."
LICENCE = "problems and solutions as in code_api (DeepSeek API / self-oss-instruct), attempts by the model, real runs"


def _tool(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _tool("run_tests", "Run the tests (python test_solution.py) and return what they print.", {}, []),
    _tool("read_file", "Show a file of the project with line numbers.",
          {"path": {"type": "string", "description": "The file to read."}}, ["path"]),
    _tool("edit_file", "Replace one exact piece of text in a file with new text.",
          {"path": {"type": "string", "description": "The file to edit."},
           "old": {"type": "string", "description": "The exact text to replace; it must occur once in the file."},
           "new": {"type": "string", "description": "The text to put in its place."}}, ["path", "old", "new"]),
    _tool("write_file", "Write a whole file, replacing what it held.",
          {"path": {"type": "string", "description": "The file to write."},
           "content": {"type": "string", "description": "The complete new content of the file."}}, ["path", "content"]),
]
TOOL_NAMES = [t["function"]["name"] for t in TOOLS]

BUGFIX_PROMPTS = (
    "The project has two files, solution.py and test_solution.py. The tests fail. Find the bug in solution.py and fix it so "
    "that all tests pass; do not change test_solution.py. Use the tools to run the tests and to read and edit files. When "
    "the tests pass, reply with a short summary of the fix.",
    "Some tests in test_solution.py fail because of a bug in solution.py. Find and fix the bug (test_solution.py must stay "
    "as it is), check that all tests pass, and then say what you changed.",
    "There is a bug in solution.py: test_solution.py does not pass. Use the tools to find it, fix it and run the tests "
    "again. Do not edit the test file. Finish with a one-sentence summary of the fix.",
)
IMPLEMENT_PROMPTS = (
    "The project has two files, solution.py and test_solution.py. The function in solution.py is not written yet. "
    "Implement it so that all tests pass; do not change test_solution.py. Use the tools to run the tests and to read and "
    "write files. When the tests pass, reply with a short summary.",
    "solution.py only has the signature and the description of a function. Write the function so that test_solution.py "
    "passes (do not edit the tests), run the tests to check it, and then say what you did.",
    "Implement the function in solution.py; test_solution.py checks it and must not be changed. Use the tools, run the "
    "tests until they pass, and finish with a short summary.",
)
RUN_FIRST = ("I start by running the tests to see what fails.", "First I run the tests.",
             "Let me run the tests to find the failure.")
READ_NEXT = ("I'll read solution.py to find the cause.", "Let me look at solution.py.", "I need to see the code in solution.py.")
READ_FIRST = ("I'll read solution.py first.", "Let me look at the code before running anything.")
RERUN = ("Now I run the tests again to check the fix.", "Let me run the tests again.", "I'll check the fix with the tests.")
DONE = ("All tests pass, so the bug is fixed.", "The tests pass now.", "All tests passed, so the fix works.")
IMPL_READ = ("I'll read solution.py to see what has to be written.", "Let me look at solution.py first.",
             "First I read solution.py.")
IMPL_RUN = ("Now I run the tests.", "Let me run the tests to check it.", "I'll run the tests.")
IMPL_DONE = ("All tests pass.", "The tests pass, so the function works.", "All tests passed.")
OTHER_WAY = ("I'll write it a different way.", "A different approach is needed.", "I'll try another way to do it.")


# ---- the environment ------------------------------------------------------------------------------------------------
_PATH = re.compile(r'File "[^"]*/([^/"]+\.py)"')


def tidy(text: str) -> str:
    """A traceback as the model reads it: file names without the sandbox's directory, no lines of ^^^ markers."""
    text = _PATH.sub(r'File "\1"', text or "")
    return "\n".join(line for line in text.strip().splitlines() if line.strip().strip("^~ "))


def run_files(files: dict, timeout: float = 10):
    """(passed, what the tool shows) for python test_solution.py with the project's files."""
    if any(code_check.UNSAFE.search(v) for k, v in files.items() if k != TEST):
        return False, ("Error: the code uses a module or call that is not allowed here (os, subprocess, open, eval, ...); "
                       "the tests were not run.")
    ok, out, err = code_check.run(files[TEST], timeout=timeout, files={k: v for k, v in files.items() if k != TEST},
                                  main=TEST, err_chars=1500)
    if err == "timeout":
        return False, f"The tests did not finish within {timeout:.0f} seconds."
    passed = ok and PASS_LINE in out
    text = out.strip() if passed else (out.strip() + "\n" + tidy(err)).strip()
    if len(text) > 900:
        text = text[-900:]
        text = text[text.find("\n") + 1:] if "\n" in text else text
    return passed, text or "The tests failed without printing anything."


def numbered(text: str) -> str:
    return "\n".join(f"{i:>3} | {line}" for i, line in enumerate(text.splitlines(), 1))


class Env:
    def __init__(self, files: dict, readonly=(TEST,)):
        self.files, self.readonly = dict(files), set(readonly)
        self.test_writes = 0

    def call(self, name, args) -> str:
        if name not in TOOL_NAMES:
            return f"Error: there is no tool {name!r}. The tools are run_tests, read_file, edit_file and write_file."
        if not isinstance(args, dict):
            return "Error: the arguments must be a JSON object."
        if name == "run_tests":
            return run_files(self.files)[1]
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return f"Error: {name} needs a path."
        path = path.strip().lstrip("./")
        if name in ("read_file", "edit_file") and path not in self.files:
            return f"Error: there is no file {path!r}. The files are {', '.join(sorted(self.files))}."
        if name == "read_file":
            return numbered(self.files[path])
        if path in self.readonly:
            self.test_writes += 1
            return f"Error: {path} is read-only. Change {SOL} instead."
        if name == "edit_file":
            old, new = args.get("old"), args.get("new")
            if not isinstance(old, str) or not isinstance(new, str) or not old:
                return "Error: edit_file needs the text to replace (old) and the new text (new)."
            n = self.files[path].count(old)
            if n == 0:
                return f"Error: the text to replace was not found in {path}."
            if n > 1:
                return f"Error: the text to replace occurs {n} times in {path}; give more of the surrounding code."
            self.files[path] = self.files[path].replace(old, new, 1)
            return f"Edited {path}."
        content = args.get("content")
        if not isinstance(content, str):
            return "Error: write_file needs the content of the file."
        if not path.endswith(".py"):
            return "Error: only Python files can be written."
        self.files[path] = content if content.endswith("\n") else content + "\n"
        return f"Wrote {path} ({len(content.splitlines())} lines)."

    def passed(self) -> bool:
        return run_files(self.files)[0]


def test_file(asserts: list, setup: str = "", style: str = "repr", fn: str | None = None) -> str:
    """The test file. style "repr": each assert reports the value the call returned; "plain": bare asserts (a failure
    shows no value); "check": HumanEval's form, the asserts inside check(candidate) and check(fn) at the end (needs fn
    and plain "assert fn(...) == ..." asserts; falls back to "plain")."""
    pre = f"from solution import *\n{setup.strip() + chr(10) if setup.strip() else ''}"
    if style == "check" and fn:
        pairs = T.simple(asserts)
        if pairs and all(c.startswith(fn + "(") for c, _ in pairs):
            body = "\n".join(f"    assert candidate{c[len(fn):]} == {v}" for c, v in pairs)
            return f"{pre}\n\ndef check(candidate):\n{body}\n\n\ncheck({fn})\nprint(\"{PASS_LINE}\")\n"
        style = "plain"
    body = "\n".join(A.with_actual(a) if style == "repr" else a for a in asserts)
    return f"{pre}\n{body}\n\nprint(\"{PASS_LINE}\")\n"


def pick_style(rng) -> str:
    """Half the training tests report the returned value, a quarter are bare asserts, a quarter HumanEval's check():
    an earlier fine-tune only knew the first and garbled its diagnosis on HumanEval's tests."""
    r = rng.random()
    return "repr" if r < 0.5 else "plain" if r < 0.75 else "check"


# ---- bugs ------------------------------------------------------------------------------------------------------------
SWAP = {"<": ("<=", ">"), "<=": ("<", ">="), ">": (">=", "<"), ">=": (">", "<="), "==": ("!=",), "!=": ("==",),
        "+": ("-",), "-": ("+",), "*": ("+",), "//": ("%",), "%": ("//",), "+=": ("-=",), "-=": ("+=",),
        "and": ("or",), "or": ("and",), "True": ("False",), "False": ("True",), "min": ("max",), "max": ("min",)}
_HEADER = ("def ", "async def ", "class ", "import ", "from ", "@")


def mutants(src: str, first_line: int = 1) -> list:
    """Single-token changes of the code from line first_line on that still parse:
    [{"src", "line", "orig_line", "new_line", "orig_tok", "new_tok"}]; header lines (def, import, ...) are left alone."""
    lines = src.splitlines(keepends=True)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []
    out = []
    same_line = [t for t in toks if t.start[0] == t.end[0] and t.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                                                                              tokenize.DEDENT, tokenize.COMMENT)]
    for k, t in enumerate(toks):
        (row, c0), (row2, c1) = t.start, t.end
        if row != row2 or row < first_line or row > len(lines):
            continue
        line = lines[row - 1]
        if line.lstrip().startswith(_HEADER):
            continue
        if t.type in (tokenize.OP, tokenize.NAME) and t.string in SWAP:
            reps = SWAP[t.string]
        elif t.type == tokenize.NUMBER and t.string.isdigit():
            k = int(t.string)
            reps = (str(k + 1),) + ((str(k - 1),) if k > 0 else ())
        elif t.type == tokenize.NAME and t.string == "not":
            reps = ("",)
        else:
            continue
        for r in reps:
            new_line = line[:c0] + line[c1:].lstrip(" ") if r == "" else line[:c0] + r + line[c1:]
            new_src = "".join(lines[:row - 1] + [new_line] + lines[row:])
            try:
                ast.parse(new_src)
            except SyntaxError:
                continue
            # the changed token with its neighbours on the line, so that "x + 1 should be x - 1" names the right "+"
            j = same_line.index(t)
            a = same_line[j - 1].start[1] if j and same_line[j - 1].start[0] == row else c0
            b = same_line[j + 1].end[1] if j + 1 < len(same_line) and same_line[j + 1].start[0] == row else c1
            ctx_orig = line[a:b]
            ctx_new = line[a:c0] + (line[c1:b].lstrip(" ") if r == "" else r + line[c1:b])
            out.append({"src": new_src, "line": row, "orig_line": line.rstrip("\n"), "new_line": new_line.rstrip("\n"),
                        "orig_tok": t.string, "new_tok": r, "ctx_orig": ctx_orig.strip(), "ctx_new": ctx_new.strip()})
    return out


def edit_args(m: dict):
    """(old, new) for edit_file that undo a mutant: the stripped line if it occurs once in the broken file, else the
    whole line with its indentation; None if neither is unique."""
    for old, new in ((m["new_line"].strip(), m["orig_line"].strip()), (m["new_line"], m["orig_line"])):
        if old and m["src"].count(old) == 1:
            return old, new
    return None


def find_bug(src: str, tests: str, rng, first_line: int = 1, tries: int = 8, extra: dict | None = None):
    """A mutant of src that makes the tests fail (not by a timeout), with the failing output; None if none of `tries`
    random candidates does. extra: other files of the project."""
    cands = mutants(src, first_line)
    rng.shuffle(cands)
    for m in cands[:tries]:
        if edit_args(m) is None:
            continue
        ok, out = run_files({SOL: m["src"], TEST: tests, **(extra or {})})
        if not ok and "did not finish" not in out and "not allowed" not in out:
            return {**m, "output": out}
    return None


def locate(m: dict) -> str:
    what = f"`{m['ctx_new']}` should be `{m['ctx_orig']}`"
    if m["new_tok"] == "":
        what += " (a `not` is missing)"
    return f"Line {m['line']}, `{m['new_line'].strip()}`, is the bug: {what}."


_SOL_LINE = re.compile(r'File "solution\.py", line (\d+)')


def diagnosis(out: str, fn: str | None = None) -> str:
    """One sentence on a failed run, from what the output shows."""
    lines = [line for line in out.strip().splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    if "did not finish" in out:
        return "The tests did not finish in time, so the code loops for too long."
    if last.startswith("AssertionError"):
        d = T.diagnosis(out)
        if fn and "candidate(" in out:
            d += f" In the tests, `candidate` is the function `{fn}`."
        return d
    where = _SOL_LINE.findall(out)
    at = f" in line {where[-1]} of solution.py" if where else ""
    return f"The tests stop with `{last[:160]}`{at}." if last else "The tests failed."


# ---- training conversations -------------------------------------------------------------------------------------------
class Conv:
    def __init__(self, files: dict, prompt: str):
        self.env, self.msgs = Env(files), [{"role": "user", "content": prompt}]

    def step(self, think, name, args, no_loss=False):
        out = self.env.call(name, args)
        m = {"role": "assistant", "think": think, "content": "", "tool_calls": [{"name": name, "arguments": args}]}
        if no_loss:
            m["no_loss"] = True
        self.msgs += [m, {"role": "tool", "content": out}]
        return out

    def answer(self, think, content):
        self.msgs.append({"role": "assistant", "think": think, "content": content})


def bugfix_conv(files: dict, m: dict, prompt: str, rng, fn: str | None = None):
    old_new = edit_args(m)
    c = Conv({**files, SOL: m["src"]}, prompt)
    if rng.random() < 0.2:
        c.step(rng.choice(READ_FIRST), "read_file", {"path": SOL})
        out = c.step("Now I run the tests to see which one fails.", "run_tests", {})
        fix_think = f"{diagnosis(out, fn)} {locate(m)}"
    else:
        out = c.step(rng.choice(RUN_FIRST), "run_tests", {})
        c.step(f"{diagnosis(out, fn)} {rng.choice(READ_NEXT)}", "read_file", {"path": SOL})
        fix_think = locate(m)
    if c.step(fix_think, "edit_file", {"path": SOL, "old": old_new[0], "new": old_new[1]}) != f"Edited {SOL}.":
        return None
    if c.step(rng.choice(RERUN), "run_tests", {}) != PASS_LINE:
        return None
    c.answer(rng.choice(DONE), f"The bug was in solution.py, line {m['line']}: `{m['new_line'].strip()}` should be "
                               f"`{m['orig_line'].strip()}`. I fixed it, and all tests pass now.")
    return c.msgs


def pair(src: str, a: str, b: str):
    """(old, new) for edit_file replacing line a by line b in src: stripped if a.strip() occurs once, else the whole
    lines if a occurs once, else None."""
    if a.strip() and src.count(a.strip()) == 1:
        return a.strip(), b.strip()
    if a and src.count(a) == 1:
        return a, b
    return None


REREAD = ("I read solution.py again.", "Let me look at the code again.", "I'll read the file again to find the real cause.")


def decoy_for(m: dict, tests: str, rng, tries: int = 10):
    """A plausible wrong fix for the broken file m["src"]: another single-token change on a different line after which
    the tests still fail (not by a timeout)."""
    cands = [d for d in mutants(m["src"]) if d["line"] != m["line"]]
    rng.shuffle(cands)
    for d in cands[:tries]:
        if pair(m["src"], d["orig_line"], d["new_line"]) is None or pair(d["src"], d["new_line"], d["orig_line"]) is None:
            continue
        ok, out = run_files({SOL: d["src"], TEST: tests})
        if not ok and "did not finish" not in out and "not allowed" not in out:
            return {**d, "output": out}
    return None


def retry_conv(files: dict, m: dict, d: dict, prompt: str, rng, fn: str | None = None):
    """A wrong fix first (not a training target), the tests still fail, the change is undone, the file read again and
    the real bug fixed: an earlier fine-tune repeated the same wrong edit instead (141 of 176 unfinished bug fixes)."""
    ap, un, fx = pair(m["src"], d["orig_line"], d["new_line"]), pair(d["src"], d["new_line"], d["orig_line"]), \
        pair(m["src"], m["new_line"], m["orig_line"])
    if None in (ap, un, fx):
        return None
    c = Conv({**files, SOL: m["src"]}, prompt)
    out = c.step(rng.choice(RUN_FIRST), "run_tests", {})
    c.step(f"{diagnosis(out, fn)} {rng.choice(READ_NEXT)}", "read_file", {"path": SOL})
    wrong = f"Line {d['line']}, `{d['orig_line'].strip()}`, looks wrong: `{d['ctx_orig']}` should be `{d['ctx_new']}`."
    if c.step(wrong, "edit_file", {"path": SOL, "old": ap[0], "new": ap[1]}, no_loss=True) != f"Edited {SOL}.":
        return None
    out2 = c.step(rng.choice(RERUN), "run_tests", {})
    if out2 == PASS_LINE:
        return None
    again = diagnosis(out2, fn)
    if c.step(f"The tests still fail. {again} My change to line {d['line']} did not fix it, so I undo it.",
              "edit_file", {"path": SOL, "old": un[0], "new": un[1]}) != f"Edited {SOL}.":
        return None
    c.step(rng.choice(REREAD), "read_file", {"path": SOL})
    if c.step(locate(m), "edit_file", {"path": SOL, "old": fx[0], "new": fx[1]}) != f"Edited {SOL}.":
        return None
    if c.step(rng.choice(RERUN), "run_tests", {}) != PASS_LINE:
        return None
    c.answer(rng.choice(DONE), f"The bug was in solution.py, line {m['line']}: `{m['new_line'].strip()}` should be "
                               f"`{m['orig_line'].strip()}`. My first change, to line {d['line']}, did not help and I undid "
                               f"it. All tests pass now.")
    return c.msgs


def editfail_conv(files: dict, m: dict, prompt: str, rng, fn: str | None = None):
    """The first edit quotes text that is not in the file (not a training target); the model reads the file again and
    copies the line exactly: an earlier fine-tune rewrote the whole file instead and lost the function (62 of 176)."""
    fx = pair(m["src"], m["new_line"], m["orig_line"])
    if fx is None:
        return None
    bad = m["orig_line"].strip() if rng.random() < 0.5 else re.sub(r"\s*([-+*/%<>=!]+)\s*", r"\1", m["new_line"].strip())
    if not bad or bad in m["src"]:
        bad = m["orig_line"].strip()
    if not bad or bad in m["src"]:
        return None
    c = Conv({**files, SOL: m["src"]}, prompt)
    out = c.step(rng.choice(RUN_FIRST), "run_tests", {})
    c.step(f"{diagnosis(out, fn)} {rng.choice(READ_NEXT)}", "read_file", {"path": SOL})
    if not c.step(locate(m), "edit_file", {"path": SOL, "old": bad, "new": fx[1]}, no_loss=True).startswith("Error: the text"):
        return None
    c.step("The text to replace was not found, so what I quoted is not exactly what the file says. I read the file again "
           "and copy the line as it is.", "read_file", {"path": SOL})
    if c.step(f"Line {m['line']} is exactly `{m['new_line'].strip()}`; I replace that.", "edit_file",
              {"path": SOL, "old": fx[0], "new": fx[1]}) != f"Edited {SOL}.":
        return None
    if c.step(rng.choice(RERUN), "run_tests", {}) != PASS_LINE:
        return None
    c.answer(rng.choice(DONE), f"The bug was in solution.py, line {m['line']}: `{m['new_line'].strip()}` should be "
                               f"`{m['orig_line'].strip()}`. I fixed it, and all tests pass now.")
    return c.msgs


def stub(signature: str, statement: str):
    if not signature or '"""' in statement or "\\" in statement:
        return None
    doc = "\n".join(("    " + line).rstrip() for line in statement.strip().splitlines())
    return f'{signature}\n    """\n{doc}\n    """\n    raise NotImplementedError\n'


def implement_conv(files: dict, prompt: str, fn: str, writes: list, rng):
    """writes: [(think, code, no_loss)]: each written in turn and tested; the conversation needs the last to pass."""
    c = Conv(files, prompt)
    c.step(rng.choice(IMPL_READ), "read_file", {"path": SOL})
    out = None
    for k, (think, code, skip) in enumerate(writes):
        if k:
            think = f"{diagnosis(out)} {rng.choice(OTHER_WAY)} {think}"
        c.step(think, "write_file", {"path": SOL, "content": code}, no_loss=skip)
        out = c.step(rng.choice(IMPL_RUN), "run_tests", {})
        if (out == PASS_LINE) != (k == len(writes) - 1):
            return None
    c.answer(rng.choice(IMPL_DONE), f"I implemented `{fn}` in solution.py, and all tests pass.")
    return c.msgs


def code_of(att):
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", att.get("content") or "", re.S)
    return m.group(1).rstrip() + "\n" if m and att.get("closed", True) and (att.get("think") or "").strip() else None


def self_tool_conv(p: dict, fail: str, ok: str, fail_think: str, ok_think: str, rng):
    """The python tool of the earlier rounds: the model's failed attempt (no_loss), the real error, a different attempt
    of its own that passes, the answer; (messages or None, task form)."""
    task, shown, form = T.make_task(p, rng)
    res = A.tool_output(T.script(fail, shown))
    if res.strip() == T.OK_LINE or A.tool_output(T.script(ok, shown)).strip() != T.OK_LINE:
        return None, form
    call = lambda code: [{"name": A.TOOL_NAME, "arguments": {"code": T.script(code, shown)}}]  # noqa: E731
    return [{"role": "user", "content": task + A.TOOL_NOTE},
            {"role": "assistant", "think": fail_think.strip(), "content": "", "tool_calls": call(fail), "no_loss": True},
            {"role": "tool", "content": res},
            {"role": "assistant", "think": f"{T.diagnosis(res)} {rng.choice(OTHER_WAY)} {ok_think.strip()}", "content": "",
             "tool_calls": call(ok)},
            {"role": "tool", "content": T.OK_LINE},
            {"role": "assistant", "think": rng.choice(T.PASS_THINK), "content": T.block(ok)}], form


def work_train(job):
    """One problem -> (conversations, counters)."""
    kind, p, extra, seed = job
    rng = random.Random(seed)
    st, out = collections.Counter(), []
    sol = p["solution"].rstrip() + "\n"
    tests = test_file(p["asserts"])
    if not run_files({SOL: sol, TEST: tests})[0]:
        st[f"{kind}: solution fails as a project file"] += 1
        return out, st

    def add(name, msgs, meta):
        if msgs is None:
            st[f"{kind}: not built"] += 1
            return
        st[f"{kind}: built"] += 1
        out.append({"id": f"{name}/{p['id']}/{kind}", "set": name, "licence": LICENCE, "messages": msgs,
                    "tools": {"agent_env": TOOLS, "agent_self": A.RUN_TOOL}.get(name), "meta": {"kind": kind, "source": p["source"], **meta}})

    fn = p.get("fn")
    if kind in ("bugfix", "bugfix2", "bugfix_retry", "bugfix_editfail"):
        style = pick_style(rng) if kind != "bugfix2" else "repr"
        tests = test_file(p["asserts"], style=style, fn=fn)
        files, first, other = {}, 1, None
        if kind == "bugfix2":
            q = extra
            qsol = q["solution"].rstrip() + "\n"
            names = lambda s: {n.name for n in ast.parse(s).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}  # noqa: E731
            if names(sol) & names(qsol):
                st["bugfix2: names collide"] += 1
                return out, st
            broken_first = rng.random() < 0.5
            tests = test_file(p["asserts"] + q["asserts"]) if rng.random() < 0.5 else test_file(q["asserts"] + p["asserts"])
            whole = sol + "\n\n" + qsol if broken_first else qsol + "\n\n" + sol
            if not run_files({SOL: whole, TEST: tests})[0]:
                st["bugfix2: combined file fails"] += 1
                return out, st
            first = 1 if broken_first else len(qsol.splitlines()) + 3
            last = len(sol.splitlines()) if broken_first else None
            cands = [m for m in mutants(whole, first) if last is None or m["line"] <= last]
            rng.shuffle(cands)
            m = None
            for c in cands[:8]:
                if edit_args(c) is None:
                    continue
                ok, o = run_files({SOL: c["src"], TEST: tests})
                if not ok and "did not finish" not in o and "not allowed" not in o:
                    m = {**c, "output": o}
                    break
        else:
            m = find_bug(sol, tests, rng)
        if m is None:
            st[f"{kind}: no failing bug found"] += 1
            return out, st
        meta = {"bug": f"{m['orig_tok']} -> {m['new_tok'] or '(removed)'}", "functions": 2 if kind == "bugfix2" else 1,
                "tests": style}
        prompt = rng.choice(BUGFIX_PROMPTS)
        if kind == "bugfix_retry":
            d = decoy_for(m, tests, rng)
            if d is None:
                st["bugfix_retry: no wrong fix that keeps the tests failing"] += 1
                return out, st
            add("agent_env", retry_conv({TEST: tests}, m, d, prompt, rng, fn), meta)
        elif kind == "bugfix_editfail":
            add("agent_env", editfail_conv({TEST: tests}, m, prompt, rng, fn), meta)
        else:
            add("agent_env", bugfix_conv({TEST: tests}, m, prompt, rng, fn if kind == "bugfix" else None), meta)
        return out, st

    # implement and self: the model's own attempts
    s = stub(p.get("signature"), p["prompt"])
    own = [(code_of(a), a) for a in extra]
    own = [(c, a) for c, a in own if c]
    passing = [(c, a) for c, a in own if run_files({SOL: c, TEST: tests})[0]]
    failing = [(c, a) for c, a in own if not run_files({SOL: c, TEST: tests})[0]]
    if kind == "implement" and s:
        files = {SOL: s, TEST: test_file(p["asserts"], style=pick_style(rng), fn=fn)}
        prompt = rng.choice(IMPLEMENT_PROMPTS)
        if failing and passing:
            (fc, fa), (pc, pa) = failing[0], passing[0]
            add("agent_env", implement_conv(files, prompt, p["fn"], [(fa["think"], fc, True), (pa["think"], pc, False)], rng),
                {"writes": "own fail, own pass"})
        elif passing:
            pc, pa = passing[0]
            add("agent_env", implement_conv(files, prompt, p["fn"], [(pa["think"], pc, False)], rng), {"writes": "own pass"})
        elif failing and rng.random() < 0.5:
            fc, fa = failing[0]
            add("agent_env", implement_conv(files, prompt, p["fn"], [(fa["think"], fc, True), (p["explanation"], sol, False)], rng),
                {"writes": "own fail, teacher"})
        else:
            add("agent_env", implement_conv(files, prompt, p["fn"], [(p["explanation"], sol, False)], rng), {"writes": "teacher"})
    if kind == "implement":
        # no tools offered: the answer is the code itself. An earlier fine-tune called the python tool in 65 of 200 MBPP
        # first replies although the question offered no tools: its only tool-free replies to the "Your code should
        # pass this test" form were the masked failed attempts of agent_fix, which teach nothing
        task, shown, form = T.make_task(p, rng)
        right = [(c, a) for c, a in passing if A.tool_output(T.script(c, p["asserts"])).strip() == T.OK_LINE]
        code, think, who = (right[0][0], right[0][1]["think"], "own") if right else (sol, p["explanation"], "teacher")
        add("agent_plain", [{"role": "user", "content": task}, {"role": "assistant", "think": think.strip(), "content": T.block(code)}],
            {"form": form, "answer": who})
    if kind == "implement" and failing and passing:
        # the same pair with the python tool: the first attempt fails the tests the task shows, the second passes all
        (fc, fa), (pc, pa) = failing[0], passing[0]
        if A.tool_output(T.script(pc, p["asserts"])).strip() == T.OK_LINE:
            msgs, form = self_tool_conv(p, fc, pc, fa["think"], pa["think"], rng)
            add("agent_self", msgs, {"form": form})
    return out, st


# ---- evaluation tasks -------------------------------------------------------------------------------------------------
def eval_tasks(seed: int = 0):
    rng = random.Random(seed)
    tasks, st = [], collections.Counter()
    he = [json.loads(line) for line in open(os.path.join(BENCH, "humaneval", "openai_humaneval", "test-00000-of-00001.jsonl"))]
    for d in he:
        src = d["prompt"] + d["canonical_solution"]
        tests = f"from solution import *\n{d['test']}\n\ncheck({d['entry_point']})\nprint(\"{PASS_LINE}\")\n"
        if not run_files({SOL: src, TEST: tests})[0]:
            st["humaneval: canonical fails as a project file"] += 1
            continue
        m = find_bug(src, tests, random.Random(f"{seed}/{d['task_id']}"), first_line=len(d["prompt"].splitlines()) + 1)
        if m is None:
            st["humaneval: no failing bug"] += 1
            continue
        tasks.append({"id": f"bugfix/{d['task_id']}", "kind": "bugfix", "source": "humaneval", "prompt": BUGFIX_PROMPTS[0],
                      "files": {SOL: m["src"], TEST: tests}, "hidden": None, "bug": {k: m[k] for k in ("line", "orig_line", "new_line")}})
    mb = [json.loads(line) for line in open(os.path.join(BENCH, "mbpp", "full", "test-00000-of-00001.jsonl"))]
    for d in mb:
        src = "\n".join(line.rstrip() for line in d["code"].replace("\r\n", "\n").replace("\r", "\n").splitlines()) + "\n"
        setup = d.get("test_setup_code") or ""
        allt = test_file(d["test_list"], setup)
        if not run_files({SOL: src, TEST: allt})[0]:
            st["mbpp: reference fails as a project file"] += 1
            continue
        m = find_bug(src, allt, random.Random(f"{seed}/mbpp/{d['task_id']}"))
        if m is not None:
            tasks.append({"id": f"bugfix/mbpp/{d['task_id']}", "kind": "bugfix", "source": "mbpp", "prompt": BUGFIX_PROMPTS[0],
                          "files": {SOL: m["src"], TEST: allt}, "hidden": None, "bug": {k: m[k] for k in ("line", "orig_line", "new_line")}})
        fn = re.match(r"\s*assert\s+(\w+)\s*\(", d["test_list"][0])
        node = None
        if fn:
            node = next((n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == fn.group(1)), None)
        s = stub(f"def {node.name}({ast.unparse(node.args)}):" if node else None, d["text"])
        if s:
            tasks.append({"id": f"implement/mbpp/{d['task_id']}", "kind": "implement", "source": "mbpp", "prompt": IMPLEMENT_PROMPTS[0],
                          "files": {SOL: s, TEST: test_file(d["test_list"][:1], setup)}, "hidden": allt, "bug": None})
        else:
            st["mbpp: no stub"] += 1
    rng.shuffle(tasks)
    return tasks, st


# ---- commands ---------------------------------------------------------------------------------------------------------
def load_pool(path):
    return {p["id"]: p for p in map(json.loads, open(path, encoding="utf-8"))}


def load_attempts(paths):
    att = collections.defaultdict(list)
    for path in paths:
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                a = json.loads(line)
                att[a["id"]].append(a)
    return att


def _failing(job):
    p, a = job
    c = code_of(a)
    return p["id"] if c and A.tool_output(T.script(c, p["asserts"])).strip() != T.OK_LINE else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("failing", "train", "eval"))
    ap.add_argument("--pool", default=os.path.join(AGENT, "pool_v2.jsonl"))
    ap.add_argument("--attempts", default=os.path.join(AGENT, "attempts.jsonl") + "," + os.path.join(AGENT, "attempts_samples.jsonl"))
    ap.add_argument("--conv", default=CONV)
    ap.add_argument("--bugfix", type=int, default=14000)
    ap.add_argument("--bugfix2", type=int, default=3500)
    ap.add_argument("--bugfix-retry", type=int, default=0, help="bug fixes with a wrong first fix that is undone")
    ap.add_argument("--bugfix-editfail", type=int, default=0, help="bug fixes whose first edit quotes text not in the file")
    ap.add_argument("--implement", type=int, default=9000, help="problems for implement tasks (those with attempts first)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pool = load_pool(args.pool)
    att = load_attempts(args.attempts.split(","))
    if args.what == "failing":
        jobs = [(pool[i], next(a for a in att[i] if a["pass"] == "greedy")) for i in sorted(att)
                if i in pool and any(a["pass"] == "greedy" for a in att[i])]
        with multiprocessing.Pool(args.workers) as mp:
            ids = [i for i in mp.imap_unordered(_failing, jobs, chunksize=16) if i]
        path = args.out or os.path.join(AGENT, "failing_ids.txt")
        open(path, "w").write("\n".join(sorted(ids)) + "\n")
        print(f"AGENT_FAILING_DONE {path}: {len(ids)} of {len(jobs)} greedy attempts fail their problem", flush=True)
        return
    if args.what == "eval":
        tasks, st = eval_tasks(args.seed)
        path = args.out or os.path.join(AGENT, "env_eval.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"AGENT_ENV_EVAL_DONE {path}: {dict(collections.Counter((t['kind'], t['source']) for t in tasks))}; {dict(st)}", flush=True)
        return
    rng = random.Random(args.seed)
    ids = sorted(pool)
    rng.shuffle(ids)
    sampled = [i for i in ids if i in att and any(a["pass"] != "greedy" for a in att[i])]
    with_att = sampled + [i for i in ids if i in att and i not in set(sampled)]
    impl = with_att[:args.implement]
    impl += [i for i in ids if i not in att][:args.implement - len(impl)]
    jobs = [("implement", pool[i], att.get(i, []), f"{args.seed}/impl/{i}") for i in impl]
    taken = set(impl)
    bug_ids = [i for i in ids if i not in taken] + impl  # problems of the implement tasks only if the rest runs out
    for i in bug_ids[:args.bugfix]:
        jobs.append(("bugfix", pool[i], None, f"{args.seed}/bug/{i}"))
    rest = bug_ids[args.bugfix:]
    for k in range(min(args.bugfix2, len(rest) // 2)):
        jobs.append(("bugfix2", pool[rest[2 * k]], pool[rest[2 * k + 1]], f"{args.seed}/bug2/{rest[2 * k]}"))
    rest = rest[2 * min(args.bugfix2, len(rest) // 2):] + bug_ids[:args.bugfix]  # reuse problems if the pool runs out
    for i in rest[:args.bugfix_retry]:
        jobs.append(("bugfix_retry", pool[i], None, f"{args.seed}/retry/{i}"))
    for i in rest[args.bugfix_retry:args.bugfix_retry + args.bugfix_editfail]:
        jobs.append(("bugfix_editfail", pool[i], None, f"{args.seed}/editfail/{i}"))
    print(f"{len(jobs)} jobs: {dict(collections.Counter(j[0] for j in jobs))}", flush=True)
    sets, st = {"agent_env": [], "agent_self": [], "agent_plain": []}, collections.Counter()
    with multiprocessing.Pool(args.workers) as mp:
        for k, (convs, s) in enumerate(mp.imap_unordered(work_train, jobs, chunksize=8)):
            st.update(s)
            for c in convs:
                sets[c["set"]].append(c)
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(jobs)} jobs, {dict((n, len(v)) for n, v in sets.items())}", flush=True)
    summary = {"jobs": len(jobs), "counts": dict(st)}
    for name, convs in sets.items():
        convs.sort(key=lambda c: c["id"])
        rng.shuffle(convs)
        path = os.path.join(args.conv, name + ".jsonl")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for c in convs:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
        summary[name] = {"conversations": len(convs),
                         "kinds": dict(collections.Counter(f"{c['meta']['kind']}/{c['meta'].get('writes', c['meta'].get('form', ''))}"
                                                           f"{'/' + c['meta']['answer'] if 'answer' in c['meta'] else ''}" for c in convs)),
                         "tests": dict(collections.Counter(c["meta"].get("tests", "") for c in convs))}
    with open(os.path.join(AGENT, "env_stats.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("AGENT_ENV_TRAIN_DONE " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
