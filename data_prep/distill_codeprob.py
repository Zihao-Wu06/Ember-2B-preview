"""The teacher writes new programming problems from snippets of real Python code .
Used by distill_gen.py (job codeprob); nothing here calls the API.

    snippet (data_prep/sft_code_seeds.py) -> teacher: one new problem, as JSON -> data/distill/codeprob.jsonl
    -> data_prep/sft_code_pool.py --gen -> data/sft/code_pool_gen.jsonl -> distill_gen.py code solves them

The problem is only a statement: the solution, and the tests that check it, come later from the code job, whose
teacher writes the tests itself for a problem that has none.
"""
from __future__ import annotations

import json
import re

PROBLEM_SYS = """You write programming problems for a Python course. The user gives you a snippet of real Python code \
and a difficulty. Write one new problem that is inspired by an idea in the snippet (a technique, a data structure, a \
kind of computation), for a student who will write the solution from scratch.

Requirements
- The problem is self-contained: everything the student needs is in the text. Never mention the snippet and never say \
"the code above".
- Choose one of two kinds.
  - function: name the function and give its exact signature, say what the arguments are and what is returned, and give \
two or three concrete examples written as `name(arguments) == result`.
  - program: say exactly what the program reads from standard input and what it prints to standard output, and give one \
example with the input and the output.
- It can be solved with the Python standard library only, in at most 40 lines.
- It is deterministic: no randomness, no current date or time, no files, no network, no user interaction other than \
standard input.
- The answer is uniquely defined. State what happens with empty input, ties and invalid values where they can occur.
- Match the difficulty. easy: a single idea in a few lines. medium: two ideas combined, or care with edge cases. hard: \
needs an algorithm or a choice of data structure.
- 60 to 250 words, plain text, no headings.

Reply with a JSON object with the keys "kind" ("function" or "program") and "problem" (the text of the problem). \
Nothing else.

Example
Difficulty: easy

Snippet:
def moving_average(values, k):
    out = []
    window_sum = 0
    for i, v in enumerate(values):
        window_sum += v
        if i >= k:
            window_sum -= values[i - k]
        if i >= k - 1:
            out.append(window_sum / k)
    return out

Reply:
{"kind": "function", "problem": "Write a function `rolling_max(values, k)` that takes a list of integers `values` and \
a positive integer `k` and returns a list with the largest value of every block of `k` consecutive elements, in order. \
If `k` is larger than the length of the list, return an empty list. Examples: `rolling_max([1, 3, 2, 5, 4], 2) == \
[3, 3, 5, 5]`, `rolling_max([4, 2, 12], 3) == [12]`, `rolling_max([1, 2], 5) == []`."}
"""

_META = re.compile(r"\b(?:snippet|the code above|given code|provided code|above code|this code)\b", re.I)
_NONDETERMINISTIC = re.compile(r"\b(?:random(?:ly)?|current (?:date|time)|today|files?|network|internet|url|http|"
                               r"database)\b", re.I)


def problem_prompt(seed: dict) -> str:
    return f"Difficulty: {seed['difficulty']}\n\nSnippet:\n```python\n{seed['code']}\n```\n\nReply:"


def check_problem(content):
    """(ok, detail, {"kind", "problem"} or None) for the teacher's reply."""
    m = re.search(r"\{.*\}", content or "", re.S)
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return False, "not JSON", None
    kind, text = d.get("kind"), d.get("problem")
    if kind not in ("function", "program") or not isinstance(text, str):
        return False, "keys are not kind and problem", None
    text = text.strip()
    if not 150 <= len(text) <= 2500:
        return False, f"{len(text)} characters", None
    if _META.search(text):
        return False, "mentions the snippet", None
    if _NONDETERMINISTIC.search(text):
        return False, "not deterministic, or uses files or a network", None
    if kind == "function" and not re.search(r"`[A-Za-z_]\w*\(", text):
        return False, "no function signature", None
    if kind == "program" and not re.search(r"standard input|stdin", text, re.I):
        return False, "does not say what the program reads", None
    return True, f"{kind}, {len(text)} characters", {"kind": kind, "problem": text}
