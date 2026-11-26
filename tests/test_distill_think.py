"""data_prep/distill_think.py: prompts and checks for the teacher jobs that write reasoning for existing data."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import distill_think as DT  # noqa: E402
from moe_llm import chat  # noqa: E402

CONV = {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        "messages": [{"role": "system", "content": "Be brief."},
                     {"role": "user", "content": "Weather in Paris?"},
                     {"role": "assistant", "think": None, "content": "",
                      "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
                     {"role": "tool", "content": '{"temp": 21}'},
                     {"role": "assistant", "think": None, "content": "It is 21 degrees in Paris."}]}


def test_backfill():
    text, n = DT.plain(CONV)
    assert n == 2 and text.startswith("[system]\nBe brief.\nAvailable functions:\n{")
    assert '[assistant reply 1]\n<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>' in text
    assert "[tool result]\n{\"temp\": 21}\n[assistant reply 2]\nIt is 21 degrees in Paris." in text
    assert "2 assistant replies" in DT.backfill_prompt(CONV)
    good = ["The user wants the weather, so I call get_weather with the city Paris.",
            "The result says 21 degrees, which I report in one sentence."]
    ok, _, thoughts = DT.check_backfill(CONV, "Here you go:\n" + json.dumps(good))
    assert ok and thoughts == good
    assert not DT.check_backfill(CONV, json.dumps(good[:1]))[0]          # one thought missing
    assert not DT.check_backfill(CONV, "no list")[0]
    assert not DT.check_backfill(CONV, json.dumps(["Too short.", good[1]]))[0]
    assert not DT.check_backfill(CONV, json.dumps(["Okay, " + good[0], good[1]]))[0]
    new = DT.with_thoughts(CONV, thoughts)
    assert [m.get("think") for m in new["messages"] if m["role"] == "assistant"] == good
    assert CONV["messages"][2]["think"] is None and new["messages"][2]["tool_calls"] == CONV["messages"][2]["tool_calls"]
    assert good[0] in chat.render(new) and chat.render(new).replace(good[0], "").replace(good[1], "") == chat.render(CONV)


def test_condense():
    conv = {"messages": [{"role": "user", "content": "What is 2 + 2?"},
                         {"role": "assistant", "think": "Let me see. " + "Two plus two is four. " * 40, "content": "4"}]}
    assert "Reasoning trace:\nLet me see." in DT.condense_prompt(conv) and "Final answer:\n4" in DT.condense_prompt(conv)
    short = "Adding the two numbers gives four. " * 5
    ok, _, new = DT.check_condense(conv, short + "\n")
    assert ok and new == short.strip()
    assert not DT.check_condense(conv, "Four.")[0]                       # almost empty
    assert not DT.check_condense(conv, conv["messages"][1]["think"])[0]  # not shorter
    assert not DT.check_condense(conv, "Wait, " + short)[0]


def test_solve_prompt_and_parts():
    assert "The answer is $\\boxed{...}$." in DT.SOLVE_SYS and "####" not in DT.SOLVE_SYS
    assert DT.Q_ANS.format(i=3, q="Who?").startswith("Answer question 3: Who?")
    m = DT.parts("math", "One pencil costs $0.40.\nSo 10 pencils cost $4.00.\nThe answer is $\\boxed{4}$.")
    assert m == {"working": "One pencil costs $0.40.\nSo 10 pencils cost $4.00.",
                 "answer": "The answer is $\\boxed{4}$.", "style": None}
    same = DT.parts("math", "Ten pencils cost 10 * 0.40 = 4. The answer is $\\boxed{4}$.")
    assert same["working"] == "Ten pencils cost 10 * 0.40 = 4." and same["answer"] == "The answer is $\\boxed{4}$."
    assert DT.parts("math", "no box here")["style"] == "no math form"
    flagged = DT.parts("math", "We need answer. Need solve the pencils problem first. The answer is $\\boxed{4}$.")
    assert flagged["style"].startswith("shorthand")
    c = DT.parts("code", "The sum of squares is one pass over the list.\n```python\ndef f(x):\n    return x\n```")
    assert c["working"] == "The sum of squares is one pass over the list." and c["answer"].startswith("```python")
    assert c["style"] is None
    assert DT.parts("code", "```python\nx = 1\n```")["style"] == "working too short"
    r = DT.parts("reading", "Where: Chapter I, the first paragraph.\nAnswer: Rome.")
    assert r == {"working": "Chapter I, the first paragraph.", "answer": "Rome.", "style": None}
    assert DT.parts("reading", "Where: X\nAnswer:")["style"] == "no reading form"
