"""Prompts and checks for the two teacher jobs that write reasoning for existing data:

  backfill   public tool-calling and chat conversations have no reasoning; the teacher writes the short thought behind
             every assistant reply, the replies themselves stay as they are
  condense   OpenThoughts' reasoning is long and full of detours; the teacher rewrites it as a short direct thought,
             question and answer stay as they are

Used by distill_pilot.py (a few samples, to measure price and quality) and distill_gen.py (the full runs). Nothing
here calls the API.
"""
from __future__ import annotations

import json
import re

BACKFILL_SYS = """You are preparing training data for a small language model that writes a short thought before \
every reply. Below is a conversation in which the assistant's replies are already written. For every assistant \
reply, write the thought the assistant had just before writing it.

Rules:
- One or two plain sentences, at most 40 words, in the first person.
- Say what the user wants and what you decide to do. For a function call: which function, and where each argument \
value comes from. If no available function fits, or a required value is missing, say that. If the conversation lists \
no functions, do not mention functions or tools at all.
- Do not quote the reply and do not mention that a reply exists. Do not add facts that are not in the conversation.
- Vary the wording. Do not begin with "Okay", "Hmm" or "Alright".
- Output only a JSON list of strings, one string per assistant reply, in order. Nothing else.

Example conversation:
[system]
You can call these functions: get_weather(city, date), convert_currency(amount, from, to)
[user]
What will the weather be like in Sydney tomorrow?
[assistant reply 1]
<tool_call>{"name": "get_weather", "arguments": {"city": "Sydney", "date": "tomorrow"}}</tool_call>
[tool result]
{"forecast": "sunny", "high_c": 24}
[assistant reply 2]
Tomorrow in Sydney it will be sunny with a high of 24 degrees Celsius.
[user]
Thanks! Can you also book me a table for dinner?
[assistant reply 3]
I'm sorry, I can't book tables. I can only look up the weather and convert currencies.

Example output:
["This is a weather question, so get_weather fits: the city is Sydney and the date is tomorrow, both given by the \
user.", "The forecast came back: sunny with a high of 24, which I should report plainly.", "Booking a table is not \
something any of my functions can do, so I should say so and name what I can do."]
"""

CONDENSE_SYS = """You are preparing training data for a small language model that reasons before it answers. Below \
are a problem, a long reasoning trace and the final answer. Rewrite the reasoning as the short, direct working of \
someone who solves the problem without detours.

Rules:
- Keep every step that the final answer depends on, in order, and any check that actually matters.
- Leave out restating the problem, false starts, hesitation and repeated checks. Where the trace makes a mistake and \
corrects it, keep only the corrected step.
- Do not add facts, do not change the method, and reach exactly the conclusion of the final answer.
- Plain sentences, no headings, no "Wait", "Hmm", "Okay" or "Let me". At most a third of the length of the trace, \
and never more than 250 words.
- Output only the rewritten reasoning. Nothing else.
"""
_BAD_OPEN = re.compile(r"^(?:Okay|Hmm+|Alright|Wait)\b", re.I)


def plain(conv: dict) -> tuple[str, int]:
    """The conversation as labelled text with numbered assistant replies, and the number of replies."""
    parts, n = [], 0
    tools = conv.get("tools")
    msgs = conv["messages"]
    if msgs and msgs[0]["role"] == "system" or tools:
        body = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        if tools:
            body += ("\n" if body else "") + "Available functions:\n" + "\n".join(json.dumps(t, ensure_ascii=False)
                                                                                 for t in tools)
        parts.append("[system]\n" + body.strip())
    for m in msgs:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            parts.append("[user]\n" + (m["content"] or "").strip())
        elif m["role"] == "tool":
            parts.append("[tool result]\n" + (m["content"] or "").strip())
        else:
            n += 1
            body = (m.get("content") or "").strip()
            for c in m.get("tool_calls") or ():
                body += ("\n" if body else "") + "<tool_call>" + json.dumps(
                    {"name": c["name"], "arguments": c.get("arguments") or {}}, ensure_ascii=False) + "</tool_call>"
            parts.append(f"[assistant reply {n}]\n" + body)
    return "\n".join(parts), n


def backfill_prompt(conv: dict) -> str:
    text, n = plain(conv)
    return f"Conversation ({n} assistant repl{'y' if n == 1 else 'ies'}):\n{text}\n\nOutput:"


def check_backfill(conv: dict, content: str):
    """(ok, detail, thoughts): a JSON list with one short thought per assistant reply."""
    n = sum(m["role"] == "assistant" for m in conv["messages"])
    m = re.search(r"\[.*\]", content or "", re.S)
    try:
        thoughts = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return False, "not a JSON list", None
    if not isinstance(thoughts, list) or len(thoughts) != n or not all(isinstance(t, str) for t in thoughts):
        return False, f"{len(thoughts) if isinstance(thoughts, list) else 'no'} thoughts for {n} replies", None
    thoughts = [t.strip() for t in thoughts]
    words = [len(t.split()) for t in thoughts]
    if min(words) < 4 or max(words) > 70:
        return False, f"thought of {min(words)}-{max(words)} words", None
    if any("<|" in t or "<tool_call>" in t or _BAD_OPEN.match(t) for t in thoughts):
        return False, "special token or a filler opening", None
    return True, f"{n} thoughts, {sum(words) / n:.0f} words each", thoughts


def with_thoughts(conv: dict, thoughts) -> dict:
    """The conversation with the thoughts set on its assistant turns (nothing else changes)."""
    it = iter(thoughts)
    msgs = [dict(m, think=next(it)) if m["role"] == "assistant" else m for m in conv["messages"]]
    return dict(conv, messages=msgs)


def condense_prompt(conv: dict) -> str:
    q, a = conv["messages"][0]["content"], conv["messages"][-1]
    return f"Problem:\n{q}\n\nReasoning trace:\n{a['think']}\n\nFinal answer:\n{a['content']}\n\nRewritten reasoning:"


def check_condense(conv: dict, content: str):
    """(ok, detail, thought): clearly shorter than the trace, not trivially short, no filler."""
    old, new = conv["messages"][-1]["think"], (content or "").strip()
    if len(new) < 120:
        return False, "almost empty", None
    if len(new) > 0.6 * len(old) or len(new.split()) > 320:
        return False, f"not shorter enough: {len(new)} of {len(old)} characters", None
    if "<|" in new or _BAD_OPEN.match(new) or len(re.findall(r"\b(?:Wait|Hmm+)\b", new)) > 0:
        return False, "filler or special token", None
    return True, f"{len(new)} of {len(old)} characters ({len(new) / len(old):.0%})", new


# ---- what the student reads: the teacher's reply split into working and answer --------------------------------------
# The teacher's own thinking (reasoning_content) is a draft in shorthand and is not used. The reply is written for the
# student: plain sentences of working, then the answer. The working becomes the thinking of the training example.
SOLVE_SYS = """You are writing worked solutions that will be used to teach a small language model to reason. A student \
reads your reply in two parts: the working first, then the answer. Write it as a clear explanation for the student.

The working
- Plain English sentences, in full, that explain the approach and the key steps in order, with the numbers and the \
result of each step. Usually 2 to 8 sentences. No notes to yourself, no shorthand, no lists of keywords.
- Write only about the problem. Never refer to these instructions, to a required form or to a length limit, and never \
say how you are going to reply.

The answer, in one of these forms.

Math problem: the last line is
The answer is $\\boxed{...}$.
Put only the final value in the box: no units, no thousands separators, fractions as \\frac{a}{b}.

Programming problem: one Python code block that solves the problem, using only the standard library. If the problem \
reads standard input, the program reads standard input and prints exactly the required output. If starter code is \
given, keep its names and signature. No example usage and no prints unless the problem asks for output. If the problem \
says that no tests are given, add a second code block after the first; its first line is `# tests` and it has 3 to 5 \
assert statements that call your solution with concrete values.

Question about a long document: follow the form that the question gives.

Be correct. Check each step before you write it down.

Example (math)
Problem: A shop sells pencils at 3 for $1.20. How much do 10 pencils cost, in dollars?
Reply:
One pencil costs $1.20 / 3 = $0.40. So 10 pencils cost 10 * $0.40 = $4.00.
The answer is $\\boxed{4}$.

Example (programming, tests given)
Problem: Write a function to return the sum of the squares of a list of numbers.
Your code should pass this test: assert sum_squares([1, 2, 3]) == 14
Reply:
The sum of squares is each number multiplied by itself and added up, so a generator that squares every element and \
passes the results to sum does the job.
```python
def sum_squares(nums):
    return sum(x * x for x in nums)
```

Example (programming, no tests given)
Problem: Write a Python function `is_palindrome` that tells whether a string reads the same in both directions, \
ignoring case.
No tests are given.
Reply:
Lowercasing the string first makes the comparison ignore case, and the string is a palindrome exactly when it equals \
its reverse.
```python
def is_palindrome(s):
    t = s.lower()
    return t == t[::-1]
```
```python
# tests
assert is_palindrome("Level")
assert not is_palindrome("abc")
assert is_palindrome("")
```
"""

Q_ANS = ("Answer question {i}: {q}\n"
         "Reply in this form and nothing else:\n"
         "Where: <one or two sentences, in your own words, naming the part of the document the answer comes from>\n"
         "Answer: <one to three sentences answering the question from the document only>")

_BOXED_ANY = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}")
_FIRST_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n.*?```", re.S)
_WHERE = re.compile(r"^\s*Where:\s*(.*?)\s*\n+\s*Answer:\s*(.*)$", re.S)
# sentence-start shorthand, and words that show the teacher talking about the format rather than the problem
_FRAGMENT = re.compile(r"(?:^|(?<=[.!?:]\s)|(?<=\n))(?:Need|Ensure|Keep|Let's parse|Okay|Hmm|Wait|Final|Output|Format)\b"
                       r"|\b(?:instructions?|required (?:form|format)|the reply)\b", re.I)


def split_math(content):
    """(working, answer) of a math reply: the text before the line that holds the last boxed answer, and that line
    as "The answer is $\\boxed{...}$." (None, None) when there is no box."""
    ms = list(_BOXED_ANY.finditer(content or ""))
    if not ms:
        return None, None
    last = ms[-1]
    start = content.rfind("\n", 0, last.start()) + 1
    pre = re.split(r"\b[Tt]he (?:final )?answer is\b", content[start:last.start()])[0]
    working = re.sub(r"[\s$]+$", "", (content[:start] + pre).strip())
    return working, "The answer is $" + last.group(0) + "$."


def split_code(content):
    """(working, answer) of a code reply: the text before the first code block, and the code block with what follows."""
    m = _FIRST_BLOCK.search(content or "")
    if not m:
        return None, None
    return content[:m.start()].strip(), content[m.start():].strip()


def split_where(content):
    """(where, answer) of a reply in the Q_ANS form, (None, None) otherwise."""
    m = _WHERE.match(content or "")
    if not m or not m.group(1).strip() or not m.group(2).strip():
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def working_problem(working, min_len=30):
    """Why a working text is not fit for a student, or None if it is."""
    if not working or len(working) < min_len:
        return "working too short"
    m = _FRAGMENT.search(working)
    return f"shorthand: {m.group(0)!r}" if m else None


def parts(kind, content):
    """The student-facing parts of a teacher reply; kind is "code", "math" or "reading". Returns working (the reasoning
    shown to the student), answer (how the reply ends) and style (None if the working is fit for a student, else the
    reason)."""
    split = {"code": split_code, "math": split_math}.get(kind, split_where)
    working, answer = split(content)
    if working is None:
        return {"working": None, "answer": None, "style": f"no {kind} form"}
    return {"working": working, "answer": answer,
            "style": working_problem(working, 10 if kind == "reading" else 30)}


# Chat conversations without functions get their own prompt: with the function-calling prompt the teacher kept writing
# "no function is needed" into the thoughts (11 of 20 in the second pilot, 2026-11-22).
BACKFILL_CHAT_SYS = """You are preparing training data for a small language model that writes a short thought before \
every reply. Below is a conversation in which the assistant's replies are already written. For every assistant \
reply, write the thought the assistant had just before writing it.

Rules:
- One or two plain sentences, at most 40 words, in the first person.
- Say what the user wants and how you will answer: the main point, the tone, or the form of the reply.
- Do not quote the reply and do not mention that a reply exists. Do not add facts that are not in the conversation.
- Vary the wording. Do not begin with "Okay", "Hmm" or "Alright".
- Output only a JSON list of strings, one string per assistant reply, in order. Nothing else.

Example conversation:
[user]
Hi there!
[assistant reply 1]
Hello! How can I help you today?
[user]
Can you suggest a name for my new cat? She is grey and very calm.
[assistant reply 2]
How about "Misty"? It suits a grey coat and a calm, gentle personality.

Example output:
["The user is just saying hello, so a warm greeting and an offer to help fits.", "They want a name that matches a \
grey, calm cat, so I'll suggest one and say briefly why it suits her."]
"""


def backfill_sys(conv: dict) -> str:
    """The backfill prompt for a conversation: the function-calling one if it lists functions, else the chat one."""
    return BACKFILL_SYS if conv.get("tools") else BACKFILL_CHAT_SYS
