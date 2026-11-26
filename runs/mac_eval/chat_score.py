"""Scoring for the chat-format evaluation (runs/mac_eval/chat_eval.py): comparing a model's answer with the reference,
running code against tests, matching function calls, overlap with a reference sentence. Pure functions, no model and
no torch, so the tests can import it.

The model's reply is first split with moe_llm/chat.py parse_assistant into thinking and content; everything here
scores the content (what the user would read), never the thinking.
"""
from __future__ import annotations

import json
import os
import re
import string
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import code_check  # noqa: E402
import distill_gen as G  # noqa: E402  last_boxed, same_answer, split_code_reply


def num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip().rstrip("."))
    except ValueError:
        return None


# ---- math --------------------------------------------------------------------------------------------------------
def gsm8k_scores(content: str, gold: str):
    """(strict, flexible): strict = the last boxed value equals the reference, flexible = the last number in the
    content does."""
    boxed = G.last_boxed(content)
    strict = boxed is not None and G.same_answer(boxed, gold)
    nums = re.findall(r"-?\d[\d,]*\.?\d*", content)
    flexible = bool(nums) and num(nums[-1]) is not None and num(nums[-1]) == num(gold)
    return strict, flexible


def math_correct(content: str, gold: str) -> bool:
    """MATH-500: the last boxed answer equals the reference after the usual LaTeX normalisation (distill_gen)."""
    return G.same_answer(G.last_boxed(content), gold)


# ---- code --------------------------------------------------------------------------------------------------------
def code_of(content: str):
    return G.split_code_reply(content)[0]


def humaneval_check(item: dict, content: str):
    """(passed, detail): the first code block of the reply, run with the official tests (code_check sandbox)."""
    code = code_of(content)
    if code is None:
        return False, "no code block"
    return code_check.check(code, {"type": "asserts", "code": item["test"] + f"\ncheck({item['entry_point']})\n"})


def mbpp_check(item: dict, content: str):
    code = code_of(content)
    if code is None:
        return False, "no code block"
    return code_check.check(code, {"type": "asserts", "code": "\n".join(item["test_list"]),
                                   "setup": item.get("test_setup_code") or ""})


# ---- function calling --------------------------------------------------------------------------------------------
def _canon(v):
    """A value as it should compare: numbers (also numeric strings) as floats, strings stripped, containers recursively."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return v.strip()
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v


def _call_key(c):
    return (str(c.get("name")), json.dumps(_canon(c.get("arguments") or {}), sort_keys=True, ensure_ascii=False))


def tool_scores(ref_calls: list, got_calls: list) -> dict:
    """What a reply did against the reference's first assistant turn. ref_calls [] means the reference makes no call
    (it asks, answers or refuses). got_calls: parse_assistant's list, where {"raw": ...} is a call whose JSON did not
    parse."""
    valid = [c for c in got_calls if "raw" not in c]
    out = {"expects_call": bool(ref_calls), "made_call": bool(got_calls), "valid_json": len(valid) == len(got_calls)}
    if ref_calls:
        out["name_ok"] = sorted(str(c.get("name")) for c in valid) == sorted(str(c["name"]) for c in ref_calls)
        out["args_ok"] = out["valid_json"] and sorted(map(_call_key, valid)) == sorted(map(_call_key, ref_calls))
    else:
        out["no_call_ok"] = not got_calls
    return out


# ---- reading answers ---------------------------------------------------------------------------------------------
def _tokens(s: str) -> list:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in ("a", "an", "the")]


def answer_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    left = list(g)
    for w in p:
        if w in left:
            left.remove(w)
            common += 1
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_recall(pred: str, gold: str) -> float:
    """Share of the reference's words (without articles and punctuation) that the answer contains."""
    g = _tokens(gold)
    p = set(_tokens(pred))
    return sum(w in p for w in g) / len(g) if g else 1.0


def found_share(content: str, wants: list) -> float:
    """Needle tasks: the share of the wanted strings that occur in the reply."""
    return sum(w in content for w in wants) / len(wants)
