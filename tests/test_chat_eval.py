"""The chat-format evaluation (runs/mac_eval/chat_eval.py, chat_score.py) and its held-out function-calling set
(data_prep/sft_eval_sets.py). A fake decoder stands in for the model; nothing here needs a GPU or the API."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "runs", "mac_eval"), os.path.join(ROOT, "data_prep"), ROOT):
    sys.path.insert(0, p)
pytest.importorskip("torch")
pytest.importorskip("tokenizers")
import chat_eval as C  # noqa: E402
import chat_score as S  # noqa: E402
import sft_eval_sets as E  # noqa: E402
from moe_llm import chat  # noqa: E402

TOK = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")


def test_scores():
    assert S.gsm8k_scores("So the total is 1,200 dollars. The answer is $\\boxed{1200}$.", "1200") == (True, True)
    assert S.gsm8k_scores("It comes to 1200 in the end.", "1200") == (False, True)
    assert S.gsm8k_scores("The answer is $\\boxed{12}$ and then 7.", "12") == (True, False)
    assert S.math_correct("so $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}$", "(3, \\frac{\\pi}{2})")
    assert not S.math_correct("no box", "3")
    he = {"test": "def check(candidate):\n    assert candidate(1, 2) == 3\n", "entry_point": "add"}
    good, bad = "Sum.\n```python\ndef add(a, b):\n    return a + b\n```", "```python\ndef add(a, b):\n    return a - b\n```"
    assert S.humaneval_check(he, good)[0] and not S.humaneval_check(he, bad)[0]
    assert S.humaneval_check(he, "no code") == (False, "no code block")
    mb = {"test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"], "test_setup_code": ""}
    assert S.mbpp_check(mb, good)[0] and not S.mbpp_check(mb, bad)[0]
    ref = [{"name": "f", "arguments": {"x": 1, "city": "Paris"}}]
    ok = S.tool_scores(ref, [{"name": "f", "arguments": {"city": " Paris ", "x": "1"}}])     # numeric string, stripped
    assert ok == {"expects_call": True, "made_call": True, "valid_json": True, "name_ok": True, "args_ok": True}
    wrong = S.tool_scores(ref, [{"name": "f", "arguments": {"x": 2, "city": "Paris"}}])
    assert wrong["name_ok"] and not wrong["args_ok"]
    assert not S.tool_scores(ref, [{"name": "g", "arguments": {}}])["name_ok"]
    assert S.tool_scores(ref, [{"raw": "{broken"}])["valid_json"] is False and S.tool_scores(ref, [])["made_call"] is False
    assert S.tool_scores([], [])["no_call_ok"] and not S.tool_scores([], [{"name": "f", "arguments": {}}])["no_call_ok"]
    assert S.answer_f1("The cat sat on the mat.", "a cat sat on a mat") == 1.0
    assert S.answer_f1("dogs", "cats") == 0.0 and 0 < S.answer_f1("the red cat", "a cat") < 1
    assert S.answer_recall("he went to Rome by train", "Rome") == 1.0 and S.answer_recall("Paris", "Rome") == 0.0
    assert S.found_share("they are 111 and 333", ["111", "222", "333", "444"]) == 0.5


def test_tools_test_set(tmp_path, monkeypatch):
    conv = tmp_path / "conv"
    conv.mkdir()
    tools = [{"type": "function", "function": {"name": "f"}}]

    def make(src, i, first_call):
        a = ({"role": "assistant", "think": None, "content": "", "tool_calls": [{"name": "f", "arguments": {"x": i}}]} if first_call
             else {"role": "assistant", "think": None, "content": "Which city?"})
        return {"id": f"{src}/{i}", "set": src, "tools": tools, "meta": {},
                "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, a,
                             {"role": "user", "content": "later"}, {"role": "assistant", "think": None, "content": "x"}]}
    for src in E.SOURCES:
        rows = [make(src, i, i % 2 == 0) for i in range(8)]
        rows.append({"id": f"{src}/no_user", "tools": tools, "messages": [{"role": "assistant", "content": "hi"}]})
        rows.append({"id": f"{src}/no_tools", "tools": None, "messages": make(src, 99, True)["messages"]})
        (conv / f"{src}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    train = tmp_path / "tools.jsonl"
    train.write_text("".join(json.dumps({"id": f"{src}/0"}) + "\n" for src in E.SOURCES))   # the training selection
    monkeypatch.setattr(E, "CONV", str(conv))
    monkeypatch.setattr(E, "TRAIN_SELECTION", str(train))
    items, report = E.build(2, 1, 100_000, 0)
    assert len(items) == 9 and all(i["id"].split("/")[1] not in ("0", "no_user", "no_tools") for i in items)
    assert sum(bool(i["ref_calls"]) for i in items) == 6
    call = next(i for i in items if i["ref_calls"])
    assert [m["role"] for m in call["messages"]] == ["system", "user"] and call["tools"] == tools
    assert call["ref_calls"][0]["name"] == "f"
    plain = next(i for i in items if not i["ref_calls"])
    assert plain["ref_text"] == "Which city?" and report["tools_glaive"] == {"with_call": 3, "without_call": 4, "chosen": 3}
    again, _ = E.build(2, 1, 100_000, 0)
    assert [i["id"] for i in again] == [i["id"] for i in items]       # seeded


def test_generate_batches_and_order(monkeypatch):
    import decode
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    seen = {}

    def fake_batch(model, caches, last_ids, max_new, done=None, eos=0):
        seen.setdefault("calls", []).append([c["n"] for c in caches])
        seen["eos"], seen["done"] = eos, done
        return [[10 + c["n"] % 7, 11, 12] for c in caches]
    monkeypatch.setattr(decode, "prefill", lambda model, ids: {"n": len(ids)})
    monkeypatch.setattr(decode, "generate_batch", fake_batch)
    monkeypatch.setattr(decode, "batch_rows", lambda n, rows: 3)
    convs = [C.user("word " * k) for k in (5, 1, 9, 3, 7)]
    outs = C.generate(None, tok, convs, 8)
    assert [len(c) for c in seen["calls"]] == [3, 2] and seen["calls"][0] == sorted(seen["calls"][0])
    assert seen["eos"] == chat.SPECIAL_IDS[chat.IM_END] and seen["done"](0, [5]) is False and seen["done"](0, [5, 0]) is True
    want = [1 + len(tok.encode(chat.render(c, generation_prompt=True)).ids) for c in convs]
    assert [o["prompt_tokens"] for o in outs] == want                  # replies come back in the order of the prompts
    assert all(o["tokens"] == 3 and isinstance(o["text"], str) for o in outs)
    # fewer rows when the prompts are long: decode.batch_rows decides
    seen.clear()
    monkeypatch.setattr(decode, "batch_rows", lambda n, rows: 1 if n > 24 else 5)   # prompt + 8 new tokens: 20, 22, 24, 26, 28
    C.generate(None, tok, convs, 8)
    assert [len(c) for c in seen["calls"]] == [3, 1, 1]


def test_run_task_and_summary(monkeypatch):
    f = chat.THINK_END
    reply = lambda think, content: f"{think}\n{f}\n{content}"  # noqa: E731
    texts = [reply("two plus two", "The answer is $\\boxed{4}$."), reply("hmm", "The answer is $\\boxed{9}$."),
             "never closes " * 5]
    monkeypatch.setattr(C, "generate", lambda model, tok, convs, max_new, rows, max_rows, label: [
        {"text": t, "ids": [], "tokens": 4 if i < 2 else max_new, "prompt_tokens": 10} for i, t in enumerate(texts)])
    items = [{"id": f"g/{i}", "conv": C.user("q"), "gold": "4", "group": "a" if i < 2 else "b"} for i in range(3)]

    class Args:
        max_new, rows_at_32k, max_rows = 100, 10, 8
    import io
    samples = io.StringIO()
    res = C.run_task("gsm8k", items, None, None, Args, samples)
    assert res["overall"]["n"] == 3 and abs(res["overall"]["strict"] - 1 / 3) < 1e-3
    assert res["overall"]["unclosed"] == round(1 / 3, 4) and res["overall"]["truncated"] == round(1 / 3, 4)
    assert res["groups"]["a"]["strict"] == 0.5 and res["groups"]["b"]["strict"] == 0.0
    rows = [json.loads(line) for line in samples.getvalue().splitlines()]
    assert rows[0]["think"] == "two plus two" and rows[0]["content"] == "The answer is $\\boxed{4}$." and rows[2]["closed"] is False
    # function calls: metrics exist for the items that expect a call or do not
    call = chat.TOOL_CALL + '\n{"name": "f", "arguments": {"x": 1}}\n' + chat.TOOL_CALL_END
    texts[:] = [reply("call f", call), reply("ask", "Which city?"), reply("call g", call)]
    items = [{"id": "t/0", "conv": C.user("q"), "ref_calls": [{"name": "f", "arguments": {"x": 1}}]},
             {"id": "t/1", "conv": C.user("q"), "ref_calls": []},
             {"id": "t/2", "conv": C.user("q"), "ref_calls": []}]
    res = C.run_task("tools", items, None, None, Args, None)["overall"]
    assert res["args_ok"] == 1.0 and res["args_ok_n"] == 1 and res["no_call_ok"] == 0.5 and res["no_call_ok_n"] == 2


BENCH = os.path.join(ROOT, "data", "raw", "bench")  # the benchmark files (the base repository's data_prep/download_raw.py)
BINS = os.path.join(ROOT, "data", "bins_main", "wiki_en")  # filler text of the long-context items
needs_bench = pytest.mark.skipif(not os.path.isdir(BENCH), reason="needs the benchmark files in data/raw/bench")
needs_bins = pytest.mark.skipif(not os.path.isdir(BINS), reason="needs the pretraining bins in data/bins_main")


@needs_bench
def test_items(tmp_path, monkeypatch):
    g = C.gsm8k_items(3)
    assert len(g) == 3 and g[0]["conv"]["messages"][0]["role"] == "user" and g[0]["gold"].replace("-", "").isdigit()
    assert len(C.gsm8k_items()) == 1319
    m = C.math500_items(2)
    assert len(m) == 2 and m[0]["gold"] and m[0]["group"].startswith("level")
    h = C.humaneval_items(2)
    assert h[0]["conv"]["messages"][0]["content"].startswith(C.HE_INSTRUCTION) and "def " in h[0]["conv"]["messages"][0]["content"]
    b = C.mbpp_items(2)
    assert "Your code should pass this test: assert" in b[0]["conv"]["messages"][0]["content"] and b[0]["data"]["test_list"]
    ev = tmp_path / "eval"
    ev.mkdir()
    (tmp_path / "doc.txt").write_text("THE DOCUMENT")
    qa = [{"question": f"Q{i}?", "evidence": "e", "where": "w", "answer": f"A{i}"} for i in range(6)]
    (ev / "longdoc_test.jsonl").write_text(json.dumps({"id": "d1", "document": "doc.txt", "qa": qa}) + "\n")
    (ev / "tools_test.jsonl").write_text(json.dumps({"id": "t1", "source": "tools_glaive", "messages": [{"role": "user", "content": "q"}],
                                                      "tools": [], "ref_calls": []}) + "\n")
    monkeypatch.setattr(C, "EVAL", str(ev))
    monkeypatch.setattr(C, "ROOT", str(tmp_path))
    ld = C.longdoc_items(per_doc=3)
    assert len(ld) == 3 and all(i["conv"]["messages"][0]["content"].startswith("THE DOCUMENT\n\nQ") for i in ld)
    assert ld == C.longdoc_items(per_doc=3)                           # seeded
    t = C.tools_items()
    assert t[0]["group"] == "tools_glaive" and t[0]["conv"]["tools"] == []


@needs_bins
def test_long_context_items():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    pk = C.passkey_items(tok, [2048], 2)
    assert len(pk) == 2 and all(i["wants"][0] in i["conv"]["messages"][0]["content"] for i in pk)
    text = pk[0]["conv"]["messages"][0]["content"]
    assert text.rstrip().endswith("Reply with the pass key only.") and 1900 < len(tok.encode(text).ids) <= 2048 + 20
    ru = C.ruler_items(tok, [2048], 1)
    assert [i["group"] for i in ru] == [f"{t}@2048" for t in C.RULER_TASKS]
    for i in ru:
        body = i["conv"]["messages"][0]["content"]
        assert all(w in body for w in i["wants"]) or i["group"].startswith("vt")   # needles are in the text
        assert "Reply with the number only" in body or "List all of them" in body
    assert len(ru[2]["wants"]) == 4 and "special magic numbers" in ru[2]["conv"]["messages"][0]["content"]
