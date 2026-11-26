"""data_prep/distill_convert.py: teacher results -> conversation sets (synthetic results, nothing calls the API)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
sys.path.insert(0, ROOT)
import distill_convert as C  # noqa: E402

WORK = "Adding the two arguments gives the sum that the problem asks for, so one return statement is enough."
CODE = "```python\ndef add(a, b):\n    return a + b\n```"


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_convert_everything(tmp_path, monkeypatch):
    d, sft = tmp_path / "distill", tmp_path / "sft"
    conv, ev = sft / "conv", sft / "eval"
    monkeypatch.setattr(C, "SFT", str(sft))
    monkeypatch.setattr(C, "decon_data", lambda: (frozenset(), {}))
    # ---- code: first pass, second pass, a shorthand working, a problem that never passed
    pool = [{"id": f"selfoss/{i}", "source": "selfoss", "licence": "odc-by", "prompt": f"Write add {i}.", "check": None}
            for i in (1, 4, 5)]
    pool += [{"id": "apps_train/2", "source": "apps", "licence": "mit", "prompt": "Read two numbers.", "check": {"type": "stdin"}},
             {"id": "mbpp_train/3", "source": "mbpp_train", "licence": "cc-by-4.0", "prompt": "Add.", "check": {"type": "asserts"}}]
    write(sft / "code_pool.jsonl", pool)
    first = [{"id": "selfoss/1", "ok": False, "content": WORK + "\n" + CODE},
             {"id": "apps_train/2", "ok": True, "content": WORK + "\n" + CODE, "effort": "none"},
             {"id": "mbpp_train/3", "ok": True, "content": WORK + "\n" + CODE + "\nIt runs in constant time."},
             {"id": "selfoss/4", "ok": True, "content": "We need answer. Need solve the problem first with a loop.\n" + CODE},
             {"id": "selfoss/5", "ok": False, "content": WORK + "\n" + CODE}]
    write(d / "code_checked.jsonl", first)
    write(d / "code_retry_checked.jsonl", [{"id": "selfoss/1", "ok": True, "content": WORK + " It is checked twice.\n" + CODE, "effort": "low"},
                                           {"id": "selfoss/5", "ok": False, "content": WORK + "\n" + CODE}])
    # ---- math
    write(conv / "gsm8k.jsonl", [{"id": "gsm8k/7", "set": "gsm8k", "messages": [{"role": "user", "content": "What is 2 + 2?"},
                                  {"role": "assistant", "think": "x", "content": "y"}]}])
    write(d / "math.jsonl", [{"id": "gsm8k/7", "ok": True, "effort": "low",
                              "content": "Two plus two is four, so adding the numbers gives four in total.\nThe answer is $\\boxed{4}$."},
                             {"id": "gsm8k/8", "ok": False, "content": "x"}])
    # ---- long documents: 12 good questions, one without evidence in the document, one without the Where / Answer form
    def qa_rows(n):
        rows = [{"question": f"Question {i}?", "evidence": "e", "evidence_in_doc": True,
                 "content": f"Where: Chapter {i}, early on.\nAnswer: Answer {i}."} for i in range(n)]
        return rows + [{"question": "Lost?", "evidence": "e", "evidence_in_doc": False, "content": "Where: x\nAnswer: y"},
                       {"question": "Plain?", "evidence": "e", "evidence_in_doc": True, "content": "Just an answer."}]
    (d / "longdocs").mkdir(parents=True)
    (d / "longdocs" / "docA.txt").write_text("DOC TEXT")
    write(d / "longdoc.jsonl", [{"id": "docA", "qa": qa_rows(12)}, {"id": "docB", "qa": qa_rows(4)}])
    write(d / "longdoc_test.jsonl", [{"id": "testA", "qa": qa_rows(11)}])
    # ---- function calling and chat: the thoughts go on the assistant turns, nothing else changes
    tool_conv = {"id": "glaive_v2/1", "set": "tools_glaive", "licence": "apache-2.0", "tools": [{"type": "function", "function": {"name": "f"}}],
                 "meta": {}, "messages": [{"role": "user", "content": "Call f."},
                                          {"role": "assistant", "think": None, "content": "", "tool_calls": [{"name": "f", "arguments": {"x": 1}}]},
                                          {"role": "tool", "content": "{\"r\": 2}"},
                                          {"role": "assistant", "think": None, "content": "It returned 2."}]}
    write(d / "backfill_in" / "tools.jsonl", [tool_conv, {**tool_conv, "id": "glaive_v2/2"}])
    write(d / "backfill_tools.jsonl", [{"id": "glaive_v2/1", "ok": True, "thoughts": ["The user wants f called with x equal to one.", "The result is two, so I report it."]},
                                       {"id": "glaive_v2/2", "ok": False, "thoughts": None}])
    chat_conv = {"id": "smoltalk/everyday/1", "set": "smoltalk", "licence": "apache-2.0", "tools": None, "meta": {"source": "everyday"},
                 "messages": [{"role": "user", "content": "Hi!"}, {"role": "assistant", "think": None, "content": "Hello!"}]}
    write(d / "backfill_in" / "chat.jsonl", [chat_conv])
    write(d / "backfill_chat.jsonl", [{"id": "smoltalk/everyday/1", "ok": True, "thoughts": ["The user greets me, so I greet back."]}])
    # ---- OpenThoughts rewritten short
    ot = {"id": "openthoughts/5", "set": "openthoughts", "licence": "apache-2.0", "tools": None, "meta": {"kind": "code"},
          "messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "think": "long " * 500, "content": "Answer."}]}
    write(conv / "openthoughts.jsonl", [ot])
    write(d / "condense.jsonl", [{"id": "openthoughts/5", "ok": True, "think": "Short direct reasoning that reaches the answer."}])

    monkeypatch.setattr(sys, "argv", ["distill_convert.py", "--distill", str(d), "--conv", str(conv), "--eval", str(ev)])
    C.main()

    code = {c["id"]: c for c in read(conv / "code_api.jsonl")}
    assert sorted(code) == ["code_api/apps_train/2", "code_api/mbpp_train/3", "code_api/selfoss/1"]
    c = code["code_api/selfoss/1"]
    assert c["messages"][0] == {"role": "user", "content": "Write add 1."} and c["meta"]["pass"] == "retry" and c["licence"] == "odc-by"
    assert c["messages"][1]["think"] == WORK + " It is checked twice." and c["messages"][1]["content"] == CODE
    assert code["code_api/mbpp_train/3"]["messages"][1]["content"].endswith("It runs in constant time.")
    assert code["code_api/apps_train/2"]["meta"] == {"source": "apps", "pass": "first", "effort": "none"}

    m = read(conv / "math_api.jsonl")
    assert len(m) == 1 and m[0]["messages"][0]["content"] == "What is 2 + 2?" and m[0]["messages"][1]["content"] == "The answer is $\\boxed{4}$."
    assert m[0]["messages"][1]["think"].startswith("Two plus two is four")

    (ld,) = read(conv / "longdoc_api.jsonl")                                  # docB has only 4 usable questions
    assert ld["id"] == "longdoc_api/docA" and len(ld["messages"]) == 24 and ld["meta"] == {"document": "docA", "questions": 12}
    assert ld["messages"][0]["content"].startswith("DOC TEXT\n\nQuestion ") and not ld["messages"][2]["content"].startswith("DOC")
    answers = [m["content"] for m in ld["messages"] if m["role"] == "assistant"]
    assert sorted(answers) == sorted(f"Answer {i}." for i in range(12)) and answers != [f"Answer {i}." for i in range(12)]
    assert all(m["think"].startswith("Chapter") for m in ld["messages"] if m["role"] == "assistant")
    (t,) = read(ev / "longdoc_test.jsonl")
    assert t["id"] == "testA" and len(t["qa"]) == 11 and t["document"].endswith(os.path.join("longdocs_test", "testA.txt"))
    assert not (conv / "longdoc_test.jsonl").exists()

    (tc,) = read(conv / "tools_api.jsonl")
    assert tc["id"] == "tools_api/glaive_v2/1" and tc["tools"] == tool_conv["tools"] and tc["meta"]["from"] == "tools_glaive"
    a = [m for m in tc["messages"] if m["role"] == "assistant"]
    assert [x["think"] for x in a] == ["The user wants f called with x equal to one.", "The result is two, so I report it."]
    assert a[0]["tool_calls"] == [{"name": "f", "arguments": {"x": 1}}] and a[1]["content"] == "It returned 2."
    (cc,) = read(conv / "chat_api.jsonl")
    assert cc["id"] == "chat_api/smoltalk/everyday/1" and cc["messages"][1]["think"] == "The user greets me, so I greet back."

    (o,) = read(conv / "openthoughts_short.jsonl")
    assert o["id"] == "openthoughts_short/5" and o["messages"][1]["think"] == "Short direct reasoning that reaches the answer."
    assert o["messages"][1]["content"] == "Answer." and o["meta"]["kind"] == "code"
    stats = json.load(open(conv / "STATS.json"))
    assert stats["code_api"]["kept"] == 3 and stats["code_api"]["dropped"]["did not pass"] == 1
    assert any(k.startswith("working: shorthand") for k in stats["code_api"]["dropped"])


def test_missing_results_are_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(C, "SFT", str(tmp_path / "sft"))
    monkeypatch.setattr(sys, "argv", ["distill_convert.py", "--distill", str(tmp_path / "d"), "--conv", str(tmp_path / "c"),
                                      "--eval", str(tmp_path / "e"), "--only", "code_api", "tools_api"])
    C.main()
    out = capsys.readouterr().out
    assert "code_api: no results yet, skipped" in out and "tools_api: no results yet, skipped" in out
    assert not (tmp_path / "c").exists()


def test_unique_keeps_one_row_per_id():
    # a job that was started twice writes some ids twice: one row per id, the first usable one wins
    rows = [{"id": "a", "ok": True, "v": 1}, {"id": "a", "ok": True, "v": 2}, {"id": "b", "ok": False, "v": 3},
            {"id": "b", "ok": True, "v": 4}, {"id": "c", "ok": False, "v": 5}, {"id": "c", "ok": False, "v": 6}]
    out = {r["id"]: r["v"] for r in C.unique(rows)}
    assert out == {"a": 1, "b": 4, "c": 5}
    assert len(C.unique(rows)) == 3
