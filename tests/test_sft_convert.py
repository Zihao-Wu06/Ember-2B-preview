"""data_prep/sft_convert.py: the dataset-specific parsers (no dataset files needed)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_prep"))
pytest.importorskip("tokenizers")
import sft_convert as S  # noqa: E402


def test_toolace_calls():
    assert S.toolace_calls('[Market Trends API(trend_type="MARKET_INDEXES", country="us")]') == [
        {"name": "Market Trends API", "arguments": {"trend_type": "MARKET_INDEXES", "country": "us"}}]
    assert S.toolace_calls("[GetCompetitions()]") == [{"name": "GetCompetitions", "arguments": {}}]
    assert S.toolace_calls('[User Feed (Video Posts) V2(username="a, b)"), f.g(from="2025-01-01", ok=false, n=null, '
                           'p={"d": [1, {"e": true}]})]') == [
        {"name": "User Feed (Video Posts) V2", "arguments": {"username": "a, b)"}},
        {"name": "f.g", "arguments": {"from": "2025-01-01", "ok": False, "n": None, "p": {"d": [1, {"e": True}]}}}]
    assert S.toolace_calls("[a(x='it''s')]"[:0] + "[a(x='y')]") == [{"name": "a", "arguments": {"x": "y"}}]
    for bad in ("plain text", "[f]=>('a'~1)", "[f(1, 2)]", "[(a=1)]"):
        with pytest.raises(Exception):
            S.toolace_calls(bad)


def test_gsm8k_parts():
    ans = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nShe sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72"
    text, steps, final = S.gsm8k_parts(ans)
    assert text == "Natalia sold 48/2 = 24 clips in May.\nShe sold 48+24 = 72 clips altogether."
    assert steps == [("48/2", "24"), ("48+24", "72")] and final == "72"
    assert S._num(72.0) == "72" and S._num(0.5) == "0.5"


def test_json_values_and_tools():
    vals = S.json_values('SYSTEM: functions -\n{"name": "a", "parameters": {}}\n\n{"name": "b"} trailing [1, 2]')
    assert vals == [{"name": "a", "parameters": {}}, {"name": "b"}, [1, 2]]
    t = S.as_tool({"name": "a", "description": "d"})
    assert t == {"type": "function", "function": {"name": "a", "description": "d",
                                                  "parameters": {"type": "object", "properties": {}}}}
    assert S.as_tool(t) == t
    with pytest.raises(ValueError):
        S.as_tool({"description": "no name"})
    assert S.boxed(" 5 ") == "The answer is $\\boxed{5}$."


def test_openthoughts_kind():
    assert S.ot_kind("Return your final response within \\boxed{}. What is 2 + 2?") == "math"
    assert S.ot_kind("Generate an executable Python function generated from the given prompt. ...") == "code"
    assert S.ot_kind("How does hypoparathyroidism affect calcium levels?") == "other"


def test_openthoughts_bands(monkeypatch):
    """openthoughts keeps thoughts up to --max-think, openthoughts_mid the next band up to OT_MID; no row is in both."""
    import types

    def row(n):
        return {"conversations": [{"from": "user", "value": "What is 2 + 2 in this problem?"},
                                  {"from": "assistant", "value": "<|begin_of_thought|>" + "x" * n + "<|end_of_thought|>"
                                                                  "<|begin_of_solution|>4<|end_of_solution|>"}]}

    rows = [row(50), row(6000), row(6001), row(S.OT_MID), row(S.OT_MID + 1)]
    monkeypatch.setattr(S, "parquet_rows", lambda pattern: iter(rows))
    a = types.SimpleNamespace(max_think=6000)
    short = [(c and c["id"], why) for c, why in S.conv_openthoughts(a)]
    mid = [(c and c["id"], why) for c, why in S.conv_openthoughts_mid(a)]
    assert [i for i, _ in short] == ["openthoughts/0", "openthoughts/1", None, None, None]
    assert [i for i, _ in mid] == [None, None, "openthoughts_mid/2", "openthoughts_mid/3", None]
    assert mid[0][1] == "in the set openthoughts" and mid[4][1] == "thought too long"


def test_rstar_parts(monkeypatch):
    code = "def main():\n    print(1)\n\nmain()"
    block = f"```python\n{code}\n```"
    # layout 1: a <think> block, then the write-up with the code
    t, ans = S.rstar_parts(f"<think>\nI work it out.\n</think>\n\nApproach: print.\n\n{block}\n\nIt prints 1.", code)
    assert t == "I work it out." and ans.startswith("Approach") and code in ans and ans.endswith("It prints 1.")
    # layout 2: running text ending in the final code block; an earlier draft block stays in the reasoning
    t, ans = S.rstar_parts(f"First try:\n```python\nx = 0\n```\nThat fails, so:\n\n{block}", code)
    assert t.endswith("That fails, so:") and "x = 0" in t and ans == block
    # the answer's code differs from the verified code, or the answer is too long: the verified code alone
    assert S.rstar_parts("<think>t</think>\n```python\nother()\n```", code) == ("t", block)
    assert S.rstar_parts("<think>t</think>\n" + "long " * 2000 + block, code) == ("t", block)
    assert S.rstar_parts("no code here", code) == (None, "no code block")
    assert S.rstar_parts("<think>t</think>x", "") == (None, "no code")
    assert S.rstar_parts("<think>a</think>b</think>" + block, code) == (None, "format")
    assert S.rstar_parts("<think>a</think>the <think> tag " + block, code) == (None, "format")
    # the converter: thought length band, ids by row number
    import types
    think = "Because the sum is fixed, moving one unit keeps it. " * 6
    rows = [{"question": "Q1", "response": f"<think>{think}</think>{block}", "code": code, "seed_source": "codeforces"},
            {"question": "Q2", "response": f"<think>{'y' * (S.OT_MID + 1)}</think>{block}", "code": code, "seed_source": "x"},
            {"question": "Q3", "response": f"<think>short</think>{block}", "code": code, "seed_source": "x"}]
    monkeypatch.setattr(S, "parquet_rows", lambda pattern: iter(rows))
    out = list(S.conv_rstar(types.SimpleNamespace(max_think=6000)))
    assert out[0][0]["id"] == "rstar/0" and out[0][0]["meta"] == {"kind": "code", "seed_source": "codeforces"}
    assert out[0][0]["messages"][1]["content"] == block
    assert [w for _, w in out] == [None, "thought too long", "length"]


def test_strip_filler():
    f = S.strip_filler
    assert f("Okay, so the question is about x. Hmm. Let me think. The answer is 3.") == \
        "The question is about x. The answer is 3."
    assert f("Okay, let's see. The problem is hard.") == "The problem is hard."
    assert f("Okay, let's tackle this problem. Hmm. So Chef has a lift.") == "Chef has a lift."
    assert f("How does it work? Hmm.\n\nPharmacogenomics is a term.") == "How does it work?\n\nPharmacogenomics is a term."
    assert f("Hmm, maybe that fails. Well, not always.") == "Maybe that fails. Not always."
    # corrections, other approaches and checks stay; so do words that only look like filler
    keep = ("Wait, no, that's wrong. Alternatively, sort first. Let me check the sample. Actually, 5 is right. "
            "Right triangles have one. Well water is cold. Let me think about the second case. Okay then.")
    assert f(keep) == keep
    # a variable or an expression at the new sentence start keeps its case
    assert f("Okay, so n is at most 5. Hmm, dp[i] is the best.") == "n is at most 5. dp[i] is the best."
    assert f("Well, a penny is a coin. Okay, so a is the array. Hmm, arr.sort() works.") == \
        "A penny is a coin. a is the array. arr.sort() works."
    assert f("It fits. Hmm, right. The file is thin. Hmm, interesting. It works.") == \
        "It fits. The file is thin. Interesting. It works."


def test_narrativeqa_helpers():
    raw = ("ï»¿The Project Gutenberg EBook of X, by Y\n\nlicence text\n\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\n\n"
           "Produced by Someone\nand the team\n\nCHAPTER I\n\n" + "The cafÃ© was full. " * 3000 +
           "\n\nEnd of the Project Gutenberg EBook of X\n\n*** END OF THIS PROJECT GUTENBERG EBOOK X ***\nmore licence")
    body = S.pg_body(S.fix_mojibake(raw))
    assert body.startswith("CHAPTER I") and body.endswith("The café was full.") and "Gutenberg" not in body
    assert S.pg_body("no marker here " * 100) is None
    # a short book: the licence after it is longer than the book; a note and an old preamble name Project Gutenberg
    short = ("header\n*** START OF THE PROJECT GUTENBERG EBOOK Y ***\n\nNote: Project Gutenberg also has an HTML version\n"
             "of this file.\n\nTHE TALE\n\n" + "Once there was a rabbit. " * 60 + "\n\nTHE END\n\n"
             "End of Project Gutenberg's The Tale\n\n*** END OF THIS PROJECT GUTENBERG EBOOK Y ***\n" + "licence " * 3000)
    body = S.pg_body(short)
    assert body.startswith("THE TALE") and body.endswith("THE END")
    assert S.fix_mojibake("a protÃ©gÃ© and a broken \udc81 byte") == "a protégé and a broken \udc81 byte"
    assert S.fix_mojibake("plain text") == "plain text"
    assert S.nqa_answer(["Rome", "Gerard escapes to Rome from Holland."]) == "Gerard escapes to Rome from Holland."
    assert S.nqa_answer(["at home", "home"]) == "At home." and S.nqa_answer([]) == ""
    assert S.nqa_names(["Who does Gerard marry?", "Margaret Brandt"]) == {"gerard", "brandt"}
