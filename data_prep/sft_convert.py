#!/usr/bin/env python
"""Public post-training datasets (data/raw/sft, data_prep/download_sft.py) -> conversations in the common format of
moe_llm/chat.py, one JSON object per line in data/sft/conv/<set>.jsonl.

    python data_prep/sft_convert.py                    # all sets (WSL: needs pyarrow), then data/sft/conv/STATS.json
    python data_prep/sft_convert.py --only gsm8k math  # some sets

A line is {"id", "set", "licence", "messages", "tools", "meta"}. An assistant message has "think" (the reasoning, None
when the dataset has none: filled in later or rendered as an empty block) and, for tool use, "tool_calls". Every
conversation must render (chat.render) and is dropped if it shares a 13-gram of words with an evaluation problem
(HumanEval, MBPP, GSM8K test, MATH-500; data_prep/decontam.py's rule). Sets:

  math            OpenMathInstruct-2 train_1M: the step-by-step solution is the reasoning, the answer a \\boxed{} line
  gsm8k           GSM8K train, the same layout (calculator annotations removed)
  gsm8k_py        GSM8K train solved with the python tool: the annotated calculations become code, the code is run
                  here and must print the reference answer
  openthoughts    OpenThoughts-114k with a thought of at most --max-think characters (most are far longer);
                  meta.kind is math / code / other
  openthoughts_mid  the same with a thought of more than --max-think and at most 12,000 characters (harder problems)
  smoltalk        SmolTalk's general subsets (no reasoning); its math, code and function-calling subsets are left out
                  (OpenMathInstruct-2, the API data and the tool sets below cover them), longalign goes to its own set
  smoltalk_long   SmolTalk longalign (long inputs, no reasoning)
  narrativeqa     NarrativeQA train, Gutenberg books only: one conversation per book with all its questions (no
                  reasoning, short human answers)
  code_selfoss    self-oss-instruct (execution-filtered): reasoning, then code and explanation
  code_rstar      rStar-Coder synthetic_sft: verified solutions with a reasoning of at most 12,000 characters
  tools_hermes    hermes-function-calling-v1 (function calling and JSON mode)
  tools_glaive    glaive-function-calling-v2
  tools_toolace   ToolACE
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import decontam  # noqa: E402
from moe_llm import chat  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw", "sft")
OUT = os.path.join(ROOT, "data", "sft", "conv")
PYTHON_TOOL = {"type": "function", "function": {
    "name": "python", "description": "Run Python code and return what it prints.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to run."}},
                   "required": ["code"]}}}
SMOLTALK_SKIP = {"numina-cot-100k", "metamathqa-50k", "self-oss-instruct", "apigen-80k", "longalign"}


def parquet_rows(pattern):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(RAW, pattern))):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            yield from pf.read_row_group(g).to_pylist()


def boxed(ans) -> str:
    return f"The answer is $\\boxed{{{str(ans).strip()}}}$."


def qa(user, think, answer):
    return [{"role": "user", "content": user}, {"role": "assistant", "think": think, "content": answer}]


def as_tool(t):
    """Any of the datasets' function descriptions -> {"type": "function", "function": {name, description, parameters}}."""
    if isinstance(t, dict) and t.get("type") == "function" and isinstance(t.get("function"), dict):
        t = t["function"]
    if not isinstance(t, dict) or not isinstance(t.get("name"), str):
        raise ValueError("tool without a name")
    return {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                             "parameters": t.get("parameters") or {"type": "object", "properties": {}}}}


def json_values(text):
    """Every top-level JSON value in text, in order (the text between them is ignored)."""
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        m = re.search(r"[\[{]", text[i:])
        if not m:
            return out
        try:
            v, end = dec.raw_decode(text, i + m.start())
            out.append(v)
            i = end
        except json.JSONDecodeError:
            i += m.start() + 1


# ---- math ----------------------------------------------------------------------------------------------------------
def conv_math(a):
    for i, r in enumerate(parquet_rows("openmathinstruct2_1m/data/*.parquet")):
        sol, ans = (r["generated_solution"] or "").strip(), (r["expected_answer"] or "").strip()
        if not ans or not 30 <= len(sol) <= a.max_think or len(r["problem"]) > 3000:
            yield None, "length"
            continue
        yield {"id": f"omi2/{i}", "messages": qa(r["problem"].strip(), sol, boxed(ans)),
               "meta": {"source": r["problem_source"]}}, None


_CALC = re.compile(r"(=\s*)?<<([^=<>]*)=([^<>]*)>>\$?[\d,\.]*")


def gsm8k_parts(answer):
    """(solution text without calculator annotations, [(expression, result)], final answer)."""
    body, final = answer.rsplit("####", 1)
    steps = [(m.group(2).strip(), m.group(3).strip()) for m in _CALC.finditer(body)]
    return re.sub(r"<<[^<>]*>>", "", body).strip(), steps, final.strip().replace(",", "")


def conv_gsm8k(a):
    for i, r in enumerate(parquet_rows("gsm8k/main/train-*.parquet")):
        text, _, final = gsm8k_parts(r["answer"])
        yield {"id": f"gsm8k_train/{i}", "messages": qa(r["question"].strip(), text, boxed(final)), "meta": {}}, None


_ARITH = re.compile(r"^[\d\.\+\-\*/\(\)\s%]+$")


def _num(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else repr(round(x, 6))


def conv_gsm8k_py(a):
    for i, r in enumerate(parquet_rows("gsm8k/main/train-*.parquet")):
        body, steps, final = gsm8k_parts(r["answer"])
        if not steps or not all(_ARITH.match(e) for e, _ in steps):
            yield None, "no plain arithmetic"
            continue
        # the plan: the solution's sentences with "= <<a*b=c>>c" cut out (the code does the arithmetic)
        plan = _CALC.sub(lambda m: "" if m.group(1) else m.group(2), r["answer"].rsplit("####", 1)[0])
        plan = re.sub(r"[ \t]+", " ", plan).strip()
        names = [f"step{k + 1}" for k in range(len(steps))]
        code = "\n".join(f"{n} = {e.replace('%', '/100')}" for n, (e, _) in zip(names, steps)) + f"\nprint({names[-1]})"
        try:
            env: dict = {}
            exec(compile(code.rsplit("\n", 1)[0], "<gsm8k>", "exec"), {"__builtins__": {}}, env)  # arithmetic only
            got = float(env[names[-1]])
            ok = abs(got - float(final)) < 1e-6
        except Exception:  # noqa: BLE001
            ok = False
        if not ok:
            yield None, "code does not give the answer"
            continue
        out = _num(got)
        msgs = [{"role": "user", "content": r["question"].strip()},
                {"role": "assistant", "think": plan, "content": "",
                 "tool_calls": [{"name": "python", "arguments": {"code": code}}]},
                {"role": "tool", "content": out},
                {"role": "assistant", "think": f"The code printed {out}.", "content": boxed(final)}]
        yield {"id": f"gsm8k_train_py/{i}", "tools": [PYTHON_TOOL], "messages": msgs, "meta": {}}, None


# ---- general -------------------------------------------------------------------------------------------------------
_OT = re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>.*?<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.S)


def ot_kind(question: str) -> str:
    """OpenThoughts marks its problem type with a fixed first sentence: "math" (78% of the rows that pass the length
    filter), "code" (competitive programming), "other" (science questions, commonsense multiple choice)."""
    if question.startswith("Return your final response within \\boxed{}"):
        return "math"
    return "code" if question.startswith("Generate an executable Python function") else "other"


# Content-free filler in OpenThoughts' reasoning ("Okay, so ...", "Hmm.", "Let me think."): removed, so the student does
# not learn to open every thought with it. "Wait", "Alternatively", "Let me check" and
# "Actually" stay: they introduce a real correction or another approach, and removing them would leave a wrong step
# next to its correction with nothing in between.
_SENT = r"(?:^|(?<=[.!?]\s))"  # the start of a sentence (with re.M also the start of a line)
_FILL_LEAD = re.compile(_SENT + r"(?:(?:Hmm+|Okay|Alright|Well|Oh right|Right)[,.] (?:[Ss]o,? )?|(?:Okay|Alright) so,? )"
                        r"(?=[A-Za-z])", re.M)
_FILL_WHOLE = re.compile(
    _SENT + r"(?:Hmm+|Okay|Alright|Well|Right"
    r"|[Ll]et me think(?: (?:about (?:this|it|that)|through (?:this|it)))?(?: (?:step by step|carefully|again))?"
    r"|[Ll]et(?:'s| me) see"
    r"|[Ll]et's think(?: (?:about (?:this|it)|through (?:this|it)))?(?: step by step)?"
    r"|[Ll]et(?:'s| me) (?:tackle|try to (?:figure out|tackle|solve|work through|understand)|figure out|work through|"
    r"think about|start by understanding|break down) (?:this|the) (?:problem|question)(?: step by step)?"
    r")[.!\u2026]+[ \t]*", re.M)


# the word an opener exposed (marked \x00): a word of two or more letters, or the article "a"; not a variable ("n is",
# "a is the array") or an expression ("dp[i]", "arr.sort()", "main()")
_FILL_CAP = re.compile(r"\x00(?:([a-z])(?=[a-z]+(?:[ ,'\u2019]|[.!?](?:\s|$)))"
                       r"|(a)(?= (?!(?:is|and|or|to|in|equals|mod|can|must|has|will|would|should|was|plus|minus|times)\b)"
                       r"[a-z]{2,}\b))")


def strip_filler(text: str) -> str:
    """The reasoning without content-free filler sentences and sentence openers; nothing else changes. A lower-case
    word that an opener leaves at the start of a sentence is capitalised (not single letters or expressions: they are
    variables)."""
    for _ in range(4):  # an opener can hide a filler sentence ("Okay, let's see.")
        new = _FILL_WHOLE.sub("", _FILL_LEAD.sub("\x00", text))
        new = _FILL_CAP.sub(lambda m: (m.group(1) or m.group(2)).upper(), new).replace("\x00", "")
        if new == text:
            break
        text = new
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text)).strip()


def conv_openthoughts(a):
    for i, r in enumerate(parquet_rows("openthoughts_114k/data/*.parquet")):
        c = r["conversations"]
        m = _OT.search(c[-1]["value"]) if len(c) == 2 and c[0]["from"] == "user" else None
        if not m:
            yield None, "format"
        elif len(m.group(1)) > a.max_think:
            yield None, "thought too long"
        else:
            yield {"id": f"openthoughts/{i}", "messages": qa(c[0]["value"].strip(), strip_filler(m.group(1).strip()),
                                                              m.group(2).strip()),
                   "meta": {"kind": ot_kind(c[0]["value"].strip())}}, None


OT_MID = 12000  # longest thought of openthoughts_mid, in characters (~3,000 tokens: still fits a 4,096-token row)


def conv_openthoughts_mid(a):
    """The next band of OpenThoughts: a thought of more than --max-think and at most OT_MID characters (the harder
    problems; the same row numbers as conv_openthoughts, so the ids do not collide)."""
    for i, r in enumerate(parquet_rows("openthoughts_114k/data/*.parquet")):
        c = r["conversations"]
        m = _OT.search(c[-1]["value"]) if len(c) == 2 and c[0]["from"] == "user" else None
        if not m:
            yield None, "format"
        elif len(m.group(1)) <= a.max_think:
            yield None, "in the set openthoughts"
        elif len(m.group(1)) > OT_MID:
            yield None, "thought too long"
        else:
            yield {"id": f"openthoughts_mid/{i}", "messages": qa(c[0]["value"].strip(), strip_filler(m.group(1).strip()),
                                                                  m.group(2).strip()),
                   "meta": {"kind": ot_kind(c[0]["value"].strip())}}, None


def _smoltalk(a, long):
    for i, r in enumerate(parquet_rows("smoltalk_all/data/all/train-*.parquet")):
        src = r["source"]
        if (src == "longalign") != long or (not long and src in SMOLTALK_SKIP):
            continue
        msgs = [{"role": m["role"], "content": m["content"]} if m["role"] != "assistant" else
                {"role": "assistant", "think": None, "content": m["content"]} for m in r["messages"]]
        if sum(len(m["content"]) for m in msgs) > (400_000 if long else a.max_chars):
            yield None, "length"
            continue
        yield {"id": f"smoltalk/{src}/{i}", "messages": msgs, "meta": {"source": src}}, None


def conv_smoltalk(a):
    yield from _smoltalk(a, False)


def conv_smoltalk_long(a):
    yield from _smoltalk(a, True)


# ---- long documents with questions ----------------------------------------------------------------------------------
_PG_START = re.compile(r"^.*(?:\*\*\* ?START OF (?:THIS|THE) PROJECT GUTENBERG|\*END\*THE SMALL PRINT).*$", re.M | re.I)
_PG_END = re.compile(r"^.*(?:\*\*\* ?END OF (?:THIS|THE) PROJECT GUTENBERG|^End of (?:the |this )?Project Gutenberg).*$",
                     re.M | re.I)
_NQA_STOP = frozenset("the a an he she it they we you his her their its this that these those who what where when why "
                      "how which and but because after before in on at to of for with from by as if while during mr mrs "
                      "miss sir lord lady dr king queen captain god".split())


_MOJI = re.compile("[\u00c2-\u00ef][\u0080-\u00bf\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u02dc"
                   "\u2013-\u203a\u20ac\u2122]{1,2}")


def _unmoji(s: str):
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return None


def fix_mojibake(text: str) -> str:
    """NarrativeQA stores UTF-8 text decoded as a one-byte code page ("ï»¿", "Ã©"): undo it, for the whole text
    when that works, otherwise sequence by sequence."""
    if "Ã" not in text and "â€" not in text and not text.startswith("ï»¿"):
        return text
    whole = _unmoji(text)
    if whole is not None:
        return whole.lstrip("\ufeff")
    text = text[3:] if text.startswith("ï»¿") else text
    return _MOJI.sub(lambda m: _unmoji(m.group(0)) or m.group(0), text)


def pg_body(text: str):
    """A Project Gutenberg file without its header and licence (None if the start marker is not in the first fifth
    of the file). Paragraphs that still name Project Gutenberg near the start or the end (credits, notes about other
    formats, the old etexts' preamble) go too."""
    starts = [m for m in _PG_START.finditer(text) if m.start() < max(len(text) // 5, 6000)]
    if not starts:
        return None
    a = starts[-1].end()
    end = _PG_END.search(text, a + 1000)
    body = text[a:end.start() if end else len(text)].strip()
    m = re.match(r"(?:Produced by|E-text prepared by|Transcribed (?:by|from))[^\n]*(?:\n[^\n]+)*\n\s*\n", body, re.I)
    body = body[m.end():].strip() if m else body
    for _ in range(6):
        i = body.rfind("Project Gutenberg", 0, 3000)
        if i < 0:
            break
        j = re.search(r"\n\s*\n", body[i:])
        body = body[i + j.end():].strip() if j else body[i + 17:]
    for _ in range(6):
        i = body.find("Project Gutenberg", max(0, len(body) - 3000))
        if i < 0:
            break
        j = [m.start() for m in re.finditer(r"\n\s*\n", body[:i])]
        body = body[:j[-1]].strip() if j else body[:i]
    return body


def nqa_names(texts) -> set:
    """Capitalised words that are not the first word of a question or answer: the names a story's questions use."""
    out = set()
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z'-]+", t)[1:]:
            if w[0].isupper() and len(w) >= 4 and w.lower() not in _NQA_STOP:
                out.add(w.lower().strip("'-"))
    return out


def nqa_answer(answers) -> str:
    """The longer reference answer (the short one is often a single word), as a sentence."""
    a = max((x.strip() for x in answers), key=len, default="")
    if not a:
        return ""
    a = a[0].upper() + a[1:]
    return a if a[-1] in ".!?\"')" else a + "."


def _book_grams(words, step=1):
    return [hash(" ".join(words[i:i + 8])) for i in range(0, len(words) - 7, step)]


def eval_book_grams() -> frozenset:
    """Hashes of every 8-word sequence of the PG19 evaluation books (data/eval/pg19_*.npz: the books the long-context
    loss is measured on). A training document that shares many of them is one of those books."""
    import numpy as np
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    grams = set()
    for split in ("test", "validation"):
        z = np.load(os.path.join(ROOT, "data", "eval", f"pg19_{split}.npz"))
        off = z["offsets"]
        for i in range(len(off) - 1):
            text = tok.decode(z["ids"][off[i]:off[i + 1]].tolist())
            grams.update(_book_grams(re.findall(r"[a-z0-9]+", text.lower())))
    return frozenset(grams)


def conv_narrativeqa(a):
    """NarrativeQA train, Project Gutenberg books only (the film scripts are scraped web pages and are not used;
    whether a book is still in copyright is not a criterion: user, 2026-11-22):
    one conversation per book, the book and the first question in the first user turn, then one turn per question;
    the answers are the annotators' (a few words to a sentence), no reasoning. Left out: books whose questions are
    about another story (the dataset matched some summaries to the wrong file: under half of the names in the
    questions and answers occur in the text), PG19 evaluation books, files without a Gutenberg start marker, and
    books of more than --max-doc-words words (a 256K window)."""
    import pyarrow.parquet as pq
    docs = {}
    for f in sorted(glob.glob(os.path.join(RAW, "narrativeqa/data/train-*.parquet"))):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16):
            for r in batch.to_pylist():
                d = r["document"]
                e = docs.get(d["id"])
                if e is None:
                    e = docs[d["id"]] = {"kind": d["kind"], "url": d["url"], "title": d["summary"]["title"], "qa": [],
                                         "text": d["text"] if d["kind"] == "gutenberg" and
                                         d["word_count"] <= a.max_doc_words else None, "words": d["word_count"]}
                e["qa"].append((r["question"]["text"].strip(), nqa_answer([x["text"] for x in r["answers"]])))
    grams = eval_book_grams()
    for doc_id, e in docs.items():
        if e["kind"] != "gutenberg":
            yield None, "film script"
            continue
        if e["text"] is None:
            yield None, "book too long"
            continue
        body = pg_body(fix_mojibake(e["text"]))
        if body is None or len(body) < 20000:
            yield None, "no Gutenberg start marker or almost no text"
            continue
        words = re.findall(r"[a-z0-9]+", body.lower())
        probes = _book_grams(words, 97)
        if probes and sum(g in grams for g in probes) / len(probes) > 0.2:
            yield None, "a PG19 evaluation book"
            continue
        qa_pairs = list(dict.fromkeys((q, x) for q, x in e["qa"] if q and x))
        names = nqa_names([t for pair in qa_pairs for t in pair])
        vocab = set(re.findall(r"[a-z][a-z'-]+", body.lower()))
        match = sum(n in vocab for n in names) / len(names) if names else 1.0
        if len(names) >= 2 and match < 0.5:
            yield None, "questions about another story"
            continue
        msgs = []
        for k, (q, x) in enumerate(qa_pairs):
            msgs += qa((body + "\n\n" + q) if k == 0 else q, None, x)
        m = re.search(r"/ebooks/(\d+)", e["url"])
        yield {"id": f"narrativeqa/{doc_id}", "messages": msgs,
               "meta": {"gutenberg": int(m.group(1)) if m else None, "title": e["title"], "questions": len(qa_pairs),
                        "words": len(words), "names_found": round(match, 2)}}, None


# ---- code ----------------------------------------------------------------------------------------------------------
def conv_code_selfoss(a):
    """self-oss-instruct (execution-filtered by its authors): the text before the code block is the reasoning, the
    code block and the explanation after it are the answer."""
    for r in parquet_rows("self_oss_instruct_50k/data/*.parquet"):
        resp = r["response"]
        k = resp.find("```")
        if k < 40 or resp.count("```") < 2 or len(resp) > a.max_chars:
            yield None, "no reasoning before the code" if k < 40 else "format"
            continue
        yield {"id": f"selfoss/{r['id']}", "messages": qa(r["instruction"].strip(), resp[:k].strip(), resp[k:].strip()),
               "meta": {"concepts": r["concepts"]}}, None


_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\n(.*?)```", re.S)
RSTAR_ANSWER = 6000  # longest answer part kept as it is, in characters; a longer one becomes the code alone


def rstar_parts(response: str, code: str):
    """(reasoning, answer) of an rStar-Coder response, or (None, why). The responses come in two layouts: a
    <think> block followed by the write-up with the code, or one running text that ends in the final code block.
    The answer must carry the dataset's verified `code`: an answer whose code differs (or that is too long) is
    replaced by that code alone."""
    r, code = response.strip(), (code or "").strip()
    if not code:
        return None, "no code"
    block = f"```python\n{code}\n```"
    if "</think>" in r:
        think, _, ans = r.partition("</think>")
        think, ans = think.replace("<think>", "", 1).strip(), ans.strip()
    else:
        fences = list(_FENCE.finditer(r))
        if not fences:
            return None, "no code block"
        think, ans = r[:fences[-1].start()].strip(), r[fences[-1].start():].strip()
    if any(tag in part for tag in ("<think>", "</think>") for part in (think, ans)) or "```" in think[-3:]:
        return None, "format"
    if code not in ans or len(ans) > RSTAR_ANSWER:
        ans = block
    return think, ans


def conv_rstar(a):
    """rStar-Coder synthetic_sft: competition-style problems written by a model from seed problems, each with a long
    reasoning and a solution that passed mutual verification on generated tests (Liu et al. 2025). Kept: Python
    answers whose reasoning has at most OT_MID characters after the filler is removed."""
    for i, r in enumerate(parquet_rows("rstar_coder/synthetic_sft/*.parquet")):
        think, ans = rstar_parts(r["response"] or "", r["code"])
        if think is None:
            yield None, ans
            continue
        if len(think) > OT_MID * 1.25:  # removing the filler never shortens a thought that much; most rows end here
            yield None, "thought too long"
            continue
        think = strip_filler(think)
        if len(think) > OT_MID:
            yield None, "thought too long"
        elif len(think) < 200 or len(r["question"]) > 6000:
            yield None, "length"
        else:
            yield {"id": f"rstar/{i}", "messages": qa(r["question"].strip(), think, ans),
                   "meta": {"kind": "code", "seed_source": r.get("seed_source")}}, None


# ---- tools ---------------------------------------------------------------------------------------------------------
_HERMES_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_HERMES_RESP = re.compile(r"</?tool_response>")


def _loads_loose(s):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)  # {'arguments': {...}, 'name': '...'}


def conv_tools_hermes(a):
    for fname in ("func-calling.json", "func-calling-singleturn.json", "glaive-function-calling-5k.json",
                  "json-mode-agentic.json", "json-mode-singleturn.json"):
        for r in json.load(open(os.path.join(RAW, "hermes_fc_v1", fname))):
            try:
                json_mode = fname.startswith("json-mode")
                tools = None if json_mode else [as_tool(t) for t in _loads_loose(r["tools"])]
                msgs = []
                for m in r["conversations"]:
                    who, v = m["from"], m["value"]
                    if who == "system":
                        if json_mode:  # the schema lives in the system turn
                            msgs.append({"role": "system", "content": v})
                    elif who == "human":
                        msgs.append({"role": "user", "content": v})
                    elif who == "gpt":
                        calls = [_loads_loose(c) for c in _HERMES_CALL.findall(v)]
                        msgs.append({"role": "assistant", "think": None, "content": _HERMES_CALL.sub("", v).strip(),
                                     "tool_calls": [{"name": c["name"], "arguments": c.get("arguments") or {}}
                                                    for c in calls] or None})
                    elif who == "tool":
                        msgs.append({"role": "tool", "content": _HERMES_RESP.sub("", v).strip()})
                    else:
                        raise ValueError(who)
                yield {"id": f"hermes/{fname[:-5]}/{r['id']}", "tools": tools, "messages": msgs,
                       "meta": {"category": r.get("category")}}, None
            except Exception as e:  # noqa: BLE001
                yield None, f"parse: {type(e).__name__}"


_GLAIVE_TURN = re.compile(r"(?:^|\n)(USER|ASSISTANT|FUNCTION RESPONSE): ")  # split points; no nested quantifiers
_GLAIVE_CALL = re.compile(r"<functioncall>\s*\{\s*\"name\":\s*\"([^\"]+)\",\s*\"arguments\":\s*(.*)\}\s*$", re.S)


def conv_tools_glaive(a):
    for i, r in enumerate(json.load(open(os.path.join(RAW, "glaive_fc_v2", "glaive-function-calling-v2.json")))):
        try:
            tools = [as_tool(t) for t in json_values(r["system"])]
            msgs = []
            parts = _GLAIVE_TURN.split(r["chat"])  # [text before, role, text, role, text, ...]
            for who, v in zip(parts[1::2], parts[2::2]):
                v = v.replace("<|endoftext|>", "").strip()
                if who == "USER":
                    msgs.append({"role": "user", "content": v})
                elif who == "FUNCTION RESPONSE":
                    msgs.append({"role": "tool", "content": v})
                elif "<functioncall>" in v:
                    m = _GLAIVE_CALL.search(v)
                    args = m.group(2).strip()
                    args = json.loads(args[1:-1] if args[:1] == "'" else args)
                    msgs.append({"role": "assistant", "think": None, "content": v[:v.index("<functioncall>")].strip(),
                                 "tool_calls": [{"name": m.group(1), "arguments": args}]})
                else:
                    msgs.append({"role": "assistant", "think": None, "content": v})
            if len(msgs) < 2:
                raise ValueError("empty")
            yield {"id": f"glaive_v2/{i}", "tools": tools or None, "messages": msgs, "meta": {}}, None
        except Exception as e:  # noqa: BLE001
            yield None, f"parse: {type(e).__name__}"


_KW = re.compile(r"\s*([A-Za-z_][\w\-\.]*)\s*=")


def _group_end(body, j):
    """Index after the ")" matching the "(" at body[j], ignoring brackets inside string literals."""
    depth, k, quote = 1, j + 1, None
    while depth:
        ch = body[k]
        if quote:
            if ch == "\\":
                k += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        k += 1
    return k


def _split_top(text):
    """Split at top-level commas (outside brackets and string literals)."""
    parts, depth, quote, start, k = [], 0, None, 0, 0
    while k < len(text):
        ch = text[k]
        if quote:
            if ch == "\\":
                k += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:k])
            start = k + 1
        k += 1
    return parts + [text[start:]] if text[start:].strip() else parts


def toolace_calls(text):
    """'[Name (With) Spaces(a="x", from=1, ok=true), Other()]' -> [{"name", "arguments"}]; raises on anything else.
    A name may contain brackets: its argument list is the first "(...)" group that is empty or starts with key= and is
    followed by a comma or the end. Values are JSON (true / false / null) or Python literals."""
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError("not a call list")
    body, calls, i = text[1:-1], [], 0
    while i < len(body):
        j = i
        while True:
            j = body.index("(", j)
            k = _group_end(body, j)
            inner = body[j + 1:k - 1]
            if (not inner.strip() or _KW.match(inner)) and body[k:].lstrip()[:1] in ("", ","):
                break
            j += 1
        name, args = body[i:j].strip(" ,\n"), {}
        if not name:
            raise ValueError("call without a name")
        for part in _split_top(inner):
            m = _KW.match(part)
            if not m:
                raise ValueError("positional argument")
            v = part[m.end():].strip()
            try:
                args[m.group(1)] = json.loads(v)
            except json.JSONDecodeError:
                args[m.group(1)] = ast.literal_eval(v)
        calls.append({"name": name, "arguments": args})
        i = k
    return calls


def conv_tools_toolace(a):
    mark = "Here is a list of functions in JSON format that you can invoke:"
    for i, r in enumerate(json.load(open(os.path.join(RAW, "toolace", "data.json")))):
        try:
            tools = [as_tool(t) for t in json_values(r["system"][r["system"].index(mark):])[0]]
            msgs = []
            for m in r["conversations"]:
                who, v = m["from"], m["value"]
                if who == "user":
                    msgs.append({"role": "user", "content": v})
                elif who == "tool":
                    msgs.append({"role": "tool", "content": v})
                elif v.strip().startswith("[") and "(" in v and msgs[-1]["role"] != "tool":
                    msgs.append({"role": "assistant", "think": None, "content": "", "tool_calls": toolace_calls(v)})
                else:
                    msgs.append({"role": "assistant", "think": None, "content": v})
            yield {"id": f"toolace/{i}", "tools": tools, "messages": msgs, "meta": {}}, None
        except Exception as e:  # noqa: BLE001
            yield None, f"parse: {type(e).__name__}"


SETS = {  # name -> (converter, licence, decontaminate)
    "math": (conv_math, "cc-by-4.0 (nvidia/OpenMathInstruct-2)", True),
    "gsm8k": (conv_gsm8k, "mit (openai/gsm8k)", True),
    "gsm8k_py": (conv_gsm8k_py, "mit (openai/gsm8k)", True),
    "openthoughts": (conv_openthoughts, "apache-2.0 (open-thoughts/OpenThoughts-114k)", True),
    "openthoughts_mid": (conv_openthoughts_mid, "apache-2.0 (open-thoughts/OpenThoughts-114k)", True),
    "smoltalk": (conv_smoltalk, "apache-2.0 (HuggingFaceTB/smoltalk)", True),
    "smoltalk_long": (conv_smoltalk_long, "apache-2.0 (HuggingFaceTB/smoltalk)", True),
    "narrativeqa": (conv_narrativeqa, "apache-2.0 (deepmind/narrativeqa; the books are Project Gutenberg texts)", False),
    "code_selfoss": (conv_code_selfoss, "odc-by (bigcode/self-oss-instruct-sc2-exec-filter-50k)", True),
    "code_rstar": (conv_rstar, "cc-by-4.0 (microsoft/rStar-Coder, synthetic_sft)", True),
    "tools_hermes": (conv_tools_hermes, "apache-2.0 (NousResearch/hermes-function-calling-v1)", False),
    "tools_glaive": (conv_tools_glaive, "apache-2.0 (glaiveai/glaive-function-calling-v2)", False),
    "tools_toolace": (conv_tools_toolace, "apache-2.0 (Team-ACE/ToolACE)", False),
}


def eval_texts():
    """Evaluation problems nothing in the SFT data may copy: HumanEval and MBPP (decontam.benchmark_texts), GSM8K test
    questions, MATH-500 problems."""
    texts = [t for ts in decontam.benchmark_texts().values() for t in ts]
    with open(os.path.join(decontam.BENCH_DIR, "gsm8k", "test.jsonl")) as f:
        texts += [json.loads(line)["question"] for line in f]
    with open(os.path.join(RAW, "math500", "test.jsonl")) as f:
        texts += [json.loads(line)["problem"] for line in f]
    return texts


_GRAMS = _SHORT = None


def _init(grams, short):
    global _GRAMS, _SHORT
    _GRAMS, _SHORT = grams, short


def _finish(job):
    """Worker: render check + decontamination of a batch; returns (lines, drop reasons, rendered characters)."""
    name, licence, decon, batch = job
    lines, drops, chars = [], collections.Counter(), []
    for c in batch:
        c = {"id": c["id"], "set": name, "licence": licence, "messages": c["messages"], "tools": c.get("tools"),
             "meta": c.get("meta") or {}}
        try:
            text = chat.render(c)
        except Exception as e:  # noqa: BLE001
            drops[f"render: {type(e).__name__}"] += 1
            continue
        if not any(m["role"] == "assistant" for m in c["messages"]) or c["messages"][-1]["role"] != "assistant":
            drops["no final assistant turn"] += 1
            continue
        if decon and decontam.contaminated(text, _GRAMS, _SHORT):
            drops["shares a 13-gram with an evaluation problem"] += 1
            continue
        lines.append(json.dumps(c, ensure_ascii=False))
        chars.append(len(text))
    return lines, drops, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SETS))
    ap.add_argument("--max-think", type=int, default=6000, help="longest reasoning kept, in characters (~1500 tokens)")
    ap.add_argument("--max-chars", type=int, default=24000, help="longest short-set conversation, in characters")
    ap.add_argument("--max-doc-words", type=int, default=200_000, help="longest NarrativeQA book, in words")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="stop each set after this many source rows (tests)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = a.out
    os.makedirs(out, exist_ok=True)
    grams = frozenset(g for t in eval_texts() for g in decontam.ngrams(decontam.words(t)))
    short = decontam.benchmark_short_statements()
    print(f"{len(grams)} evaluation 13-grams", flush=True)
    stats_path = os.path.join(out, "STATS.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(grams, short)) as pool:
        for name in a.only:
            fn, licence, decon = SETS[name]
            t0, seen, drops, chars, kept = time.time(), 0, collections.Counter(), [], 0
            early = collections.Counter()  # dropped by the converter (filled from the pool's feeder thread)

            def jobs():
                nonlocal seen
                batch = []
                for c, why in fn(a):
                    seen += 1
                    if c is None:
                        early[why] += 1
                    else:
                        batch.append(c)
                        if len(batch) == 2000:
                            yield name, licence, decon, batch
                            batch = []
                    if a.limit and seen >= a.limit:
                        break
                if batch:
                    yield name, licence, decon, batch

            tmp = os.path.join(out, name + ".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for lines, d, ch in pool.imap(_finish, jobs()):
                    drops.update(d)
                    chars += ch
                    kept += len(lines)
                    if lines:
                        f.write("\n".join(lines) + "\n")
            os.replace(tmp, os.path.join(out, name + ".jsonl"))
            drops.update(early)
            chars.sort()
            q = (lambda p: chars[int(p * (len(chars) - 1))]) if chars else (lambda p: 0)
            stats[name] = {"rows": seen, "kept": kept, "dropped": dict(drops), "licence": licence,
                           "chars": {"total": sum(chars), "median": q(.5), "p90": q(.9), "max": q(1)},
                           "max_think": a.max_think, "limit": a.limit}
            print(f"{name}: {kept}/{seen} kept, {sum(chars) / 1e6:.1f}M chars (median {q(.5)}, p90 {q(.9)}), "
                  f"dropped {dict(drops)} [{time.time() - t0:.0f}s]", flush=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=1, ensure_ascii=False)
    print("SFT_CONVERT_DONE", flush=True)


if __name__ == "__main__":
    main()
