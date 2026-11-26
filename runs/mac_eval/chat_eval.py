#!/usr/bin/env python
"""Chat-format evaluation of a post-trained checkpoint: the model is
prompted with the chat template (moe_llm/chat.py: ChatML, generation prompt ending in "<think>"), writes its thinking
and its reply greedily, and only the reply (content) is scored. The same batched decoding as passk.py / long_ctx.py.

    # 4090 (WSL), after the post-training run, on an exported checkpoint:
    python runs/mac_eval/chat_eval.py --export runs/mac_eval/sft_0000123 --device cuda --moe-impl bmm \
        --tasks gsm8k,math500,humaneval,mbpp,tools,longdoc --out runs/mac_eval/chat_sft_0000123.json
    python runs/mac_eval/chat_eval.py --export ... --tasks passkey,ruler --lengths 4096,32768,131072 --n-long 20
    python runs/mac_eval/chat_eval.py --check --export ...      # batched rows must equal single-sequence decoding

tasks (generation limit in tokens, thinking included):
  gsm8k      all 1319 test problems, zero-shot, the question alone as the user turn (1536). strict = the last boxed
             value, flexible = the last number of the reply
  math500    MATH-500 (data/raw/sft/math500), last boxed answer against the reference after LaTeX normalisation (2048)
  humaneval  164 problems: "Complete the following Python function." and the prompt in a code block; the first code
             block of the reply runs with the official tests in code_check's sandbox (1024)
  mbpp       the 500 test problems (task ids 11-510) as "<text>\\nYour code should pass this test: <assert>" (1024)
  tools      function-calling decisions (data/sft/eval/tools_test.jsonl, data_prep/sft_eval_sets.py): per item the model
             writes the first assistant turn; with a reference call: made a call, right function(s), same arguments;
             without: made no call (512)
  longdoc    the held-out documents (data/sft/eval/longdoc_test.jsonl, distill_convert.py): document and question in the
             user turn, --per-doc questions per document; F1 and recall of the reference answer's words (512)
  passkey    the pass key hidden in filler text, --lengths x --n-long depths; share found (256)
  ruler      RULER-style needles (single, multikey, multivalue, variable tracking) in held-out text, asked as a chat
             question; share of the wanted strings found (384)
Results are the means per task (and per group for the long-context tasks), plus per task the share of replies whose
thinking never closed (unclosed), that hit the generation limit (truncated) and the mean tokens written. Every reply is
kept in <out>_samples.jsonl, thinking and content apart.

Not comparable with the base-model numbers of gen_eval.py / ruler_lite.py (few-shot completions of a base model):
this measures the post-trained assistant, so it is compared with the pre-post-training checkpoint only on the
tasks that have a chat form (run both through this script, the base model with an empty thinking block being a poor
chat model, so expect low numbers there).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
import chat_score as S  # noqa: E402
from moe_llm import chat  # noqa: E402

BENCH = os.path.join(ROOT, "data", "raw", "bench")
EVAL = os.path.join(ROOT, "data", "sft", "eval")
MAX_NEW = {"gsm8k": 1536, "math500": 2048, "humaneval": 1024, "mbpp": 1024, "tools": 512, "longdoc": 512,
           "passkey": 256, "ruler": 384}
HE_INSTRUCTION = "Complete the following Python function."
PASSKEY_TAIL = "\n\nWhat is the pass key? Reply with the pass key only."
RULER_QUESTIONS = {
    "Q_ONE": "\n\nWhat is the special magic number for {key} mentioned in the provided text? Reply with the number only.",
    "Q_ALL": "\n\nWhat are all the special magic numbers for {key} mentioned in the provided text? List all of them.",
    "VT_Q": "\n\nFind all variables that are assigned the value {val} in the text above. List all of them."}
RULER_TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")
# prompts of 32K tokens and more: the 'bmm' MoE implementation (fast with many short rows) needs [experts, capacity, d]
# buffers and ran the 4090 out of memory while prefilling a 32K prompt in the rehearsal (2026-11-22)
LONG_TASKS = ("passkey", "ruler", "longdoc")


def read_jsonl(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


# ---- the items of every task: {"id", "conv", ...what scoring needs} -------------------------------------------------
def gsm8k_items(limit=0):
    data = read_jsonl(BENCH, "gsm8k", "test.jsonl")[:limit or None]
    return [{"id": f"gsm8k/{i}", "conv": user(d["question"]), "gold": d["answer"].split("####")[-1].strip().replace(",", "")}
            for i, d in enumerate(data)]


def math500_items(limit=0):
    data = read_jsonl(ROOT, "data", "raw", "sft", "math500", "test.jsonl")[:limit or None]
    return [{"id": f"math500/{i}", "conv": user(d["problem"]), "gold": d["answer"], "group": f"level {d.get('level')}"}
            for i, d in enumerate(data)]


def humaneval_items(limit=0):
    data = read_jsonl(BENCH, "humaneval", "openai_humaneval", "test-00000-of-00001.jsonl")[:limit or None]
    return [{"id": d["task_id"], "conv": user(f"{HE_INSTRUCTION}\n\n```python\n{d['prompt'].rstrip()}\n```"), "data": d}
            for d in data]


def mbpp_items(limit=0):
    data = read_jsonl(BENCH, "mbpp", "full", "test-00000-of-00001.jsonl")[:limit or None]
    return [{"id": f"mbpp/{d['task_id']}", "conv": user(f"{d['text']}\nYour code should pass this test: {d['test_list'][0]}"),
             "data": d} for d in data]


def tools_items(limit=0):
    path = os.path.join(EVAL, "tools_test.jsonl")
    if not os.path.exists(path):
        sys.exit(f"{path} is missing: run data_prep/sft_eval_sets.py")
    rows = read_jsonl(path)[:limit or None]
    return [{"id": r["id"], "conv": {"messages": r["messages"], "tools": r["tools"]}, "ref_calls": r["ref_calls"],
             "group": r["source"]} for r in rows]


def longdoc_items(limit=0, per_doc=4, seed=0):
    path = os.path.join(EVAL, "longdoc_test.jsonl")
    if not os.path.exists(path):
        sys.exit(f"{path} is missing: it is written by data_prep/distill_convert.py after the teacher asked its questions")
    items = []
    for r in read_jsonl(path)[:limit or None]:
        doc = open(os.path.join(ROOT, r["document"]), encoding="utf-8").read()
        for k, q in enumerate(random.Random(f"{seed}/{r['id']}").sample(r["qa"], min(per_doc, len(r["qa"])))):
            items.append({"id": f"{r['id']}/{k}", "conv": user(doc + "\n\n" + q["question"]), "gold": q["answer"]})
    return items


@contextlib.contextmanager
def patched(module, **attrs):
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def passkey_items(tok, lengths, n, seed=0):
    import long_ctx
    items = []
    with patched(long_ctx, TAIL=PASSKEY_TAIL):
        for length in lengths:
            rng = random.Random(f"{seed}/passkey/{length}")
            for k in range(n):
                key = str(rng.randint(10000, 99999))
                ids = long_ctx.passkey_ids(tok, length, (k + 0.5) / n, key)  # ids start with the EOS that stands for BOS
                items.append({"id": f"passkey/{length}/{k}", "group": str(length), "conv": user(tok.decode(ids[1:])),
                              "wants": [key]})
    return items


def ruler_items(tok, lengths, n, seed=0):
    import ruler_lite
    sents = ruler_lite.haystack_sentences(tok)
    items = []
    with patched(ruler_lite, **RULER_QUESTIONS):
        for length in lengths:
            for task in RULER_TASKS:
                rng = random.Random(f"{seed}/ruler/{task}/{length}")
                for k in range(n):
                    ids, wants, _ = ruler_lite.sample(task, tok, sents, length, rng)
                    items.append({"id": f"ruler/{task}/{length}/{k}", "group": f"{task}@{length}",
                                  "conv": user(tok.decode(ids[1:])), "wants": wants})
    return items


# ---- scoring ----------------------------------------------------------------------------------------------------
def score(task, item, parsed):
    """(metrics: name -> bool or float, detail or None) of one parsed reply."""
    content = parsed["content"]
    if task == "gsm8k":
        strict, flexible = S.gsm8k_scores(content, item["gold"])
        return {"strict": strict, "flexible": flexible}, None
    if task == "math500":
        return {"correct": S.math_correct(content, item["gold"])}, None
    if task in ("humaneval", "mbpp"):
        ok, why = (S.humaneval_check if task == "humaneval" else S.mbpp_check)(item["data"], content)
        return {"pass": ok}, None if ok else why
    if task == "tools":
        return S.tool_scores(item["ref_calls"], parsed["tool_calls"]), None
    if task == "longdoc":
        return {"f1": S.answer_f1(content, item["gold"]), "recall": S.answer_recall(content, item["gold"])}, None
    return {"found": S.found_share(content, item["wants"])}, None  # passkey, ruler


def summarize(rows):
    """Mean of every metric over the rows that have it (the function-calling metrics exist for a part of the items)."""
    out = {"n": len(rows)}
    for key in sorted({k for r in rows for k in r}):
        vals = [r[key] for r in rows if key in r]
        out[key] = round(sum(map(float, vals)) / len(vals), 4)
        if len(vals) != len(rows):
            out[key + "_n"] = len(vals)
    return out


# ---- generation ---------------------------------------------------------------------------------------------------
def generate(model, tok, convs, max_new, rows_at_32k=10, max_rows=128, label=""):
    """Greedy replies to conversations (messages up to the last user turn, optional "tools"): a list, in the order of
    convs, of {"text" (special tokens written out), "ids", "tokens", "prompt_tokens"}. Prompts of similar length share a
    decode.generate_batch() call (at most max_rows, fewer for long prompts: decode.batch_rows); a row ends at <|im_end|>
    or the EOS."""
    import decode
    im_end = chat.SPECIAL_IDS[chat.IM_END]
    ids = [[chat.SPECIAL_IDS[chat.EOS]] + tok.encode(chat.render(c, generation_prompt=True)).ids for c in convs]
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    outs, i, t0 = [None] * len(ids), 0, time.time()
    while i < len(order):
        n = min(max_rows, len(order) - i)
        while n > 1 and n > decode.batch_rows(len(ids[order[i + n - 1]]) + max_new, rows_at_32k):
            n -= 1
        chunk = order[i:i + n]
        caches = [decode.prefill(model, ids[j][:-1]) for j in chunk]
        gen = decode.generate_batch(model, caches, [ids[j][-1] for j in chunk], max_new,
                                    done=lambda b, o: o[-1] == chat.SPECIAL_IDS[chat.EOS], eos=im_end)
        for j, g in zip(chunk, gen):
            outs[j] = {"ids": g, "text": tok.decode(g, skip_special_tokens=False), "tokens": len(g),
                       "prompt_tokens": len(ids[j])}
        i += n
        print(f"  {label}: {i}/{len(order)} replies [{time.time() - t0:.0f}s]", flush=True)
    return outs


def run_task(task, items, model, tok, args, samples):
    max_new = args.max_new or MAX_NEW[task]
    t0 = time.time()
    outs = generate(model, tok, [it["conv"] for it in items], max_new, args.rows_at_32k, args.max_rows, task)
    rows, groups = [], {}
    for it, o in zip(items, outs):
        p = chat.parse_assistant(o["text"])
        metrics, detail = score(task, it, p)
        row = {**metrics, "unclosed": not p["closed"], "truncated": o["tokens"] >= max_new, "tokens": o["tokens"]}
        rows.append(row)
        if it.get("group"):
            groups.setdefault(it["group"], []).append(row)
        if samples is not None:
            samples.write(json.dumps({"task": task, "id": it["id"], "prompt_tokens": o["prompt_tokens"], "think": p["think"],
                                      "content": p["content"], "tool_calls": p["tool_calls"], "closed": p["closed"],
                                      "tokens": o["tokens"], "metrics": metrics, "detail": detail,
                                      "raw_end": o["text"][-240:]},
                                     ensure_ascii=False) + "\n")
            samples.flush()
    res = {"overall": summarize(rows), "seconds": round(time.time() - t0)}
    if groups:
        res["groups"] = {g: summarize(r) for g, r in sorted(groups.items())}
    return res


def set_moe_impl(model, impl):
    from moe_llm.layers.moe import MoE
    for m in model.modules():
        if isinstance(m, MoE):
            m.impl = impl


def check(model, tok):
    """Batched rows must equal single-sequence decoding of the same prompt (up to the first EOS token)."""
    import decode
    items = gsm8k_items(3)
    batched = generate(model, tok, [it["conv"] for it in items], 24, 10, 8, "check")
    eos, im_end = chat.SPECIAL_IDS[chat.EOS], chat.SPECIAL_IDS[chat.IM_END]
    for it, o in zip(items, batched):
        prompt = [eos] + tok.encode(chat.render(it["conv"], generation_prompt=True)).ids
        ref = decode.generate_ids(model, prompt, 24, eos=im_end)
        cut = lambda xs: xs[:xs.index(eos)] if eos in xs else xs  # noqa: E731
        same = cut(ref) == cut(o["ids"])
        print(f"{it['id']}: batched == single-sequence: {same}\n  {tok.decode(o['ids'], skip_special_tokens=False)[:160]!r}",
              flush=True)
        if not same:
            raise SystemExit("CHAT_EVAL_CHECK_FAILED")
    print("CHAT_EVAL_CHECK_OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--moe-impl", default="loop", help="'bmm' is usually faster on the GPU with many rows (used for "
                    "the short-prompt tasks; passkey, ruler and longdoc always use 'loop')")
    ap.add_argument("--tasks", default="gsm8k,math500,humaneval,mbpp,tools,longdoc")
    ap.add_argument("--limit", type=int, default=0, help="at most this many items per task (0: all)")
    ap.add_argument("--max-new", type=int, default=0, help="generation limit for every task (default: per task)")
    ap.add_argument("--max-rows", type=int, default=128, help="most replies decoded together")
    ap.add_argument("--rows-at-32k", type=int, default=10, help="replies decoded together for a 32K-token prompt")
    ap.add_argument("--per-doc", type=int, default=4, help="longdoc: questions per held-out document")
    ap.add_argument("--lengths", default="4096,32768,131072", help="passkey, ruler: prompt lengths in tokens")
    ap.add_argument("--n-long", type=int, default=20, help="passkey: depths per length; ruler: samples per task and length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import mac_lm
    model, tok, meta = mac_lm.load(args.export, args.device)
    set_moe_impl(model, args.moe_impl)
    if args.check:
        return check(model, tok)
    chat.check_tokenizer(tok)
    lengths = [int(x) for x in args.lengths.split(",")]
    builders = {"gsm8k": lambda: gsm8k_items(args.limit), "math500": lambda: math500_items(args.limit),
                "humaneval": lambda: humaneval_items(args.limit), "mbpp": lambda: mbpp_items(args.limit),
                "tools": lambda: tools_items(args.limit),
                "longdoc": lambda: longdoc_items(args.limit, args.per_doc, args.seed),
                "passkey": lambda: passkey_items(tok, lengths, args.n_long, args.seed),
                "ruler": lambda: ruler_items(tok, lengths, args.n_long, args.seed)}
    samples = open(args.out.replace(".json", "_samples.jsonl"), "w", encoding="utf-8") if args.out else None
    res = {"step": meta["step"], "tokens": meta["tokens"], "args": vars(args), "tasks": {}}
    for task in args.tasks.split(","):
        items = builders[task]()
        set_moe_impl(model, "loop" if task in LONG_TASKS else args.moe_impl)
        print(f"{task}: {len(items)} prompts", flush=True)
        res["tasks"][task] = run_task(task, items, model, tok, args, samples)
        print(json.dumps({task: res["tasks"][task]["overall"]}), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=1)
    print("CHAT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
