#!/usr/bin/env python
"""Synthetic long-input conversations for post-training, no API needed:
data/sft/conv/long_multidoc.jsonl and long_records.jsonl, in the format of moe_llm/chat.py.

    python data_prep/sft_long_synth.py            # needs data/sft/conv/smoltalk.jsonl (sft_convert.py)

  long_multidoc   many short documents (SmolTalk's smol-summarize inputs: emails, posts) numbered "Document k"; the
                  user asks for a one-sentence summary of one of them and the answer is that document's own summary
                  from the dataset. The reasoning names the document by its first words. Any length, by the number
                  of documents; the asked document sits at a uniformly random depth; in 30% the question comes first.
  long_records    a table of records "ID | owner | city | item | code" (random values, so nothing can be guessed);
                  three kinds of question: one field of one record, one field of three records, the IDs of all
                  records with a given owner (2 to 6 of them). The reasoning quotes the rows it used.

These are deliberately not the evaluation's tasks (runs/mac_eval/ruler_lite.py: "special magic number" needles and
variable chains in running text, passkeys in filler): the surface, the haystack and the questions differ, so RULER
and the passkey test stay held-out tests of retrieval. Lengths are approximate here (measured ratios: 24 tokens per
record row, 4.9 characters per token in the documents); the packer (data_prep/sft_long_packs.py) tokenizes and uses
the exact lengths.
"""
from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONV = os.path.join(ROOT, "data", "sft", "conv")
# (approximate tokens, how many conversations)
MULTIDOC = ((4000, 1500), (8000, 1500), (16000, 800), (32000, 400), (64000, 200), (115000, 60))
RECORDS = ((3000, 6000), (6000, 5000), (12000, 600), (30000, 200), (60000, 80), (115000, 30))
OWNERS = ("Avery Brandt Calla Dorian Elias Farah Gideon Hana Ines Jonas Kiran Leona Marek Nadia Orin Petra Quill Rhea "
          "Soren Talia Ugo Vera Wren Xavi Yara Zane Aldo Bryn Cato Dara Eamon Fenn Gwen Hale Ivo Juno Koa Lark Milo Noor "
          "Odell Pia Remy Sage Tove Uri Vale Wynn Yuki Zora").split()
CITIES = ("Arden Bellmore Corvin Dunmere Eastbay Fallow Glenhaven Harrow Ilford Juniper Kestrel Lowmoor Marlow Northam "
          "Oakhurst Penrose Quarry Redfern Stanwick Thornby Upland Verity Westmere Yarrow").split()
ITEMS = ("anchor basket candle drum easel funnel goblet hammer inkwell jar kettle ladder mallet needle oar pulley "
         "quiver rudder saddle trowel urn vise whistle yoke").split()


def summarize_pairs(limit=0):
    """(document text, its one-sentence summary) from SmolTalk's smol-summarize."""
    out = []
    with open(os.path.join(CONV, "smoltalk.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"smol-summarize"' not in line:
                continue
            m = json.loads(line)["messages"]
            if len(m) == 3 and 200 <= len(m[1]["content"]) <= 4000 and len(m[2]["content"]) <= 400:
                out.append((m[1]["content"].strip(), m[2]["content"].strip()))
                if limit and len(out) >= limit:
                    break
    return out


def multidoc(pairs, n_tokens, rng, i):
    docs, chars = [], 0
    while chars < n_tokens * 4.9 or len(docs) < 3:  # measured: 4.9 characters per token in these documents
        d = pairs[rng.randrange(len(pairs))]
        docs.append(d)
        chars += len(d[0]) + 20
    k = rng.randrange(len(docs))
    body = "\n\n".join(f"Document {j + 1}\n{d[0]}" for j, d in enumerate(docs))
    ask = f"Give the main point of Document {k + 1} in one very short sentence, with essential details such as dates or places."
    user = (ask + "\n\n" + body) if rng.random() < 0.3 else (body + "\n\n" + ask)
    first = " ".join(docs[k][0].split()[:8])
    think = f'Document {k + 1} is the one that begins "{first}". I summarize only that document.'
    return {"id": f"long_multidoc/{i}", "set": "long_multidoc", "licence": "apache-2.0 (HuggingFaceTB/smoltalk), arranged here",
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "think": think, "content": docs[k][1]}],
            "tools": None, "meta": {"approx_tokens": n_tokens, "documents": len(docs), "target": k + 1,
                                    "depth": round(k / max(1, len(docs) - 1), 3)}}


def records(n_tokens, rng, i):
    n = max(8, n_tokens // 24)  # a row is 24 tokens: the tokenizer splits digits one by one
    owners = rng.sample(OWNERS, min(len(OWNERS), max(6, n // 4)))
    ids = rng.sample(range(100000, 999999), n)
    rows = [{"id": f"R{x}", "owner": rng.choice(owners), "city": rng.choice(CITIES), "item": rng.choice(ITEMS),
             "code": str(rng.randrange(10000, 99999))} for x in ids]
    line = lambda r: f"{r['id']} | {r['owner']} | {r['city']} | {r['item']} | {r['code']}"  # noqa: E731
    table = "ID | owner | city | item | code\n" + "\n".join(line(r) for r in rows)
    kind = rng.choice(("one", "three", "owner"))
    field = rng.choice(("city", "item", "code", "owner"))
    if kind == "one":
        r = rng.choice(rows)
        q = f"What is the {field} of record {r['id']}?"
        think = f"The row for {r['id']} is: {line(r)}"
        ans = f"The {field} of record {r['id']} is {r[field]}."
    elif kind == "three":
        pick = rng.sample(rows, 3)
        q = f"What are the {field} values of records {pick[0]['id']}, {pick[1]['id']} and {pick[2]['id']}?"
        think = "The rows are:\n" + "\n".join(line(r) for r in pick)
        ans = "\n".join(f"{r['id']}: {r[field]}" for r in pick)
    else:
        by = {}
        for r in rows:
            by.setdefault(r["owner"], []).append(r)
        ok = [o for o, rs in by.items() if 2 <= len(rs) <= 6]
        if not ok:
            return records(n_tokens, rng, i)
        o = rng.choice(ok)
        q = f"List the IDs of all records whose owner is {o}."
        think = f"The rows with owner {o} are:\n" + "\n".join(line(r) for r in by[o])
        ans = ", ".join(r["id"] for r in by[o])
    user = (q + "\n\n" + table) if rng.random() < 0.3 else (table + "\n\n" + q)
    return {"id": f"long_records/{i}", "set": "long_records", "licence": "generated (data_prep/sft_long_synth.py)",
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "think": think, "content": ans}],
            "tools": None, "meta": {"approx_tokens": n_tokens, "rows": n, "kind": kind}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=CONV)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every count (tests: 0.01)")
    ap.add_argument("--pairs-limit", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(0)
    pairs = summarize_pairs(a.pairs_limit)
    print(f"{len(pairs)} summarize documents", flush=True)
    for name, plan, make in (("long_multidoc", MULTIDOC, lambda n, i: multidoc(pairs, n, rng, i)),
                             ("long_records", RECORDS, lambda n, i: records(n, rng, i))):
        i, chars = 0, 0
        with open(os.path.join(a.out, name + ".jsonl.tmp"), "w", encoding="utf-8") as f:
            for n_tokens, count in plan:
                for _ in range(max(1, int(count * a.scale))):
                    c = make(int(n_tokens * rng.uniform(0.75, 1.0)), i)
                    chars += sum(len(m["content"]) for m in c["messages"])
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
                    i += 1
        os.replace(os.path.join(a.out, name + ".jsonl.tmp"), os.path.join(a.out, name + ".jsonl"))
        print(f"{name}: {i} conversations, {chars / 1e6:.1f}M characters", flush=True)
    print("SFT_LONG_SYNTH_DONE", flush=True)


if __name__ == "__main__":
    main()
