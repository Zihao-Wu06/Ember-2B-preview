#!/usr/bin/env python
"""Documents for the real long-document questions:
1000 documents, of which 50 are held out as a test set.

    python data_prep/sft_longdocs.py

writes
    data/distill/longdocs/<id>.txt        950 training documents: PG19 training books and long web articles, each cut
                                          to 70K-110K tokens at a paragraph break (the questions and answers of a
                                          document must fit a 128K training window with it)
    data/distill/longdocs_test/<id>.txt   50 test documents: PG19 validation and test books, which no training or
                                          pre-training data contains
    data/distill/longdocs_manifest.json   id, source, tokens, characters, starting offset of every document

and the questions for them come from `distill_gen.py longdoc` (training) and `distill_gen.py longdoc --docs test`.

A document is kept only if it reads like running prose (letters, English function words, no repeated lines, no tables,
not a reference work). A test document that shares more than 2% of its 8-word sequences with any training document
(another edition of the same book) is replaced. The selection is seeded: running it again gives the same files, and it
refuses to run once data/distill/longdoc.jsonl exists (the questions refer to these files) unless --force is given.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BINS = os.path.join(ROOT, "data", "bins_main")
OUT = os.path.join(ROOT, "data", "distill")
TOK = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
STOP = frozenset("the of and to a in that is was he for it with as his on be at by i this had not are but".split())
REFERENCE = re.compile(r"complete works|collected works|encyclop|dictionary|concordance|catalogue|index of|"
                       r"project gutenberg", re.I)


def judge(text: str):
    """None if the text reads like running prose, else the reason it does not."""
    n = len(text)
    if n < 150_000:
        return "too short"
    if sum(c.isalpha() for c in text) / n < 0.70:
        return "few letters"
    if sum(ord(c) > 127 for c in text) / n > 0.03:
        return "not English"
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 25_000 or sum(w.lower() in STOP for w in words) / len(words) < 0.28:
        return "not running English prose"
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if len(set(lines)) / len(lines) < 0.85:
        return "repeated lines"
    if sum("|" in x for x in lines) / len(lines) > 0.03:
        return "tables"
    if REFERENCE.search(text[:3000]):
        return "reference work or licence text"
    return None


def paragraph_cut(text: str, truncated: bool) -> str:
    """A document cut off at the token limit ends at the last paragraph break of its final 4,000 characters."""
    if truncated and "\n\n" in text[-4000:]:
        text = text[:text.rfind("\n\n")]
    return text.strip()


def shingles(text: str, step: int = 1, n: int = 8) -> np.ndarray:
    """Hashes of the text's 8-word sequences (every step-th); equal only within one process."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    return np.fromiter((hash(" ".join(w[i:i + n])) for i in range(0, len(w) - n + 1, step)), dtype=np.int64)


def head_key(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text[:600]).lower().encode()).hexdigest()


def pick_training(src, n, min_tokens, rng, tok, a, seen, why):
    """n documents of one pre-training source (data/bins_main/<src>) that pass the checks."""
    from moe_llm.data import TokenStream
    docs = np.load(os.path.join(BINS, src, "long_docs.npz"))["docs"]
    order = [i for i in range(len(docs)) if docs[i, 1] >= min_tokens]
    rng.shuffle(order)
    ts, out = TokenStream(os.path.join(BINS, src)), []
    for i in order:
        if len(out) >= n:
            break
        start, length = int(docs[i, 0]), int(docs[i, 1])
        limit = rng.randint(a.len_lo, a.len_hi)
        text = paragraph_cut(tok.decode(ts.read(start, min(length, limit)).tolist()), length > limit)
        reason = judge(text)
        key = head_key(text)
        if reason is None and key in seen:
            reason = "same beginning as another document"
        if reason:
            why[f"{src}: {reason}"] += 1
            continue
        seen.add(key)
        out.append({"id": f"{src}_{i:05d}", "source": src, "offset": start, "text": text})
    return out


def pick_test(n, rng, tok, a, train_hashes, why):
    """n PG19 validation / test books (data/eval/pg19_*.npz) that pass the checks and are not another edition of a
    training document."""
    cands = []
    for split in ("test", "validation"):
        z = np.load(os.path.join(ROOT, "data", "eval", f"pg19_{split}.npz"))
        ids, off, titles = z["ids"], z["offsets"], z["titles"]
        for i in range(len(off) - 1):
            if off[i + 1] - off[i] >= a.min_tokens_book and not REFERENCE.search(str(titles[i])):
                cands.append((split, i, ids[off[i]:off[i + 1]], str(titles[i])))
    rng.shuffle(cands)
    out = []
    for split, i, ids, title in cands:
        if len(out) >= n:
            break
        limit = rng.randint(a.len_lo, a.len_hi)
        text = paragraph_cut(tok.decode(ids[:limit].tolist()), len(ids) > limit)
        reason = judge(text)
        if reason is None and np.isin(shingles(text), train_hashes).mean() > 0.02:
            reason = "shares text with a training document"
        if reason:
            why[f"test: {reason}"] += 1
            continue
        out.append({"id": f"pg19eval_{split}_{i:03d}", "source": f"pg19_{split}", "offset": 0, "text": text,
                    "title": title})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pg19", type=int, default=620, help="training documents from the PG19 training books")
    ap.add_argument("--n-web", type=int, default=330, help="training documents from long web articles (en_web)")
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--min-tokens-book", type=int, default=60_000)
    ap.add_argument("--min-tokens-web", type=int, default=45_000)
    ap.add_argument("--len-lo", type=int, default=70_000, help="a document is cut to a length drawn from lo..hi tokens")
    ap.add_argument("--len-hi", type=int, default=110_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    done = os.path.join(a.out, "longdoc.jsonl")
    if os.path.exists(done) and os.path.getsize(done) and not a.force:
        sys.exit(f"{done} exists: its questions refer to the current documents; use --force to select again")
    from tokenizers import Tokenizer
    tok, rng, why, t0 = Tokenizer.from_file(TOK), random.Random(a.seed), collections.Counter(), time.time()
    seen = set()
    train = pick_training("pg19", a.n_pg19, a.min_tokens_book, rng, tok, a, seen, why)
    print(f"pg19 books: {len(train)} chosen [{time.time() - t0:.0f}s]", flush=True)
    train += pick_training("en_web", a.n_web, a.min_tokens_web, rng, tok, a, seen, why)
    print(f"{len(train)} training documents [{time.time() - t0:.0f}s]", flush=True)
    train_hashes = np.unique(np.concatenate([shingles(d["text"], 11) for d in train]))
    test = pick_test(a.n_test, rng, tok, a, train_hashes, why)
    print(f"{len(test)} test documents [{time.time() - t0:.0f}s]", flush=True)
    manifest = {"seed": a.seed, "train": [], "test": [], "rejected": dict(why)}
    for split, docs in (("longdocs", train), ("longdocs_test", test)):
        d = os.path.join(a.out, split)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".txt"):
                os.remove(os.path.join(d, f))
        toks = [len(e.ids) for e in tok.encode_batch([x["text"] for x in docs])]
        for x, n_tok in zip(docs, toks):
            with open(os.path.join(d, x["id"] + ".txt"), "w", encoding="utf-8") as f:
                f.write(x["text"])
            manifest["train" if split == "longdocs" else "test"].append(
                {k: x[k] for k in ("id", "source", "offset") if k in x} | {"tokens": n_tok, "chars": len(x["text"]),
                                                                           **({"title": x["title"]} if "title" in x else {})})
    with open(os.path.join(a.out, "longdocs_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    for name in ("train", "test"):
        m = manifest[name]
        t = sorted(x["tokens"] for x in m)
        by = collections.Counter(x["source"] for x in m)
        print(f"{name}: {len(m)} documents {dict(by)}, tokens min {t[0]} median {t[len(t) // 2]} max {t[-1]}, "
              f"total {sum(t) / 1e6:.1f}M")
    print("rejected:", dict(why))
    print("SFT_LONGDOCS_DONE", flush=True)


if __name__ == "__main__":
    main()
