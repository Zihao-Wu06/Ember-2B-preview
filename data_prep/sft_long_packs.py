#!/usr/bin/env python
"""Long and medium conversations -> the long: source of the SFT mix:
data/bins_sft/sft_long/NNN.bin plus long_docs.npz, the index moe_llm.data.MixtureLoader reads for "long:sft_long".

    python data_prep/sft_long_packs.py                         # packs of up to 131072 tokens
    python data_prep/sft_long_packs.py --pack-len 65536 --name sft_long64k   # for training at 64K on the 4090
    python data_prep/sft_long_packs.py --name sft_hard --sets --medium sft_code_pub sft_code_rstar sft_reason   # below

The loader gives a "long:" source one training window per document of more than half a window, starting at the EOS
before it. A conversation of 4K-64K tokens would not qualify, and in a plain source it would be cut into rows of 4096
tokens. So conversations are packed: every index entry is a pack of whole conversations (each followed by EOS)
totalling at most --pack-len tokens, filled first-fit from the longest conversation down and then shuffled inside
the pack. A long sample thus trains with its full context; what precedes it in the window is other conversations.

Input: the conversation sets in SETS that exist (rendered and tokenized here; --sets picks some or none) and the
conversations data_prep/sft_tokenize.py set aside in <bins>/_medium/<source>.npz for the sources in --medium.
Conversations longer than a pack are left out; packs that end up at half a pack or less (they would give no window)
are left out too and reported.

sft_hard (2026-11-23): the hard problems of sft_code_pub, sft_code_rstar and sft_reason whose conversation is longer
than a row (thoughts of up to 12,000 characters), about 30 whole conversations to a pack of 131072 tokens. It
is a source of its own so that its weight is set apart from the long-document packs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from moe_llm import chat  # noqa: E402
from moe_llm.data import LONG_INDEX  # noqa: E402

CONV = os.path.join(ROOT, "data", "sft", "conv")
BINS = os.path.join(ROOT, "data", "bins_sft")
TOK = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
# smoltalk_long and narrativeqa have no reasoning: they are rendered with an empty thinking block
SETS = (("long_multidoc", "empty"), ("long_records", "empty"), ("longdoc_api", "empty"), ("smoltalk_long", "empty"),
        ("narrativeqa", "empty"))
MEDIUM = ("sft_chat", "sft_code", "sft_general", "sft_tools")  # the default packs' set-aside conversations


def pack(lengths, cap):
    """First-fit decreasing: lists of item indices, each list's lengths summing to at most cap; items longer than
    cap are not placed. Bins are searched in order, so the result is deterministic."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins, room = [], []
    for i in order:
        n = lengths[i]
        if n > cap:
            continue
        for b in range(len(bins)):
            if room[b] >= n:
                bins[b].append(i)
                room[b] -= n
                break
        else:
            bins.append([i])
            room.append(cap - n)
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", default=BINS)
    ap.add_argument("--conv", default=CONV)
    ap.add_argument("--name", default="sft_long")
    ap.add_argument("--pack-len", type=int, default=131072)
    ap.add_argument("--limit", type=int, default=0, help="first N conversations of each set (tests)")
    ap.add_argument("--sets", nargs="*", default=[n for n, _ in SETS], help="conversation sets to pack (may be none)")
    ap.add_argument("--medium", nargs="*", default=list(MEDIUM), help="sources whose _medium/<source>.npz is packed")
    a = ap.parse_args()
    if set(a.sets) - {n for n, _ in SETS}:
        ap.error(f"unknown sets {sorted(set(a.sets) - {n for n, _ in SETS})}")
    from tokenizers import Tokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = Tokenizer.from_file(TOK)
    chat.check_tokenizer(tok)
    t0, items, stats = time.time(), [], {}
    for name, no_think in SETS:
        if name not in a.sets:
            continue
        path = os.path.join(a.conv, name + ".jsonl")
        if not os.path.exists(path):
            print(f"  set {name} is not there yet, skipped", flush=True)
            continue
        st = {"conversations": 0, "tokens": 0, "without_reasoning_dropped": 0}
        with open(path, encoding="utf-8") as f:
            batch = []

            def flush():
                for enc in tok.encode_batch(batch):
                    items.append(np.asarray(enc.ids + [0], dtype=np.uint16))
                    st["conversations"] += 1
                    st["tokens"] += len(enc.ids) + 1
                batch.clear()
            for k, line in enumerate(f):
                if a.limit and k >= a.limit:
                    break
                c = json.loads(line)
                if no_think == "drop" and not all((m.get("think") or "").strip() for m in c["messages"]
                                                  if m["role"] == "assistant"):
                    st["without_reasoning_dropped"] += 1
                    continue
                batch.append(chat.render(c))
                if len(batch) >= 64:
                    flush()
            flush()
        stats[name] = st
        print(f"  {name}: {st}", flush=True)
    med_dir = os.path.join(a.bins, "_medium")
    for f in sorted(m + ".npz" for m in a.medium):
        if not os.path.exists(os.path.join(med_dir, f)):
            continue
        z = np.load(os.path.join(med_dir, f))
        off, ids = z["offsets"], z["ids"]  # read once: every z["ids"] reads the whole array again and keeps it alive
        for i in range(len(off) - 1):
            items.append(ids[off[i]:off[i + 1]])
        stats["medium/" + f[:-4]] = {"conversations": len(off) - 1, "tokens": int(off[-1])}
        print(f"  medium/{f[:-4]}: {stats['medium/' + f[:-4]]}", flush=True)
    lengths = [len(x) for x in items]
    too_long = sum(n > a.pack_len for n in lengths)
    packs = pack(lengths, a.pack_len)
    rng = random.Random(0)
    rng.shuffle(packs)
    kept = [p for p in packs if sum(lengths[i] for i in p) > a.pack_len // 2]
    out_dir = os.path.join(a.bins, a.name)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".bin") or f == LONG_INDEX:
            os.remove(os.path.join(out_dir, f))
    docs, parts, pos = [], [np.zeros(1, dtype=np.uint16)], 1  # a leading EOS: the first pack needs one before it
    for p in kept:
        rng.shuffle(p)
        n = sum(lengths[i] for i in p)
        docs.append((pos, n))
        parts += [items[i] for i in p]
        pos += n
    stream = np.concatenate(parts)
    files = {}
    for k, s in enumerate(range(0, len(stream), 64_000_000)):
        name = f"{k:03d}.bin"
        stream[s:s + 64_000_000].tofile(os.path.join(out_dir, name))
        files[name] = int(len(stream[s:s + 64_000_000]))
    np.savez(os.path.join(out_dir, LONG_INDEX), docs=np.asarray(docs, dtype=np.int64).reshape(-1, 2),
             total=np.int64(len(stream)), min_len=np.int64(a.pack_len // 2 + 1))
    fill = [n / a.pack_len for _, n in docs]
    meta = {"source": a.name, "tokens": int(len(stream)), "files": files, "eos_id": 0, "dtype": "uint16",
            "tokenizer_sha256": hashlib.sha256(open(TOK, "rb").read()).hexdigest(), "pack_len": a.pack_len,
            "packs": len(docs), "mean_fill": round(float(np.mean(fill)), 4) if fill else 0,
            "conversations_longer_than_a_pack": too_long, "packs_too_small": len(packs) - len(kept), "sets": stats}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"{a.name}: {len(docs)} packs of up to {a.pack_len} tokens (mean fill {meta['mean_fill']:.1%}), "
          f"{len(stream) / 1e6:.1f}M tokens; left out: {too_long} conversations longer than a pack, "
          f"{len(packs) - len(kept)} packs too small [{time.time() - t0:.0f}s]", flush=True)
    print("SFT_LONG_PACKS_DONE", flush=True)


if __name__ == "__main__":
    main()
