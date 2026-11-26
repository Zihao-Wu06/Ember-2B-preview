#!/usr/bin/env python
"""Conversations (data/sft/conv/*.jsonl, moe_llm/chat.py format) -> token streams the trainer reads
: data/bins_sft/<source>/NNN.bin, uint16, EOS after every conversation, like the
pretraining bins, so scripts/train.py and moe_llm.data.MixtureLoader need no change.

    python data_prep/sft_tokenize.py                       # every source in SOURCES
    python data_prep/sft_tokenize.py --only sft_math --limit 2000 --out /tmp/bins_test

A source is a list of (conversation set, options): "max_rows" keeps a seeded random sample, "no_think" says what to
do with a conversation in which an assistant turn has no reasoning: "empty" renders an empty thinking block, "drop"
leaves the conversation out; "where" keeps only conversations whose meta field has one of the listed values;
"min_think" drops conversations whose last reply has a shorter reasoning (characters). Sets whose file does not exist
yet (the API data) are skipped with a note.

OpenThoughts: sft_general has its code and "other" problems with the reasoning rewritten short
(its math part, 78%, repeats the domain OpenMathInstruct-2 covers, with longer thoughts and harder
problems). Code and reasoning are to be strong without further API calls, so the data that is
on disk and was left out goes into two sources of its own (sft_code_pub, sft_reason; the mix decides how much).

Conversations of more than --short-max tokens do not go into the stream: the trainer cuts plain sources into rows of
4096 tokens, which would split them. They are written to <out>/_medium/<source>.npz (tokens + offsets) for the long
packs (data_prep/sft_long_packs.py). About every 100th conversation of each set is moved to the end of the stream
(each set up to an equal share of --val-tokens), so the loader's held-out tail (the last 2M tokens or 10%) covers
every set.
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

CONV = os.path.join(ROOT, "data", "sft", "conv")
OUT = os.path.join(ROOT, "data", "bins_sft")
TOK = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
SHARD = 64_000_000
SOURCES = {
    "sft_math": [("math", {}), ("gsm8k", {})],
    # code: the teacher's answers that passed the tests (code_api, with a working in full sentences). The public
    # self-oss-instruct answers (code_selfoss) are left out since the teacher data exists: their "thinking" is one
    # opening sentence ("Here's a concise implementation:"), and the plan says samples without a process are not used
    "sft_code": [("code_api", {})],
    # more code without the API (2026-11-23): the public self-oss-instruct answers whose reasoning is a real
    # paragraph (at least 200 characters; 58% of them; the same instructions as code_api's "selfoss" part, so these
    # are second answers), and OpenThoughts' competition problems with a thought of 6,000-12,000 characters
    "sft_code_pub": [("code_selfoss", {"min_think": 200}), ("openthoughts_mid", {"where": {"kind": ["code"]}})],
    # rStar-Coder synthetic_sft : competition-style problems with a
    # verified Python solution, the ones whose reasoning has at most 12,000 characters (21% of the 398,183)
    "sft_code_rstar": [("code_rstar", {})],
    # harder reasoning without the API: OpenThoughts' math (thoughts up to 12,000 characters) and its "other"
    # problems of 6,000-12,000 characters, and the teacher's own math answers (math_api)
    "sft_reason": [("openthoughts", {"where": {"kind": ["math"]}}),
                   ("openthoughts_mid", {"where": {"kind": ["math", "other"]}}), ("math_api", {})],
    # general reasoning: OpenThoughts' code and "other" problems with the reasoning rewritten short by the teacher
    # (openthoughts_short: 4,096 characters of reasoning on average became 596; the answers are unchanged). The long
    # original traces (set "openthoughts") are no longer used.
    "sft_general": [("openthoughts_short", {}), ("general_api", {}), ("smoltalk", {"no_think": "drop"})],
    # everyday conversation (sft_select.py): chat keeps the replies as they are with an empty thinking block,
    # chat_api is the quarter for which the teacher wrote a short thought
    "sft_chat": [("chat", {}), ("chat_api", {})],
    "sft_tools": [("gsm8k_py", {}), ("tools_api", {}), ("tools_glaive", {"no_think": "drop"}),
                  ("tools_hermes", {"no_think": "drop"}), ("tools_toolace", {"no_think": "drop"})],
    # small-agent data (2026-11-24, data_prep/agent_traj.py): code written, run with the python tool, fixed after a real
    # error, answered; and the same repair as a plain exchange with the user's error report
    "sft_agent_tool": [("agent_tool", {})],
    "sft_agent_fix": [("agent_fix", {})],
    # a small project as the environment (2026-11-24, data_prep/agent_env.py): run the tests, read, edit, write files;
    # and failed attempts followed by a different attempt of the model's own with the python tool
    "sft_agent_env": [("agent_env", {})],
    "sft_agent_self": [("agent_self", {})],
    # the same tasks with no tools offered, answered with the code (an earlier fine-tune called a tool without one)
    "sft_agent_plain": [("agent_plain", {})],
}


def has_think(conv) -> bool:
    return all((m.get("think") or "").strip() for m in conv["messages"] if m["role"] == "assistant")


def read_set(name, opt, limit):
    """Rendered conversations of one set as a list of strings (None if the file is missing)."""
    path = os.path.join(CONV, name + ".jsonl")
    if not os.path.exists(path):
        return None, {}
    keep, n, dropped, filtered, where, min_think = [], 0, 0, 0, opt.get("where"), opt.get("min_think", 0)
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += 1
            if limit and n > limit:
                break
            if where or min_think:
                conv = json.loads(line)
                meta = conv.get("meta") or {}
                if where and any(meta.get(k) not in vals for k, vals in where.items()):
                    filtered += 1
                    continue
                if len(conv["messages"][-1].get("think") or "") < min_think:
                    filtered += 1
                    continue
            if opt.get("no_think", "empty") == "drop" and '"think": null' in line:
                if not has_think(json.loads(line)):
                    dropped += 1
                    continue
            keep.append(line)
    if opt.get("max_rows") and len(keep) > opt["max_rows"]:
        keep = random.Random(0).sample(keep, opt["max_rows"])
    return keep, {"rows": n - (1 if limit and n > limit else 0), "without_reasoning_dropped": dropped,
                  "filtered_out": filtered}


class Writer:
    """Token shards NNN.bin and, if any conversation has no-loss turns (moe_llm.chat.render_parts), NNN.mask of the
    same length (uint8, 1 = this token is not a training target), which moe_llm.data applies to the targets."""

    def __init__(self, d):
        self.d, self.buf, self.mbuf, self.n_buf, self.files, self.total = d, [], [], 0, {}, 0
        self.masked, self.mask_tokens = False, 0
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.endswith(".bin") or f.endswith(".tmp") or f.endswith(".mask"):
                os.remove(os.path.join(d, f))

    def add(self, ids: np.ndarray, mask: np.ndarray | None = None):
        self.buf.append(ids)
        self.mbuf.append(mask if mask is not None else np.zeros(len(ids), dtype=np.uint8))
        if mask is not None and mask.any():
            self.masked = True
            self.mask_tokens += int(mask.sum())
        self.n_buf += len(ids)
        if self.n_buf >= SHARD:
            self.flush()

    def _write(self, name, arr):
        arr.tofile(os.path.join(self.d, name + ".tmp"))
        os.replace(os.path.join(self.d, name + ".tmp"), os.path.join(self.d, name))

    def flush(self):
        if not self.buf:
            return
        name = f"{len(self.files):03d}.bin"
        arr = np.concatenate(self.buf)
        self._write(name, arr)
        if self.masked:
            self._write(name[:-4] + ".mask", np.concatenate(self.mbuf))
        self.files[name] = int(len(arr))
        self.total += len(arr)
        self.buf, self.mbuf, self.n_buf = [], [], 0

    def finish(self):
        """Flush; if a later shard turned out to have a mask, the earlier ones get an all-zero mask (the loader wants
        a mask for every shard of a stream or for none)."""
        self.flush()
        if self.masked:
            for name, n in self.files.items():
                if not os.path.exists(os.path.join(self.d, name[:-4] + ".mask")):
                    self._write(name[:-4] + ".mask", np.zeros(n, dtype=np.uint8))


def token_mask(enc, spans, n):
    """uint8 [n]: 1 for the tokens of an encoding that lie inside one of the character spans."""
    mask = np.zeros(n, dtype=np.uint8)
    if spans:
        off = enc.offsets
        k = 0
        for t, (a, b) in enumerate(off):
            while k < len(spans) and a >= spans[k][1]:
                k += 1
            if k == len(spans):
                break
            if a >= spans[k][0] and b <= spans[k][1] and b > a:
                mask[t] = 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--short-max", type=int, default=3500, help="longest conversation kept in the stream, in tokens")
    ap.add_argument("--val-tokens", type=int, default=2_200_000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="read only the first N conversations of each set (tests)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    tok = Tokenizer.from_file(TOK)
    chat.check_tokenizer(tok)
    tok_sha = hashlib.sha256(open(TOK, "rb").read()).hexdigest()
    os.makedirs(os.path.join(a.out, "_medium"), exist_ok=True)
    for src in a.only:
        t0, w = time.time(), Writer(os.path.join(a.out, src))
        tail, tail_tokens, medium, sets = [], 0, [], {}
        present = sum(os.path.exists(os.path.join(CONV, n + ".jsonl")) for n, _ in SOURCES[src])
        for name, opt in SOURCES[src]:
            set_tail = 0  # each set gives the tail an equal share
            lines, st = read_set(name, opt, a.limit)
            if lines is None:
                print(f"  {src}: set {name} is not there yet, skipped", flush=True)
                continue
            st.update(kept=0, tokens=0, medium=0, bad=0)
            for b in range(0, len(lines), a.batch):
                texts, spans = [], []
                for line in lines[b:b + a.batch]:
                    try:
                        conv = json.loads(line)
                        texts.append(chat.render(conv))
                        spans.append(chat.no_loss_spans(conv) if '"no_loss": true' in line else None)
                    except Exception:  # noqa: BLE001
                        st["bad"] += 1
                for k, enc in enumerate(tok.encode_batch(texts)):
                    ids = np.asarray(enc.ids + [0], dtype=np.uint16)
                    if len(ids) > a.short_max:
                        medium.append(ids)  # the long packs carry no mask: such a conversation trains in full
                        st["medium"] += 1
                        continue
                    mask = token_mask(enc, spans[k], len(ids)) if spans[k] else None
                    st["kept"] += 1
                    st["tokens"] += len(ids)
                    if (b + k) % 100 == 0 and set_tail < a.val_tokens / max(1, present):
                        tail.append((ids, mask))
                        tail_tokens += len(ids)
                        set_tail += len(ids)
                    else:
                        w.add(ids, mask)
            sets[name] = st
            print(f"  {src}/{name}: {st}", flush=True)
        for ids, mask in tail:
            w.add(ids, mask)
        w.finish()
        if medium:
            off = np.concatenate([[0], np.cumsum([len(m) for m in medium])]).astype(np.int64)
            np.savez(os.path.join(a.out, "_medium", src + ".npz"), ids=np.concatenate(medium), offsets=off)
        meta = {"source": src, "tokens": w.total, "files": w.files, "eos_id": 0, "dtype": "uint16",
                "tokenizer_sha256": tok_sha, "sets": sets, "tail_tokens": tail_tokens,
                "medium_conversations": len(medium), "medium_tokens": int(sum(len(m) for m in medium)),
                "short_max": a.short_max, "no_loss_tokens": w.mask_tokens}
        with open(os.path.join(a.out, src, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"{src}: {w.total / 1e6:.1f}M tokens in {len(w.files)} files, {len(medium)} longer conversations "
              f"({meta['medium_tokens'] / 1e6:.1f}M tokens) set aside [{time.time() - t0:.0f}s]", flush=True)
    print("SFT_TOKENIZE_DONE", flush=True)


if __name__ == "__main__":
    main()
