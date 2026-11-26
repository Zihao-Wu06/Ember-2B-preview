"""data_prep/sft_tokenize.py: conversations -> EOS-separated token streams the loader reads."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
pytest.importorskip("tokenizers")
if not os.path.exists(os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")):
    pytest.skip("tokenizer not built", allow_module_level=True)
import sft_tokenize as T  # noqa: E402
from moe_llm import chat  # noqa: E402
from moe_llm.data import MixtureLoader  # noqa: E402


def _conv(i, think="because", long=False):
    q = f"question {i} " + ("word " * 5000 if long else "")
    return {"id": f"t/{i}", "messages": [{"role": "user", "content": q},
                                         {"role": "assistant", "think": think, "content": f"answer {i}"}]}


def test_streams(tmp_path, monkeypatch):
    conv, out = tmp_path / "conv", tmp_path / "bins"
    conv.mkdir()
    a = [_conv(i) for i in range(250)] + [_conv(900, long=True)]
    b = [_conv(1000 + i, think=None if i % 2 else "t") for i in range(40)]
    c = [dict(_conv(2000 + i), meta={"kind": ("math", "code", "other")[i % 3]}) for i in range(30)]
    for name, rows in (("a", a), ("b", b), ("c", c)):
        with open(conv / f"{name}.jsonl", "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(T, "CONV", str(conv))
    monkeypatch.setattr(T, "SOURCES", {"s": [("a", {}), ("missing", {}), ("b", {"no_think": "drop"})],
                                       "e": [("b", {})],
                                       "w": [("c", {"where": {"kind": ("code", "other")}})],
                                       "m": [("b", {"min_think": 1}), ("c", {"min_think": 8, "where": {"kind": ("code",)}})]})
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(out), "--val-tokens", "60"])
    T.main()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(T.TOK)
    ids = np.fromfile(out / "s" / "000.bin", dtype=np.uint16)
    docs = tok.decode(ids.tolist(), skip_special_tokens=False).split("<|endoftext|>")[:-1]
    meta = json.load(open(out / "s" / "meta.json"))
    assert meta["sets"]["a"] == {"rows": 251, "without_reasoning_dropped": 0, "filtered_out": 0, "kept": 250,
                                 "tokens": meta["sets"]["a"]["tokens"], "medium": 1, "bad": 0}
    assert meta["sets"]["b"]["without_reasoning_dropped"] == 20 and meta["sets"]["b"]["kept"] == 20
    assert meta["tokens"] == len(ids) and "missing" not in meta["sets"]
    want = [chat.render(c) for c in a[:250]] + [chat.render(c) for c in b if c["messages"][1]["think"]]
    assert sorted(docs) == sorted(want)  # every kept conversation exactly once, nothing else
    # the held-out tail: the first of each hundred conversations, each set up to half of --val-tokens (30 tokens):
    # conversations 0 and 100 of set a (26 tokens each: the second one starts below the cap), then set b's first
    assert docs[-3:] == [chat.render(a[0]), chat.render(a[100]), chat.render(b[0])]
    z = np.load(out / "_medium" / "s.npz")
    assert len(z["offsets"]) == 2 and tok.decode(z["ids"].tolist(), skip_special_tokens=False) == chat.render(a[250]) + "<|endoftext|>"
    # "empty": conversations without reasoning stay, with an empty thinking block
    e = tok.decode(np.fromfile(out / "e" / "000.bin", dtype=np.uint16).tolist(), skip_special_tokens=False)
    assert e.count("<|endoftext|>") == 40 and f"{chat.THINK}\n\n{chat.THINK_END}" in e
    # "where": only conversations whose meta field has a listed value
    wm = json.load(open(out / "w" / "meta.json"))["sets"]["c"]
    assert wm["filtered_out"] == 10 and wm["kept"] == 20
    wdocs = tok.decode(np.fromfile(out / "w" / "000.bin", dtype=np.uint16).tolist(), skip_special_tokens=False)
    assert sorted(wdocs.split("<|endoftext|>")[:-1]) == sorted(chat.render(r) for r in c if r["meta"]["kind"] != "math")
    # "min_think": conversations whose reasoning is shorter are left out (None counts as empty); combines with "where"
    mm = json.load(open(out / "m" / "meta.json"))["sets"]
    assert mm["b"]["filtered_out"] == 20 and mm["b"]["kept"] == 20
    assert mm["c"]["filtered_out"] == 30 and mm["c"]["kept"] == 0  # "because" has 7 characters
    # the trainer's loader reads the stream
    ld = MixtureLoader(str(out), {"s": 1.0}, seq_len=64, batch_size=2, val_tokens=100)
    x, y = ld.next_batch()
    assert x.shape == (2, 64) and (x[:, 1:] == y[:, :-1]).all()
