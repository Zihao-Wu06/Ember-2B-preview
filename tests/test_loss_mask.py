"""No-loss turns: an assistant turn marked {"no_loss": true} is rendered as before, tokenized with a mask
(data_prep/sft_tokenize.py), and the loader turns the masked targets into the loss's ignore value (moe_llm.data)."""
import json
import os
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
from moe_llm import chat  # noqa: E402
from moe_llm.data import IGNORE, MixtureLoader, TokenStream  # noqa: E402

CONV = {"messages": [{"role": "user", "content": "Fix it."},
                     {"role": "assistant", "think": "First idea.", "content": "", "no_loss": True,
                      "tool_calls": [{"name": "python", "arguments": {"code": "print(1/0)"}}]},
                     {"role": "tool", "content": "ZeroDivisionError: division by zero"},
                     {"role": "assistant", "think": "Dividing by zero fails.", "content": "```python\nprint(1)\n```"}],
        "tools": [{"type": "function", "function": {"name": "python", "parameters": {}}}]}


def test_render_parts_join_to_render_and_mark_only_the_flagged_turn():
    parts = chat.render_parts(CONV)
    text = chat.render(CONV)
    assert "".join(t for t, _ in parts) == text
    marked = [t for t, skip in parts if skip]
    assert len(marked) == 1 and "First idea." in marked[0] and "print(1/0)" in marked[0] and marked[0].endswith(chat.IM_END + "\n")
    assert "Dividing by zero" not in marked[0] and not marked[0].startswith(chat.IM_START)
    (a, b), = chat.no_loss_spans(CONV)
    assert text[a:b] == marked[0]
    plain = {**CONV, "messages": [{k: v for k, v in m.items() if k != "no_loss"} for m in CONV["messages"]]}
    assert chat.render(plain) == text and chat.no_loss_spans(plain) == []


def test_tokenize_writes_a_mask_that_covers_exactly_the_flagged_turn(tmp_path, monkeypatch):
    import sft_tokenize as ST
    from tokenizers import Tokenizer
    conv_dir = tmp_path / "conv"
    conv_dir.mkdir()
    plain = {**CONV, "messages": [{k: v for k, v in m.items() if k != "no_loss"} for m in CONV["messages"]]}
    with open(conv_dir / "agent_tool.jsonl", "w") as f:
        for i in range(6):
            f.write(json.dumps(CONV if i % 2 else plain) + "\n")
    with open(conv_dir / "agent_fix.jsonl", "w") as f:
        f.write(json.dumps(plain) + "\n")
    monkeypatch.setattr(ST, "CONV", str(conv_dir))
    monkeypatch.setattr(sys, "argv", ["sft_tokenize.py", "--only", "sft_agent_tool", "sft_agent_fix", "--out", str(tmp_path / "bins")])
    ST.main()
    d = tmp_path / "bins" / "sft_agent_tool"
    ids = np.fromfile(d / "000.bin", dtype=np.uint16)
    mask = np.fromfile(d / "000.mask", dtype=np.uint8)
    assert len(ids) == len(mask) and json.load(open(d / "meta.json"))["no_loss_tokens"] == int(mask.sum()) > 0
    assert not os.path.exists(tmp_path / "bins" / "sft_agent_fix" / "000.mask")  # a source without flagged turns: no mask
    tok = Tokenizer.from_file(ST.TOK)
    ends = [i for i, t in enumerate(ids) if t == 0]
    start, seen = 0, 0
    for e in ends:
        m = mask[start:e + 1]
        if m.any():
            seen += 1
            idx = np.flatnonzero(m)
            assert (np.diff(idx) == 1).all()  # one contiguous run
            body = tok.decode(ids[start:e + 1][m.astype(bool)].tolist(), skip_special_tokens=False)
            assert "First idea." in body and "print(1/0)" in body and "Dividing by zero" not in body and "Fix it." not in body
            rest = tok.decode(ids[start:e + 1][~m.astype(bool)].tolist(), skip_special_tokens=False)
            assert "Dividing by zero" in rest and "First idea." not in rest
        start = e + 1
    assert seen == 3


def _bins(d, n, mask=None):
    os.makedirs(d, exist_ok=True)
    np.arange(1, n + 1, dtype=np.uint16).tofile(os.path.join(d, "000.bin"))
    if mask is not None:
        mask.astype(np.uint8).tofile(os.path.join(d, "000.mask"))


def test_loader_ignores_masked_targets_and_leaves_inputs_alone(tmp_path):
    n = 4000
    mask = np.zeros(n, dtype=np.uint8)
    mask[100:160] = 1
    mask[2500:2510] = 1
    _bins(str(tmp_path / "a"), n, mask)
    _bins(str(tmp_path / "b"), n)
    s = TokenStream(str(tmp_path / "a"))
    assert s.read_mask(98, 5).tolist() == [False, False, True, True, True] and TokenStream(str(tmp_path / "b")).read_mask(0, 4) is None
    plain = MixtureLoader(str(tmp_path), {"b": 1.0}, 64, 2, seed=0)
    x, y = plain.next_batch()
    assert (y == x + 1).all()  # no mask: targets are the next tokens, as before
    ld = MixtureLoader(str(tmp_path), {"a": 1.0}, 64, 2, seed=0)
    hit = 0
    for _ in range(ld.n_train_windows["a"] // 2):
        x, y = ld.next_batch()
        ign = y == IGNORE
        hit += int(ign.sum())
        # the token with value v sits at stream position v - 1, so the target of input v is at position v
        assert (y[~ign] == (x + 1)[~ign]).all()
        assert ign.tolist() == torch.from_numpy(mask[x.numpy().clip(max=n - 1)]).bool().tolist()
    assert hit > 0
    for x, y in ld.val_batches("a", 2):
        assert ((y == IGNORE) | (y == x + 1)).all()


def test_one_row_per_batch_keeps_the_inputs_free_of_the_ignore_value(tmp_path):
    """The trainer's long windows come one row per batch: there x and y were views of one buffer, and masking the
    targets wrote the ignore value into the inputs (the fine-tuning of 2026-11-24 15:16 crashed in the Engram lookup)."""
    n = 4000
    mask = np.zeros(n, dtype=np.uint8)
    mask[10:3000:3] = 1
    _bins(str(tmp_path / "a"), n, mask)
    ld = MixtureLoader(str(tmp_path), {"a": 1.0}, 64, 1, seed=0)
    hit = 0
    for _ in range(ld.n_train_windows["a"]):
        x, y = ld.next_batch()
        assert int(x.min()) >= 1 and tuple(x.shape) == (1, 64)
        ign = y == IGNORE
        hit += int(ign.sum())
        assert (y[~ign] == (x + 1)[~ign]).all() and ign[0].tolist() == mask[x[0].numpy().clip(max=n - 1)].astype(bool).tolist()
    assert hit > 0
    for x, y in ld.val_batches("a", 3):
        assert int(x.min()) >= 1


def test_a_shard_without_its_mask_is_refused(tmp_path):
    d = str(tmp_path / "a")
    _bins(d, 300, np.zeros(300))
    np.arange(1, 301, dtype=np.uint16).tofile(os.path.join(d, "001.bin"))
    try:
        TokenStream(d)
    except FileNotFoundError:
        return
    raise AssertionError("a stream with a mask for only one of its shards was accepted")


def test_training_runs_on_a_masked_source(tmp_path):
    from test_longctx import ENV, _train_cmd
    n = 6000
    rng = np.random.default_rng(0)
    d = str(tmp_path / "bins" / "a")
    os.makedirs(d)
    rng.integers(1, 200, n).astype(np.uint16).tofile(os.path.join(d, "000.bin"))
    m = np.zeros(n, dtype=np.uint8)
    m[::3] = 1
    m.tofile(os.path.join(d, "000.mask"))
    out = str(tmp_path / "run")
    r = subprocess.run(_train_cmd(out, str(tmp_path / "bins"), {"a": 1.0}, 3, ("--seq-len", "32")), env=ENV,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    log = [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"ce"' in l]
    assert len(log) == 3 and all(np.isfinite(x["ce"]) for x in log)
