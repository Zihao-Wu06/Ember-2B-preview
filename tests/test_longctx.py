"""Context extension: log-n attention scaling, long-document windows in the loader, the
long-document index, train.py --init-from."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
from moe_llm.config import PRESETS  # noqa: E402
from moe_llm.data import LONG_INDEX, MixtureLoader, long_window_starts  # noqa: E402
from moe_llm.layers.attention import logn_scale  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def tiny(**kw):
    return PRESETS["tiny"]().replace(engram_offload=False, **kw)


def test_logn_scale_values():
    s = logn_scale(3, 10, 4, "cpu", torch.float64)  # queries 8..10 of 10 see 8, 9, 10 keys
    assert torch.allclose(s, torch.tensor([math.log(n) / math.log(4) for n in (8, 9, 10)], dtype=torch.float64))
    assert torch.equal(logn_scale(5, 5, 8, "cpu", torch.float32), torch.ones(5))  # n <= base: no scaling


@torch.no_grad()
def test_logn_is_exact_up_to_base_and_changes_beyond():
    torch.manual_seed(0)
    plain = MoELLM(tiny()).eval()
    logn = MoELLM(tiny(attn_logn=16)).eval()
    logn.load_state_dict(plain.state_dict())
    ids = torch.randint(1, 300, (1, 24))
    assert torch.equal(plain(ids[:, :16]), logn(ids[:, :16]))  # T <= 16: the same computation
    a, b = plain(ids), logn(ids)
    assert torch.equal(a[:, :16], b[:, :16])  # causal: the first 16 positions see at most 16 keys
    assert (a[:, 16:] - b[:, 16:]).abs().max() > 1e-4


@torch.no_grad()
def test_decode_session_matches_full_forward_with_logn():
    import decode
    torch.manual_seed(0)
    model = MoELLM(tiny(attn_logn=8)).eval()
    ids = torch.randint(1, 300, (1, 40))
    diff, scale = decode.check_against_full(model, ids, split=(5, 1, 3, 1, 10))
    assert diff < 1e-4 * max(1.0, scale)


def test_causal_lower_right_without_the_stray_allocation():
    import decode
    from torch.nn.attention.bias import causal_lower_right as torch_lr
    q, k, v = (torch.randn(1, 2, n, 16) for n in (5, 12, 12))
    q = q[:, :, :5]
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=torch_lr(5, 12))
    got = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=decode.causal_lower_right(5, 12))
    assert torch.equal(ref, got)
    b = decode.causal_lower_right(65536, 1 << 20)  # torch's would ask for 512 GB of address space
    assert b.seq_len_q == 65536 and b.seq_len_kv == 1 << 20 and b.numel() == 0


def test_long_window_starts():
    docs = np.array([[10, 100], [200, 33], [300, 32], [0, 500], [1000, 250]])  # (first token, length)
    st = long_window_starts(docs, train_end=1200, seq_len=64)
    # doc 1: EOS at 9, then every 64 while more than 32 tokens remain: 9, 73 (100-64=36 > 32), not 137 (100-128<0)
    # doc 2: 33 > 32 -> 199; doc 3: 32 is not more than half -> none; doc 4 starts at 0 (no EOS before) -> none
    # doc 5: 999, 1063 (250-64=186), 1127 (250-128=122) would end at 1127+65=1192 <= 1200 -> kept, 1191 too late
    assert st.tolist() == [9, 73, 199, 999, 1063, 1127]


def _doc_bins(root, name, lengths, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for n in lengths:
        parts += [rng.integers(1, 500, n, dtype=np.uint16), np.zeros(1, dtype=np.uint16)]
    arr = np.concatenate(parts)
    os.makedirs(os.path.join(root, name), exist_ok=True)
    half = len(arr) // 2  # two files: the stream offsets must run across them
    arr[:half].tofile(os.path.join(root, name, "000.bin"))
    arr[half:].tofile(os.path.join(root, name, "001.bin"))
    return arr


def test_long_docs_index_and_loader_windows_resume_exactly(tmp_path):
    from long_docs import index_source
    lengths = [20, 300, 15, 90, 40, 700, 10, 120] * 6 + [5] * 200  # the tail of short docs is held out
    arr = _doc_bins(tmp_path, "a", lengths)
    row = index_source(str(tmp_path / "a"), min_len=64, val_tokens=600)
    z = np.load(tmp_path / "a" / LONG_INDEX)
    assert int(z["total"]) == len(arr) and row["total"] == len(arr)
    for first, n in z["docs"].tolist():
        assert n >= 64 and arr[first - 1] == 0 and arr[first + n] == 0 and (arr[first:first + n] != 0).all()
    assert sorted(set(z["docs"][:, 1].tolist())) == [90, 120, 300, 700]

    w = {"a": 0.5, "long:a": 0.5}
    L1 = MixtureLoader(str(tmp_path), w, seq_len=64, batch_size=4, seed=3, val_tokens=600)
    starts = set(L1.starts["long:a"].tolist())
    assert L1.n_train_windows["long:a"] == len(starts) > 0
    seen_long = 0
    batches = [L1.next_batch() for _ in range(8)]
    for x, y in batches:
        assert torch.equal(x[:, 1:], y[:, :-1])
        for row_ in x:
            hits = [s for s in starts if np.array_equal(arr[s:s + 64], row_.numpy())]
            seen_long += bool(hits)
    assert seen_long > 0
    assert list(L1.val_batches("long:a", 2)) == [] and len(list(L1.val_batches("a", 2))) == 2
    assert "long:a" in L1.planned_epochs(100)

    L2 = MixtureLoader(str(tmp_path), w, seq_len=64, batch_size=4, seed=3, val_tokens=600)
    for _ in range(3):
        L2.next_batch()
    L3 = MixtureLoader(str(tmp_path), w, seq_len=64, batch_size=4, seed=3, val_tokens=600)
    L3.load_state_dict(L2.state_dict())
    for i in range(3, 8):
        assert torch.equal(L3.next_batch()[0], batches[i][0])


def test_loader_refuses_missing_or_stale_index(tmp_path):
    _doc_bins(tmp_path, "a", [100, 50] * 40)
    with pytest.raises(FileNotFoundError):
        MixtureLoader(str(tmp_path), {"long:a": 1.0}, seq_len=64, batch_size=2, val_tokens=100)
    np.savez(tmp_path / "a" / LONG_INDEX, docs=np.zeros((0, 2), dtype=np.int64), total=np.int64(7))
    with pytest.raises(ValueError):
        MixtureLoader(str(tmp_path), {"long:a": 1.0}, seq_len=64, batch_size=2, val_tokens=100)


def test_short_rows():
    from moe_llm.data import short_rows
    arr = torch.arange(1, 2 * 64 + 2).view(1, -1)[:, :129]  # one window of 128 tokens + the next one
    x, y = arr[:, :-1], arr[:, 1:]
    xs, ys = short_rows(x, y, ["a"], 32)
    assert xs.shape == ys.shape == (4, 32)
    assert torch.equal(xs.flatten(), x.flatten()) and torch.equal(ys, xs + 1)  # each row: targets = inputs shifted
    for names, short in ((["long:a"], 32), (["a"], 0), (["a", "long:a"], 32)):  # a long row, or off: unchanged
        xs2, ys2 = short_rows(x.expand(len(names), -1), y.expand(len(names), -1), names, short)
        assert xs2.shape == ys2.shape == (len(names), 128)


def test_loader_records_row_sources(tmp_path):
    _doc_bins(tmp_path, "a", [20, 300, 15, 90, 40, 700, 10, 120] * 6 + [5] * 200)
    from long_docs import index_source
    index_source(str(tmp_path / "a"), min_len=64, val_tokens=600)
    L = MixtureLoader(str(tmp_path), {"a": 0.5, "long:a": 0.5}, seq_len=64, batch_size=3, seed=1, val_tokens=600)
    starts = set(L.starts["long:a"].tolist())
    arr = np.concatenate([np.fromfile(tmp_path / "a" / f, dtype=np.uint16) for f in ("000.bin", "001.bin")])
    for _ in range(6):
        x, _ = L.next_batch()
        assert len(L.last_names) == 3
        for name, row in zip(L.last_names, x):
            if name == "long:a":
                assert any(np.array_equal(arr[s:s + 64], row.numpy()) for s in starts)


def test_longctx_s1_mix():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train as T
    m, base = T.MIXES["longctx_s1"], T.MIXES["reasoning_anneal_code_s2"]
    assert abs(sum(m.values()) - 1) < 1e-9
    long = {n[len("long:"):]: w for n, w in m.items() if n.startswith("long:")}
    assert abs(sum(long.values()) - 1 / 3) < 1e-9 and set(long) == set(T.LONG_WINDOWS_32K)
    assert set(long) <= set(base) | {"code"}  # every long source is one the main run trained on
    assert all(abs(m[n] - w * 2 / 3) < 1e-12 for n, w in base.items())


def test_longctx_s2_mix():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train as T
    m, base = T.MIXES["longctx_s2"], T.MIXES["reasoning_anneal_code_s2"]
    assert abs(sum(m.values()) - 1) < 1e-9
    long = {n[len("long:"):]: w for n, w in m.items() if n.startswith("long:")}
    assert abs(sum(long.values()) - 1 / 3) < 1e-9 and set(long) == set(T.LONG_WINDOWS_64K) == set(T.LONG_WINDOWS_32K)
    assert all(abs(m[n] - w * 2 / 3) < 1e-12 for n, w in base.items())
    assert all(T.LONG_WINDOWS_64K[n] < T.LONG_WINDOWS_32K[n] for n in long)  # fewer documents are that long


def test_longctx_s3_s4_mixes():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train as T
    base = T.MIXES["reasoning_anneal_code_s2"]
    for name, windows, books in (("longctx_s3", T.LONG_WINDOWS_128K, 0.6), ("longctx_s4", T.LONG_WINDOWS_256K, 0.7)):
        m = T.MIXES[name]
        assert abs(sum(m.values()) - 1) < 1e-9
        assert all(abs(m[n] - w * 2 / 3) < 1e-12 for n, w in base.items())
        long = {n[len("long:"):]: w for n, w in m.items() if n.startswith("long:")}
        assert abs(sum(long.values()) - 1 / 3) < 1e-9 and abs(long.pop("pg19") - books / 3) < 1e-12
        assert set(long) == set(windows)
        tot = sum(windows.values())  # the other sources in proportion to their windows
        assert all(abs(w - (1 - books) / 3 * windows[n] / tot) < 1e-12 for n, w in long.items())
    assert all(T.LONG_WINDOWS_256K[n] < T.LONG_WINDOWS_128K[n] < T.LONG_WINDOWS_64K[n] for n in T.LONG_WINDOWS_256K)


def _train_cmd(out, data, mix, steps, extra=()):
    return [sys.executable, os.path.join(ROOT, "scripts", "train.py"), "--preset", "tiny",
            "--override", json.dumps({"engram_offload": False}), "--data", data, "--canon", "none",
            "--mix", json.dumps(mix), "--out", out, "--micro-bsz", "2", "--global-tokens", "128",
            "--steps", str(steps), "--lr", "3e-3", "--warmup", "2", "--log-every", "1", "--eval-every", "0",
            "--ckpt-every", str(steps), *extra]


def test_init_from_starts_a_new_run_from_the_weights(tmp_path):
    data = str(tmp_path / "bins")
    _doc_bins(data, "a", [20, 300, 15, 90, 40, 700, 10, 120] * 8 + [5] * 100)
    subprocess.run([sys.executable, os.path.join(ROOT, "data_prep", "long_docs.py"), "--bins", data,
                    "--min-len", "64", "--val-tokens", "600"], check=True, env=ENV, capture_output=True, text=True)
    base = str(tmp_path / "base")
    subprocess.run(_train_cmd(base, data, {"a": 1.0}, 4, ("--seq-len", "32")), check=True, env=ENV,
                   capture_output=True, text=True)
    ck = sorted(f for f in os.listdir(base) if f.startswith("ckpt_") and f.endswith(".pt"))[-1]
    ext = str(tmp_path / "ext")
    cmd = _train_cmd(ext, data, {"a": 0.5, "long:a": 0.5}, 3,
                     ("--seq-len", "64", "--override", json.dumps({"engram_offload": False, "attn_logn": 32}),
                      "--init-from", os.path.join(base, ck)))
    r = subprocess.run(cmd, check=True, env=ENV, capture_output=True, text=True)
    assert "initialized from" in r.stdout and "seq_len 64" in r.stdout
    log = [json.loads(l) for l in open(os.path.join(ext, "log.jsonl")) if '"ce"' in l]
    assert [x["step"] for x in log] == [0, 1, 2]
    # the new run starts from the trained weights: its first loss is far below a fresh model's (~ln vocab)
    fresh = str(tmp_path / "fresh")
    subprocess.run(_train_cmd(fresh, data, {"a": 0.5, "long:a": 0.5}, 1, ("--seq-len", "64")), check=True, env=ENV,
                   capture_output=True, text=True)
    first_fresh = [json.loads(l) for l in open(os.path.join(fresh, "log.jsonl")) if '"ce"' in l][0]["ce"]
    assert log[0]["ce"] < first_fresh - 0.05
    # re-running the same command resumes the new run (its own checkpoint wins over --init-from)
    r2 = subprocess.run(cmd, check=True, env=ENV, capture_output=True, text=True)
    assert "resumed from" in r2.stdout


def test_train_short_rows_end_to_end(tmp_path):
    data = str(tmp_path / "bins")
    _doc_bins(data, "a", [20, 300, 15, 90, 40, 700, 10, 120] * 8 + [5] * 100)
    subprocess.run([sys.executable, os.path.join(ROOT, "data_prep", "long_docs.py"), "--bins", data,
                    "--min-len", "64", "--val-tokens", "600"], check=True, env=ENV, capture_output=True, text=True)
    out = str(tmp_path / "run")
    r = subprocess.run(_train_cmd(out, data, {"a": 0.5, "long:a": 0.5}, 4, ("--seq-len", "64", "--short-seq-len", "16")),
                       check=True, env=ENV, capture_output=True, text=True)
    assert "short rows: micro-batches from plain sources as [8, 16]" in r.stdout
    log = [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"ce"' in l]
    assert [x["step"] for x in log] == [0, 1, 2, 3] and all(math.isfinite(x["ce"]) for x in log)
    bad = subprocess.run(_train_cmd(str(tmp_path / "bad"), data, {"a": 1.0}, 1, ("--seq-len", "64", "--short-seq-len", "24")),
                         env=ENV, capture_output=True, text=True)
    assert bad.returncode != 0 and "must divide" in bad.stderr


def test_short_micro_batches_weigh_like_one(tmp_path):
    """--short-micro-bsz splits the short rows into smaller micro-batches (here without recompute) that weigh their
    share of the step's tokens: step 0's CE (same weights, same tokens) is the same as with one micro-batch."""
    data = str(tmp_path / "bins")
    _doc_bins(data, "a", [20, 300, 15, 90, 40, 700, 10, 120] * 8 + [5] * 100)
    common = ("--seq-len", "64", "--short-seq-len", "16", "--grad-ckpt", "--ckpt-frac", "1.0")
    logs = {}
    for name, extra in (("one", ()), ("split", ("--short-micro-bsz", "2", "--short-ckpt-frac", "0"))):
        out = str(tmp_path / name)
        r = subprocess.run(_train_cmd(out, data, {"a": 1.0}, 2, common + extra), check=True, env=ENV,
                           capture_output=True, text=True)
        logs[name] = [json.loads(l) for l in open(os.path.join(out, "log.jsonl")) if '"ce"' in l]
        if name == "split":
            assert "short rows in micro-batches of 2 rows, recompute fraction 0" in r.stdout
    a, b = logs["one"], logs["split"]
    assert abs(a[0]["ce"] - b[0]["ce"]) < 1e-4 * max(1.0, a[0]["ce"])
    assert abs(a[1]["ce"] - b[1]["ce"]) < 0.05  # the MoE balance term depends on the split, so step 1 differs a little
    bad = subprocess.run(_train_cmd(str(tmp_path / "bad"), data, {"a": 1.0}, 1, common + ("--short-micro-bsz", "3")),
                         env=ENV, capture_output=True, text=True)
    assert bad.returncode != 0 and "must divide" in bad.stderr


def test_profile_flag_profiles_one_step_boundary(tmp_path):
    data = str(tmp_path / "bins")
    _doc_bins(data, "a", [20, 300, 15, 90, 40, 700, 10, 120] * 8 + [5] * 100)
    out = str(tmp_path / "run")
    os.makedirs(out)
    open(os.path.join(out, "PROFILE"), "w").close()
    r = subprocess.run(_train_cmd(out, data, {"a": 1.0}, 3, ("--seq-len", "32", "--global-tokens", "256")),
                       check=True, env=ENV, capture_output=True, text=True)
    assert "profile: " in r.stdout and not os.path.exists(os.path.join(out, "PROFILE"))
    table = open(os.path.join(out, "profile_step1.txt")).read()
    trace = open(os.path.join(out, "profile_step1.json")).read()
    for label in ("step.finite_sync", "opt.step", "opt.clip_grad_norm", "step.data_wait"):
        assert label in trace, label
    assert "Name" in table


def test_sft_mix():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train as T
    m = T.MIXES["sft_v1"]
    assert abs(sum(m.values()) - 1) < 1e-9 and all(w > 0 for w in m.values())
    short = {n: w for n, w in m.items() if not n.startswith("long:")}
    assert short == {"sft_math": 0.29, "sft_code": 0.26, "sft_chat": 0.04, "sft_general": 0.03, "sft_tools": 0.05}
    assert m["long:sft_long"] == 0.20
    plain = {n: w for n, w in m.items() if n.startswith("long:") and n != "long:sft_long"}
    s3 = {n: w for n, w in T.MIXES["longctx_s3"].items() if n.startswith("long:")}
    assert set(plain) == set(s3) and abs(sum(plain.values()) - 0.13) < 1e-9
    assert abs(plain["long:pg19"] / sum(plain.values()) - 0.6) < 1e-9  # the same split as S3's long part
    m64 = T.MIXES["sft_v1_64k"]  # the 4090's version differs only in the name of the pack source
    assert {("long:sft_long" if k == "long:sft_long64k" else k): v for k, v in m64.items()} == m


def test_sft_v2_mix():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train as T
    for name, pack in (("sft_v2", "long:sft_long"), ("sft_v2_64k", "long:sft_long64k")):
        m = T.MIXES[name]
        assert abs(sum(m.values()) - 1) < 1e-9 and all(w > 0 for w in m.values()), name
        assert m[pack] == 0.27
        assert {n: w for n, w in m.items() if n.startswith("sft_")} == T.SFT_V2_SHORT
        plain = {n: w for n, w in m.items() if n.startswith("long:") and n != pack}
        assert abs(sum(plain.values()) - 0.13) < 1e-9
    # no source is read more than about 4.5 times at 1.5B tokens (token counts of 2026-11-23)
    tokens = {"sft_math": 465.9e6, "sft_code": 31.1e6, "sft_chat": 61.8e6, "sft_general": 3.7e6, "sft_tools": 22.8e6}
    for n, w in T.SFT_V2_SHORT.items():
        assert w * 1.5e9 / tokens[n] <= 4.5, n
    assert 0.27 * 1.5e9 / 249.7e6 < 2


def test_sft_v3_mix():
    """sft_v3: v2 plus the data it left out (public code, rStar-Coder, harder reasoning, hard-problem packs, plain code)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import train as T
    tokens = {"sft_math": 465.9e6, "sft_reason": 70.3e6, "sft_code": 31.1e6, "sft_code_pub": 14.1e6,
              "sft_code_rstar": 145.0e6, "sft_chat": 61.8e6, "sft_general": 3.7e6,
              "sft_tools": 22.8e6}  # data/bins_sft/*/meta.json, 2026-11-23
    hard_tokens, hard_code = 116.5e6, 61.3e6 / 116.5e6  # the hard packs and the share of code in them
    for name, pack, hard, pack_tokens in (("sft_v3", "long:sft_long", "long:sft_hard", 249.7e6),
                                          ("sft_v3_64k", "long:sft_long64k", "long:sft_hard64k", 161.9e6)):
        m = T.MIXES[name]
        assert abs(sum(m.values()) - 1) < 1e-9 and all(w > 0 for w in m.values()), name
        assert {n: w for n, w in m.items() if n.startswith("sft_")} == T.SFT_V3_SHORT
        assert m[pack] == T.SFT_V3_PACKS == 0.17 and m[hard] == T.SFT_V3_HARD == 0.10
        plain_long = sum(w for n, w in m.items() if n.startswith("long:") and n not in (pack, hard))
        plain_code = sum(m[n] for n in T.PLAIN_CODE)
        assert abs(plain_long - 0.06) < 1e-9 and abs(plain_code - 0.04) < 1e-9
        for n, w in T.SFT_V3_SHORT.items():   # no source is read more than about 3.5 times at 1.5B tokens
            assert w * 1.5e9 / tokens[n] <= 3.5, n
        assert m[pack] * 1.5e9 / pack_tokens < 2 and m[hard] * 1.5e9 / hard_tokens < 2
        code = m["sft_code"] + m["sft_code_pub"] + m["sft_code_rstar"] + m[hard] * hard_code
        reason = m["sft_math"] + m["sft_reason"] + m[hard] * (1 - hard_code)
        assert 0.28 < code < 0.30 and 0.34 < reason < 0.36 and abs(code + plain_code - 0.33) < 0.005


def test_sft_v3_norstar_control_mix():
    """The sixth rehearsal's control: the mix before rStar-Coder, with hard packs built without it."""
    import train as T
    m = T.MIXES["sft_v3_norstar_64k"]
    assert abs(sum(m.values()) - 1) < 1e-9 and "sft_code_rstar" not in m and "long:sft_hard64k" not in m
    assert m["long:sft_hard64k_norstar"] == 0.06 and m["long:sft_long64k"] == 0.19 and m["sft_math"] == 0.30
    assert abs(sum(m[n] for n in T.PLAIN_CODE) - 0.06) < 1e-12
    full = T.MIXES["sft_v3_64k"]  # the two mixes read the same sources, apart from rStar-Coder and the hard packs
    assert set(m) - {"long:sft_hard64k_norstar"} == set(full) - {"sft_code_rstar", "long:sft_hard64k"}


def test_agent_v1_mix_is_the_sft_mix_plus_the_agent_sources():
    import train as T
    m, base = T.MIXES["agent_v1_64k"], T.MIXES["sft_v3_64k"]
    assert abs(sum(m.values()) - 1) < 1e-9 and m["sft_agent_tool"] == 0.18 and m["sft_agent_fix"] == 0.045
    assert set(m) == set(base) | {"sft_agent_tool", "sft_agent_fix"}
    assert all(abs(m[n] - 0.775 * w) < 1e-12 for n, w in base.items())
