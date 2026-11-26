"""scripts/to_safetensors.py: an export becomes safetensors files that hold the same tensors, an index and a config."""
import json
import os
import struct
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import to_safetensors as S  # noqa: E402

NP = {"F32": np.float32, "I64": np.int64, "BOOL": np.bool_}


def read_all(path):
    """Every tensor of a .safetensors file, read from the format's definition alone."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        head = json.loads(f.read(n))
        data = f.read()
    out, spans = {}, []
    for k, v in head.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        spans.append((a, b))
        out[k] = torch.from_numpy(np.frombuffer(data[a:b], dtype=NP[v["dtype"]]).reshape(v["shape"]).copy())
    spans.sort()
    assert n % 8 == 0 and spans[0][0] == 0 and spans[-1][1] == len(data)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))  # no gap, no overlap
    return out


def make_export(d):
    g = torch.Generator().manual_seed(0)
    shards = {"model-0000.pt": {"embed.weight": torch.randn(6, 4, generator=g), "engram.canon": torch.arange(7)},
              "model-0001.pt": {"blocks.0.w": torch.randn(3, 5, generator=g), "flag": torch.tensor([True, False]),
                                "scalar": torch.tensor(2.5)}}
    os.makedirs(d)
    index = {}
    for name, sd in shards.items():
        torch.save(sd, os.path.join(d, name))
        index.update({k: name for k in sd})
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"step": 350, "cfg": {"d_model": 4, "max_seq_len": 65536, "engram_offload": True}, "shards": list(shards),
                   "index": index}, f)
    tok = os.path.join(d, "tok.json")
    with open(tok, "w") as f:
        f.write('{"tokenizer": true}')
    return {k: v for sd in shards.values() for k, v in sd.items()}, tok


def test_convert_keeps_every_tensor_and_writes_index_and_config(tmp_path):
    want, tok = make_export(str(tmp_path / "export"))
    out = str(tmp_path / "hf")
    r = S.convert(str(tmp_path / "export"), out, max_bytes=120, tokenizer=tok, repo="example.org/repo")
    files = sorted(f for f in os.listdir(out) if f.endswith(".safetensors"))
    assert r["files"] == len(files) > 1 and r["tensors"] == len(want)
    assert files[0] == f"model-00001-of-{len(files):05d}.safetensors"
    index = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    got = {}
    for f in files:
        part = read_all(os.path.join(out, f))
        assert all(index["weight_map"][k] == f for k in part)
        got.update(part)
    assert set(got) == set(want) == set(index["weight_map"])
    for k, v in want.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape and torch.equal(got[k], v)
    assert index["metadata"]["total_size"] == r["bytes"] == sum(v.numel() * v.element_size() for v in want.values())
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["step"] == 350 and cfg["d_model"] == 4 and cfg["max_seq_len"] == 131072 and cfg["engram_offload"] is False
    assert "example.org/repo" in cfg["architecture"]
    assert open(os.path.join(out, "tokenizer.json")).read() == '{"tokenizer": true}'
    assert not any(f.endswith(".tmp") for f in os.listdir(out))


def test_plan_closes_a_file_before_it_passes_the_limit():
    assert S.plan({"a": 60, "b": 60, "c": 10, "d": 200, "e": 1}, 100) == [["a"], ["b", "c"], ["d"], ["e"]]


def test_files_load_with_the_safetensors_library(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    want, tok = make_export(str(tmp_path / "export"))
    out = str(tmp_path / "hf")
    S.convert(str(tmp_path / "export"), out, max_bytes=10 ** 9, tokenizer=tok, repo="example.org/repo")
    got = st.load_file(os.path.join(out, "model-00001-of-00001.safetensors"))
    assert set(got) == set(want) and all(torch.equal(got[k], want[k]) for k in want)
