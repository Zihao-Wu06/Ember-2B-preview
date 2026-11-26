#!/usr/bin/env python
"""An export of inference weights (runs/mac_eval/export_weights.py: meta.json + model-NNNN.pt shards) -> the files
published on the Hugging Face Hub: model-0000k-of-0000n.safetensors, model.safetensors.index.json, config.json and
tokenizer.json.

    python scripts/to_safetensors.py --export runs/mac_eval/agent --out hf/Ember-2B-preview

The safetensors files are written directly (an 8-byte header length, a JSON header, the raw little-endian tensor
bytes), so the conversion needs nothing but torch; `safetensors.torch.load_file` reads them. One shard of the export
is in memory at a time. config.json holds the model config of the export with the two run-only fields set for
inference (max_seq_len 131072: the context the model was extended to, no layer reads it; engram_offload false: the
Engram tables are ordinary parameters named engram.tables.<i>, as in the export).
"""
import argparse
import json
import os
import shutil
import struct

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTYPES = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.float64: "F64", torch.int64: "I64",
          torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}


def plan(sizes: dict, max_bytes: int) -> list:
    """Tensor names grouped into output files, in the given order: a file is closed before it would pass max_bytes."""
    groups, cur, size = [], [], 0
    for name, nb in sizes.items():
        if cur and size + nb > max_bytes:
            groups.append(cur)
            cur, size = [], 0
        cur.append(name)
        size += nb
    if cur:
        groups.append(cur)
    return groups


def header(tensors: dict) -> bytes:
    """The safetensors header of the tensors in this order: name -> dtype, shape, byte range; padded to 8 bytes."""
    h, off = {"__metadata__": {"format": "pt"}}, 0
    for name, t in tensors.items():
        nb = t.numel() * t.element_size()
        h[name] = {"dtype": DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + nb]}
        off += nb
    raw = json.dumps(h, separators=(",", ":")).encode("utf-8")
    return raw + b" " * (-len(raw) % 8)


def write_file(path: str, tensors: dict) -> int:
    """Write one .safetensors file; returns the number of tensor bytes."""
    head, n = header(tensors), 0
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for t in tensors.values():
            a = t.detach().contiguous().cpu().numpy()
            if a.dtype.byteorder == ">":
                a = a.byteswap().view(a.dtype.newbyteorder("<"))
            f.write(a.tobytes())
            n += a.nbytes
    os.replace(path + ".tmp", path)
    return n


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def convert(export: str, out: str, max_bytes: int, tokenizer: str, repo: str) -> dict:
    with open(os.path.join(export, "meta.json")) as f:
        meta = json.load(f)
    os.makedirs(out, exist_ok=True)
    # sizes first (one export shard in memory at a time), then the split, then the files
    sizes, where = {}, {}
    for shard in meta["shards"]:
        sd = torch.load(os.path.join(export, shard), map_location="cpu", weights_only=True)
        for k, v in sd.items():
            sizes[k] = v.numel() * v.element_size()
            where[k] = shard
        del sd
    groups = plan(sizes, max_bytes)
    index, total, loaded, loaded_name = {}, 0, None, None
    for i, names in enumerate(groups):
        fname = f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        for k in names:
            if where[k] != loaded_name:
                loaded = torch.load(os.path.join(export, where[k]), map_location="cpu", weights_only=True)
                loaded_name = where[k]
            tensors[k] = loaded[k]
        total += write_file(os.path.join(out, fname), tensors)
        index.update({k: fname for k in names})
        print(f"  {fname}: {len(names)} tensors, {sum(sizes[k] for k in names) / 2 ** 30:.2f} GiB", flush=True)
        del tensors
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": index}, f, indent=2)
    cfg = {**meta["cfg"], "max_seq_len": 131072, "engram_offload": False}
    config = {"model_type": "ember_moe", "architecture": f"MoELLM ({repo}, moe_llm/model.py)", "torch_dtype": "float32",
              "step": meta["step"], **cfg}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copyfile(tokenizer, os.path.join(out, "tokenizer.json"))
    return {"files": len(groups), "tensors": len(index), "bytes": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="directory written by runs/mac_eval/export_weights.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-gb", type=float, default=2.0)
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "data", "tokenizer", "tokenizer.json"))
    ap.add_argument("--repo", default="github.com/Zihao-Wu06/Ember-2B-base", help="where the model code is; named in config.json")
    args = ap.parse_args()
    r = convert(args.export, args.out, int(args.max_shard_gb * 2 ** 30), args.tokenizer, args.repo)
    print(f"wrote {r['files']} safetensors files, {r['tensors']} tensors, {r['bytes'] / 2 ** 30:.2f} GiB -> {args.out}")
    print("TO_SAFETENSORS_DONE")


if __name__ == "__main__":
    main()
