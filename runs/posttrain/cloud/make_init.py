#!/usr/bin/env python
"""The starting weights of the post-training run: the S3 checkpoint without its optimizer states, the output rows of
the chat format's tokens replaced by the fitted ones (runs/posttrain/special_rows.pt, probe_special_rows.py;
runs/posttrain/special_init.py says why: S3 has one very negative row for all of them, and a run that starts from it
needs thousands of steps before it closes a thought). train.py --init-from --init-no-opt reads the result.

    python runs/posttrain/cloud/make_init.py S3_CKPT OUT             # Engram tables stay in the model state (GPU)
    python runs/posttrain/cloud/make_init.py S3_CKPT OUT --offload   # tables in sd["engram_offload"] (the 4090)

The S3 checkpoint has the tables GPU-resident (engram.tables.<i> in the model state). Without --offload that layout
is kept: the B300 run uses it ({"engram_offload": false}). With --offload the result holds the same tensors as
runs/posttrain/slim_for_4090.py --special-rows writes (the 4090 rehearsals' starting point).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.dirname(HERE))


def convert(sd: dict, fitted: dict, new_ids: list, offload: bool) -> dict:
    """{"model", "engram_offload", "step"} from a checkpoint with GPU-resident Engram tables."""
    model = sd["model"]
    names = sorted((k for k in model if k.startswith("engram.tables.")), key=lambda k: int(k.rsplit(".", 1)[1]))
    assert names and sd.get("engram_offload") is None, "no GPU-resident Engram tables in this checkpoint"
    head = model["lm_head.weight"].clone()
    assert fitted["ids"] == new_ids and tuple(fitted["rows"].shape) == (len(new_ids), head.shape[1])
    head[new_ids] = fitted["rows"].to(head.dtype)
    out = {k: v for k, v in model.items() if not (offload and k.startswith("engram.tables."))}
    out["lm_head.weight"] = head
    return {"model": out, "engram_offload": [{"weight": model[k]} for k in names] if offload else None,
            "step": sd["step"]}


def main():
    import special_init
    from slim_for_4090 import SyncedFile
    src, dst = sys.argv[1:3]
    rows = os.path.join(os.path.dirname(HERE), "special_rows.pt")
    sd = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    out = convert(sd, torch.load(rows, map_location="cpu", weights_only=False), special_init.NEW_IDS,
                  "--offload" in sys.argv)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    f = SyncedFile(dst + ".tmp")  # fsync and cache drop every 256 MB (a plain 8 GB write took this WSL down)
    torch.save(out, f)
    f.close()
    os.replace(dst + ".tmp", dst)
    print(f"MAKE_INIT_DONE {dst}: step {sd['step']}, {len(out['model'])} tensors, Engram tables "
          f"{'offloaded' if out['engram_offload'] else 'in the model state'}, rows of tokens {special_init.NEW_IDS} "
          f"fitted, {os.path.getsize(dst) / 2 ** 30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
