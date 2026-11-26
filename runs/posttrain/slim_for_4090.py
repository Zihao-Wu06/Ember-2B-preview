#!/usr/bin/env python
"""A checkpoint of the cloud run (S3 on the B300: Engram tables GPU-resident, as nn.Parameters engram.tables.<i> in the
model state) -> a weights-only init checkpoint for the 4090 (Engram tables offloaded to the CPU: sd["engram_offload"] is
a list of {"weight": ...}, the model state has no engram.tables.*). train.py --init-from --init-no-opt reads exactly
this: the weights, fp32 as stored, and the step; the lazy-Adam moments of the tables start at zero (weights-only state).
The optimizer states are dropped (7.9 GB instead of 15.6 GB).

    python runs/posttrain/slim_for_4090.py <ckpt_dir>/ckpt_0002862.pt <ckpt_dir>/init_for_4090.pt
    python runs/posttrain/slim_for_4090.py SRC DST --init-special    # also: output rows of the chat tokens from EOS's row

--special-rows runs/posttrain/special_rows.pt: the rows fitted on the model's own hidden states
(runs/posttrain/probe_special_rows.py) instead of EOS's row (implies --init-special).

--init-special: the S3 head has one and the same, very negative row for <|im_start|>, <|im_end|> and the reserved tokens
(pre-training never saw them), so an SFT run needs thousands of steps before it learns to write </think> and <|im_end|>;
runs/posttrain/special_init.py says what is done and why. The rows are the only thing changed.

The output is written with an fsync and a cache drop every 256 MB: on this WSL a 15 GB `cp` piled up dirty pages and
twice took the whole virtual machine down (2026-11-22).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYNC = 256 * 2 ** 20


class SyncedFile:
    def __init__(self, path):
        self.f, self.pending = open(path, "wb"), 0

    def write(self, b):
        self.f.write(b)
        self.pending += len(b)
        if self.pending >= SYNC:
            self.sync()
        return len(b)

    def sync(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        os.posix_fadvise(self.f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.pending = 0

    def flush(self):
        self.f.flush()

    def close(self):
        self.sync()
        self.f.close()


def main():
    src, dst = sys.argv[1:3]
    rows_path = sys.argv[sys.argv.index("--special-rows") + 1] if "--special-rows" in sys.argv else None
    special = "--init-special" in sys.argv or rows_path is not None
    sd = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    model = sd["model"]
    names = sorted((k for k in model if k.startswith("engram.tables.")), key=lambda k: int(k.rsplit(".", 1)[1]))
    assert names and sd.get("engram_offload") is None, "no GPU-resident Engram tables in this checkpoint"
    out = {"model": {k: v for k, v in model.items() if not k.startswith("engram.tables.")},
           "engram_offload": [{"weight": model[k]} for k in names], "step": sd["step"]}
    if special:
        import special_init
        head = out["model"]["lm_head.weight"].clone()
        if rows_path:
            fitted = torch.load(rows_path, map_location="cpu", weights_only=False)
            assert fitted["ids"] == special_init.NEW_IDS and fitted["rows"].shape == (len(special_init.NEW_IDS), head.shape[1])
            head[special_init.NEW_IDS] = fitted["rows"].to(head.dtype)
        else:
            head[special_init.NEW_IDS] = special_init.init_rows(head).to(head.dtype)
        out["model"]["lm_head.weight"] = head
        print(f"output rows of tokens {special_init.NEW_IDS} set from {rows_path or 'the EOS row'}", flush=True)
    f = SyncedFile(dst + ".tmp")
    torch.save(out, f)
    f.close()
    os.replace(dst + ".tmp", dst)
    print(f"SLIM_FOR_4090_DONE {dst}: step {sd['step']}, {len(out['model'])} tensors and {len(names)} Engram tables, "
          f"{os.path.getsize(dst) / 2 ** 30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
