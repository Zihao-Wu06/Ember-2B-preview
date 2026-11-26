#!/usr/bin/env python
"""Single-GPU pretraining: bf16 autocast, gradient accumulation, WSD schedule with data annealing in
the decay phase (--anneal-mix), exact resume.

Ablation example (4090):
  python scripts/train.py --preset ablation:C_kda_engram --out runs/abl_C --total-tokens 4e8 \
      --seq-len 4096 --micro-bsz 4 --global-tokens 262144 --lr 3e-3 --grad-ckpt
Main run (filtered phase-1 + phase-2 data; ablations use data/bins only so their data never changes),
under scripts/supervise.py, which restarts it after crashes, stalls and slowdowns:
  python scripts/supervise.py --out runs/main -- python scripts/train.py --preset main_2b \
      --data data/bins_main --out runs/main --ckpt-dir /root/ckpt/main --ckpt-every 50 --backup-every 12 \
      --total-tokens 2.5e10 --micro-bsz 2 --global-tokens 524288 --lr 2e-3 --grad-ckpt --ckpt-frac 0.5 \
      --offload-opt --fuse --muon-bf16 --anneal-mix reasoning_anneal
  (settings chosen on the 4090 on 2026-11-10, runs/expert_check/report.md: fusion on, Block AttnRes (the main_2b
   default), half the layers recomputed, routed experts recomputed in the other half and optimizer states in
   pinned CPU memory between steps; peak 20.25 GB and flat, 12.2k tok/s in the pre-flight, 7% faster than
   recomputing every layer. All of this is exact: only recompute and data movement. Real training keeps the
   fp32 gradients of all micro-batches; keep the peak <= ~21 GB. A checkpoint is ~15.6 GB:
   ~10 s on the WSL ext4 disk vs ~2.5 min on /mnt/f, so checkpoints go to the WSL disk and every
   12th one is copied to runs/main/backup in the background.)
Re-running the same command resumes from the newest checkpoint in --ckpt-dir, --out or --out/backup.
Creating <out>/SAVE_AND_EXIT (or sending SIGTERM / SIGUSR1) makes it checkpoint at the end of the
current step and exit with code 75.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
import signal
import sys
import time

import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # MoE shapes vary
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from moe_llm.budget import flops_per_token, param_report  # noqa: E402
from moe_llm.config import PRESETS, ablation_ladder  # noqa: E402
from moe_llm.data import MixtureLoader, short_rows  # noqa: E402
from moe_llm.layers.common import release_compile_workers  # noqa: E402
from moe_llm.model import MoELLM  # noqa: E402
from moe_llm.optim import MUON_CHUNK_ELEMS, OptimizerBundle, wsd_decay_start, wsd_lr  # noqa: E402

# Goal = reasoning, English only: English 57%, math 22%, code 21%.
MIXES = {
    "reasoning": {"en_web": 0.40, "textbook": 0.12, "wiki_en": 0.05, "math": 0.15, "math_infi": 0.07,
                  "code_py": 0.10, "code": 0.07, "code_web": 0.04},
    # LR-decay phase ("annealing", as in Llama 3 / OLMo 2 / SmolLM2): more math, textbooks and Python,
    # less web: English 47%, math 28%, code 25%. In the 25B-token main run (20B stable + 5B decay) every
    # source is at 0.56-0.86 passes when the decay starts and at most ~1.1 passes at the end.
    "reasoning_anneal": {"en_web": 0.24, "textbook": 0.18, "wiki_en": 0.05, "math": 0.20, "math_infi": 0.08,
                         "code_py": 0.14, "code": 0.07, "code_web": 0.04},
    # LR-decay mixes with the high-quality code sources (data_prep/sources.py CODE_ANNEAL, 2026-11-12):
    # Stack-Edu (educational score >= 3), OpenCoder's algorithmic corpus and snippets, all decontaminated
    # against HumanEval / MBPP. They replace code_web (prose: its code blocks were lost in FineWeb's text
    # extraction) and github-code-clean. Sources that only these mixes use join the loader after the stable
    # sources, so the stable phase keeps its exact batches (tests/test_data.py).
    # English 42%, math 28%, code 30%:
    "reasoning_anneal_code": {"en_web": 0.21, "textbook": 0.16, "wiki_en": 0.05, "math": 0.20, "math_infi": 0.08,
                              "code_py": 0.04, "edu_python": 0.12, "edu_code": 0.06, "opc_algo": 0.05,
                              "opc_snippet": 0.03},
    # English and math as in reasoning_anneal (47% / 28%), code 25%:
    "reasoning_anneal_code25": {"en_web": 0.24, "textbook": 0.18, "wiki_en": 0.05, "math": 0.20, "math_infi": 0.08,
                                "code_py": 0.04, "edu_python": 0.10, "edu_code": 0.05, "opc_algo": 0.04,
                                "opc_snippet": 0.02},
    # Stage 2 of the main run (2026-11-12: reasoning and code as the model's strengths), from the
    # switch at step ~12.2k to the LR decay (--mid-mix): English 43%, math 28%, code 29% (real code: code_web is
    # prose and dropped; 20 points of high-quality code). math2 / math_infi2 / edu_python2 / edu_code2 are more
    # shards of the same datasets (data_prep/sources.py STAGE2); each dataset's weight is split between its old
    # and new part so that both end the run at about the same number of passes (all sources <= ~1 pass).
    # The order matters: sources new in this mix are appended to the loader's list in this order.
    "reasoning_s2": {"en_web": 0.280, "textbook": 0.110, "wiki_en": 0.040, "math": 0.094, "math_infi": 0.037,
                     "code_py": 0.050, "code": 0.040, "edu_python": 0.037, "edu_code": 0.035, "opc_algo": 0.020,
                     "opc_snippet": 0.010, "math2": 0.086, "math_infi2": 0.063, "edu_python2": 0.073,
                     "edu_code2": 0.025},
    # reasoning_anneal_code with the same split between the old and new parts (English 42%, math 28%, code 30%)
    "reasoning_anneal_code_s2": {"en_web": 0.21, "textbook": 0.16, "wiki_en": 0.05, "math": 0.104, "math_infi": 0.030,
                                 "code_py": 0.04, "edu_python": 0.040, "edu_code": 0.035, "opc_algo": 0.05,
                                 "opc_snippet": 0.03, "math2": 0.096, "math_infi2": 0.050, "edu_python2": 0.080,
                                 "edu_code2": 0.025},
}

# Context extension, stage S1 (32K on the 4090): 2/3 the main run's final mix
# (keeps short-context quality), 1/3 documents of more than 16K tokens ("long:<source>" windows that start at a
# document, moe_llm.data), each source weighted by its number of 32K windows outside the held-out tail, so every long
# document is read about equally often (~0.2 times in 2B tokens). The counts are data_prep/long_docs.py's output on
# data/bins_main (2026-11-19); code_web (prose), opc_* and textbook have (almost) no documents that long.
# Long part: code 49%, English 24%, math 27%.
LONG_WINDOWS_32K = {"code": 25596, "code_py": 16943, "edu_code": 1185, "edu_code2": 812, "edu_python": 1576,
                    "edu_python2": 3030, "en_web": 22552, "wiki_en": 1737, "math": 12788, "math2": 7840,
                    "math_infi": 3390, "math_infi2": 3363}
MIXES["longctx_s1"] = {
    **{n: w * 2 / 3 for n, w in MIXES["reasoning_anneal_code_s2"].items()},
    **{f"long:{n}": c / sum(LONG_WINDOWS_32K.values()) / 3 for n, c in LONG_WINDOWS_32K.items()},
}
# Stage S2, 64K (64K fits the 4090 with full recompute, runs/longctx_dry64):
# the same split, the long part from documents of more than 32K tokens (the long: windows at --seq-len 65536 take
# documents of at least half a window), weighted by each source's 64K windows outside the held-out tail (the same
# count as data_prep/long_docs.py, 2026-11-20): 1.39B window tokens, each read ~0.36 times in 1.5B tokens.
# Long part: code 47.5%, English 19.2%, math 33.3%.
LONG_WINDOWS_64K = {"code": 6307, "code_py": 2823, "edu_code": 221, "edu_code2": 138, "edu_python": 191,
                    "edu_python2": 397, "en_web": 3857, "wiki_en": 224, "math": 3434, "math2": 2158,
                    "math_infi": 749, "math_infi2": 713}
MIXES["longctx_s2"] = {
    **{n: w * 2 / 3 for n, w in MIXES["reasoning_anneal_code_s2"].items()},
    **{f"long:{n}": c / sum(LONG_WINDOWS_64K.values()) / 3 for n, c in LONG_WINDOWS_64K.items()},
}

# Stages S3 (128K) and S4 (256K), for a rented B300 (runs/longctx_s3/cloud/): the same
# 2/3 short mix; the long third adds PG19 books (data_prep/sources.py "pg19", never evaluated on: the evals use PG19's
# own test split). By window count PG19 would be 88% (128K) / 93% (256K) of the long part, so it is capped (as in
# per-source length upsampling, Fu et al. 2024: keep the domain mix, upsample long documents within each domain):
# PG19 60% / 70%, the other sources 40% / 30% split by their own window counts (each of their windows is then read
# ~0.5 / ~1.0 times in 1.5B tokens). Window counts: long_window_starts() at 131072 / 262144 tokens over the held-out
# tails' complement of data/bins_main (2026-11-21). No repo-level code yet; code has no document >= 128K tokens.
LONG_WINDOWS_128K = {"code": 206, "code_py": 29, "edu_code": 5, "edu_code2": 4, "edu_python": 8, "edu_python2": 7,
                     "en_web": 663, "wiki_en": 19, "math": 990, "math2": 625, "math_infi": 142, "math_infi2": 138}
LONG_WINDOWS_256K = {"en_web": 26, "math": 310, "math2": 177, "math_infi": 29, "math_infi2": 30}


def _long_mix(windows: dict, books: float) -> dict:
    other = sum(windows.values())
    return {**{n: w * 2 / 3 for n, w in MIXES["reasoning_anneal_code_s2"].items()},
            "long:pg19": books / 3, **{f"long:{n}": (1 - books) * c / other / 3 for n, c in windows.items()}}


MIXES["longctx_s3"] = _long_mix(LONG_WINDOWS_128K, 0.6)
MIXES["longctx_s4"] = _long_mix(LONG_WINDOWS_256K, 0.7)


# Post-training SFT at 128K: chat-format sources from data/bins_sft
# (data_prep/sft_tokenize.py, sft_long_packs.py; --data data/bins_sft,<pretraining bins>). By tokens: math 25%, code
# 25%, general reasoning 12%, tool calls 5%, long instruction packs 20% (real long-document questions, multi-document
# and record lookups, the 4K-8K retrieval samples), and 13% plain long documents as in S3's long part (keeps what the
# context extension learned). The weights are provisional until the API data exists: a source's weight should not
# make the run read it more than about 3 times.
def _sft_mix(short: dict, packs: float, plain_long: float, pack: str = "sft_long") -> dict:
    long_part = {n: w for n, w in MIXES["longctx_s3"].items() if n.startswith("long:")}
    z = sum(long_part.values())
    return {**short, f"long:{pack}": packs, **{n: plain_long * w / z for n, w in long_part.items()}}


# general is 3%: the only reasoning data for it so far is OpenThoughts' non-math part (about 4,900 conversations);
# math at 29% of 1.5B tokens is 0.93 passes over sft_math (provisional until the API data is in)
# chat is 4%: about 80,000 SmolTalk conversations, roughly one pass
SFT_V1_SHORT = {"sft_math": 0.29, "sft_code": 0.26, "sft_chat": 0.04, "sft_general": 0.03, "sft_tools": 0.05}
MIXES["sft_v1"] = _sft_mix(SFT_V1_SHORT, 0.20, 0.13)
# the same mix with the packs of 65,536 tokens, for the 4090's 64K windows (rehearsal and comparison runs):
# data_prep/sft_long_packs.py --pack-len 65536 --name sft_long64k
MIXES["sft_v1_64k"] = _sft_mix(SFT_V1_SHORT, 0.20, 0.13, pack="sft_long64k")

# sft_v2 (2026-11-23, API data in): the weights re-derived from the token counts that exist, for 1.5B tokens in all.
#   source        tokens    v1 weight -> passes    v2 weight -> passes
#   sft_math      465.9M    0.29 -> 0.9            0.42 -> 1.35
#   sft_code       31.1M    0.26 -> 12.5           0.09 -> 4.3   (teacher answers that passed their tests; 86,504 convs)
#   sft_chat       61.8M    0.04 -> 1.0            0.04 -> 1.0
#   sft_general     3.7M    0.03 -> 12.0           0.006 -> 2.4  (OpenThoughts, reasoning rewritten short)
#   sft_tools      22.8M    0.05 -> 3.3            0.044 -> 2.9
#   long packs    249.7M    0.20 -> 1.2            0.27 -> 1.6   (128K packs; 161.9M tokens in the 64K packs: 2.5 passes)
# code stays the most repeated source: its share is set by the data (3 passes would be 6%), the per-source held-out
# loss of the run shows when it starts to overfit.
SFT_V2_SHORT = {"sft_math": 0.42, "sft_code": 0.09, "sft_chat": 0.04, "sft_general": 0.006, "sft_tools": 0.044}
MIXES["sft_v2"] = _sft_mix(SFT_V2_SHORT, 0.27, 0.13)
MIXES["sft_v2_64k"] = _sft_mix(SFT_V2_SHORT, 0.27, 0.13, pack="sft_long64k")

# sft_v3 (2026-11-23: code and reasoning are to be strong, with no further API calls): everything on disk
# that sft_v2 left out goes in, plus rStar-Coder's synthetic_sft (a free download), and
# math's filler share goes to them. For 1.5B tokens in all:
#   source           tokens    weight -> passes
#   sft_code          31.1M    0.07  -> 3.4   the teacher's answers that passed their tests (short reasoning)
#   sft_code_pub      14.1M    0.03  -> 3.2   self-oss-instruct answers with a real reasoning, OpenThoughts code
#   sft_code_rstar   145.0M    0.14  -> 1.45  rStar-Coder: verified competition solutions, reasoning <= 12,000 chars
#   long:sft_hard    116.5M    0.10  -> 1.3   the conversations of more than 3,500 tokens of sft_code_pub,
#                                             sft_code_rstar and sft_reason, packed whole: 53% code, 47% math
#                                             (899 packs at 128K, 1,819 at 64K)
#   plain code          >9B    0.04  -> ~0    pre-training code as in the main run's last mix (keeps plain code
#                                             modelling while the rest of the run is chat format)
#   sft_math         465.9M    0.22  -> 0.71
#   sft_reason        70.3M    0.08  -> 1.7   OpenThoughts' math (thoughts <= 12,000 characters) and "other", math_api
#   sft_chat          61.8M    0.04  -> 1.0
#   sft_general        3.7M    0.006 -> 2.4
#   sft_tools         22.8M    0.044 -> 2.9
#   long packs       249.7M    0.17  -> 1.0   (64K packs: 161.9M, 1.6 passes)
#   plain long           -     0.06
# Code is 33% (29% chat format with a reasoning, 4% plain), reasoning 35% (math, reason, the math half of the hard
# packs). By conversations the teacher's short-reasoning code is still seen most (about 294,000 against 116,000 of
# rStar-Coder's), so short tasks keep a short reasoning and competition problems get the long one.
SFT_V3_SHORT = {"sft_math": 0.22, "sft_reason": 0.08, "sft_code": 0.07, "sft_code_pub": 0.03, "sft_code_rstar": 0.14,
                "sft_chat": 0.04, "sft_general": 0.006, "sft_tools": 0.044}
SFT_V3_PACKS, SFT_V3_HARD, SFT_V3_PLAIN_CODE, SFT_V3_PLAIN_LONG = 0.17, 0.10, 0.04, 0.06
PLAIN_CODE = ("code_py", "edu_python", "edu_code", "opc_algo", "opc_snippet", "edu_python2", "edu_code2")


def _sft_v3(pack: str, hard: str, short: dict = SFT_V3_SHORT, packs: float = SFT_V3_PACKS, hard_w: float = SFT_V3_HARD,
            plain_code: float = SFT_V3_PLAIN_CODE, plain_long: float = SFT_V3_PLAIN_LONG) -> dict:
    code = {n: MIXES["reasoning_anneal_code_s2"][n] for n in PLAIN_CODE}
    z = sum(code.values())
    return {**_sft_mix(short, packs, plain_long, pack=pack), f"long:{hard}": hard_w,
            **{n: plain_code * w / z for n, w in code.items()}}


MIXES["sft_v3"] = _sft_v3("sft_long", "sft_hard")
MIXES["sft_v3_64k"] = _sft_v3("sft_long64k", "sft_hard64k")
# Control of the sixth rehearsal: the mix as it was before rStar-Coder was downloaded (the first
# sft_v3), i.e. the recipe that would be trained without it. No sft_code_rstar, and hard packs built without its long
# conversations (sft_long_packs.py --name sft_hard64k_norstar --sets --medium sft_code_pub sft_reason: 62.8M tokens).
SFT_V3_NORSTAR_SHORT = {"sft_math": 0.30, "sft_reason": 0.09, "sft_code": 0.09, "sft_code_pub": 0.03, "sft_chat": 0.04,
                        "sft_general": 0.006, "sft_tools": 0.044}
MIXES["sft_v3_norstar_64k"] = _sft_v3("sft_long64k", "sft_hard64k_norstar", short=SFT_V3_NORSTAR_SHORT, packs=0.19,
                                      hard_w=0.06, plain_code=0.06, plain_long=0.09)

# Small-agent fine-tuning on the 4090 from the SFT model (the goal is a model that writes code, runs
# it, reads the error and fixes it; data_prep/agent_traj.py). 22.5% the new multi-step conversations, 77.5% the SFT
# mix unchanged (64K packs), so that what the SFT run taught is replayed while the new behaviour is learned. The
# shares follow from the data that exists (13.5M and 3.1M tokens) and the rule of at most about 3 passes: in 300 steps
# (157M tokens) sft_agent_tool is read 2.3 times and sft_agent_fix 2.5 times (the first choice, 0.22 / 0.08, was made
# before the data existed and would have read the fix set 4.5 times).
AGENT_V1 = {"sft_agent_tool": 0.18, "sft_agent_fix": 0.045}
MIXES["agent_v1_64k"] = {**AGENT_V1, **{n: w * (1 - sum(AGENT_V1.values())) for n, w in MIXES["sft_v3_64k"].items()}}


def get_mix(spec: str) -> dict:
    return MIXES[spec] if spec in MIXES else json.loads(spec)


def get_config(args):
    if args.preset.startswith("ablation:"):
        cfg = ablation_ladder()[args.preset.split(":", 1)[1]]
    else:
        cfg = PRESETS[args.preset]()
    if args.override:
        cfg = cfg.replace(**json.loads(args.override))
    return cfg


EXIT_RESTART = 75  # checkpoint saved on request (SAVE_AND_EXIT / SIGTERM): restart to continue


def _ckpt_step(path):
    return int(os.path.basename(path)[len("ckpt_"):-len(".pt")])


def latest_ckpt(*dirs):
    """Newest checkpoint over several directories (e.g. --ckpt-dir, --out, --out/backup)."""
    c = [p for d in dirs if d for p in glob.glob(os.path.join(d, "ckpt_*.pt"))]
    return max(c, key=_ckpt_step) if c else None


def backup_ckpt(src, dst_dir, keep):
    """Copy a finished checkpoint into dst_dir (atomic rename) and keep only the newest `keep`."""
    t0 = time.time()
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    for old in sorted(glob.glob(os.path.join(dst_dir, "ckpt_*.pt")))[:-keep]:
        os.remove(old)
    return f"{dst} ({time.time() - t0:.1f}s)"


def heartbeat(out, **kw):
    """<out>/heartbeat.json: progress signal for scripts/supervise.py (atomic rewrite)."""
    tmp = os.path.join(out, "heartbeat.json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"t": time.time(), "pid": os.getpid(), **kw}, fh)
    os.replace(tmp, os.path.join(out, "heartbeat.json"))


def _ckpt_object(step, model, opt, loader_state, cfg, args):
    opt.sync()  # offloaded weights and states: their copies into pinned memory have landed
    obj = {"step": step, "model": model.state_dict(), "opt": opt.state_dict(), "loader": loader_state,
           "cfg": cfg.to_dict(), "args": vars(args), "rng": torch.get_rng_state(),
           "engram_offload": model.engram.offload_state_dict() if model.engram is not None else None}
    if torch.cuda.is_available():
        obj["cuda_rng"] = torch.cuda.get_rng_state()
    return obj


def _write_ckpt(obj, path, out, keep):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)  # atomic: a crash never leaves a half-written checkpoint
    for old in sorted(glob.glob(os.path.join(out, "ckpt_*.pt")))[:-keep]:
        os.remove(old)


def save_ckpt(out, step, model, opt, loader_state, cfg, args, keep=2):
    path = os.path.join(out, f"ckpt_{step:07d}.pt")
    _write_ckpt(_ckpt_object(step, model, opt, loader_state, cfg, args), path, out, keep)
    return path


# ------------------------------------------------------------------ background checkpoint (--async-ckpt)
def _snapshot(o, stable, refs, name=""):
    """The checkpoint object for a background write. Tensors in `stable` (data_ptr of the pinned buffers holding the
    fp32 masters and optimizer states, and of the offloaded Engram tables: nothing writes them before the next
    optimizer step) are kept by reference and listed in `refs`; every other tensor is copied now (CUDA tensors to the
    host, CPU tensors cloned), so the forward passes that run during the write cannot change what is saved."""
    if torch.is_tensor(o):
        if not o.is_cuda and o.numel() and o.data_ptr() in stable:
            refs.append((name, o))
            return o
        return o.detach().cpu() if o.is_cuda else o.detach().clone()
    if isinstance(o, dict):
        return type(o)((k, _snapshot(v, stable, refs, f"{name}.{k}")) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return type(o)(_snapshot(v, stable, refs, f"{name}.{i}") for i, v in enumerate(o))
    return o


def _checksum(t) -> int:
    v = t.detach().contiguous().view(-1)
    if not v.numel():
        return 0
    return int(v.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[v.element_size()])
               .sum(dtype=torch.int64))


def _same_tensors(a, b, name=""):
    """Number of tensors and bytes in a, all bit-identical (and of the same dtype) in b; raises otherwise."""
    if torch.is_tensor(a):
        if not (torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)):
            raise RuntimeError(f"checkpoint verify: {name} differs")
        return 1, a.numel() * a.element_size()
    if isinstance(a, dict):
        if not isinstance(b, dict) or list(a.keys()) != list(b.keys()):
            raise RuntimeError(f"checkpoint verify: keys of {name} differ")
        res = [_same_tensors(v, b[k], f"{name}.{k}") for k, v in a.items()]
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise RuntimeError(f"checkpoint verify: length of {name} differs")
        res = [_same_tensors(v, w, f"{name}.{i}") for i, (v, w) in enumerate(zip(a, b))]
    else:
        if a != b:
            raise RuntimeError(f"checkpoint verify: {name} differs ({a!r} vs {b!r})")
        return 0, 0
    return sum(r[0] for r in res), sum(r[1] for r in res)


class BackgroundSave:
    """One checkpoint written on a thread while the next step's forward / backward passes run. join() must be
    called before anything writes the by-reference tensors again, i.e. before the next optimizer step (train.py
    does it right after the last backward). verify (--ckpt-verify, for tests): the by-reference tensors must not
    have changed between the snapshot and join(), and the file read back must equal the snapshot bit for bit."""

    def __init__(self, out, step, model, opt, loader_state, cfg, args, keep=2, verify=False):
        import threading
        t0 = time.time()
        self.path = os.path.join(out, f"ckpt_{step:07d}.pt")
        obj = _ckpt_object(step, model, opt, loader_state, cfg, args)
        stable = set(opt.stable_host_ptrs())
        if model.engram is not None and getattr(model.engram, "cpu_tables", None) is not None:
            for t in model.engram.cpu_tables:
                stable.update(x.data_ptr() for x in (t.weight, t.m, t.v))
        self.refs = []
        self.obj = _snapshot(obj, stable, self.refs)
        self.snap_s = time.time() - t0  # what the main thread pays for a background save (plus any wait in join())
        self.sums = [_checksum(t) for _, t in self.refs] if verify else None
        self.verify, self.err, self.write_s = verify, None, None
        self.ref_gb = sum(t.numel() * t.element_size() for _, t in self.refs) / 2 ** 30
        self.thread = threading.Thread(target=self._run, args=(out, keep), name="ckpt")  # not a daemon: a crash of
        self.thread.start()                                                             # the main thread lets it finish

    def _run(self, out, keep):
        t0 = time.time()
        try:
            _write_ckpt(self.obj, self.path, out, keep)
            self.write_s = time.time() - t0
        except BaseException as e:  # reported by join()
            self.err = e

    def join(self):
        """Wait for the write; returns (seconds the caller waited, verify note)."""
        t0 = time.time()
        self.thread.join()
        waited = time.time() - t0
        if self.err is not None:
            raise RuntimeError(f"background checkpoint {self.path} failed") from self.err
        note = ""
        if self.verify:
            t1 = time.time()
            for (name, t), s in zip(self.refs, self.sums):
                if _checksum(t) != s:
                    raise RuntimeError(f"checkpoint verify: {name} changed while {self.path} was being written")
            back = torch.load(self.path, map_location="cpu", weights_only=False, mmap=True)
            n, nbytes = _same_tensors(self.obj, back)
            note = (f"; verified: {n} tensors, {nbytes / 2 ** 30:.2f} GB identical in the file, {len(self.refs)} "
                    f"by-reference tensors unchanged ({time.time() - t1:.1f}s)")
        self.obj = self.refs = self.sums = None
        return waited, note


@torch.no_grad()
def evaluate(model, loader, n_batches, device, amp):
    if model.engram is not None and model.engram.offload:
        model.engram.flush()  # a queued Engram table update must land before the lookups here
    model.eval()
    res = {}
    for name in loader.names:
        tot, n = 0.0, 0
        for x, y in loader.val_batches(name, n_batches):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                _, info = model(x.to(device), y.to(device))
            tot += float(info["ce"]) * x.numel()
            n += x.numel()
        if n:  # long:<source> mixture entries have no validation batches of their own
            res[name] = tot / n
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ablation_small")
    ap.add_argument("--override", default="", help="JSON dict of ModelConfig fields")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "bins"))
    ap.add_argument("--canon", default=os.path.join(ROOT, "data", "tokenizer", "canon_map.npy"))
    ap.add_argument("--mix", default="reasoning", help="name in MIXES or a JSON dict")
    ap.add_argument("--anneal-mix", default="", help="data mix for the LR-decay phase (name in MIXES or a JSON "
                    "dict); empty = keep --mix throughout")
    ap.add_argument("--mid-mix", default="", help="data mix from optimizer step --mid-from until the LR decay "
                    "(name in MIXES or a JSON dict); empty = keep --mix until the decay")
    ap.add_argument("--mid-from", type=int, default=0, help="first optimizer step of --mid-mix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--short-seq-len", type=int, default=0,
                    help="context extension: micro-batches from plain (not long:) sources train as rows of this many "
                         "tokens (moe_llm.data.short_rows); 0 = every row at --seq-len")
    ap.add_argument("--short-micro-bsz", type=int, default=0,
                    help="with --short-seq-len: split those rows into micro-batches of this many rows (0 = one "
                         "micro-batch); each weighs its share of the step's tokens")
    ap.add_argument("--short-ckpt-frac", type=float, default=None,
                    help="with --short-micro-bsz: fraction of layers the short micro-batches recompute (default "
                         "--ckpt-frac); the full-length micro-batches keep --ckpt-frac")
    ap.add_argument("--micro-bsz", type=int, default=2)
    ap.add_argument("--global-tokens", type=int, default=524288, help="tokens per optimizer step")
    ap.add_argument("--total-tokens", type=float, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--final-lr-frac", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--engram-lr-mult", type=float, default=5.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--ckpt-frac", type=float, default=1.0, help="with --grad-ckpt: fraction of layers recomputed")
    ap.add_argument("--no-expert-recompute", action="store_true", help="with --ckpt-frac < 1: keep the routed-expert "
                    "buffers of the other layers too (memory then grows with routing imbalance; for comparisons)")
    ap.add_argument("--ckpt-ffn-frac", type=float, default=0.0, help="with --grad-ckpt: this fraction of the "
                    "recomputed layers recompute only their FFN part and keep the mixer's activations")
    ap.add_argument("--offload-act-frac", type=float, default=0.0, help="with --grad-ckpt: this fraction of the "
                    "would-be recomputed layers is not recomputed; their saved activations go to pinned CPU memory "
                    "and come back before their backward (exact; costs PCIe bandwidth and host RAM)")
    ap.add_argument("--fuse", "--compile", dest="compile", action="store_true",
                    help="fuse the memory-bound helpers with torch.compile (same math; also env MOE_FUSE=1)")
    ap.add_argument("--muon-bf16", action="store_true", help="bf16 Muon momentum (saves ~2.6 GB on main_2b)")
    ap.add_argument("--offload-opt", action="store_true", help="keep optimizer states in pinned CPU memory between "
                    "steps (~3.75 GB more GPU memory for activations on main_2b; exact, only data movement)")
    ap.add_argument("--offload-master", action="store_true", help="fp32 weights of the linear layers and experts in "
                    "pinned CPU memory between steps, persistent bf16 copies on the GPU (~2.7 GB less GPU memory, "
                    "no weight casts per micro-batch; exact)")
    ap.add_argument("--fused-grad-acc", action="store_true", help="with --offload-master: the weight-gradient GEMMs "
                    "of the bias-free linear layers and of the routed experts (moe_impl=triton) add their fp32 result "
                    "straight into the fp32 gradient buffers (no bf16 rounding of each micro-batch's weight gradient, "
                    "no separate add kernels; rounding-level change towards the exact sum, activations unchanged)")
    ap.add_argument("--opt-adamw-last", action="store_true", help="optimizer step: copy back and update the Muon "
                    "parameters first and the AdamW ones last (with --offload-opt the step no longer idles ~45 ms "
                    "on main_2b until the AdamW moments of the embedding and LM head have arrived; exact: the same "
                    "updates in another order)")
    ap.add_argument("--engram-early", action="store_true", help="queue the CPU Adam update of the offloaded Engram "
                    "tables right after the last backward of a step instead of after the GPU optimizer step has been "
                    "queued: it overlaps that work instead of delaying the next forward (exact: it reads only the "
                    "tables' own gradients and moments)")
    ap.add_argument("--engram-native", action="store_true", help="run the offloaded Engram tables' lazy Adam in a "
                    "small C kernel (moe_llm/row_adam_native.py) instead of the torch.compile one: bit-identical "
                    "values, but it releases the GIL, so with --engram-early the update overlaps the main thread's "
                    "queuing of the GPU optimizer work")
    ap.add_argument("--engram-native-verify", action="store_true", help="with --engram-native (tests): every update "
                    "also runs the torch.compile kernel on copies of the touched rows and compares bit for bit")
    ap.add_argument("--opt-flat", action="store_true", help="with --offload-master --offload-opt --fused-grad-acc: masters, gradient buffers, bf16 copies and offloaded states in flat buffers (one copy per update segment instead of per tensor; bit-identical)")
    ap.add_argument("--state-min-bytes", type=int, default=0, help="with --offload-opt: optimizer state tensors "
                    "smaller than this stay on the GPU instead of travelling to pinned memory and back every step "
                    "(exact; 0 = offload every state)")
    ap.add_argument("--async-ckpt", action="store_true", help="write the periodic checkpoints on a background "
                    "thread while the next step's forward / backward passes run, and wait for the write before the "
                    "next optimizer step (same checkpoint contents; the checkpoint at a stop request and the final "
                    "one are written as before)")
    ap.add_argument("--ckpt-verify", action="store_true", help="with --async-ckpt (tests): check that nothing "
                    "changed the by-reference tensors during each background write and that the file read back "
                    "equals the snapshot bit for bit")
    ap.add_argument("--muon-impl", default="loop", choices=["loop", "batched", "batched_fused"],
                    help="Muon's Newton-Schulz: loop = one per parameter (default, the original code); batched = the "
                    "parameters of one shape stacked into chunks, one per chunk (per element the same ops; the reload "
                    "copies chunk by chunk); batched_fused = batched with each iteration's scalar terms in the GEMM "
                    "epilogues (fewer roundings, rounding-level change closer to the exact iteration)")
    ap.add_argument("--muon-chunk-elems", type=int, default=0, help="with a batched --muon-impl: elements per stacked "
                    "Newton-Schulz chunk (0 = moe_llm.optim.MUON_CHUNK_ELEMS, 2^25; smaller = less optimizer-step memory)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=500, help="0 = never save (benchmarks)")
    ap.add_argument("--ckpt-dir", default="", help="checkpoint directory (default --out); the WSL ext4 disk "
                    "writes ~15x faster than /mnt/f (1.6 GB/s vs 107 MB/s)")
    ap.add_argument("--backup-every", type=int, default=0,
                    help="with --ckpt-dir: copy every N-th checkpoint to <out>/backup in the background")
    ap.add_argument("--keep-backups", type=int, default=4)
    ap.add_argument("--peak-tflops", type=float, default=165.0, help="for MFU (4090 bf16 dense)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default="", help="start a NEW run from this checkpoint file: its weights (and, "
                    "unless --init-no-opt, optimizer states) with this run's schedule, mix and sequence length from "
                    "step 0 (context extension). Ignored once the run has checkpoints of its own "
                    "(re-running the command then resumes as usual)")
    ap.add_argument("--init-no-opt", action="store_true", help="with --init-from: fresh optimizer states")
    args = ap.parse_args()
    if args.short_seq_len and (args.short_seq_len >= args.seq_len or args.seq_len % args.short_seq_len):
        ap.error("--short-seq-len must divide --seq-len and be shorter")
    if args.short_micro_bsz and (not args.short_seq_len
                                 or (args.micro_bsz * args.seq_len // args.short_seq_len) % args.short_micro_bsz):
        ap.error("--short-micro-bsz needs --short-seq-len and must divide the number of short rows per micro-batch")
    if args.short_ckpt_frac is not None and not args.short_micro_bsz:
        ap.error("--short-ckpt-frac needs --short-micro-bsz")
    if args.fused_grad_acc and not args.offload_master:
        ap.error("--fused-grad-acc needs --offload-master (it accumulates into the offload_master gradient buffers)")

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = args.ckpt_dir or args.out
    os.makedirs(ckpt_dir, exist_ok=True)
    backup_dir = os.path.join(args.out, "backup")
    for d in (ckpt_dir, backup_dir):  # partial writes of a process killed while saving (~15 GB each)
        for t in glob.glob(os.path.join(d, "ckpt_*.pt.tmp")):
            os.remove(t)
    stop_flag = os.path.join(args.out, "SAVE_AND_EXIT")
    if os.path.exists(stop_flag):
        os.remove(stop_flag)  # stale request from an earlier process
    stop = {"why": None}

    def _request_stop(signum, frame):
        stop["why"] = signal.Signals(signum).name

    for name in ("SIGTERM", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _request_stop)
    heartbeat(args.out, step=-1, phase="start")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = get_config(args)
    cfg.to_json(os.path.join(args.out, "config.json"))
    model = MoELLM(cfg).to(device)
    if model.engram is not None and os.path.exists(args.canon):
        model.engram.set_canon_map(np.load(args.canon))
    model.set_checkpointing(args.ckpt_frac if args.grad_ckpt else 0.0, expert_recompute=not args.no_expert_recompute,
                            ffn_frac=args.ckpt_ffn_frac, offload_frac=args.offload_act_frac)
    if args.compile:
        from moe_llm.layers.common import set_compile
        set_compile(True)
    opt = OptimizerBundle(model, lr=args.lr, weight_decay=args.wd, engram_lr_mult=args.engram_lr_mult,
                          muon_state_dtype=torch.bfloat16 if args.muon_bf16 else torch.float32,
                          offload_states=args.offload_opt and device.type == "cuda", offload_master=args.offload_master,
                          fused_grad_acc=args.fused_grad_acc, adamw_last=args.opt_adamw_last,
                          muon_impl=args.muon_impl, muon_chunk_elems=args.muon_chunk_elems or MUON_CHUNK_ELEMS,
                          engram_early=args.engram_early, state_min_bytes=args.state_min_bytes,
                          engram_native=args.engram_native, engram_native_verify=args.engram_native_verify,
                          flat=args.opt_flat)
    tpm = args.micro_bsz * args.seq_len
    accum = max(1, args.global_tokens // tpm)
    tps = accum * tpm
    total = args.steps or math.ceil(args.total_tokens / tps)
    decay_start = wsd_decay_start(total, args.decay_frac)
    mix = get_mix(args.mix)
    anneal_mix = get_mix(args.anneal_mix) if args.anneal_mix else None
    mid_mix = get_mix(args.mid_mix) if args.mid_mix else None
    if mid_mix and not 0 <= args.mid_from <= decay_start:
        raise SystemExit(f"--mid-from {args.mid_from} must lie in [0, {decay_start}] (the LR decay starts at step "
                         f"{decay_start})")
    # the anneal mix starts with the LR decay; loader steps count micro-batches
    loader = MixtureLoader(args.data, mix, args.seq_len, args.micro_bsz, seed=args.seed, anneal_weights=anneal_mix,
                           anneal_from=decay_start * accum if anneal_mix else None, mid_weights=mid_mix,
                           mid_from=args.mid_from * accum if mid_mix else None)
    rep = param_report(cfg)
    flops_tok = flops_per_token(cfg, args.seq_len, rep)["total"]

    start = 0
    ck = latest_ckpt(ckpt_dir, args.out, backup_dir)
    if ck:
        t_load = time.time()
        # mmap: tensors are paged in from the file while being copied into place, so the ~15.6 GB main-run
        # checkpoint never sits in RAM next to the 8 GB of Engram CPU tables (WSL has ~25 GB).
        sd = torch.load(ck, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        loader.load_state_dict(sd["loader"])
        if model.engram is not None:
            model.engram.load_offload_state_dict(sd["engram_offload"])
        torch.set_rng_state(sd["rng"])
        if "cuda_rng" in sd and amp:
            torch.cuda.set_rng_state(sd["cuda_rng"])
        start = sd["step"]
        del sd
        print(f"resumed from {ck} at step {start} ({time.time() - t_load:.1f}s)", flush=True)
    elif args.init_from:
        t_load = time.time()
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False, mmap=True)
        model.load_state_dict(sd["model"])
        if model.engram is not None:
            model.engram.load_offload_state_dict(sd["engram_offload"])
        if not args.init_no_opt:
            opt.load_state_dict(sd["opt"])
        src_step = sd["step"]
        del sd
        print(f"initialized from {args.init_from} (its step {src_step}; optimizer states "
              f"{'fresh' if args.init_no_opt else 'loaded'}); this run starts at step 0 with its own schedule, mix and "
              f"seq_len {args.seq_len} ({time.time() - t_load:.1f}s)", flush=True)
    opt.begin()  # offload_master: bf16 copies from the (loaded) weights, fp32 masters to pinned CPU memory

    print(f"model total {rep['total'] / 1e9:.3f}B, active {rep['active_matmul'] / 1e6:.0f}M | "
          f"{total} steps x {tps} tokens (accum {accum}) = {total * tps / 1e9:.2f}B tokens | "
          f"opt groups {opt.groups}", flush=True)
    sched = f"mix {args.mix}"
    if mid_mix:
        sched += f", mid mix {args.mid_mix} from step {args.mid_from}" + \
            (" [active]" if args.mid_from <= start < decay_start else "")
    if anneal_mix:
        sched += f", anneal mix {args.anneal_mix} from step {decay_start} (LR decay)" + \
            (" [active]" if start >= decay_start else "")
    print(f"data: {sched} | planned passes per source: " +
          " ".join(f"{n} {e:.2f}" for n, e in loader.planned_epochs(total * accum).items()), flush=True)
    if args.short_seq_len:
        print(f"short rows: micro-batches from plain sources as [{args.micro_bsz * args.seq_len // args.short_seq_len}, "
              f"{args.short_seq_len}], long: windows at [{args.micro_bsz}, {args.seq_len}]"
              + (f"; short rows in micro-batches of {args.short_micro_bsz} rows, recompute fraction "
                 f"{args.ckpt_frac if args.short_ckpt_frac is None else args.short_ckpt_frac}" if args.short_micro_bsz
                 else ""), flush=True)
    print(model.recompute_summary(), flush=True)
    if args.offload_opt:
        print(f"optimizer states: {'pinned CPU memory between steps' if opt.offload_states else 'on device (no CUDA)'}",
              flush=True)
    if args.opt_adamw_last:
        print("optimizer step order: Muon first, AdamW last", flush=True)
    if args.engram_early and model.engram is not None and model.engram.offload:
        print("Engram CPU update: queued right after the last backward of each step (--engram-early)", flush=True)
    if args.engram_native and model.engram is not None and model.engram.offload:
        print("Engram CPU update kernel: " + ("native C, GIL released (--engram-native"
                                              + (", verified against the torch.compile kernel" if args.engram_native_verify else "")
                                              + ")" if opt.engram_native else "torch.compile (the native kernel is unavailable)"),
              flush=True)
    if args.opt_flat:
        if opt.flat:
            xfer = (f"masters in {len(opt._fpin.blocks)} pinned blocks ({opt._fpin.nbytes / 1e9:.2f} GB)"
                    if opt._flat_xfer and opt._fpin is not None else "transfers per tensor (pinned memory)")
            print(f"optimizer: flat layout ({len(opt._segments)} update segments, {opt._fspace.numel / 1e9:.3f}B elements "
                  f"per master / gradient / copy buffer; {xfer})", flush=True)
        else:
            print("optimizer: flat layout requested but off (see the [optim] note)", flush=True)
    if args.state_min_bytes and opt.offload_states:
        print(f"optimizer states below {args.state_min_bytes} bytes stay on the GPU (--state-min-bytes)", flush=True)
    if args.async_ckpt:
        print("periodic checkpoints: written in the background during the next step (--async-ckpt"
              + (", verified" if args.ckpt_verify else "") + ")", flush=True)
    if args.muon_impl != "loop" and opt.muon is not None:
        chunks = [c for gi in range(len(opt.muon.param_groups)) for c in opt.muon._chunks(gi)]
        print(f"Muon Newton-Schulz: {args.muon_impl}, {len(chunks)} chunks for {opt.groups['muon']} parameters"
              + (" (scalar terms in the GEMM epilogues)" if args.muon_impl == "batched_fused" else ""), flush=True)
    if args.offload_master:
        print(f"weights: fp32 masters of {len(opt._cast)} linear/expert tensors in "
              f"{'pinned CPU memory between steps' if amp else 'place (no CUDA)'}, persistent bf16 copies for the "
              "forward", flush=True)
        print("weight gradients: " + ("fp32, accumulated inside the weight-gradient GEMMs (bias-free linear layers"
                                      + (", routed experts" if cfg.moe_impl == "triton" else "") + ")"
                                      if args.fused_grad_acc else "bf16 per micro-batch, added into fp32 buffers"),
              flush=True)
    from moe_llm.layers.common import fusion_on
    print(f"fusion {'on' if fusion_on() and amp else 'off'} (norms, gates, SwiGLU, short convs, residual add + norm)",
          flush=True)
    if cfg.fuse_proj:
        print(f"projections: fused, {len(model.fused_proj_groups())} groups (per KDA layer q/k/v/f_down/g_down/b_proj, "
              f"per attention layer q/k/v/gate: one GEMM each for the forward and dx; KDA q/k/v short convs in one "
              f"function)" + (f"; {len(opt._groups)} stacked persistent copies" if opt._groups else "")
              + (f"; fp32 dW GEMM {'per member' if cfg.fuse_proj_dw == 'member' else 'per stack'}"
                 if args.fused_grad_acc else ""), flush=True)
    if cfg.conv_impl != "compiled":
        print(f"KDA short convs: conv_impl {cfg.conv_impl} (fla's {'Gluon' if cfg.conv_impl == 'fla_gluon' else 'Triton'}"
              f" causal_conv1d kernels on CUDA" + ("; inactive with fuse_proj" if cfg.fuse_proj else "")
              + "; the Engram conv stays compiled)", flush=True)
    print(f"LM head + cross-entropy: ce_impl {cfg.ce_impl}, chunk {model.ce_chunk}", flush=True)
    if cfg.residual != "standard" and cfg.attnres_impl == "fused":
        from moe_llm.layers.attnres_fused import describe
        print(f"residual {cfg.residual}: AttnRes op {describe(device.type)}", flush=True)
    elif cfg.residual != "standard":
        from moe_llm.layers.attnres import attnres_compiled
        print(f"residual {cfg.residual}: AttnRes op {'torch.compile' if attnres_compiled() and amp else 'eager'}",
              flush=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    model.train()
    nan_skips = 0

    # Read the next step's token windows on a background thread while the GPU computes the current
    # step (data lives on /mnt/f; 9P latency spikes would otherwise stall the GPU). The loader state
    # saved in checkpoints is the one right after the CURRENT step's batches, so resume stays exact.
    from concurrent.futures import ThreadPoolExecutor
    io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="data")

    def fetch_step():
        """The step's micro-batches as (x, y, short): short = plain-source rows of --short-seq-len tokens, split into
        micro-batches of --short-micro-bsz rows if that is set (each then weighs its share of the step's tokens)."""
        b = []
        for _ in range(accum):
            x, y = loader.next_batch()
            xs, ys = short_rows(x, y, loader.last_names, args.short_seq_len)
            short = xs.shape[1] != x.shape[1]
            if short and args.short_micro_bsz:
                b += [(xa, ya, True) for xa, ya in zip(xs.split(args.short_micro_bsz), ys.split(args.short_micro_bsz))]
            else:
                b.append((xs, ys, short))
        if device.type == "cuda":  # pinned: the per-micro-batch copies to the GPU are asynchronous (no host sync)
            b = [(x.pin_memory(), y.pin_memory(), s) for x, y, s in b]
        return b, loader.state_dict()

    # --short-ckpt-frac: the short micro-batches recompute this fraction of the layers, the others --ckpt-frac
    # (MoELLM._call decides per forward, so switching between micro-batches is safe)
    rec_frac = {False: args.ckpt_frac, True: args.ckpt_frac if args.short_ckpt_frac is None else args.short_ckpt_frac}
    rec_now = [args.ckpt_frac]

    def set_recompute(short):
        f = rec_frac[short]
        if args.grad_ckpt and f != rec_now[0]:
            model.set_checkpointing(f, expert_recompute=not args.no_expert_recompute, ffn_frac=args.ckpt_ffn_frac,
                                    offload_frac=args.offload_act_frac)
            rec_now[0] = f

    pending = io_pool.submit(fetch_step)
    loader_state_after_step = None
    bk_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backup")
    backups = []
    n_saved = 0

    def drain_backups(wait=False):
        for f in list(backups):
            if wait or f.done():
                try:
                    print("backup", f.result(), flush=True)
                except Exception as e:  # a failed copy must not stop training: the checkpoint itself is intact
                    print(f"backup FAILED: {e!r}", flush=True)
                backups.remove(f)

    bg_save = None  # a BackgroundSave in flight (--async-ckpt)

    def saved(path):
        nonlocal n_saved
        n_saved += 1
        drain_backups()
        if args.backup_every and os.path.abspath(ckpt_dir) != os.path.abspath(args.out) \
                and n_saved % args.backup_every == 0:
            backups.append(bk_pool.submit(backup_ckpt, path, backup_dir, args.keep_backups))

    def finish_save():
        """Wait for the background checkpoint, if any. Called before the next optimizer step and before exiting."""
        nonlocal bg_save
        if bg_save is None:
            return
        h, bg_save = bg_save, None
        waited, note = h.join()
        print(f"saved {h.path} ({h.write_s:.1f}s in the background, {h.ref_gb:.1f} GB by reference; main thread: "
              f"snapshot {h.snap_s:.2f}s, waited {waited:.2f}s{note})", flush=True)
        saved(h.path)

    def checkpoint(step_done, background=False):
        nonlocal bg_save
        finish_save()
        heartbeat(args.out, step=step_done - 1, phase="ckpt")
        t_save = time.time()
        if background:
            bg_save = BackgroundSave(ckpt_dir, step_done, model, opt, loader_state_after_step, cfg, args,
                                     verify=args.ckpt_verify)
            return bg_save.path
        path = save_ckpt(ckpt_dir, step_done, model, opt, loader_state_after_step, cfg, args)
        print(f"saved {path} ({time.time() - t_save:.1f}s)", flush=True)
        saved(path)
        return path

    # Diagnostics. Log steps print where the step's time goes (CUDA events on the compute stream). `touch <out>/PROFILE`
    # profiles the next step boundary (from the end of a step's micro-batches to the second micro-batch of the next
    # step) with torch.profiler: <out>/profile_step<k>.json (chrome trace) and .txt (op table); the flag is removed.
    rf = torch.profiler.record_function  # labels in the trace; negligible cost when not profiling
    prof_flag = os.path.join(args.out, "PROFILE")
    prof, prof_left, prev_opt_ev = None, 0, None

    def start_profile():
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        p = torch.profiler.profile(activities=acts)
        p.__enter__()
        return p

    def stop_profile(p, step):
        if device.type == "cuda":
            torch.cuda.synchronize()
        p.__exit__(None, None, None)
        base = os.path.join(args.out, f"profile_step{step}")
        p.export_chrome_trace(base + ".json")
        key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        with open(base + ".txt", "w") as fh:
            fh.write(p.key_averages().table(sort_by=key, row_limit=80))
        print(f"profile: {base}.json / .txt (end of step {step - 1} to micro-batch 2 of step {step})", flush=True)

    def cuda_event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    path = None
    for step in range(start, total):
        t0 = time.time()
        lr = wsd_lr(step, total, args.lr, args.warmup, args.decay_frac, args.final_lr_frac)
        ce_acc = torch.zeros((), device=device)
        aux_acc = torch.zeros((), device=device)
        loss_sum = torch.zeros((), device=device)
        with rf("step.data_wait"):
            batches, loader_state_after_step = pending.result()  # token ids only: a few MB
        t_data = time.time() - t0
        ev = {"start": cuda_event()} if amp else {}
        pending = io_pool.submit(fetch_step) if step + 1 < total else None
        if mid_mix and step == args.mid_from and step < decay_start:
            print(f"data mid: mix {args.mid_mix} from step {step}", flush=True)
        if anneal_mix and step == decay_start:
            print(f"data anneal: mix {args.anneal_mix} from step {step}", flush=True)
        offload = model.engram is not None and model.engram.offload
        if offload:  # Engram rows for micro-batch 0; later ones are prefetched one ahead
            with rf("step.engram_prefetch0"):
                model.engram.prefetch(batches[0][0])
        for i, (x, y, short) in enumerate(batches):
            if offload and i + 1 < len(batches):
                model.engram.prefetch(batches[i + 1][0])
            set_recompute(short)
            div = tps / x.numel()  # = accum for a full micro-batch (as before); more for a split short one
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss, info = model(x, y)
            (loss / div).backward()
            ce_acc += info["ce"] / div
            aux_acc += info["aux"] / div
            loss_sum = loss_sum + loss.detach().float()
            if prof is not None:
                prof_left -= 1
                if prof_left == 0:
                    stop_profile(prof, step)
                    prof = None
        if amp:
            ev["loop"] = cuda_event()
        if prof is None and os.path.exists(prof_flag):
            os.remove(prof_flag)
            prof, prof_left = start_profile(), 2
        t1 = time.time()
        # one host sync per step instead of one per micro-batch (each stalled the GPU between forward and
        # backward); a non-finite micro-batch still runs its backward, then the whole step is skipped as before
        with rf("step.finite_sync"):
            finite = bool(torch.isfinite(loss_sum))
        t_sync = time.time() - t1
        if not finite:
            opt.zero_grad()
            if model.engram is not None and model.engram.offload:
                model.engram.flush()
                for t in model.engram.cpu_tables:
                    t.grad.zero_()
                    t.touched.zero_()
            nan_skips += 1
            print(f"step {step}: non-finite loss, skipped ({nan_skips})", flush=True)
            if nan_skips > 10:
                raise RuntimeError("too many non-finite steps")
            continue
        t2 = time.time()
        with rf("opt.finish_save"):
            finish_save()  # a background checkpoint reads the masters and states that this optimizer step rewrites
        with rf("opt.start_engram_update"):
            opt.start_engram_update(lr)  # --engram-early: the Engram CPU update overlaps the GPU optimizer work
        with rf("opt.clip_grad_norm"):
            gnorm = opt.clip_grad_norm(args.clip)  # == clip_grad_norm_ over model.parameters()
        with rf("opt.step"):
            opt.step(lr)
        with rf("opt.zero_grad"):
            opt.zero_grad()
        with rf("opt.router_bias"):
            moe = model.update_router_bias()
        t_opt = time.time() - t2
        if amp:
            ev["opt"] = cuda_event()
        is_log = step % args.log_every == 0 or step == total - 1
        if amp and is_log:  # other steps do not wait for the optimizer step: the next one starts meanwhile
            torch.cuda.synchronize()
        dt = time.time() - t0
        if is_log and amp:
            gap = prev_opt_ev.elapsed_time(ev["start"]) / 1e3 if prev_opt_ev is not None else float("nan")
            print(f"timing: data wait {t_data:.2f}s, GPU gap after the previous step {gap:.2f}s, micro-batches "
                  f"{ev['start'].elapsed_time(ev['loop']) / 1e3:.2f}s, optimizer {ev['loop'].elapsed_time(ev['opt']) / 1e3:.2f}s "
                  f"(CPU: finite-check wait {t_sync:.2f}s, optimizer calls {t_opt:.2f}s)", flush=True)
        if amp:
            prev_opt_ev = ev["opt"]
        if is_log:
            moe = {k: int(v) if k.endswith("dead_experts") else float(v) for k, v in moe.items()}
            rec = {"step": step, "tokens": (step + 1) * tps, "lr": lr, "ce": float(ce_acc), "aux": float(aux_acc),
                   "gnorm": float(gnorm), "tok_s": tps / dt, "mfu": tps / dt * flops_tok / (args.peak_tflops * 1e12),
                   "mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if amp else 0.0, **moe}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
            if model.offloader is not None:
                st = model.offloader.stats
                print(f"activation offload: {st['bytes'] / 2 ** 30 / max(1, (step - start + 1) * accum):.2f} GB per "
                      f"micro-batch, host buffers {st['host_bytes'] / 2 ** 30:.2f} GB, prefetch misses {st['late']}",
                      flush=True)
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step == total - 1):
            heartbeat(args.out, step=step, phase="eval")
            ev = evaluate(model, loader, args.eval_batches, device, amp)
            # always weighted by --mix, so the number stays comparable across the mid and anneal switches
            ev["weighted"] = sum(ev[n] * mix.get(n, 0.0) for n in ev) / sum(mix.get(n, 0.0) for n in ev)
            log.write(json.dumps({"step": step, "eval": ev}) + "\n")
            log.flush()
            print("eval", {k: round(v, 4) for k, v in ev.items()}, flush=True)
        if is_log:  # nothing is compiling now (new shapes later recompile on demand): free the idle compile workers
            release_compile_workers()
        path = None
        if args.ckpt_every > 0 and ((step + 1) % args.ckpt_every == 0 or step == total - 1):
            path = checkpoint(step + 1, background=args.async_ckpt and step + 1 < total)
        if stop["why"] is None and os.path.exists(stop_flag):
            stop["why"] = "SAVE_AND_EXIT"
        heartbeat(args.out, step=step, dt=dt, phase="train")  # dt excludes eval and checkpoint time
        if stop["why"] is not None and step + 1 < total:
            path = path or checkpoint(step + 1)
            finish_save()  # this step's periodic checkpoint may still be on its way to the disk
            drain_backups(wait=True)
            print(f"{stop['why']}: checkpoint {path}; exiting with code {EXIT_RESTART}", flush=True)
            sys.exit(EXIT_RESTART)
    finish_save()
    drain_backups(wait=True)
    if path and os.path.abspath(ckpt_dir) != os.path.abspath(args.out) \
            and not os.path.exists(os.path.join(backup_dir, os.path.basename(path))):
        print("backup", backup_ckpt(path, backup_dir, args.keep_backups), flush=True)  # final model on --out
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
