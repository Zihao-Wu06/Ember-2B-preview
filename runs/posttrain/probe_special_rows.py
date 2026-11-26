#!/usr/bin/env python
"""Output rows for the chat format's new tokens, fitted on the model's own hidden states.

    python runs/posttrain/probe_special_rows.py --export runs/mac_eval/s3_0002862 --device cuda

Why: copying EOS's row (special_init.py) lifted </think> after a thought from 1e-5 to 0.1%-3% and the second rehearsal
(38 steps) from 1% to 4%-19%, but the eight new tokens still start almost alike (their rows differ by noise only), so
the model ends a thought with <|im_end|>, EOS or <|im_start|> instead of </think>. Here each new token's row is the
solution of a ridge regression on hidden states of the SFT conversations, teacher-forced through the S3 model:
  - where the token is the next one in the conversation (positives): its logit should be `margin_pos` above the best
    logit of any ordinary token;
  - elsewhere (ordinary positions, and the positions where another new token belongs): `margin_neg` below it.
Hidden states are the input of the output head, so the fitted rows are exactly what the head needs to produce
those logits. The rows are scored on conversations the fit has not seen: probability and top-1 rate at the
token's own positions, and the mass the new tokens take at ordinary positions.

Writes runs/posttrain/special_rows.pt: {"ids", "rows" [8, d] fp32, "report"}; slim_for_4090.py --special-rows uses it.
"""
import argparse
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import mac_lm  # noqa: E402
from moe_llm import chat  # noqa: E402
from special_init import EOS_ID, NEW_IDS  # noqa: E402

# set -> conversations (fit + test); the sets without thinking get an empty thinking block from chat.render
SETS = {"math": 60, "gsm8k": 30, "code_selfoss": 50, "openthoughts": 30, "chat": 40, "tools_glaive": 30, "tools_hermes": 20}


def conversations(tok, per_scale, max_tokens=2200):
    """{set: [ids, ...]}: the first conversations of each set that fit max_tokens, EOS in front as in the stream."""
    out = {}
    for name, n in SETS.items():
        got = []
        with open(os.path.join(ROOT, "data", "sft", "conv", name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                ids = [EOS_ID] + tok.encode(chat.render(json.loads(line))).ids
                if len(ids) <= max_tokens:
                    got.append(ids)
                    if len(got) == int(n * per_scale):
                        break
        out[name] = got
    return out


@torch.inference_mode()
def collect(model, convs, rng, neg_per_conv=24):
    """Hidden states (CPU, fp32), the target token and the best ordinary logit at every position that is used: all
    positions whose next token is a new one, and neg_per_conv others."""
    dev = next(model.parameters()).device
    W = model.lm_head.weight
    ordinary = torch.ones(W.shape[0], dtype=torch.bool, device=dev)
    ordinary[NEW_IDS + [EOS_ID]] = False
    Wo = W[ordinary]
    hs, tgt, best = [], [], []
    for ids in convs:
        h, _ = model.hidden(torch.tensor([ids], device=dev))
        h = h[0].float()
        nxt = torch.tensor(ids[1:] + [-1], device=dev)
        pos = [j for j in range(len(ids) - 1) if ids[j + 1] in NEW_IDS]
        rest = [j for j in range(len(ids) - 1) if ids[j + 1] not in NEW_IDS and ids[j + 1] != EOS_ID and j > 4]
        pos += rng.sample(rest, min(neg_per_conv, len(rest)))
        sel = torch.tensor(pos, device=dev)
        hs.append(h[sel].cpu())
        tgt.append(nxt[sel].cpu())
        best.append((h[sel] @ Wo.T).max(-1).values.cpu())
    return torch.cat(hs), torch.cat(tgt), torch.cat(best)


def fit(H, tgt, best, margin_pos, margin_neg, ridge, pos_weight):
    """[8, d] rows: per new token a weighted ridge regression of its target logit on the hidden state."""
    rows = []
    d = H.shape[1]
    for t in NEW_IDS:
        pos = tgt == t
        y = torch.where(pos, best + margin_pos, best + margin_neg)
        wgt = torch.where(pos, torch.tensor(pos_weight), torch.tensor(1.0))
        Xw = H * wgt.sqrt()[:, None]
        A = Xw.T @ Xw + ridge * wgt.sum() * torch.eye(d)
        rows.append(torch.linalg.solve(A, Xw.T @ (y * wgt.sqrt())))
    return torch.stack(rows)


@torch.inference_mode()
def score(model, rows, H, tgt):
    """Per new token: mean probability and top-1 rate at its own positions; the mass all new tokens take at ordinary
    positions (full softmax with the fitted rows in place)."""
    dev = next(model.parameters()).device
    W = model.lm_head.weight
    keep = W[NEW_IDS].clone()
    W[NEW_IDS] = rows.to(dev)
    out = {}
    try:
        Hd = H.to(dev)
        lp = torch.empty(len(Hd), dtype=torch.float32, device=dev)
        logits_new = torch.empty(len(Hd), len(NEW_IDS), device=dev)
        top = torch.empty(len(Hd), dtype=torch.long, device=dev)
        probs = torch.empty(len(Hd), len(NEW_IDS), device=dev)
        for a in range(0, len(Hd), 512):
            lg = Hd[a:a + 512] @ W.T
            ls = lg.log_softmax(-1)
            probs[a:a + 512] = ls[:, NEW_IDS].exp()
            top[a:a + 512] = lg.argmax(-1)
        T = tgt.to(dev)
        for t in NEW_IDS:
            m = T == t
            if int(m.sum()):
                out[f"token {t}"] = {"positions": int(m.sum()), "p": round(float(probs[m][:, NEW_IDS.index(t)].mean()), 3),
                                     "top1": round(float((top[m] == t).float().mean()), 3)}
        ordinary = ~torch.isin(T, torch.tensor(NEW_IDS, device=dev))
        out["ordinary positions"] = {"positions": int(ordinary.sum()), "mass of the new tokens": round(float(probs[ordinary].sum(1).mean()), 5),
                                     "top-1 is a new token": round(float(torch.isin(top[ordinary], torch.tensor(NEW_IDS, device=dev)).float().mean()), 5)}
    finally:
        W[NEW_IDS] = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="runs/mac_eval/s3_0002862")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--margin-pos", type=float, default=3.0)
    ap.add_argument("--margin-neg", type=float, default=-8.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--out", default="runs/posttrain/special_rows.pt")
    a = ap.parse_args()
    model, tok, meta = mac_lm.load(a.export, a.device)
    model.requires_grad_(False)
    fit_convs = conversations(tok, 1.0)
    test_convs = conversations(tok, 2.0)   # the first conversations of each set are the fit's, the rest are new ones
    rng = random.Random(0)
    fit_ids = [i for v in fit_convs.values() for i in v]
    test_ids = [i for k, v in test_convs.items() for i in v[len(fit_convs[k]):]]
    print(f"{len(fit_ids)} conversations to fit on, {len(test_ids)} held out", flush=True)
    Hf, tf, bf = collect(model, fit_ids, rng)
    Ht, tt, bt = collect(model, test_ids, rng)
    print(f"{len(Hf)} fit positions ({int(torch.isin(tf, torch.tensor(NEW_IDS)).sum())} positives), {len(Ht)} held-out positions", flush=True)
    report = {}
    from special_init import init_rows
    eos_rows = init_rows(model.lm_head.weight.detach().cpu()).float()
    report["EOS rows (special_init.py), held out"] = score(model, eos_rows, Ht, tt)
    rows = fit(Hf, tf, bf, a.margin_pos, a.margin_neg, a.ridge, a.pos_weight)
    report["fitted rows, fit data"] = score(model, rows, Hf, tf)
    report["fitted rows, held out"] = score(model, rows, Ht, tt)
    print(json.dumps(report, indent=1), flush=True)
    torch.save({"ids": NEW_IDS, "rows": rows, "report": report, "args": vars(a)}, a.out)
    print("PROBE_SPECIAL_ROWS_DONE", a.out, tuple(rows.shape), "row norms", [round(float(r.norm()), 2) for r in rows], flush=True)


if __name__ == "__main__":
    main()
