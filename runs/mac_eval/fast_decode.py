#!/usr/bin/env python
"""Fast decoding of one conversation on the GPU: the per-token step is a short sequence of fused kernels
(fast_kernels.py) recorded once as a CUDA graph and replayed, in fp32.

The ordinary decoder (decode.py) runs the model's own layers one token at a time: about 3,000 small GPU kernels per
token, launched from Python, so the GPU is one-third busy and a 4090 gives about 23 tokens/s. Here
  - a KDA layer is three kernels (one matmul for all its input projections, one kernel for the convs, gates, state
    update and read-out of every head, one matmul for the output projection);
  - a MoE layer reads its routed experts' weights in place (no per-expert loop, no gathered copies);
  - Block AttnRes depth attention and the norm that follows it are one kernel, and the residual adds are folded into
    the kernels that produce their operands;
  - a full-attention layer is three kernels as well: its keys and values sit in buffers of max_len positions and one
    kernel reads only the positions in use, so a step costs what the conversation so far costs, not max_len;
  - all state lives in fixed buffers that the step updates in place, the next token is chosen on the GPU, and the
    CPU reads back one integer per token.
  - the Engram lookup is one kernel per Engram layer and an Engram block is a matmul and one kernel.
Nothing of the model's own layer code runs in a step; decode.py is used to read the prompt (prefill).

    dec = FastDecoder(model)                       # one per model, reused across prompts
    ids = dec.generate(prompt_ids, max_new=512, eos=im_end)

    python runs/mac_eval/fast_decode.py --export <export dir> --out speed.json     # check against decode.py, then time

Results agree with decode.generate_ids to rounding (sums are taken in a different order than the matmul library's):
the same tokens in practice; check() reports the largest logit difference and any token that differs.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import decode  # noqa: E402
import fast_kernels as fk  # noqa: E402
from moe_llm.layers.attention import Attention  # noqa: E402
from moe_llm.layers.kda import KDA  # noqa: E402
from moe_llm.layers.moe import MoE  # noqa: E402


class FastDecoder:
    def __init__(self, model, max_len=8192, warm=2, tune=None, replay_prompt=2048):
        cfg = model.cfg
        if cfg.residual != "block_attnres" or cfg.residual_bf16:
            raise NotImplementedError("fast decoding: Block AttnRes in fp32 only")
        self.model, self.cfg, self.max_len, self.warm = model, cfg, max_len, warm
        self.dev = next(model.parameters()).device
        self.N = N = 1
        d, L = cfg.d_model, cfg.n_layers
        self.lpb = cfg.attnres_layers_per_block
        self.graph, self.debug = None, None
        # prompts up to this many tokens are read by replaying the recorded step token by token (once it exists);
        # longer ones by decode.prefill, which reads a prompt in parallel but has a fixed cost of a few tenths of a second
        self.replay_prompt = replay_prompt
        self.tune = tune or {}  # kernel block sizes: {"gate_up": {...}, "down": {...}} (fast_kernels launchers)
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=self.dev)  # noqa: E731
        self.tok = z(N, 1, dtype=torch.long)
        self.blk = z((L + self.lpb - 1) // self.lpb, N, d)  # block outputs b_0 .. b_{n-1}
        self.partial, self.p, self.xn, self.hmix, self.tmp, self.y = (z(N, d) for _ in range(6))
        self.idx0 = z(N, dtype=torch.long)
        self.pos = z(N, dtype=torch.long)  # tokens in the attention caches
        self.hist, self.hpos = z(max_len, dtype=torch.long), z(1, dtype=torch.long)  # the tokens chosen so far
        self.logits = z(N, cfg.vocab_size)
        eng = model.engram
        if eng is not None:
            if eng.orders != (2, 3) or eng.offload:
                raise NotImplementedError("fast decoding: Engram orders (2, 3), tables on the GPU")
            self.idring = z(N, 4, dtype=torch.long)  # canonical ids, position t in slot t % 4
            self.e = [z(N, eng.d_mem) for _ in eng.layers]
            self.cE = z(N, d)
        self.layers = []
        for blk in model.blocks:
            m, f, lay = blk.mixer, blk.ffn, {}
            if blk.engram is not None:
                eb = blk.engram
                lay["ekv"] = torch.cat([eb.w_k.weight, eb.w_v.weight]).contiguous()  # [2d, d_mem]
                lay["kv"] = z(N, 2 * d)
                lay["ecw"] = eb.conv_w[:, 0, :].contiguous()
                lay["ering"] = z(N, 16, d)
                if eb.conv_w.shape[-1] != 4:
                    raise NotImplementedError("fast decoding: Engram conv size 4")
            if isinstance(m, KDA):
                if m.fuse_proj:
                    raise NotImplementedError("fast decoding: fuse_proj")
                lay["wcat"] = torch.cat(m.proj_weights()).contiguous()  # [3D + 2r + H, d] = [q | k | v | f_down | g_down | b]
                lay["cw"] = torch.cat([c.weight[:, 0, :] for c in (m.q_conv, m.k_conv, m.v_conv)]).contiguous()
                lay["S"] = z(N, m.H, m.K, m.K)
                lay["win"] = z(N, 4, 3 * m.D)  # a ring: the q|k|v inputs of position t in slot t % 4
                lay["yproj"] = z(N, lay["wcat"].shape[0])
                lay["o"] = z(N, m.D)
            elif isinstance(m, Attention):
                if m.fuse_proj or m.rope is not None or m.q_norm is None or m.gate_proj is None or m.Hq == m.Hkv:
                    raise NotImplementedError("fast decoding: this attention configuration")
                lay["wcat"] = torch.cat(m.proj_weights()).contiguous()  # [q | k | v | gate]
                lay["yproj"] = z(N, lay["wcat"].shape[0])
                lay["o"] = z(N, m.Hq * m.hd)
                lay["kb"], lay["vb"] = z(N, m.Hkv, max_len, m.hd), z(N, m.Hkv, max_len, m.hd)
            else:
                raise NotImplementedError(f"fast decoding: mixer {type(m).__name__}")
            if isinstance(f, MoE):
                lay["h"] = z(N * f.k, f.w_gate.shape[2])
                lay["idx"], lay["w"] = z(N * f.k, dtype=torch.long), z(N * f.k)
                if f.shared is not None:
                    lay["sg"], lay["su"] = (lin.weight.t().contiguous()[None] for lin in (f.shared.gate, f.shared.up))
                    lay["sd"] = f.shared.down.weight.t().contiguous()[None]
                    lay["hs"] = z(N, lay["sg"].shape[2])
            else:  # the dense FFN of layer 0, in the experts' layout
                lay["sg"], lay["su"] = (lin.weight.t().contiguous()[None] for lin in (f.mlp.gate, f.mlp.up))
                lay["sd"] = f.mlp.down.weight.t().contiguous()[None]
                lay["hs"] = z(N, lay["sg"].shape[2])
            self.layers.append(lay)

    # ------------------------------------------------------------------ state
    def _reset(self):
        """The state before the first token: zero states and windows, the id ring at the "before the start" id."""
        for blk, lay in zip(self.model.blocks, self.layers):
            for key in ("S", "win", "ering"):
                if key in lay:
                    lay[key].zero_()
        if self.model.engram is not None:
            self.idring.fill_(self.model.engram.pad_id)
        self.pos.zero_()

    def _load(self, prompt_ids):
        """The state after the prompt minus its last token goes into the fixed buffers; returns the steps there is room for."""
        n = len(prompt_ids) - 1
        if n >= self.max_len:
            raise ValueError(f"the prompt has {n + 1} tokens; max_len is {self.max_len}")
        if self.graph is not None and self.debug is None and n <= self.replay_prompt:
            self._reset()
            for t in prompt_ids[:-1]:  # the step's own choice of the next token is overwritten by the prompt's
                self.tok.fill_(t)
                self.graph.replay()
            self.hpos.zero_()
            self.tok.fill_(prompt_ids[-1])
            return self.max_len - n
        cache = decode.prefill(self.model, prompt_ids[:-1])
        for blk, lay in zip(self.model.blocks, self.layers):
            m = blk.mixer
            if isinstance(m, KDA):
                lay["S"].copy_(cache.pop((id(m), "S")))
                wins = [cache.pop((id(c), "x")) for c in (m.q_conv, m.k_conv, m.v_conv)]
                lay["_wins"] = torch.cat(wins, -1)  # [N, <= 3, 3D]: the positions up to n - 1
            else:
                k, v = cache.pop((id(m), "kv"))
                lay["kb"][:, :, :n].copy_(k)
                lay["vb"][:, :, :n].copy_(v)
        self.pos.fill_(n)
        for blk, lay in zip(self.model.blocks, self.layers):
            w = lay.pop("_wins", None)
            if w is not None:
                lay["win"].zero_()  # positions before the start read as zeros, as the conv's padding does
                for j in range(w.shape[1]):
                    lay["win"][:, (n - w.shape[1] + j) % 4].copy_(w[:, j])
            if blk.engram is not None:
                w = cache.pop((id(blk.engram), "x"))  # the conv's last inputs, positions n - len .. n - 1
                lay["ering"].zero_()
                for j in range(w.shape[1]):
                    lay["ering"][:, (n - w.shape[1] + j) % 16].copy_(w[:, j])
        if self.model.engram is not None:
            ids = cache.pop((id(self.model.engram), "ids"))  # the last two token ids
            self.idring.fill_(self.model.engram.pad_id)
            for j in range(ids.shape[1]):
                self.idring[:, (n - ids.shape[1] + j) % 4].copy_(self.model.engram.canon[ids[:, j]])
        assert not cache, list(cache)
        self.hpos.zero_()
        self.tok.fill_(prompt_ids[-1])
        return self.max_len - n

    # ------------------------------------------------------------------ one token
    def _kda(self, m, lay, x, out):
        torch.mm(x, lay["wcat"].t(), out=lay["yproj"])
        fk.kda_head(lay["yproj"], lay["win"], self.pos, lay["cw"], m.f_up.weight, m.g_up.weight, m.g_up.bias, m.A_log, m.dt_bias,
                    m.o_norm_weight, lay["S"], lay["o"], m.H, m.K, m.r, m.o_eps, **self.tune.get("kda", {}))
        torch.mm(lay["o"], m.o_proj.weight.t(), out=out)

    def _attn(self, m, lay, x, out):
        torch.mm(x, lay["wcat"].t(), out=lay["yproj"])
        fk.attn_step(lay["yproj"], m.q_norm.weight, m.k_norm.weight, m.q_norm.eps, lay["kb"], lay["vb"], self.pos,
                     lay["o"], m.Hq, m.Hkv, m.hd, m.logn, **self.tune.get("attn", {}))
        torch.mm(lay["o"], m.o_proj.weight.t(), out=out)

    def _ffn(self, f, lay, x, p, out):
        """out = p + ffn(x)."""
        if isinstance(f, MoE):
            # the router's logits as the layer computes them, then its choice in one kernel
            idx, w = lay["idx"], lay["w"]
            fk.select(F.linear(x, f.router.weight), f.e_bias, idx, w, f.k, f.routed_scale)
            gu, dn = self.tune.get("gate_up", {}), self.tune.get("down", {})
            if f.shared is not None:  # routed and shared experts together: two kernels
                fk.gate_up2(x, f.w_gate, f.w_up, idx, lay["h"], lay["sg"], lay["su"], lay["hs"], f.k, **gu)
                fk.down2(lay["h"], f.w_down, idx, w, lay["hs"], lay["sd"], p, out, f.k, **dn)
            else:
                fk.gate_up(x, f.w_gate, f.w_up, idx, lay["h"], f.k, **gu)
                fk.down(lay["h"], f.w_down, idx, w, p, out, f.k, **dn)
        else:
            fk.gate_up(x, lay["sg"], lay["su"], self.idx0, lay["hs"], 1)
            fk.down(lay["hs"], lay["sd"], self.idx0, None, p, out, 1)

    def _step(self):
        """Read the buffers, run the model on the input token, write the new state, the logits and the next token."""
        model, L, lpb = self.model, self.cfg.n_layers, self.lpb
        if True:
            emb = model.embed(self.tok)[:, 0]
            eng_m = model.engram
            if eng_m is not None:
                for li in range(len(eng_m.layers)):
                    fk.engram_lookup(self.tok, self.idring, self.pos, eng_m.canon, eng_m.mult[li], eng_m.mods[li],
                                     eng_m.offs[li], eng_m.tables[li], self.e[li], eng_m.K, write=li == 0)
            nb = 0
            for i, (blk, lay) in enumerate(zip(model.blocks, self.layers)):
                boundary = i % lpb == 0
                if boundary:
                    if i == 0:
                        self.blk[0].copy_(emb)
                    nb += 1  # the previous layer wrote its output into blk[nb - 1]
                eng = blk.engram is not None
                # --- the mixer's input
                if i == 0:
                    x = blk.norm1(emb)
                elif eng:
                    fk.depth_norm(self.blk, nb, blk.res_mix.g, blk.res_mix.w, blk.res_mix.eps,
                                  e1=None if boundary else self.partial, h=self.hmix)
                    eb, cE = blk.engram, self.cE
                    torch.mm(self.e[eng_m.layers.index(i)], lay["ekv"].t(), out=lay["kv"])
                    fk.engram_block(self.hmix, lay["kv"], eb.norm_q.weight, eb.norm_k.weight, eb.norm_c.weight, lay["ecw"],
                                    lay["ering"], self.pos, blk.norm1.weight, cE, self.xn, eb.dilation, eb.signed_sqrt,
                                    eb.norm_q.eps, eb.norm_c.eps, blk.norm1.eps)
                    x = self.xn
                else:
                    fk.depth_norm(self.blk, nb, blk.res_mix.g, blk.res_mix.w, blk.res_mix.eps,
                                  e1=None if boundary else self.partial, xn=self.xn, nw=blk.norm1.weight, eps_n=blk.norm1.eps, **self.tune.get("depth", {}))
                    x = self.xn
                # --- the mixer
                if isinstance(blk.mixer, KDA):
                    self._kda(blk.mixer, lay, x, self.y)
                else:
                    self._attn(blk.mixer, lay, x, self.y)
                y = self.y
                if eng:
                    y.add_(cE)
                # --- p = partial + y (y alone at a block start), and the FFN's input
                if boundary:
                    fk.depth_norm(self.blk, nb, blk.res_ffn.g, blk.res_ffn.w, blk.res_ffn.eps, e1=y, xn=self.xn,
                                  nw=blk.norm2.weight, eps_n=blk.norm2.eps, **self.tune.get("depth", {}))
                    p = y
                else:
                    fk.depth_norm(self.blk, nb, blk.res_ffn.g, blk.res_ffn.w, blk.res_ffn.eps, e1=self.partial, e2=y,
                                  last=self.p, xn=self.xn, nw=blk.norm2.weight, eps_n=blk.norm2.eps, **self.tune.get("depth", {}))
                    p = self.p
                # --- the FFN; its output plus p is the new partial sum, or the next block's b_k
                out = self.blk[nb] if (i + 1) % lpb == 0 and i + 1 < L else self.partial
                self._ffn(blk.ffn, lay, self.xn, p, out)
                if self.debug is not None:
                    self.debug.append(out.clone())
            fk.depth_norm(self.blk, nb, model.out_res.g, model.out_res.w, model.out_res.eps, e1=self.partial,
                          xn=self.xn, nw=model.final_norm.weight, eps_n=model.final_norm.eps)
            torch.mm(self.xn, model.lm_head.weight.t(), out=self.logits)
            nxt = self.logits.argmax(-1)
            self.pos.add_(1)
            self.tok.copy_(nxt.view(-1, 1))
            self.hist.index_copy_(0, self.hpos, nxt)
            self.hpos.add_(1)

    # ------------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(self, prompt_ids, max_new, eos=0, temperature=0.0, seed=0, use_graph=True, read_every=8):
        """Token ids after the prompt: greedy, or sampled at `temperature` (the choice is then made outside the
        recorded step, from its logits). Stops before `eos`, after max_new tokens or when the context is full.
        Greedy tokens are read back from the GPU `read_every` at a time (one synchronisation per read); up to
        read_every - 1 steps may run past the end of the reply, their tokens are dropped."""
        room = self._load(prompt_ids)
        steps = min(max_new, room - 1)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        out, i = [], 0
        while i < steps:
            eager = not use_graph or (self.graph is None and i < self.warm)
            if not eager and self.graph is None:
                torch.cuda.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._step()  # recorded, not run
            k = 1 if eager or temperature > 0 else min(read_every, steps - i)
            for _ in range(k):
                self._step() if eager else self.graph.replay()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(self.logits / temperature, -1), 1, generator=g)
                self.tok.copy_(nxt)
                toks = [int(nxt)]
            else:
                toks = self.hist[i:i + k].tolist()
            for t in toks:
                if t == eos:
                    return out
                out.append(t)
            i += k
        return out


# ---------------------------------------------------------------------------------------------- checks
@torch.inference_mode()
def reference_steps(model, prompt_ids, n):
    """The ordinary decoder's logits and every layer's output for n greedy steps after the prompt."""
    dev = next(model.parameters()).device
    per_layer, logits, toks = [], [], []
    orig = type(model.blocks[0]).forward_block_attnres

    def spy(self, *a, **k):
        res = orig(self, *a, **k)
        per_layer[-1].append(res[0][:, -1].clone())
        return res
    type(model.blocks[0]).forward_block_attnres = spy
    try:
        with decode.session():
            per_layer.append([])
            h, _ = model.hidden(torch.tensor([prompt_ids], device=dev))
            for _ in range(n):
                lg = model.lm_head(h[:, -1]).float()
                logits.append(lg.clone())
                t = int(lg.argmax())
                toks.append(t)
                per_layer.append([])
                h, _ = model.hidden(torch.tensor([[t]], device=dev))
    finally:
        type(model.blocks[0]).forward_block_attnres = orig
    return toks, logits, per_layer


@torch.inference_mode()
def check_layers(model, dec, prompt_ids, steps=3):
    """Largest difference of every layer's output between the fast step (run eagerly) and the ordinary decoder."""
    toks, logits, per_layer = reference_steps(model, prompt_ids, steps)
    dec._load(prompt_ids)
    worst = []
    for s in range(steps):
        dec.debug = []
        dec._step()
        ref = per_layer[s] if s else [x for x in per_layer[0]]  # step 0 of the fast path = the prompt's last token
        diffs = [float((a - b).abs().max()) for a, b in zip(dec.debug, ref)]
        scale = [float(b.abs().max()) for b in ref]
        worst.append({"step": s, "max_layer_diff": max(diffs), "at_layer": diffs.index(max(diffs)),
                      "over_1e-4": [(i, round(x, 5)) for i, x in enumerate(diffs) if x > 1e-4][:8], "layer_scale": round(max(scale), 2),
                      "logit_diff": float((dec.logits - logits[s]).abs().max()), "same_token": int(dec.tok) == toks[s]})
        dec.tok.fill_(toks[s])  # follow the reference's tokens
    dec.debug = None
    return worst


@torch.inference_mode()
def check_graph(model, dec, prompt_ids, steps=40):
    """The recorded step replayed, fed the ordinary decoder's tokens: the largest logit difference per step."""
    toks, logits, _ = reference_steps(model, prompt_ids, steps)
    dec.generate(prompt_ids, dec.warm + 1)  # records the graph if there is none yet
    dec._load(prompt_ids)
    diffs, same = [], 0
    for s in range(steps):
        dec.graph.replay()
        diffs.append(float((dec.logits - logits[s]).abs().max()))
        same += int(dec.tok) == toks[s]
        dec.tok.fill_(toks[s])
    return {"steps": steps, "max_logit_diff": max(diffs), "median_logit_diff": sorted(diffs)[steps // 2], "same_argmax": same}


# the questions of the self-check (python fast_decode.py --export ...): no data files are needed
QUESTIONS = (
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h?",
    "Tom has 3 boxes with 12 pencils each. He gives 9 pencils away. How many pencils does he have left?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "A shop sells apples at 4 for $3. How much do 20 apples cost?",
    "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.",
    "The sum of three consecutive integers is 72. What is the largest of them?",
)


def now():
    torch.cuda.synchronize()
    return time.time()


@torch.inference_mode()
def main():
    import mac_lm
    from moe_llm import chat
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--layers-only", action="store_true", help="only the per-layer comparison")
    ap.add_argument("--steps", type=int, default=3, help="steps of the per-layer comparison")
    ap.add_argument("--tune", default="", help="JSON: kernel block sizes, e.g. {\"gate_up\": {\"bd\": 128}}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, tok, meta = mac_lm.load(a.export, "cuda")
    eos, im_end = chat.SPECIAL_IDS[chat.EOS], chat.SPECIAL_IDS[chat.IM_END]
    prompts = [[eos] + tok.encode(chat.render({"messages": [{"role": "user", "content": q}]}, generation_prompt=True)).ids
               for q in QUESTIONS[:a.prompts]]
    dec = FastDecoder(model, a.max_len, tune=json.loads(a.tune) if a.tune else None)
    res = {"export": a.export, "gpu": torch.cuda.get_device_name(0), "max_len": a.max_len, "layers": [], "check": [], "speed": {}}
    for w in check_layers(model, dec, prompts[0], a.steps):
        res["layers"].append(w)
        print("layers", w, flush=True)
    for i, p in enumerate(prompts[:3]):
        r = check_graph(model, dec, p)
        res.setdefault("graph", []).append(r)
        print("graph", i, r, flush=True)
    if a.layers_only:
        return
    for i, p in enumerate(prompts):
        ref = decode.generate_ids(model, p, a.new, eos=im_end)
        got = dec.generate(p, a.new, eos=im_end)
        same = next((k for k, (x, y) in enumerate(zip(ref, got)) if x != y), None)
        if same is None and len(ref) != len(got):
            same = min(len(ref), len(got))
        r = {"prompt_tokens": len(p), "reference_tokens": len(ref), "fast_tokens": len(got), "first_difference": same}
        res["check"].append(r)
        print("check", i, r, flush=True)
    p = prompts[0]
    runs = []
    for rep in range(4):
        t0 = now()
        out = dec.generate(p, a.new, eos=-1)
        runs.append(round(a.new / (now() - t0), 1))
        assert len(out) == a.new
    res["speed"]["with_prompt"] = runs[1:]
    for name, every in (("decode_only", 8), ("decode_only_read_every_token", 1)):
        dec._load(p)
        t0 = now()
        for i in range(0, a.new, every):
            for _ in range(every):
                dec.graph.replay()
            dec.hist[i:i + every].tolist()
        res["speed"][name] = round(a.new / (now() - t0), 1)
    print("speed", res["speed"], flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print("FAST_DECODE_DONE")


if __name__ == "__main__":
    main()
