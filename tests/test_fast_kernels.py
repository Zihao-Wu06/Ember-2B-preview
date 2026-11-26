"""runs/mac_eval/fast_kernels.py against plain torch on random data (GPU only: the kernels are Triton kernels)."""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runs", "mac_eval"))
sys.path.insert(0, ROOT)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def fk():
    return pytest.importorskip("fast_kernels")


def rnd(*shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).cuda()


def close(a, b, tol=2e-5):
    return float((a - b).abs().max()) <= tol * max(1.0, float(b.abs().max()))


@pytest.mark.parametrize("N,K,E,D,I", [(1, 4, 32, 1024, 512), (1, 1, 1, 1024, 768), (3, 2, 8, 256, 2816)])
@pytest.mark.parametrize("gu,dn", [({}, {}), ({"bn": 8, "bd": 64, "warps": 4}, {"bn": 16, "bi": 64, "warps": 2})])
def test_gate_up_and_down_match_the_routed_expert_formula(fk, N, K, E, D, I, gu, dn):
    x, wg, wu, wd = rnd(N, D, seed=1), rnd(E, D, I, seed=2, scale=0.05), rnd(E, D, I, seed=3, scale=0.05), rnd(E, I, D, seed=4, scale=0.05)
    idx = torch.stack([torch.randperm(E, generator=torch.Generator().manual_seed(9 + t))[:K] for t in range(N)]).cuda().reshape(-1)
    w, add = rnd(N * K, seed=5).abs(), rnd(N, D, seed=6)
    h = torch.zeros(N * K, I, device="cuda")
    out = torch.zeros(N, D, device="cuda")
    fk.gate_up(x, wg, wu, idx, h, K, **gu)
    xs = x.repeat_interleave(K, 0)
    h_ref = F.silu(torch.bmm(xs[:, None], wg[idx])) * torch.bmm(xs[:, None], wu[idx])
    assert close(h, h_ref[:, 0])
    fk.down(h_ref[:, 0].contiguous(), wd, idx, w, add, out, K, **dn)
    y_ref = add + (torch.bmm(h_ref, wd[idx])[:, 0] * w[:, None]).view(N, K, D).sum(1)
    assert close(out, y_ref)
    fk.down(h_ref[:, 0].contiguous(), wd, idx, None, None, out, K, **dn)
    assert close(out, torch.bmm(h_ref, wd[idx])[:, 0].view(N, K, D).sum(1))


def test_select_is_the_routers_choice(fk):
    N, E, K = 5, 32, 4
    for rep in range(300):  # many times: a race between the threads of a kernel shows only now and then
        logits, bias = rnd(N, E, seed=rep), rnd(E, seed=1000 + rep, scale=0.01)
        idx, w = torch.zeros(N * K, dtype=torch.long, device="cuda"), torch.zeros(N * K, device="cuda")
        fk.select(logits, bias, idx, w, K, 1.5)
        s = torch.sigmoid(logits)
        ref = torch.topk(s + bias, K, dim=-1).indices
        wr = s.gather(1, ref)
        wr = wr / (wr.sum(-1, keepdim=True) + 1e-20) * 1.5
        assert torch.equal(idx.view(N, K), ref) and close(w.view(N, K), wr, 1e-6), rep


@pytest.mark.parametrize("nb,mode", [(1, 0), (3, 1), (8, 2), (2, 2)])
def test_depth_norm_matches_depth_attention_then_rms_norm(fk, nb, mode):
    from moe_llm.layers.attnres import _depth_attn
    N, D, MAXB = 2, 1024, 8
    blk, e1, e2 = rnd(MAXB, N, D, seed=1), rnd(N, D, seed=2), rnd(N, D, seed=3)
    g, w, nw = rnd(D, seed=4).abs() + 0.5, rnd(D, seed=5, scale=0.2), rnd(D, seed=6).abs() + 0.5
    last, h, xn = (torch.zeros(N, D, device="cuda") for _ in range(3))
    srcs = [blk[i] for i in range(nb)]
    if mode:
        srcs.append(e1 if mode == 1 else e1 + e2)
    fk.depth_norm(blk, nb, g, w, 1e-5, e1=e1 if mode else None, e2=e2 if mode == 2 else None,
                  last=last if mode == 2 else None, h=h, xn=xn, nw=nw, eps_n=1e-6)
    ref = _depth_attn(w, g, 1e-5, *srcs) if len(srcs) > 1 else srcs[0]
    assert close(h, ref)
    assert close(xn, F.rms_norm(ref, (D,), nw, 1e-6))
    if mode == 2:
        assert close(last, e1 + e2, 1e-7)


@pytest.mark.parametrize("rep", range(40))  # repeated: see test_select_is_the_routers_choice
def test_kda_head_matches_the_layers_recurrence(fk, rep):
    from moe_llm.layers import kda
    N, H, K, R = 2, 8, 128, 128
    D = H * K
    YW = 3 * D + 2 * R + H
    y, win = rnd(N, YW, seed=100 + rep), rnd(N, 3, 3 * D, seed=200 + rep)
    cw, fup, gup, gb = rnd(3 * D, 4, seed=3, scale=0.5), rnd(D, R, seed=4, scale=0.05), rnd(D, R, seed=5, scale=0.05), rnd(D, seed=6)
    alog, dtb, onw = rnd(H, seed=7).abs(), rnd(D, seed=8), rnd(K, seed=9).abs() + 0.5
    S0 = rnd(N, H, K, K, seed=10, scale=0.1)
    S, o = S0.clone(), torch.zeros(N, D, device="cuda")
    pos = torch.tensor([7, 7], device="cuda")  # the three inputs before position 7 sit in ring slots 0, 1, 2; slot 3 is written
    ring = torch.zeros(N, 4, 3 * D, device="cuda")
    for j in range(3):
        ring[:, (7 - 3 + j) % 4] = win[:, j]
    fk.kda_head(y, ring, pos, cw, fup, gup, gb, alog, dtb, onw, S, o, H, K, R, 1e-5)
    # the layer's own steps (runs/mac_eval/decode.py kda_forward, T = 1)
    full = torch.cat([win, y[:, None, :3 * D]], 1)                       # [N, 4, 3D]
    conv = F.silu((full * cw.t()[None]).sum(1))                           # causal depthwise conv over the window, SiLU
    q, k, v = (conv[:, i * D:(i + 1) * D].view(N, 1, H, K) for i in range(3))
    fd, gd, beta = y[:, 3 * D:3 * D + R], y[:, 3 * D + R:3 * D + 2 * R], y[:, 3 * D + 2 * R:]
    g = (fd @ fup.t()).view(N, 1, H, K)
    gk = kda.kda_gate(g, alog, dtb.view(H, K))
    o_ref, S_ref = kda.naive_recurrent_kda(kda.l2norm(q), kda.l2norm(k), v, gk, torch.sigmoid(beta)[:, None],
                                           initial_state=S0, output_final_state=True)
    gate = (gd @ gup.t() + gb).view(N, 1, H, K)
    of = o_ref.float()
    o_ref = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + 1e-5) * onw * torch.sigmoid(gate)
    assert close(S, S_ref) and close(o, o_ref.reshape(N, D))
    assert torch.equal(ring[:, 7 % 4], y[:, :3 * D])                               # this token's inputs joined the ring
    assert all(torch.equal(ring[:, (7 - 3 + j) % 4], win[:, j]) for j in range(3))  # the others are untouched


@pytest.mark.parametrize("pos", [0, 5, 64, 200])
def test_attn_step_matches_scaled_dot_product_attention(fk, pos):
    N, Hq, Hkv, hd, maxt, logn = 1, 8, 2, 128, 512, 16
    QD, KVD = Hq * hd, Hkv * hd
    y = rnd(N, 2 * QD + 2 * KVD, seed=1)
    qw, kw = rnd(hd, seed=2).abs() + 0.5, rnd(hd, seed=3).abs() + 0.5
    kb, vb = rnd(N, Hkv, maxt, hd, seed=4), rnd(N, Hkv, maxt, hd, seed=5)
    kb0, vb0 = kb.clone(), vb.clone()
    o = torch.zeros(N, QD, device="cuda")
    p = torch.tensor([pos], device="cuda")
    fk.attn_step(y, qw, kw, 1e-6, kb, vb, p, o, Hq, Hkv, hd, logn)
    q = F.rms_norm(y[:, :QD].view(N, Hq, hd), (hd,), qw, 1e-6)
    k = F.rms_norm(y[:, QD:QD + KVD].view(N, Hkv, hd), (hd,), kw, 1e-6)
    v = y[:, QD + KVD:QD + 2 * KVD].view(N, Hkv, hd)
    q = q * max(1.0, math.log(pos + 1) / math.log(logn))
    keys = torch.cat([kb0[:, :, :pos], k[:, :, None]], 2).repeat_interleave(Hq // Hkv, 1)
    vals = torch.cat([vb0[:, :, :pos], v[:, :, None]], 2).repeat_interleave(Hq // Hkv, 1)
    ref = F.scaled_dot_product_attention(q[:, :, None], keys, vals)[:, :, 0]
    ref = ref * torch.sigmoid(y[:, QD + 2 * KVD:].view(N, Hq, hd))
    assert close(o, ref.reshape(N, QD))
    assert close(kb[:, :, pos], k) and close(vb[:, :, pos], v)                    # the new key and value are appended
    assert torch.equal(kb[:, :, :pos], kb0[:, :, :pos]) and torch.equal(kb[:, :, pos + 1:], kb0[:, :, pos + 1:])


@pytest.mark.parametrize("N,K,E,D,I,IS", [(1, 4, 32, 1024, 512, 768), (2, 2, 8, 256, 64, 96), (1, 3, 8, 512, 256, 128)])
def test_gate_up2_and_down2_add_the_shared_expert(fk, N, K, E, D, I, IS):
    for rep in range(10):
        x, wg, wu, wd = (rnd(*sh, seed=10 * rep + i, scale=sc) for i, (sh, sc) in
                         enumerate((((N, D), 1.0), ((E, D, I), 0.05), ((E, D, I), 0.05), ((E, I, D), 0.05))))
        sg, su, sd = rnd(1, D, IS, seed=rep + 50, scale=0.05), rnd(1, D, IS, seed=rep + 60, scale=0.05), rnd(1, IS, D, seed=rep + 70, scale=0.05)
        idx = torch.stack([torch.randperm(E, generator=torch.Generator().manual_seed(rep * 7 + t))[:K] for t in range(N)]).cuda().reshape(-1)
        w, add = rnd(N * K, seed=rep + 80).abs(), rnd(N, D, seed=rep + 90)
        h, hs, out = torch.zeros(N * K, I, device="cuda"), torch.zeros(N, IS, device="cuda"), torch.zeros(N, D, device="cuda")
        fk.gate_up2(x, wg, wu, idx, h, sg, su, hs, K)
        xs = x.repeat_interleave(K, 0)
        h_ref = (F.silu(torch.bmm(xs[:, None], wg[idx])) * torch.bmm(xs[:, None], wu[idx]))[:, 0]
        hs_ref = F.silu(x @ sg[0]) * (x @ su[0])
        assert close(h, h_ref) and close(hs, hs_ref), rep
        fk.down2(h_ref.contiguous(), wd, idx, w, hs_ref.contiguous(), sd, add, out, K)
        ref = add + (torch.bmm(h_ref[:, None], wd[idx])[:, 0] * w[:, None]).view(N, K, D).sum(1) + hs_ref @ sd[0]
        assert close(out, ref), rep


def test_engram_lookup_gives_the_layers_rows(fk):
    from types import SimpleNamespace
    from moe_llm.layers.engram import EngramMemory
    cfg = SimpleNamespace(engram_layers=(1, 5), engram_orders=(2, 3), engram_heads=8, engram_d_head=40, engram_dim=640,
                          engram_offload=False, vocab_size=1000, engram_rows=5000, engram_seed=0, init_std=0.02)
    eng = EngramMemory(cfg).cuda()
    eng.canon.copy_(torch.randperm(1000, generator=torch.Generator().manual_seed(0)).cuda() % 700)  # a many-to-one map
    for rep in range(30):
        ids = torch.randint(0, 1000, (1, 9), generator=torch.Generator().manual_seed(rep)).cuda()
        ref = eng(ids)                                                     # {layer: [1, 9, 640]}
        pos = torch.tensor([8], device="cuda")
        ring = torch.zeros(1, 4, dtype=torch.long, device="cuda")
        ring[0, 7 % 4], ring[0, 6 % 4] = eng.canon[ids[0, 7]], eng.canon[ids[0, 6]]
        for li, layer in enumerate(eng.layers):
            e = torch.zeros(1, 640, device="cuda")
            fk.engram_lookup(ids[:, 8].contiguous(), ring, pos, eng.canon, eng.mult[li], eng.mods[li], eng.offs[li],
                             eng.tables[li], e, 8, write=li == 0)
            assert torch.equal(e, ref[layer][:, 8]), (rep, li)
        assert int(ring[0, 0]) == int(eng.canon[ids[0, 8]])               # position 8 joined the ring (slot 0)


def test_engram_block_matches_the_layer_and_the_norm(fk):
    from types import SimpleNamespace
    from moe_llm.layers.engram import EngramBlock
    d = 1024
    cfg = SimpleNamespace(d_model=d, engram_dim=640, init_std=0.02, norm_eps=1e-6, engram_conv_size=4, engram_orders=(2, 3),
                          engram_signed_sqrt=True)
    eb = EngramBlock(cfg, 1).cuda()
    with torch.no_grad():
        eb.conv_w.copy_(rnd(d, 1, 4, seed=1, scale=0.3))
        for i, n in enumerate((eb.norm_q, eb.norm_k, eb.norm_c)):
            n.weight.copy_(rnd(d, seed=2 + i).abs() + 0.5)
    n1 = rnd(d, seed=9).abs() + 0.5
    T = 14
    for rep in range(10):
        h, e = rnd(1, T, d, seed=20 + rep), rnd(1, T, 640, seed=40 + rep)
        with torch.no_grad():
            ref = eb(h, e)                                                 # [1, T, d], causal
            kv = torch.cat([e @ eb.w_k.weight.t(), e @ eb.w_v.weight.t()], -1)
        ring = torch.zeros(1, 16, d, device="cuda")
        c, xn = torch.zeros(1, d, device="cuda"), torch.zeros(1, d, device="cuda")
        for t in range(T):                                                 # token by token through the ring
            fk.engram_block(h[:, t].contiguous(), kv[:, t].contiguous(), eb.norm_q.weight, eb.norm_k.weight, eb.norm_c.weight,
                            eb.conv_w[:, 0, :].contiguous(), ring, torch.tensor([t], device="cuda"), n1, c, xn,
                            eb.dilation, True, 1e-6, 1e-5, 1e-6)
            assert close(c, ref[:, t]), (rep, t)
            assert close(xn, F.rms_norm(h[:, t] + ref[:, t], (d,), n1, 1e-6)), (rep, t)
