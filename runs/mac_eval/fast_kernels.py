"""Triton kernels for decoding one token at a time (runs/mac_eval/fast_decode.py): each replaces the dozens of small
GPU kernels a component launches per token with one or two, in fp32, without reading anything back to the CPU and
with shapes that do not depend on the data, so that a whole step can be recorded as a CUDA graph and replayed.

  select          the MoE router's choice from its logits (the logits stay torch's own matmul, as in the layer):
                  sigmoid, the top K by score + bias, the normalised weights
  gate_up         h[r] = silu(x[t] @ W_gate[e]) * (x[t] @ W_up[e]) for every (token, expert) assignment r, the expert's
                  weights read in place through the index (no gathered copy)
  down            out[t] = add[t] + sum_k w[t,k] * h[t*K+k] @ W_down[e]
  gate_up2, down2 the same for a layer's routed experts and its shared expert in one launch each
  kda_head        one KDA step for one head: the q/k/v short convs (and their windows), the L2 norms, the decay and
                  output gates' up-projections, the delta-rule state update, the read-out and the gated RMS norm
  depth_norm      Block AttnRes depth attention over the block outputs (and a last source), then optionally RMSNorm
  engram_lookup   the hashed n-gram rows of one Engram layer for the token (canonical ids kept in a small ring)
  engram_block    an Engram layer's gate, dilated conv and the norm that follows
  attn_step       one token of a full-attention layer: QK-norm, the softmax over the stored keys read in place
                  (only the positions in use), the output gate; the new key and value are appended

Every kernel follows the order of operations of the layer it replaces where that is cheap, but a sum over 1024
products is not added up in the same order as the matmul library does, so results agree with the ordinary decoder to
rounding (about 1e-5 on logits of size 10), not bit for bit.

A rule every kernel here keeps: within one launch no memory location is both read and written unless exactly one
thread holds that element. Triton may give a small tensor's element (and every scalar) to several threads, in
different warps, which do not run in step, so a thread could read what another has already overwritten. Windows
are therefore kept as rings (a step writes the slot it does not read) and outputs are written once.
tests/test_fast_kernels.py repeats each kernel many times against torch for that reason.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_kernel(x_ptr, wg_ptr, wu_ptr, idx_ptr, h_ptr, D: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
                    BN: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)   # assignment: token r // K, expert idx[r]
    pn = tl.program_id(1)  # block of BN output columns
    e = tl.load(idx_ptr + r).to(tl.int64)
    t = r // K
    offs_n = pn * BN + tl.arange(0, BN)
    g = tl.zeros([BN], dtype=tl.float32)
    u = tl.zeros([BN], dtype=tl.float32)
    base = e * D * I
    for d0 in range(0, D, BD):
        offs_d = d0 + tl.arange(0, BD)
        xv = tl.load(x_ptr + t * D + offs_d)
        p = base + offs_d[:, None] * I + offs_n[None, :]
        g += tl.sum(xv[:, None] * tl.load(wg_ptr + p), axis=0)
        u += tl.sum(xv[:, None] * tl.load(wu_ptr + p), axis=0)
    tl.store(h_ptr + r * I + offs_n, g * tl.sigmoid(g) * u)


@triton.jit
def _down_kernel(h_ptr, wd_ptr, idx_ptr, w_ptr, add_ptr, out_ptr, D: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
                 BN: tl.constexpr, BI: tl.constexpr, HAS_W: tl.constexpr, HAS_ADD: tl.constexpr):
    t = tl.program_id(0)
    pn = tl.program_id(1)
    offs_d = pn * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for k in range(K):
        r = t * K + k
        e = tl.load(idx_ptr + r).to(tl.int64)
        part = tl.zeros([BN], dtype=tl.float32)
        for i0 in range(0, I, BI):
            offs_i = i0 + tl.arange(0, BI)
            hv = tl.load(h_ptr + r * I + offs_i)
            wd = tl.load(wd_ptr + e * I * D + offs_i[:, None] * D + offs_d[None, :])
            part += tl.sum(hv[:, None] * wd, axis=0)
        if HAS_W:
            part = part * tl.load(w_ptr + r)
        acc += part
    if HAS_ADD:
        acc = tl.load(add_ptr + t * D + offs_d) + acc
    tl.store(out_ptr + t * D + offs_d, acc)


@triton.jit
def _gate_up2_kernel(x_ptr, wg_ptr, wu_ptr, idx_ptr, h_ptr, sg_ptr, su_ptr, hs_ptr, D: tl.constexpr, I: tl.constexpr,
                     IS: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    """gate_up for a token's K routed experts (rows 0 .. K-1 of the token, widths I, weights [E, D, I]) and its shared
    expert (row K, width IS, weights [1, D, IS]) in one launch: h [N*K, I] and hs [N, IS]."""
    r2 = tl.program_id(0)
    pn = tl.program_id(1)
    t = r2 // (K + 1)
    k = r2 % (K + 1)
    sh = k == K
    r = t * K + tl.minimum(k, K - 1)
    e = tl.load(idx_ptr + r).to(tl.int64)
    offs_n = pn * BN + tl.arange(0, BN)
    m_r = (offs_n < I) & (k < K)
    m_s = (offs_n < IS) & sh
    g = tl.zeros([BN], dtype=tl.float32)
    u = tl.zeros([BN], dtype=tl.float32)
    base = e * D * I
    for d0 in range(0, D, BD):
        offs_d = d0 + tl.arange(0, BD)
        xv = tl.load(x_ptr + t * D + offs_d)
        pr = base + offs_d[:, None] * I + offs_n[None, :]
        ps = offs_d[:, None] * IS + offs_n[None, :]
        wg = tl.load(wg_ptr + pr, mask=m_r[None, :], other=0.0) + tl.load(sg_ptr + ps, mask=m_s[None, :], other=0.0)
        wu = tl.load(wu_ptr + pr, mask=m_r[None, :], other=0.0) + tl.load(su_ptr + ps, mask=m_s[None, :], other=0.0)
        g += tl.sum(xv[:, None] * wg, axis=0)
        u += tl.sum(xv[:, None] * wu, axis=0)
    val = g * tl.sigmoid(g) * u
    tl.store(h_ptr + r * I + offs_n, val, mask=m_r)
    tl.store(hs_ptr + t * IS + offs_n, val, mask=m_s)


@triton.jit
def _down2_kernel(h_ptr, wd_ptr, idx_ptr, w_ptr, hs_ptr, sd_ptr, add_ptr, out_ptr, D: tl.constexpr, I: tl.constexpr,
                  IS: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, BI: tl.constexpr):
    """out[t] = add[t] + sum_k w[t,k] * h[t*K+k] @ W_down[e] + hs[t] @ W_shared_down (sd: [1, IS, D])."""
    t = tl.program_id(0)
    pn = tl.program_id(1)
    offs_d = pn * BN + tl.arange(0, BN)
    acc = tl.zeros([BN], dtype=tl.float32)
    for k in range(K):
        r = t * K + k
        e = tl.load(idx_ptr + r).to(tl.int64)
        part = tl.zeros([BN], dtype=tl.float32)
        for i0 in range(0, I, BI):
            offs_i = i0 + tl.arange(0, BI)
            hv = tl.load(h_ptr + r * I + offs_i)
            part += tl.sum(hv[:, None] * tl.load(wd_ptr + e * I * D + offs_i[:, None] * D + offs_d[None, :]), axis=0)
        acc += part * tl.load(w_ptr + r)
    part = tl.zeros([BN], dtype=tl.float32)
    for i0 in range(0, IS, BI):
        offs_i = i0 + tl.arange(0, BI)
        hv = tl.load(hs_ptr + t * IS + offs_i)
        part += tl.sum(hv[:, None] * tl.load(sd_ptr + offs_i[:, None] * D + offs_d[None, :]), axis=0)
    acc += part
    tl.store(out_ptr + t * D + offs_d, tl.load(add_ptr + t * D + offs_d) + acc)


@triton.jit
def _kda_head_kernel(y_ptr, win_ptr, pos_ptr, cw_ptr, fup_ptr, gup_ptr, gb_ptr, alog_ptr, dtb_ptr, onw_ptr, s_ptr, o_ptr,
                     YW: tl.constexpr, D: tl.constexpr, R: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
                     SCALE: tl.constexpr, EPS_O: tl.constexpr):
    """y: [N, YW] = [q | k | v | f_down | g_down | b] (D, D, D, R, R, H); win: [N, 4, 3D] a ring of the q|k|v inputs,
    position t in slot t % 4; pos: [N] int64, this token's position; cw: [3D, 4] conv weights; fup, gup: [D, R]; gb, dtb: [D]; alog: [H]; onw: [K];
    s: [N, H, K, K] the state, updated in place (each element by the one thread that holds it); o: [N, D] the
    gated, normalised read-out."""
    n = tl.program_id(0)
    h = tl.program_id(1)
    ch = h * K + tl.arange(0, K)

    # --- q, k, v: SiLU(causal depthwise conv over the last four inputs). The inputs of position t are in ring slot
    # t % 4; this step reads the three before pos and writes its own into slot pos % 4, which holds the input of
    # four steps ago: no address is both read and written (an element may be handled by more than one thread)
    pos = tl.load(pos_ptr + n)
    s0 = (n * 4 + (pos + 1) % 4) * 3 * D  # position pos - 3
    s1 = (n * 4 + (pos + 2) % 4) * 3 * D
    s2 = (n * 4 + (pos + 3) % 4) * 3 * D
    sc = (n * 4 + pos % 4) * 3 * D

    c_q = ch
    cur = tl.load(y_ptr + n * YW + c_q)
    a = tl.load(win_ptr + s0 + c_q) * tl.load(cw_ptr + c_q * 4 + 0)
    a = a + tl.load(win_ptr + s1 + c_q) * tl.load(cw_ptr + c_q * 4 + 1)
    a = a + tl.load(win_ptr + s2 + c_q) * tl.load(cw_ptr + c_q * 4 + 2)
    a = a + cur * tl.load(cw_ptr + c_q * 4 + 3)
    q = a * tl.sigmoid(a)
    tl.store(win_ptr + sc + c_q, cur)

    c_k = D + ch
    cur = tl.load(y_ptr + n * YW + c_k)
    a = tl.load(win_ptr + s0 + c_k) * tl.load(cw_ptr + c_k * 4 + 0)
    a = a + tl.load(win_ptr + s1 + c_k) * tl.load(cw_ptr + c_k * 4 + 1)
    a = a + tl.load(win_ptr + s2 + c_k) * tl.load(cw_ptr + c_k * 4 + 2)
    a = a + cur * tl.load(cw_ptr + c_k * 4 + 3)
    kk = a * tl.sigmoid(a)
    tl.store(win_ptr + sc + c_k, cur)

    c_v = 2 * D + ch
    cur = tl.load(y_ptr + n * YW + c_v)
    a = tl.load(win_ptr + s0 + c_v) * tl.load(cw_ptr + c_v * 4 + 0)
    a = a + tl.load(win_ptr + s1 + c_v) * tl.load(cw_ptr + c_v * 4 + 1)
    a = a + tl.load(win_ptr + s2 + c_v) * tl.load(cw_ptr + c_v * 4 + 2)
    a = a + cur * tl.load(cw_ptr + c_v * 4 + 3)
    v = a * tl.sigmoid(a)
    tl.store(win_ptr + sc + c_v, cur)

    # --- L2-normalised q (scaled) and k
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * SCALE
    kk = kk * tl.rsqrt(tl.sum(kk * kk, axis=0) + 1e-6)

    # --- the two rank-R up-projections of this head's channels
    offs_r = tl.arange(0, R)
    fd = tl.load(y_ptr + n * YW + 3 * D + offs_r)
    gd = tl.load(y_ptr + n * YW + 3 * D + R + offs_r)
    wp = ch[:, None] * R + offs_r[None, :]
    g_raw = tl.sum(tl.load(fup_ptr + wp) * fd[None, :], axis=1)
    gate = tl.sum(tl.load(gup_ptr + wp) * gd[None, :], axis=1) + tl.load(gb_ptr + ch)

    # --- decay per channel, beta per head
    z = g_raw + tl.load(dtb_ptr + ch)
    sp = tl.where(z > 20.0, z, tl.log(1.0 + tl.exp(tl.minimum(z, 20.0))))  # softplus, torch's threshold
    decay = tl.exp(-tl.exp(tl.load(alog_ptr + h)) * sp)
    beta = tl.sigmoid(tl.load(y_ptr + n * YW + 3 * D + 2 * R + h))

    # --- S <- (I - beta k k^T) Diag(decay) S + beta k v^T ; o = S^T q
    offs_v = tl.arange(0, K)
    sp_ptr = s_ptr + ((n * H + h) * K + offs_v[:, None]) * K + offs_v[None, :]  # [K rows (key), K cols (value)]
    S = tl.load(sp_ptr) * decay[:, None]
    res = v - tl.sum(kk[:, None] * S, axis=0)
    S = S + (beta * kk)[:, None] * res[None, :]
    tl.store(sp_ptr, S)
    o = tl.sum(q[:, None] * S, axis=0)

    # --- rmsnorm(o) * weight * sigmoid(gate)
    o = o * tl.rsqrt(tl.sum(o * o, axis=0) / K + EPS_O) * tl.load(onw_ptr + offs_v)
    tl.store(o_ptr + n * D + ch, o * tl.sigmoid(gate))


@triton.jit
def _depth_norm_kernel(blk_ptr, e1_ptr, e2_ptr, g_ptr, w_ptr, nw_ptr, last_ptr, h_ptr, xn_ptr,
                       D: tl.constexpr, NT: tl.constexpr, NB: tl.constexpr, LAST: tl.constexpr,
                       STORE_LAST: tl.constexpr, STORE_H: tl.constexpr, DO_NORM: tl.constexpr,
                       EPS: tl.constexpr, EPS_N: tl.constexpr):
    """Sources: the first NB rows of blk [MAXB, N, D] (NT = N), then (LAST = 1) e1 or (LAST = 2) e1 + e2 (stored to last_ptr
    if STORE_LAST). h = sum_i softmax_i((g * w) . v_i / rms(v_i)) v_i -> h_ptr (STORE_H); RMSNorm(h) * nw -> xn_ptr."""
    n = tl.program_id(0)
    offs = tl.arange(0, D)
    slot = tl.arange(0, 16)
    q = tl.load(g_ptr + offs) * tl.load(w_ptr + offs)
    logits = tl.full([16], float("-inf"), dtype=tl.float32)
    for i in range(NB):
        v = tl.load(blk_ptr + (i * NT + n) * D + offs)
        r = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
        logits = tl.where(slot == i, tl.sum(v * q, axis=0) * r, logits)
    if LAST > 0:
        v = tl.load(e1_ptr + n * D + offs)
        if LAST == 2:
            v = v + tl.load(e2_ptr + n * D + offs)
        if STORE_LAST:
            tl.store(last_ptr + n * D + offs, v)
        r = tl.rsqrt(tl.sum(v * v, axis=0) / D + EPS)
        logits = tl.where(slot == NB, tl.sum(v * q, axis=0) * r, logits)
    ex = tl.exp(logits - tl.max(logits, axis=0))
    a = ex / tl.sum(ex, axis=0)
    out = tl.zeros([D], dtype=tl.float32)
    for i in range(NB):
        v = tl.load(blk_ptr + (i * NT + n) * D + offs)
        out = out + tl.sum(tl.where(slot == i, a, 0.0), axis=0) * v
    if LAST > 0:
        v = tl.load(e1_ptr + n * D + offs)
        if LAST == 2:
            v = v + tl.load(e2_ptr + n * D + offs)
        out = out + tl.sum(tl.where(slot == NB, a, 0.0), axis=0) * v
    if STORE_H:
        tl.store(h_ptr + n * D + offs, out)
    if DO_NORM:
        tl.store(xn_ptr + n * D + offs, out * tl.rsqrt(tl.sum(out * out, axis=0) / D + EPS_N) * tl.load(nw_ptr + offs))


@triton.jit
def _attn_step_kernel(y_ptr, qw_ptr, kw_ptr, kb_ptr, vb_ptr, pos_ptr, o_ptr, YW: tl.constexpr, HQ: tl.constexpr,
                      HKV: tl.constexpr, HD: tl.constexpr, MAXT: tl.constexpr, BT: tl.constexpr, LOGN: tl.constexpr,
                      EPS: tl.constexpr):
    """One query token of one head against the keys stored so far and its own key. y: [N, YW] = [q | k | v | gate]
    (HQ*HD, HKV*HD, HKV*HD, HQ*HD); kb, vb: [N, HKV, MAXT, HD], the first pos[n] positions filled (keys after QK-norm);
    the new key and value are written at position pos[n] (by the first query head of each key head). o: [N, HQ*HD],
    the attention output times sigmoid(gate). The softmax is accumulated block by block (running maximum)."""
    n = tl.program_id(0)
    hq = tl.program_id(1)
    kv = hq // (HQ // HKV)
    offs = tl.arange(0, HD)
    QD = HQ * HD
    KVD = HKV * HD
    q = tl.load(y_ptr + n * YW + hq * HD + offs)
    q = q * tl.rsqrt(tl.sum(q * q, axis=0) / HD + EPS) * tl.load(qw_ptr + offs)
    kn = tl.load(y_ptr + n * YW + QD + kv * HD + offs)
    kn = kn * tl.rsqrt(tl.sum(kn * kn, axis=0) / HD + EPS) * tl.load(kw_ptr + offs)
    vn = tl.load(y_ptr + n * YW + QD + KVD + kv * HD + offs)
    pos = tl.load(pos_ptr + n)
    if LOGN > 0:  # the query is scaled by max(1, ln(keys it sees) / ln(LOGN))
        q = q * tl.maximum(tl.log((pos + 1).to(tl.float32)) / tl.log(LOGN * 1.0), 1.0)
    scale = 1.0 / tl.sqrt(HD * 1.0)
    m = tl.sum(q * kn, axis=0) * scale  # the token's own key
    s = m * 0.0 + 1.0
    acc = vn
    base = (n * HKV + kv) * MAXT
    for t0 in range(0, pos, BT):
        offs_t = t0 + tl.arange(0, BT)
        mask = offs_t < pos
        ptr = (base + offs_t[:, None]) * HD + offs[None, :]
        kt = tl.load(kb_ptr + ptr, mask=mask[:, None], other=0.0)
        lg = tl.where(mask, tl.sum(kt * q[None, :], axis=1) * scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(lg, axis=0))
        al = tl.exp(m - m_new)
        pr = tl.exp(lg - m_new)
        vt = tl.load(vb_ptr + ptr, mask=mask[:, None], other=0.0)
        acc = acc * al + tl.sum(pr[:, None] * vt, axis=0)
        s = s * al + tl.sum(pr, axis=0)
        m = m_new
    gate = tl.load(y_ptr + n * YW + QD + 2 * KVD + hq * HD + offs)
    tl.store(o_ptr + n * QD + hq * HD + offs, acc / s * tl.sigmoid(gate))
    if hq % (HQ // HKV) == 0:
        tl.store(kb_ptr + (base + pos) * HD + offs, kn)
        tl.store(vb_ptr + (base + pos) * HD + offs, vn)


@triton.jit
def _select_kernel(lg_ptr, bias_ptr, idx_ptr, w_ptr, E: tl.constexpr, K: tl.constexpr, SCALE: tl.constexpr):
    """MoE._select for one token: scores = sigmoid(router logits); the K experts with the largest score + bias, best
    first; their weights = score / (sum of the K scores + 1e-20) * SCALE. lg: [N, E]; idx (int64), w: [N * K]."""
    t = tl.program_id(0)
    offs = tl.arange(0, E)
    s = tl.sigmoid(tl.load(lg_ptr + t * E + offs))
    sel = s + tl.load(bias_ptr + offs)
    rank = offs * 0 - 1  # the place (0 = best) of every chosen expert, -1 for the others
    for k in range(K):
        hit = offs == tl.argmax(sel, axis=0)
        rank = tl.where(hit, k, rank)
        sel = tl.where(hit, float("-inf"), sel)
    tot = tl.sum(tl.where(rank >= 0, s, 0.0), axis=0)
    for k in range(K):  # every output is written once and never read here
        hit = rank == k
        tl.store(idx_ptr + t * K + k, tl.sum(tl.where(hit, offs, 0), axis=0).to(tl.int64))
        tl.store(w_ptr + t * K + k, tl.sum(tl.where(hit, s, 0.0), axis=0) / (tot + 1e-20) * SCALE)


@triton.jit
def _engram_lookup_kernel(tok_ptr, ring_ptr, pos_ptr, canon_ptr, mult_ptr, mods_ptr, offs_ptr, table_ptr, e_ptr,
                          K: tl.constexpr, DH: tl.constexpr, DHP: tl.constexpr, WRITE: tl.constexpr):
    """EngramMemory for one token and one Engram layer, orders (2, 3): the 2-gram and 3-gram hashes of the canonical
    ids (K heads each, mix % prime + offset) and the K * 2 table rows, side by side in e [N, 2K * DH].
    ring: [N, 4] the canonical ids of the last positions, position t in slot t % 4; with WRITE this token's is added."""
    n = tl.program_id(0)
    pos = tl.load(pos_ptr + n)
    x0 = tl.load(canon_ptr + tl.load(tok_ptr + n))
    x1 = tl.load(ring_ptr + n * 4 + (pos + 3) % 4)
    x2 = tl.load(ring_ptr + n * 4 + (pos + 2) % 4)
    if WRITE:
        tl.store(ring_ptr + n * 4 + pos % 4, x0)
    mix2 = (x0 * tl.load(mult_ptr + 0)) ^ (x1 * tl.load(mult_ptr + 1))
    mix3 = mix2 ^ (x2 * tl.load(mult_ptr + 2))
    hh = tl.arange(0, 2 * K)
    rows = tl.where(hh < K, mix2, mix3) % tl.load(mods_ptr + hh) + tl.load(offs_ptr + hh)
    cols = tl.arange(0, DHP)
    mask = (cols < DH)[None, :] & (hh >= 0)[:, None]
    vals = tl.load(table_ptr + rows[:, None] * DH + cols[None, :], mask=mask, other=0.0)
    tl.store(e_ptr + n * 2 * K * DH + hh[:, None] * DH + cols[None, :], vals, mask=mask)


@triton.jit
def _engram_block_kernel(h_ptr, kv_ptr, nq_ptr, nk_ptr, nc_ptr, cw_ptr, ring_ptr, pos_ptr, n1_ptr, c_ptr, xn_ptr,
                         D: tl.constexpr, DIL: tl.constexpr, SIGNED: tl.constexpr, EPS: tl.constexpr,
                         EPS_C: tl.constexpr, EPS_1: tl.constexpr):
    """EngramBlock for one token, then the layer's norm1: kv [N, 2D] = [W_k e | W_v e]; the gate sigmoid(signed sqrt of
    <norm_q(h), norm_k(k)> / sqrt(D)) times v; SiLU of the dilated causal conv (kernel 4) of its norm, plus itself -> c;
    RMSNorm(h + c) -> xn. ring [N, 16, D]: the conv's inputs, position t in slot t % 16 (this token's is added)."""
    n = tl.program_id(0)
    offs = tl.arange(0, D)
    h = tl.load(h_ptr + n * D + offs)
    k = tl.load(kv_ptr + n * 2 * D + offs)
    v = tl.load(kv_ptr + n * 2 * D + D + offs)
    hq = h * tl.rsqrt(tl.sum(h * h, axis=0) / D + EPS) * tl.load(nq_ptr + offs)
    kk = k * tl.rsqrt(tl.sum(k * k, axis=0) / D + EPS) * tl.load(nk_ptr + offs)
    s = tl.sum(hq * kk, axis=0) / tl.sqrt(D * 1.0)
    if SIGNED:
        s = tl.where(s > 0, 1.0, tl.where(s < 0, -1.0, 0.0)) * tl.sqrt(tl.maximum(tl.abs(s), 1e-6))
    vt = tl.sigmoid(s) * v
    xc = vt * tl.rsqrt(tl.sum(vt * vt, axis=0) / D + EPS_C) * tl.load(nc_ptr + offs)
    pos = tl.load(pos_ptr + n)
    a = tl.load(ring_ptr + (n * 16 + (pos + 16 - 3 * DIL) % 16) * D + offs) * tl.load(cw_ptr + offs * 4 + 0)
    a = a + tl.load(ring_ptr + (n * 16 + (pos + 16 - 2 * DIL) % 16) * D + offs) * tl.load(cw_ptr + offs * 4 + 1)
    a = a + tl.load(ring_ptr + (n * 16 + (pos + 16 - DIL) % 16) * D + offs) * tl.load(cw_ptr + offs * 4 + 2)
    a = a + xc * tl.load(cw_ptr + offs * 4 + 3)
    tl.store(ring_ptr + (n * 16 + pos % 16) * D + offs, xc)
    c = a * tl.sigmoid(a) + vt
    tl.store(c_ptr + n * D + offs, c)
    hc = h + c
    tl.store(xn_ptr + n * D + offs, hc * tl.rsqrt(tl.sum(hc * hc, axis=0) / D + EPS_1) * tl.load(n1_ptr + offs))


# ---------------------------------------------------------------------------------------------- launchers
def _fit(block: int, dim: int) -> int:
    """The largest power of two <= block that divides dim (a kernel's loop reads whole blocks)."""
    while dim % block:
        block //= 2
    return block


def gate_up(x, wg, wu, idx, h, K, bn=16, bd=512, warps=8):  # block sizes: the fastest of a sweep on a 4090
    """x [N, D]; wg, wu [E, D, I]; idx [N*K] int64; h [N*K, I] (written)."""
    E, D, I = wg.shape
    bn, bd = _fit(bn, I), _fit(bd, D)
    _gate_up_kernel[(idx.numel(), I // bn)](x, wg, wu, idx, h, D=D, I=I, K=K, BN=bn, BD=bd, num_warps=warps)


def down(h, wd, idx, w, add, out, K, bn=8, bi=256, warps=4):
    """h [N*K, I]; wd [E, I, D]; idx [N*K]; w [N*K] or None; add [N, D] or None; out [N, D] (written)."""
    E, I, D = wd.shape
    bn, bi = _fit(bn, D), _fit(bi, I)
    _down_kernel[(out.shape[0], D // bn)](h, wd, idx, w if w is not None else h, add if add is not None else out, out,
                                         D=D, I=I, K=K, BN=bn, BI=bi, HAS_W=w is not None, HAS_ADD=add is not None,
                                         num_warps=warps)


def kda_head(y, win, pos, cw, fup, gup, gb, alog, dtb, onw, S, o, H, K, R, eps_o, warps=4):
    """One KDA step, see the kernel. win is the ring [N, 4, 3D]; pos [N] int64 is not advanced here."""
    N, D = o.shape
    _kda_head_kernel[(N, H)](y, win, pos, cw, fup, gup, gb, alog, dtb, onw, S, o, YW=y.shape[1], D=D, R=R, H=H, K=K,
                             SCALE=K ** -0.5, EPS_O=eps_o, num_warps=warps)


def depth_norm(blk, nb, g, w, eps, e1=None, e2=None, last=None, h=None, xn=None, nw=None, eps_n=0.0, warps=4):
    """Depth attention over blk[:nb] ([MAXB, N, D]) and the last source (e1, or e1 + e2), see the kernel."""
    MAXB, N, D = blk.shape
    mode = 0 if e1 is None else (1 if e2 is None else 2)
    d = g  # an unused pointer argument must still be a tensor
    _depth_norm_kernel[(N,)](blk, e1 if e1 is not None else d, e2 if e2 is not None else d, g, w,
                             nw if nw is not None else d, last if last is not None else d, h if h is not None else d,
                             xn if xn is not None else d, D=D, NT=N, NB=nb, LAST=mode,
                             STORE_LAST=last is not None, STORE_H=h is not None, DO_NORM=xn is not None,
                             EPS=eps, EPS_N=eps_n, num_warps=warps)


def attn_step(y, qw, kw, eps, kb, vb, pos, o, Hq, Hkv, hd, logn, bt=128, warps=8):
    """One token through a full-attention layer's cache, see the kernel. pos [N] int64 is not advanced here."""
    N, _, maxt, _ = kb.shape
    _attn_step_kernel[(N, Hq)](y, qw, kw, kb, vb, pos, o, YW=y.shape[1], HQ=Hq, HKV=Hkv, HD=hd, MAXT=maxt, BT=bt,
                               LOGN=int(logn), EPS=eps, num_warps=warps)


def select(logits, bias, idx, w, K, scale):
    """Expert choice and weights from the router's logits [N, E] into idx [N*K] (int64) and w [N*K]."""
    N, E = logits.shape
    _select_kernel[(N,)](logits, bias, idx, w, E=E, K=K, SCALE=float(scale), num_warps=1)


def gate_up2(x, wg, wu, idx, h, sg, su, hs, K, bn=16, bd=1024, warps=8):
    """The routed experts and the shared expert of every token in one launch (see gate_up; sg, su: [1, D, IS])."""
    E, D, I = wg.shape
    IS = sg.shape[2]
    bn, bd = _fit(_fit(bn, I), IS), _fit(bd, D)
    _gate_up2_kernel[(x.shape[0] * (K + 1), max(I, IS) // bn)](x, wg, wu, idx, h, sg, su, hs, D=D, I=I, IS=IS, K=K,
                                                               BN=bn, BD=bd, num_warps=warps)


def down2(h, wd, idx, w, hs, sd, add, out, K, bn=8, bi=512, warps=4):
    """out = add + the weighted routed experts' down projections + the shared expert's (see down; sd: [1, IS, D])."""
    E, I, D = wd.shape
    IS = sd.shape[1]
    bn, bi = _fit(bn, D), _fit(_fit(bi, I), IS)
    _down2_kernel[(out.shape[0], D // bn)](h, wd, idx, w, hs, sd, add, out, D=D, I=I, IS=IS, K=K, BN=bn, BI=bi,
                                           num_warps=warps)


def engram_lookup(tok, ring, pos, canon, mult, mods, offs, table, e, K, write):
    """One Engram layer's rows for every token (orders 2 and 3), see the kernel. tok [N] int64, e [N, 2K * d_head]."""
    DH = table.shape[1]
    _engram_lookup_kernel[(tok.numel(),)](tok, ring, pos, canon, mult, mods, offs, table, e, K=K, DH=DH,
                                          DHP=triton.next_power_of_2(DH), WRITE=bool(write), num_warps=1)


def engram_block(h, kv, nq, nk, nc, cw, ring, pos, n1, c, xn, dil, signed, eps, eps_c, eps_1, warps=4):
    """EngramBlock and the layer's norm1, see the kernel."""
    N, D = h.shape
    assert 3 * dil < 16, "the conv ring holds 16 positions"
    _engram_block_kernel[(N,)](h, kv, nq, nk, nc, cw, ring, pos, n1, c, xn, D=D, DIL=dil, SIGNED=bool(signed), EPS=eps,
                               EPS_C=eps_c, EPS_1=eps_1, num_warps=warps)
