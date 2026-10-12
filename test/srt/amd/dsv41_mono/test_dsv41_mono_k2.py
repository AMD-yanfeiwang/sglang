"""Check K2 (attention back, FFN half) after K1 against a torch reference on gfx950 (torchrun, TP2 or TP4).

The step is a synthetic decode / DSpark verify step with SGLang's metadata: paged 584-B record pools (SWA 256 rows
a page; c2 128, c1 256), each token's window slots newest first and its compressed slots (a random top-512 of the
request's compressed rows, physical slots), their counts, and a pad row without keys. Weights are random in the
checkpoint formats. Each stage is compared on the kernel's own input ("local") and along an fp64 chain from the
dequantized records ("chain"): the keys K2 looked up (exact), the per-head attention output (sink, scale
1 / sqrt(512)), XO (inverse RoPE, MXFP8), X8B (wo_a, MXFP8), the wo_b partial and its TP all-reduce. The layer
outputs are compared with the FFN launch run from the reference partial, and bit for bit with the FFN launch run
from K2's own partial. Then K1 + K2 of two widths in one graph are replayed, and K2's time a layer is measured.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from test_dsv41_mono_ffn import (
    HC,
    HIDDEN,
    cos,
    deq_fp8_block,
    finish,
    make_weights,
    mono_weights,
    rel,
    routing,
    seam_inputs,
)
from test_dsv41_mono_k1_drift import rope_table

RECORD, DATA, ALIGN = 584, 576, 576
HEAD, NOPE, Q_RANK = 512, 448, 1280
WINDOW, TOPK, KEYS = 128, 512, 640
O_RANK, N_HEADS, O_GROUPS = 1024, 64, 8
SWA_ROWS = 256  # SGLang's SWA page; a c{r} page holds 256 / r rows
MIX = 24
LOG2E = 1.4426950408889634
FLT_MIN = torch.finfo(torch.float32).tiny
# width -> (requests, tokens a request, pad rows); the first request's context is a short one
STEPS = {
    1: (1, 1, 0),
    6: (1, 6, 0),
    8: (7, 1, 1),
    12: (2, 6, 0),
    24: (4, 6, 0),
    25: (4, 6, 1),
    26: (4, 6, 2),
    48: (8, 6, 0),
}
SHORT = {1: 300, 6: 40, 8: 90, 12: 700, 24: 3, 25: 60, 26: 1000, 48: 5}


# ---------------------------------------------------------------- pools and the step
def paged_cache(pages, rows, dev):
    """[pages, rows, 584] u8 over pages of round_up(rows * 584, 576) bytes (SGLang's V4 pages)."""
    page = -(-rows * RECORD // ALIGN) * ALIGN
    raw = torch.zeros(pages * page, dtype=torch.uint8, device=dev)
    return torch.as_strided(raw, (pages, rows, RECORD), (page, RECORD, 1))


def flat(cache):
    n = cache.untyped_storage().nbytes() - cache.storage_offset()
    return torch.as_strided(cache, (n,), (1,))


def rec_index(cache, slots):
    """Byte indices into ``flat(cache)`` of the slots' 576 data bytes and 8 scale bytes."""
    rows, page = cache.shape[1], cache.stride(0)
    b, o = slots // rows, slots % rows
    d0 = b * page + o * DATA
    s0 = b * page + rows * DATA + o * 8
    ar = torch.arange(DATA, device=slots.device)
    return d0[:, None] + ar, s0[:, None] + ar[:8]


def write_records(cache, slots, kv):
    """fp8_ds_mla records of roped bf16 rows: the 448 nope dims e4m3 with a UE8M0 a 64, the rope dims bf16."""
    nope = kv.float()[:, :NOPE].view(-1, 7, 64)
    exp = torch.ceil(torch.log2(nope.abs().amax(-1).clamp_min(1e-4) / 448.0))
    q = (nope * torch.exp2(-exp)[..., None]).clamp(-448, 448).reshape(-1, NOPE)
    data = torch.cat(
        [
            q.to(torch.float8_e4m3fn).view(torch.uint8),
            kv[:, NOPE:].contiguous().view(torch.uint8),
        ],
        1,
    )
    scales = torch.zeros(kv.shape[0], 8, dtype=torch.uint8, device=kv.device)
    scales[:, :7] = (exp + 127).clamp(0, 255).to(torch.uint8)
    di, si = rec_index(cache, slots.long())
    f = flat(cache)
    f[di.reshape(-1)] = data.reshape(-1)
    f[si.reshape(-1)] = scales.reshape(-1)


def dequant(cache, slots):
    """The slots' records as fp64 [n, 512]: nope e4m3 x 2^(code - 127) a 64, rope bf16."""
    di, si = rec_index(cache, slots.long())
    f = flat(cache)
    data, sc = f[di], f[si]
    nope = data[:, :NOPE].contiguous().view(torch.float8_e4m3fn).double()
    nope = nope * torch.exp2(sc[:, :7].double() - 127).repeat_interleave(64, 1)
    rope = data[:, NOPE:].contiguous().view(torch.bfloat16).double()
    return torch.cat([nope, rope], 1)


def rope_rows(kv, positions, cos_sin):
    """GPT-J RoPE of the last 64 dims (fp32 math, bf16 out)."""
    cs = cos_sin[positions.long()]
    c, s = cs[:, :32], cs[:, 32:]
    x = kv.float()
    e, o = x[:, NOPE::2].clone(), x[:, NOPE + 1 :: 2].clone()
    x[:, NOPE::2] = e * c - o * s
    x[:, NOPE + 1 :: 2] = e * s + o * c
    return x.to(torch.bfloat16)


def normed_kv(n, kv_norm, g, dev):
    x = torch.randn(n, HEAD, generator=g).to(dev)
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)) * kv_norm.float()).to(
        torch.bfloat16
    )


def paged_pool(pages, rows, g, dev):
    """A pool for each request's logical pages ``pages[r]``: distinct random physical pages (page 0 unused) ->
    (cache, slot(r, row))."""
    phys = (torch.randperm(sum(len(p) for p in pages), generator=g) + 1).tolist()
    maps = [{lp: phys.pop() for lp in sorted(p)} for p in pages]
    cache = paged_cache(1 + sum(len(p) for p in pages), rows, dev)
    return cache, lambda r, i: maps[r][i // rows] * rows + i % rows


def decode_step(M, ratio, kv_norm, cos_sin, seed, dev):
    """R requests x T tokens (and pad rows without keys) past random contexts, with SGLang's metadata: each
    token's window slots (newest first) and its compressed pool slots (all of its compressed rows up to 512,
    else a random 512), their counts. The window history and the selected compressed rows are written."""
    R, T, pads = STEPS[M]
    g = torch.Generator().manual_seed(seed)
    ctx = [int(c) for c in torch.randint(2000, 40000, (R,), generator=g)]
    ctx[0] = SHORT[M]
    pos = [ctx[r] + t for r in range(R) for t in range(T)] + [0] * pads
    t2r = [r for r in range(R) for _ in range(T)] + [0] * pads
    real = range(R * T)
    sels = []
    for i in real:
        cand = (pos[i] + 1) // ratio
        sels.append(
            torch.randperm(cand, generator=g)[: min(max(cand, 1), TOPK)].tolist()
        )
    first = [max(0, c - WINDOW - T) for c in ctx]
    swa, slot = paged_pool(
        [{p // SWA_ROWS for p in range(first[r], ctx[r] + T)} for r in range(R)],
        SWA_ROWS,
        g,
        dev,
    )
    used = [set() for _ in range(R)]
    for i in real:
        used[t2r[i]].update(sels[i])
    rows = SWA_ROWS // ratio
    comp, cslot = paged_pool([{j // rows for j in u} for u in used], rows, g, dev)
    sidx = torch.full((M, WINDOW), -1, dtype=torch.int32)
    slen = torch.zeros(M, dtype=torch.int32)
    cidx = torch.full((M, TOPK), -1, dtype=torch.int32)
    comp_len = torch.zeros(M, dtype=torch.int32)
    for i in real:
        n = min(pos[i] + 1, WINDOW)
        sidx[i, :n] = torch.tensor([slot(t2r[i], pos[i] - j) for j in range(n)])
        slen[i] = n
        cidx[i, : len(sels[i])] = torch.tensor([cslot(t2r[i], j) for j in sels[i]])
        comp_len[i] = len(sels[i])
    sidx[min(1, R * T - 1), 2] = -1  # an absent window entry
    if comp_len[0] > 3:
        cidx[0, 1 : int(comp_len[0]) : 3] = -1  # absent entries among a token's top-k
    hist = [(r, p) for r in range(R) for p in range(first[r], ctx[r])]
    hp = torch.tensor([p for _, p in hist], device=dev)
    kv = rope_rows(normed_kv(len(hist), kv_norm, g, dev), hp, cos_sin)
    write_records(swa, torch.tensor([slot(r, p) for r, p in hist], device=dev), kv)
    for r in range(R):
        idx = torch.tensor(sorted(used[r]), dtype=torch.int64)
        kv = rope_rows(
            normed_kv(len(idx), kv_norm, g, dev), (idx * ratio).to(dev), cos_sin
        )
        write_records(
            comp, torch.tensor([cslot(r, int(j)) for j in idx], device=dev), kv
        )
    slots = [slot(t2r[i], pos[i]) for i in real] + [-1] * pads
    return dict(
        M=M,
        positions=torch.tensor(pos, dtype=torch.int64, device=dev),
        slot_mapping=torch.tensor(
            slots, dtype=torch.int64, device=dev
        ),  # K1's KV insert (pad: none)
        swa_cache=swa,
        swa_indices=sidx.to(dev),
        swa_lens=slen.to(dev),
        comp_cache=comp,
        comp_indices=cidx.to(dev),
        comp_lens=comp_len.to(dev),
    )


def k2_args(step):
    return (step["positions"], step["swa_cache"], step["swa_indices"], step["swa_lens"],
            step["comp_cache"], step["comp_indices"], step["comp_lens"])  # fmt: skip


# ---------------------------------------------------------------- weights
def seam_weights(dev, seed):
    """The attention seam's mHC and norm weights (alike on every rank)."""
    g = torch.Generator(device=dev).manual_seed(seed)

    def rnd(*shape, s=1.0):
        return torch.randn(*shape, generator=g, device=dev) * s

    return dict(fn=rnd(MIX, HC * HIDDEN, s=0.01), scale=rnd(3, s=0.5).abs() + 0.5,
                base=rnd(MIX, s=0.1), norm=(1 + rnd(HIDDEN, s=0.1)).to(torch.bfloat16))  # fmt: skip


def attn_weights(tp, rank, dev, ratio, cos_sin, seed):
    """One layer's attention weights on this rank in vLLM's loaded layout: e4m3 [N, K], E8M0 32 x 32 blocks."""
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import AttnWeights

    gr = torch.Generator(device=dev).manual_seed(
        4001 + 7919 * seed
    )  # replicated: wqkv, the norms
    gs = torch.Generator(device=dev).manual_seed(
        5001 + 7919 * seed + rank
    )  # this rank's shards
    H, G = N_HEADS // tp, O_GROUPS // tp

    def fp8(n, k, lo, hi, g):
        w = (
            torch.randn(n, k, generator=g, device=dev)
            .clamp(-6, 6)
            .to(torch.float8_e4m3fn)
        )
        s = torch.randint(
            lo, hi, (n // 32, k // 32), generator=g, device=dev, dtype=torch.uint8
        )
        return w, s

    wqkv, wqkv_s = fp8(1792, HIDDEN, 116, 126, gr)
    # wq_b's codes: q rows whose scores span flat to peaked softmaxes
    wq_b, wq_b_s = fp8(H * HEAD, Q_RANK, 119, 124, gs)
    wo_a, wo_a_s = fp8(G * O_RANK, H // G * HEAD, 118, 123, gs)
    wo_b, wo_b_s = fp8(HIDDEN, G * O_RANK, 118, 123, gs)
    q_norm = (1 + torch.randn(Q_RANK, generator=gr, device=dev) * 0.1).to(
        torch.bfloat16
    )
    kv_norm = (torch.randn(HEAD, generator=gr, device=dev).abs() * 2 + 0.05).to(
        torch.bfloat16
    )
    sink = torch.randn(H, generator=gs, device=dev) * 2
    return AttnWeights(
        layer_id=21, wqkv=wqkv, wqkv_scale=wqkv_s, q_norm=q_norm, kv_norm=kv_norm, wq_b=wq_b,
        wq_b_scale=wq_b_s, wo_a=wo_a, wo_a_scale=wo_a_s, wo_b=wo_b, wo_b_scale=wo_b_s, attn_sink=sink,
        cos_sin=cos_sin, ratio=ratio,
    )  # fmt: skip


def run_k1(k1, r, a, sw, inputs, step):
    """#43497's K1 for the step: the attention seam, wqkv_a, the norms, the KV insert, wq_b -> r.q etc."""
    res, x, post, comb, pre = inputs
    swa = step["swa_cache"]
    k1(
        res.data_ptr(), x.data_ptr(), post.data_ptr(), comb.data_ptr(), pre.data_ptr(),
        sw["fn"].data_ptr(), sw["scale"].data_ptr(), sw["base"].data_ptr(), sw["norm"].data_ptr(),
        r.res_mid.data_ptr(), r.post_a.data_ptr(), r.comb_a.data_ptr(), r.pre_a.data_ptr(),
        a.wqkv.data_ptr(), a.wqkv_scale.data_ptr(), a.q_norm.data_ptr(), a.kv_norm.data_ptr(),
        a.wq_b.data_ptr(), a.wq_b_scale.data_ptr(), a.cos_sin.data_ptr(), step["positions"].data_ptr(),
        step["slot_mapping"].data_ptr(), swa.data_ptr(), swa.stride(0), swa.shape[1], r.q.data_ptr(),
        r.scratch(step["M"]).data_ptr(), r.epoch.data_ptr(), stream=torch.cuda.current_stream(),
    )  # fmt: skip


# ---------------------------------------------------------------- the reference
def key_table(step):
    """The keys of the step as vLLM's key table (window keys, then compressed): KT [M, 640] int32 bits (slot,
    slot | 1 << 31 compressed, -1 none), key counts, compressed counts."""
    slen, comp_len = step["swa_lens"].long(), step["comp_lens"].long()
    sw, cw = step["swa_indices"].long(), step["comp_indices"].long()
    dev = sw.device
    win = torch.where(
        (torch.arange(WINDOW, device=dev)[None] < slen[:, None]) & (sw >= 0), sw, -1
    )
    cmp = torch.where(
        (torch.arange(TOPK, device=dev)[None] < comp_len[:, None]) & (cw >= 0),
        cw | (1 << 31),
        -1,
    )
    kt = torch.cat([win, cmp], 1)
    kt = torch.where(kt >= 2**31, kt - 2**32, kt)
    return kt.int(), (slen + comp_len).int(), comp_len.int()


def attention_ref(q, kt, step, sink):
    """fp64 attention of every head over the token's keys: K = V = the dequantized records, softmax of
    q.k / sqrt(512) with the sink in the denominator -> [M, H, 512]."""
    M = q.shape[0]
    kt64 = kt.long()
    valid = kt64 != -1
    is_c = valid & (kt64 < 0)
    slot = kt64 & 0x7FFFFFFF
    K = torch.zeros(M, KEYS, HEAD, dtype=torch.float64, device=q.device)
    for cache, sel in ((step["swa_cache"], valid & ~is_c), (step["comp_cache"], is_c)):
        if bool(sel.any()):
            K[sel] = dequant(cache, slot[sel])
    s = torch.einsum("thd,tkd->thk", q.double(), K) * HEAD**-0.5
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    sk = sink.double()[None]
    m = torch.maximum(s.amax(-1), sk)
    e = torch.exp(s - m[..., None])
    den = e.sum(-1) + torch.exp(sk - m)
    return torch.einsum("thk,tkd->thd", e, K) / den[..., None]


def split_combine(part, pstat, step, ck, sink):
    """The kernel's split partials (bf16 [M, chunks, H, 512], base-2 (max, sum)) merged in fp64 with the sink,
    over the chunks holding listed keys (window chunks below 128, compressed from 128)."""
    nchunk = part.shape[1]
    k0 = torch.arange(nchunk, device=part.device)[None] * ck
    slen, comp_len = step["swa_lens"].long()[:, None], step["comp_lens"].long()[:, None]
    live = torch.where(k0 < WINDOW, k0 < slen, k0 - WINDOW < comp_len)
    lv = live[..., None]
    mc = torch.where(lv, pstat[..., 0].double(), float("-inf"))
    lc = torch.where(lv, pstat[..., 1].double(), 0.0)
    s2 = sink.double()[None] * LOG2E
    m = torch.maximum(mc.amax(1), s2)
    w = torch.exp2(mc - m[:, None])
    den = torch.exp2(s2 - m) + (lc * w).sum(1)
    pv = torch.where(lv[..., None], part.double(), 0.0)
    return (pv * w[..., None]).sum(1) / den[..., None]


def mx8(x):
    """vLLM's MXFP8 of each 32: (e4m3 values as fp32, codes) with code = clamp(ceil_exp(RN(amax / 448)), 0, 254)."""
    g = x.float().reshape(*x.shape[:-1], -1, 32)
    q = (g.abs().amax(-1, keepdim=True).clamp_min(FLT_MIN) / 448.0).contiguous()
    bits = q.view(torch.int32)
    code = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()).clamp(0, 254)
    v = (g * torch.exp2(127.0 - code.float())).clamp(-448, 448).to(torch.float8_e4m3fn)
    return v.float().reshape(x.shape), code.squeeze(-1)


def mx8_deq(v, code):
    g = v.float().reshape(*v.shape[:-1], -1, 32)
    return (g * torch.exp2(code.float()[..., None] - 127)).reshape(v.shape)


def inv_rope(o, pos, cos_sin):
    """Inverse GPT-J RoPE of dims 448.. (fp32): e' = e cos + o sin, o' = o cos - e sin."""
    y = o.float().clone()
    cs = cos_sin[pos.long()][:, None]
    c, s = cs[..., :32], cs[..., 32:]
    e, od = y[..., NOPE::2].clone(), y[..., NOPE + 1 :: 2].clone()
    y[..., NOPE::2] = e * c + od * s
    y[..., NOPE + 1 :: 2] = od * c - e * s
    return y


def wo_a_ref(xo, a, G):
    """[M, H * 512] -> [M, G * 1024]: each group's eight heads times its wo_a rows (fp64)."""
    w = deq_fp8_block(a.wo_a, a.wo_a_scale).double()
    k = xo.shape[1] // G
    return torch.cat(
        [
            xo[:, g * k : (g + 1) * k].double() @ w[g * O_RANK : (g + 1) * O_RANK].T
            for g in range(G)
        ],
        1,
    )


def wo_b_ref(x8b, a):
    return x8b.double() @ deq_fp8_block(a.wo_b, a.wo_b_scale).double().T


def rank_sum(parts):
    """The attention all-reduce: the ranks' bf16 partials summed in fp32 in rank order -> bf16."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


def codes_equal(a_v, a_c, b_v, b_c):
    """Share of fp8 values (and of group codes) that agree."""
    return (a_v == b_v).float().mean().item(), (a_c == b_c).float().mean().item()


def chain(o, pos, cos_sin, a, G):
    """The reference chain from an attention output [M, H, 512]: bf16, inverse RoPE, MXFP8 (XO), wo_a, bf16,
    MXFP8 (X8B), wo_b -> bf16 (this rank's partial). Returns the dequantized XO and X8B and the partial."""
    M, H = o.shape[:2]
    xo = mx8_deq(*mx8(inv_rope(o.bfloat16(), pos, cos_sin).reshape(M, H * HEAD)))
    x8b = mx8_deq(*mx8(wo_a_ref(xo, a, G).bfloat16()))
    return xo, x8b, wo_b_ref(x8b, a).bfloat16()


def perturbed(o_ref, o_k, seed):
    """The control's attention output: o_ref with random relative noise of K2's attention error's size."""
    g = torch.Generator(device=o_ref.device).manual_seed(seed)
    d = o_ref * torch.randn(
        o_ref.shape, generator=g, device=o_ref.device, dtype=o_ref.dtype
    )
    return o_ref + d * ((o_k - o_ref).norm() / d.norm())


# ---------------------------------------------------------------- one step
def check_step(r, mw, wraw, a, sw, M, tp, rank, dev, cos_sin, seed):
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.attention.back import (
        chunk_keys,
    )
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.attention.plan import Dims
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.layer import (
        MonoBuild,
        build_mono_k1,
        peer_half_bytes,
        scratch_layout,
    )

    d = Dims(tp)
    H, G = d.heads, d.groups
    step = decode_step(M, a.ratio, a.kv_norm, cos_sin, seed, dev)
    _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
    g = torch.Generator(device=dev).manual_seed(77 + M)
    x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    k1 = build_mono_k1(MonoBuild(M, tp))
    run_k1(k1, r, a, sw, (res, x, post, comb, pre), step)
    outs = r.k2(mw, a, M, *k2_args(step))
    torch.cuda.synchronize()
    ep = int(r.epoch[0].item()) - 1  # the epoch K2 ran with (its CTA 0 moved it on)
    lay, sc = scratch_layout(M, tp), r.scratch(M)

    def region(name, dtype, *shape):
        o, n = lay[name]
        return sc[o : o + n].view(dtype).view(*shape).clone()

    ck = chunk_keys(d, M)
    nchunk = -(-KEYS // ck)
    part_k = region("part", torch.bfloat16, M, nchunk, H, HEAD)
    pstat = region("pstat", torch.float32, M, nchunk, H, 2)
    xo_v = region("xo", torch.float8_e4m3fn, M, H * HEAD).float()
    xo_c = region("xos", torch.int32, M, H * HEAD // 32)
    x8b_v = region("x8b", torch.float8_e4m3fn, M, G * O_RANK).float()
    x8b_c = region("x8bs", torch.int32, M, G * O_RANK // 32)
    # every rank's wo_b partial as K2 pushed it, from this rank's ATTN peer region (epoch parity ep)
    par = (ep & 1) * peer_half_bytes(tp)
    pairs = (
        r.peer.bytes[par : par + tp * M * HIDDEN // 2 * 8]
        .view(torch.int32)
        .view(tp, M, HIDDEN // 2, 2)
    )
    tags_ok = bool((pairs[..., 1] == 2 * ep + 2).all())
    parts_k = pairs[..., 0].contiguous().view(torch.bfloat16).view(tp, M, HIDDEN)
    part_own = parts_k[rank].contiguous()
    q = r.q[:M].clone()
    kt_k = r.kt[: M * KEYS].view(M, KEYS).clone()
    klen_k, clen_k = r.klen[:M].clone(), r.klen[M : 2 * M].clone()
    k2_outs = [t.clone() for t in outs]
    mids = (
        r.res_mid[:M].clone(),
        r.post_a[:M].clone(),
        r.comb_a[:M].clone(),
        r.pre_a[:M].clone(),
    )

    # the reference chain from the fp64 attention, and the control: the same chain from the fp64 attention
    # perturbed by noise of K2's attention error's size (what any attention that close would show downstream)
    kt_r, klen_r, clen_r = key_table(step)
    stats = {}
    stats["keys exact"] = bool(
        torch.equal(kt_k, kt_r)
        and torch.equal(klen_k, klen_r)
        and torch.equal(clen_k, clen_r)
    )
    sink = a.attn_sink[:H]
    o_ref = attention_ref(q, kt_r, step, sink)
    o_k = split_combine(part_k, pstat, step, ck, sink)
    pos = step["positions"]
    xo_r, x8b_r, part_ref = chain(o_ref, pos, cos_sin, a, G)
    xo_t, x8b_t, part_ctl = chain(perturbed(o_ref, o_k, seed), pos, cos_sin, a, G)
    xl_v, xl_c = mx8(inv_rope(o_k.bfloat16(), pos, cos_sin).reshape(M, H * HEAD))
    xo_k = mx8_deq(xo_v, xo_c)
    zl_v, zl_c = mx8(wo_a_ref(xo_k, a, G).bfloat16())
    x8b_k = mx8_deq(x8b_v, x8b_c)
    part_loc = wo_b_ref(x8b_k, a).bfloat16()

    def gathered(p):
        ps = [torch.empty_like(p) for _ in range(tp)]
        dist.all_gather(ps, p.contiguous())
        return ps

    red_ref, red_ctl, red_k = (
        rank_sum(gathered(part_ref)),
        rank_sum(gathered(part_ctl)),
        rank_sum(list(parts_k)),
    )

    def cr(u, v):
        return cos(u, v), rel(u, v)

    stats["attn"] = cr(o_k, o_ref)
    for name, got, ref_, ctl, loc in (
        ("xo", xo_k, xo_r, xo_t, mx8_deq(xl_v, xl_c)),
        ("x8b", x8b_k, x8b_r, x8b_t, mx8_deq(zl_v, zl_c)),
        ("partial", part_own, part_ref, part_ctl, part_loc),
        ("reduced", red_k, red_ref, red_ctl, None),
    ):
        stats[name] = dict(
            chain=cr(got, ref_),
            control=cr(ctl, ref_),
            local=cr(got, loc) if loc is not None else None,
        )
    stats["xo codes local"] = codes_equal(xo_v, xo_c, xl_v, xl_c)
    stats["x8b codes local"] = codes_equal(x8b_v, x8b_c, zl_v, zl_c)
    stats["peer tags"] = tags_ok

    # the layer's outputs: the FFN launch from K2's own partial (bit for bit), from the reference partial and
    # from the control's
    res_mid, post_a, comb_a, pre_a = mids
    seam = (res_mid, post_a.view(M, HC, 1), comb_a, pre_a)
    off = lay["normed"][0]

    def ffn_from(p):
        o = [t.clone() for t in r.ffn(mw, p.contiguous(), *seam)]
        torch.cuda.synchronize()
        normed = sc[off : off + M * HIDDEN * 2].view(torch.bfloat16).view(M, HIDDEN)
        return o, routing(normed, wraw)[0].sort(1).values

    own, ids_k = ffn_from(part_own)
    stats["FFN(own partial) bit-exact"] = all(
        torch.equal(u, v) for u, v in zip(own, k2_outs)
    )
    ref, ids_r = ffn_from(part_ref)
    ctl, ids_c = ffn_from(part_ctl)

    def out_stats(o, ids):
        tok = F.cosine_similarity(o[0].float(), ref[0].float(), dim=1)
        # a token routed to other experts has another MoE output altogether: compare the ones routed alike
        same = (ids == ids_r).all(1)
        return dict(
            out=cr(o[0], ref[0]),
            out_same=cr(o[0][same], ref[0][same]) if bool(same.any()) else (0.0, 1.0),
            tok=tok.min().item(),
            tok_same=tok[same].min().item() if bool(same.any()) else float("nan"),
            routed_same=same.float().mean().item(),
            residual=cr(o[1], ref[1]),
            mixes=max(
                (u.float() - v.float()).abs().max().item()
                for u, v in zip(o[2:], ref[2:])
            ),
        )

    stats["layer"] = out_stats(k2_outs, ids_k)
    stats["layer control"] = out_stats(ctl, ids_c)
    stats["|attn| / |residual|"] = (
        red_k.float().norm() / k2_outs[1].float().norm()
    ).item()
    return stats


def within(got, ctl):
    """A chain deviation no worse than the control's (1.5 x, plus half a point)."""
    return got[0] >= 1 - 1.5 * (1 - ctl[0]) - 1e-4 and got[1] <= 1.5 * ctl[1] + 0.005


def verdict(st):
    """The step's failures: the exact parts exact, the attention within 1% of fp64, every stage on its own
    input within its rounding, and every chain deviation within the control's."""
    bad = [
        k
        for k in ("keys exact", "peer tags", "FFN(own partial) bit-exact")
        if not st[k]
    ]
    if not (st["attn"][0] >= 0.9999 and st["attn"][1] <= 0.01):
        bad.append("attn")
    for name in ("xo codes local", "x8b codes local"):
        if st[name][0] < 0.99 or st[name][1] < 0.99:
            bad.append(name)
    if not (
        st["partial"]["local"][0] >= 0.99999 and st["partial"]["local"][1] <= 0.005
    ):
        bad.append("partial local")
    for name in ("xo", "x8b", "partial", "reduced"):
        if not within(st[name]["chain"], st[name]["control"]):
            bad.append(name)
    lk, lc = st["layer"], st["layer control"]
    for name in ("out_same", "residual"):
        if not within(lk[name], lc[name]):
            bad.append(name)
    if not (lk["tok_same"] >= 0.99 and lk["routed_same"] >= 0.5):
        bad.append("token cos")
    return bad


def fmt(st):
    def cr(v):
        return f"cos {v[0]:.6f} rel {100 * v[1]:.3f}%"

    lines = [f"    attn     {cr(st['attn'])} (vs fp64)"]
    for k in ("xo", "x8b", "partial", "reduced"):
        v = st[k]
        s = f"    {k:8s} chain {cr(v['chain'])} | control {cr(v['control'])}"
        if v["local"] is not None:
            s += f" | local {cr(v['local'])}"
        lines.append(s)
    lk, lc = st["layer"], st["layer control"]
    for k, name in (
        ("out", "out"),
        ("out_same", "out (routed alike)"),
        ("residual", "residual"),
    ):
        lines.append(f"    {name:8s} chain {cr(lk[k])} | control {cr(lc[k])}")
    lines += [
        f"    token cos min {lk['tok']:.5f} (same experts {lk['tok_same']:.5f}, {100 * lk['routed_same']:.0f}% "
        f"routed alike; control {lc['tok']:.5f}) | mixes max|d| {lk['mixes']:.2e} (control {lc['mixes']:.2e}) "
        f"| |attn| / |residual| {st['|attn| / |residual|']:.3f}",
        f"    codes equal (value, group code): XO {st['xo codes local'][0]:.4f} {st['xo codes local'][1]:.4f}, "
        f"X8B {st['x8b codes local'][0]:.4f} {st['x8b codes local'][1]:.4f}",
        f"    keys exact {st['keys exact']}, peer tags {st['peer tags']}, FFN launch from K2's partial "
        f"bit-exact {st['FFN(own partial) bit-exact']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- replay and timing
def replay(r, mw, sw, tp, rank, dev, cos_sin, widths, replays):
    """K1 + K2 of every width in one graph (the ratios alternating), replayed: every replay's outputs bit for bit
    the eager run's."""
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.layer import (
        MonoBuild,
        build_mono_k1,
    )

    steps = []
    for i, M in enumerate(widths):
        a = attn_weights(tp, rank, dev, 2 - i % 2, cos_sin, seed=10 + i)
        step = decode_step(M, a.ratio, a.kv_norm, cos_sin, 100 + M, dev)
        _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
        g = torch.Generator(device=dev).manual_seed(99 + M)
        x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.5).to(torch.bfloat16)
        k1 = build_mono_k1(MonoBuild(M, tp))
        steps.append((M, a, step, (res, x, post, comb, pre), k1, r._outs(M, res)))

    def run_all():
        for M, a, step, inputs, k1, outs in steps:
            run_k1(k1, r, a, sw, inputs, step)
            r.k2(mw, a, M, *k2_args(step), outs=outs)

    run_all()
    torch.cuda.synchronize()
    ref = [t.clone() for s in steps for t in s[5]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    bad = 0
    for i in range(replays):
        graph.replay()
        if i % 50 == 49 or i == replays - 1:
            torch.cuda.synchronize()
            got = [t for s in steps for t in s[5]]
            bad += sum(not torch.equal(u, v) for u, v in zip(got, ref))
    del graph
    return bad


def timing(r, tp, rank, dev, cos_sin, widths, layers, sets, iters):
    """K2's (and the FFN launch's, and K1 + K2's) time a layer: ``layers`` launches back to back in one graph,
    each layer with its own attention weights and its own copy of the KV pools, ``sets`` MoE weight sets
    cycled."""
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.layer import (
        MonoBuild,
        build_mono_k1,
    )

    sw = seam_weights(dev, 3)
    moe = []
    for s_ in range(sets):
        w, _, _ = make_weights(tp, rank, dev, seed=50 + s_)
        moe.append(mono_weights(w))
        del w
        torch.cuda.empty_cache()
    out = {}
    for M in widths:
        res_ = {}
        for ratio in (2, 1):
            attn = [
                attn_weights(tp, rank, dev, ratio, cos_sin, seed=200 + i)
                for i in range(layers)
            ]
            step = decode_step(M, ratio, attn[0].kv_norm, cos_sin, 300 + M, dev)
            _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
            g = torch.Generator(device=dev).manual_seed(98 + M)
            x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.5).to(
                torch.bfloat16
            )
            k1 = build_mono_k1(MonoBuild(M, tp))
            inputs = (res, x, post, comb, pre)
            run_k1(k1, r, attn[0], sw, inputs, step)
            copies = []
            for _ in range(layers):
                st = dict(step)
                for name in ("swa_cache", "comp_cache"):
                    c = paged_cache(st[name].shape[0], st[name].shape[1], dev)
                    flat(c).copy_(flat(st[name]))
                    st[name] = c
                copies.append(st)
            outs = [r._outs(M, res) for _ in range(layers)]
            part = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.1).to(
                torch.bfloat16
            )
            seam = (
                r.res_mid[:M],
                r.post_a[:M].view(M, HC, 1),
                r.comb_a[:M],
                r.pre_a[:M],
            )

            def k2s():
                for i in range(layers):
                    r.k2(moe[i % sets], attn[i], M, *k2_args(copies[i]), outs=outs[i])

            def ffns():
                for i in range(layers):
                    r.ffn(moe[i % sets], part, *seam, outs=outs[i])

            def layer_pairs():
                for i in range(layers):
                    run_k1(k1, r, attn[i], sw, inputs, copies[i])
                    r.k2(moe[i % sets], attn[i], M, *k2_args(copies[i]), outs=outs[i])

            for name, fn in (
                (f"K2 (c{ratio} pool)", k2s),
                ("FFN launch", ffns),
                ("K1 + K2", layer_pairs),
            ):
                if name != f"K2 (c{ratio} pool)" and ratio == 1:
                    continue
                fn()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                for _ in range(5):
                    gr.replay()
                torch.cuda.synchronize()
                dist.barrier()
                t0, t1 = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                t0.record()
                for _ in range(iters):
                    gr.replay()
                t1.record()
                torch.cuda.synchronize()
                res_[name] = t0.elapsed_time(t1) * 1e3 / (iters * layers)
                del gr
        out[M] = res_
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="1,6,8,48")
    ap.add_argument("--ratios", default="1,2")
    ap.add_argument("--replays", type=int, default=300)
    ap.add_argument("--replay-widths", default="6,48")
    ap.add_argument("--time-widths", default="6,12,24,48")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument(
        "--sets", type=int, default=8, help="MoE weight sets in the timing graph"
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--skip", default="", help="comma list of: numerics, replay, timing"
    )
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.attention.plan import Dims
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import DSV41MonoLayer

    skip = set(filter(None, args.skip.split(",")))
    wraw, _, _ = make_weights(tp, rank, dev, seed=0)
    mw = mono_weights(wraw)
    mw.check(tp)
    r = DSV41MonoLayer(tp, rank, cpu, dev)
    cos_sin = rope_table(dev)
    sw = seam_weights(dev, 1)
    fails = []

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    if "numerics" not in skip:
        for ratio in [int(v) for v in args.ratios.split(",")]:
            a = attn_weights(tp, rank, dev, ratio, cos_sin, seed=ratio)
            a.check(Dims(tp))
            for M in [int(v) for v in args.widths.split(",")]:
                t0 = time.time()
                st = check_step(
                    r, mw, wraw, a, sw, M, tp, rank, dev, cos_sin, seed=7 + M + ratio
                )
                bad = verdict(st)
                log(f"TP{tp} c{ratio} pool M={M:2d} {'PASS' if not bad else 'FAIL ' + ','.join(bad)} "
                    f"({time.time() - t0:.0f} s)\n{fmt(st)}")  # fmt: skip
                fails += [f"ratio {ratio} M={M} {b}" for b in bad]
    if "replay" not in skip:
        widths = [int(v) for v in args.replay_widths.split(",")]
        bad = replay(r, mw, sw, tp, rank, dev, cos_sin, widths, args.replays)
        log(f"replay: K1 + K2 of widths {widths} in one graph, {args.replays} replays, outputs sampled every 50: "
            f"{bad} mismatching checks")  # fmt: skip
        if bad:
            fails.append("replay")
    if "timing" not in skip:
        del mw
        torch.cuda.empty_cache()
        widths = [int(v) for v in args.time_widths.split(",")]
        t = timing(
            r, tp, rank, dev, cos_sin, widths, args.layers, args.sets, args.iters
        )
        for M, v in t.items():
            log(f"timing TP{tp} M={M:2d}: " + ", ".join(f"{k} {us:.1f} us" for k, us in v.items())
                + f" a layer ({args.layers} layers a graph, {args.sets} MoE weight sets)")  # fmt: skip
    finish(fails, dev)


if __name__ == "__main__":
    main()
