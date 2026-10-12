# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K2 ``back``: one DeepSeek-V4.1 attention layer from the rotated query to the
rank's wo_b partial.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S rows:

    split   (S x chunks of CK keys): flash-decoding over one chunk of a token's
            keys (its window list, then its compressed list: SGLang's
            physical slots, looked up in place, ``key_rec``) for
            every head of the rank. The token's q in LDS; each wave one
            (head tile, 16-key tile): its fp8_ds_mla rows loaded once,
            dequantized into its QK operands and kept raw in LDS; QK (bf16
            MFMA), the chunk's max / sum per head, bf16 p in LDS; PV over the
            chunk's raw rows (wave w: dims 64 w ..) -> the chunk's partial
            (bf16) and its (max, sum) per head, published
    combine (S x H): a (token, head)'s chunks merged with the sink folded in,
            bf16, inverse GPT-J RoPE of dims 448.., vLLM's MXFP8 of each 32
            -> XO (published)
    wo_a    (G x 1024 / 32, two 16-row tiles): the group's XO rows times wo_a
            -> bf16 -> MXFP8 of the 32 rows -> X8B (published)
    wo_b    (320, 16 rows): X8B times wo_b -> bf16 -> ``out``

Hand-offs are plain device-scope data behind tagged flags (half the bytes of a
tagged pair per value). Adapted from ATOM's V4.1 ``attn_post`` (K2a), which
reads a bf16 pool and ends in a peer all-reduce.
"""

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.srt.models.deepseek_common.amd.dsv41_mono.common.mx import clamp_fp8
from sglang.srt.models.deepseek_common.amd.dsv41_mono.common.ops import (
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fresh,
    hw_exp2,
    lane_gather,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    xshfl,
)
from sglang.srt.models.deepseek_common.amd.dsv41_mono.common.plan import first_task
from sglang.srt.models.deepseek_common.amd.dsv41_mono.stages.gemv import (
    ld_f32,
    lds_row,
    plus,
    tile_rows,
    token_tiles,
)

from .device import (
    BLOCKS,
    CM_DEV,
    THREADS,
    WAVES,
    Mailbox,
    fp8x8_bf16,
    gload,
    ld_i32,
    mx_code,
    mx_mul,
    pow2,
)
from .gemv import RED, gemv_loads, gemv_mfmas, rows_loads, rows_store
from .plan import (
    DATA,
    HALF,
    HEAD_DIM,
    HEAD_TILE,
    KEYS,
    NOPE,
    ROPE,
    ROWS,
    TOPK,
    WINDOW,
    WOA_ROWS,
    Dims,
    cdiv,
)

NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
REC = DATA + 8  # a raw record row in LDS: 576 data bytes + 8 scale bytes
Q_ROW = (
    HEAD_DIM // 2 + 4
)  # a q row in LDS (i32 words): 512 bf16 + 16 B, no bank aliasing
P_PAD = 2  # i32 words past a P row: 16 head rows read together hit distinct banks
# wo_a's K order: waves 0..3 / 4..7 take the two 16-row tiles, a contiguous K
# range a wave, summed in wave order
WOA_SPLIT_K = WAVES * ROWS // WOA_ROWS


@dataclass(frozen=True)
class BackBuild:
    tokens: int
    tp: int


def head_groups(d: Dims, s: int) -> int:
    """Wave groups a split unit's head tiles are spread over: two (a tile each,
    64-key chunks, twice the units) while the units fit the grid, else one (a
    wave's keys loaded and dequantized once for every head tile)."""
    return 2 if d.head_tiles == 2 and s * cdiv(KEYS, 64) <= BLOCKS else 1


def chunk_keys(d: Dims, s: int) -> int:
    """A split unit's keys: a 16-key tile a wave of a head group."""
    return 16 * (WAVES // head_groups(d, s))


def p_row(d: Dims, s: int) -> int:
    """A P row (one head's chunk of bf16 p) in LDS, i32 words."""
    return chunk_keys(d, s) // 2 + P_PAD


@traced
def publish(c, region, i):
    """This CTA's stores visible device-wide, then the flag pair (region, i)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["mb"].put(region, i, fx.Int32(1))


@traced
def await_flags(c, region, n, index):
    """Thread u < n polls flag ``index(u)``; then the CTA proceeds."""
    tid = c["tid"]
    if tid < n:
        c["mb"].poll([(region, index(fx.min(tid, n - 1)), 1)])
    gpu.barrier()


# ---------------------------------------------------------------- keys
# A token's keys: 0 .. WINDOW its window list (SGLang's swa_page_indices, SWA cache slots), WINDOW .. KEYS its
# compressed list (c{r}_sparse_page_indices, compressed cache slots), each valid up to its count (swa_topk_lengths,
# c{r}_sparse_topk_lengths). Window first: neither list's address waits on a count.


def key_lens(c, t):
    """Token t's (window, compressed) key counts."""
    a = c["args"]
    return ld_i32(a["slen"], t), ld_i32(a["comp_len"], t)


def chunk_live(k0, lens):
    """Whether the chunk of keys from ``k0`` (a multiple of the chunk size, which divides WINDOW) holds a listed key."""
    slen, comp_len = lens
    if isinstance(k0, int):
        return k0 < slen if k0 < WINDOW else k0 - WINDOW < comp_len
    return ((k0 < WINDOW) & (k0 < slen)) | ((k0 >= WINDOW) & (k0 - WINDOW < comp_len))


def key_rec(c, t, k0, k, lens):
    """(data address, scale address, valid, key table value) of token t's key
    k of the chunk from ``k0``: its list entry, a slot of that list's cache; an
    entry past the list's count or of -1 is none (it reads a zero record: its
    scores are masked, its values zeros). The key table value is vLLM's: slot |
    1 << 31 for a compressed row, a window slot, -1 for none."""
    a = c["args"]
    slen, comp_len = lens
    # from the chunk, uniform: the list's buffer resource must be
    win = k0 < WINDOW
    j = win.select(k, k - WINDOW)
    row = win.select(fx.Int32(WINDOW), fx.Int32(TOPK))
    idx = ld_i32(win.select(a["sidx"], a["cidx"]), t * row + j)
    ok = (j < win.select(slen, comp_len)) & (idx >= 0)
    slot = ok.select(idx, fx.Int32(0))
    blk = win.select(a["swa_block"], a["comp_block"])
    stride = win.select(a["swa_stride"], a["comp_stride"])
    base = win.select(a["swa"], a["comp"])
    b = slot // blk
    off = slot % blk
    page = base + fx.Int64(b) * stride
    z = a["zrec"]
    data = ok.select(page + fx.Int64(off * DATA), z)
    sc = ok.select(page + fx.Int64(blk * DATA + off * 8), z + fx.Int64(DATA))
    val = ok.select(win.select(slot, slot | fx.Int32(-2147483648)), fx.Int32(-1))
    return data, sc, ok, val


def _scale_byte(scw, blk64):
    return (scw[blk64 // 4] >> (blk64 % 4 * 8)) & 0xFF


def _fp8_pair_f32(word, scale, hi):
    v = rocdl.cvt_scalef32_pk_f32_fp8(
        fx.Vector.make_type(2, fx.Float32),
        as_ir_value(fx.Int32(word)),
        as_ir_value(fx.Float32(scale)),
        hi,
    )
    v = fx.Vector(v)
    return [v[0], v[1]]


# ---------------------------------------------------------------- split


@traced
def stage_split(c, unit):
    """Flash-decoding over chunk ``unit % nchunk`` of token ``unit // nchunk``,
    every head of the rank -> the chunk's partial and (max, sum) per head. Wave
    (head group hg, key tile kt): keys 16 kt .. of the chunk (loaded and
    dequantized once) against the group's head tiles; then dims 64 w .. of the
    PV over every key and head."""
    tid, lane, wave, d, s = c["tid"], c["lane"], c["wave"], c["d"], c["S"]
    a, sl = c["args"], c["sl"]
    nchunk = c["nchunk"]
    t = unit // nchunk
    ch = unit % nchunk
    ck = chunk_keys(d, s)
    ngrp = head_groups(d, s)
    nkt = WAVES // ngrp  # 16-key tiles a chunk
    nht = d.head_tiles // ngrp  # head tiles a wave
    hgi = wave // nkt
    kt = wave % nkt
    g = lane // 16
    r16 = lane % 16
    k0 = ch * ck
    # the key counts, this lane's key and q: independent loads, out together
    lens = key_lens(c, t)
    key = k0 + kt * 16 + r16
    data, sc, okv, kval = key_rec(c, t, k0, key, lens)
    qrow = d.heads * HEAD_DIM // 8
    qv = []
    for i in range_constexpr(cdiv(qrow, THREADS)):
        u = fresh(tid) + THREADS * i
        qv.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["q"]),
                    (t * d.q_rows) // 2 + fx.min(u, qrow - 1) * 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
        )
    # the key table and counts as vLLM's KT / KLEN (read by no stage)
    if (g == 0) & (hgi == 0):
        bo.buffer_store(kval, rsrc(a["kt"]), t * KEYS + key)
    if (ch == 0) & (tid == 0):
        bo.buffer_store(lens[0] + lens[1], rsrc(a["klen"]), t)
        bo.buffer_store(lens[1], rsrc(a["klen"]), s + t)
    if chunk_live(k0, lens):
        # a key's 448 e4m3 as 7 x 16 B (lane g: dims 64 c + 16 g ..), its 64
        # bf16 rope dims as 2 x 16 B (32 c + 8 g ..), its 7 scale bytes
        scw = gload(sc, words=2)
        nope = [
            gload(data + fx.Int64(64 * c4 + 16 * g), words=4)
            for c4 in range(NOPE // 64)
        ]
        rope = [
            gload(data + fx.Int64(NOPE + (cc * 32 + 8 * g) * 2), words=4)
            for cc in range(ROPE // 32)
        ]
        # q of every head into LDS (bf16 rows of 512 + pad), 16 B a thread
        for i in range_constexpr(cdiv(qrow, THREADS)):
            u = fresh(tid) + THREADS * i
            if u < qrow:
                fx.ptr_store(
                    qv[i],
                    sl["q"]
                    + (u // (HEAD_DIM // 8)) * Q_ROW
                    + (u % (HEAD_DIM // 8)) * 4,
                )
        # the raw rows to LDS for the PV (head group 0's waves)
        if hgi == 0:
            rawb = sl["raw"] + (kt * 16 + r16) * (REC // 4)
            for c4 in range_constexpr(NOPE // 64):
                fx.ptr_store(nope[c4], rawb + (64 * c4 + 16 * g) // 4)
            for cc in range_constexpr(ROPE // 32):
                fx.ptr_store(rope[cc], rawb + (NOPE + (cc * 32 + 8 * g) * 2) // 4)
            if g == 0:
                fx.ptr_store(scw, rawb + DATA // 4)
        gpu.barrier()
        # QK: A = the keys (dequantized once), B = q of a head tile (LDS); the
        # contraction's 32-dim chunk 2 c + h of the nope dims is {64 c + 16 g +
        # 8 h ..} (lane g's 16 B load), q read in the same order
        scale = fx.Int32(a["qk_scale"]).bitcast(fx.Float32)
        st = sl["st"]
        accs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(nht)]
        for c4 in range_constexpr(NOPE // 64):
            ksc = pow2(_scale_byte(scw, c4))
            for h2 in range_constexpr(2):
                kw = fp8x8_bf16(nope[c4][2 * h2], nope[c4][2 * h2 + 1], ksc).bitcast(
                    fx.BFloat16
                )
                for ht in range_constexpr(nht):
                    qb = sl["q"] + ((hgi * nht + ht) * HEAD_TILE + r16) * Q_ROW
                    qw = fx.Vector(
                        fx.ptr_load(
                            qb + 32 * c4 + 8 * g + 4 * h2,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    accs[ht] = mfma_bf16(kw, qw.bitcast(fx.BFloat16), accs[ht])
        for cc in range_constexpr(ROPE // 32):
            kw = rope[cc].bitcast(fx.BFloat16)
            for ht in range_constexpr(nht):
                qb = sl["q"] + ((hgi * nht + ht) * HEAD_TILE + r16) * Q_ROW
                qw = fx.Vector(
                    fx.ptr_load(
                        qb + (NOPE + cc * 32 + 8 * g) // 2,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                accs[ht] = mfma_bf16(kw, qw.bitcast(fx.BFloat16), accs[ht])
        # scores of keys 4 g + i (C rows), head r16 of a tile (C column)
        okw = okv.select(fx.Int32(1), fx.Int32(0))
        valid = [lane_gather(okw, 4 * g + i) != 0 for i in range(4)]
        svs = []
        for ht in range_constexpr(nht):
            sv = [
                fx.Float32(valid[i].select(accs[ht][i] * scale, fx.Float32(NEG)))
                for i in range(4)
            ]
            mw = butterfly(
                fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])), (32, 16), fx.max
            )
            if g == 0:
                fx.ptr_store(mw, st + ((hgi * nht + ht) * nkt + kt) * HEAD_TILE + r16)
            svs.append(sv)
        gpu.barrier()
        lst = st + d.heads * nkt
        for ht in range_constexpr(nht):
            hti = hgi * nht + ht
            m = fx.ptr_load(st + (hti * nkt) * HEAD_TILE + r16)
            for j in range_constexpr(1, nkt):
                m = fx.max(m, fx.ptr_load(st + (hti * nkt + j) * HEAD_TILE + r16))
            sv = svs[ht]
            p = [
                fx.Float32(valid[i].select(hw_exp2(sv[i] - m), fx.Float32(0.0)))
                for i in range(4)
            ]
            lw = (p[0] + p[2]) + (p[1] + p[3])
            lw = lw + xshfl(lw, 32)
            lw = lw + xshfl(lw, 16)
            pw = (
                fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
            )
            pbase = (
                sl["p"] + (hti * HEAD_TILE + r16) * p_row(d, s) + (kt * 16 + 4 * g) // 2
            )
            fx.ptr_store(fx.Vector.from_elements([pw[0], pw[1]], fx.Int32), pbase)
            if g == 0:
                fx.ptr_store(lw, lst + (hti * nkt + kt) * HEAD_TILE + r16)
        gpu.barrier()
        # chunk stats: (max, sum) per head
        if tid < d.heads:
            hts = tid // HEAD_TILE
            hr = tid % HEAD_TILE
            mm = fx.ptr_load(st + (hts * nkt) * HEAD_TILE + hr)
            ll = fx.ptr_load(lst + (hts * nkt) * HEAD_TILE + hr)
            for j in range_constexpr(1, nkt):
                mm = fx.max(mm, fx.ptr_load(st + (hts * nkt + j) * HEAD_TILE + hr))
                ll = ll + fx.ptr_load(lst + (hts * nkt + j) * HEAD_TILE + hr)
            bo.buffer_store(
                fx.Vector.from_elements([mm, ll], fx.Float32),
                rsrc(c["pstat"]),
                ((t * nchunk + ch) * d.heads + tid) * 2,
                cache_modifier=CM_DEV,
            )
        # PV: wave w dims 64 w + 4 r16 + dt (dt = 0..3: one LDS word holds the
        # four of a key); the rope wave (dims 448..) reads bf16 pairs. Lane
        # (r16, g) takes keys 4 g + j of a 16-key tile: A[dim][key]
        is_rope = wave == WAVES - 1
        dbase = 64 * wave
        sh = (wave % 4) * 8
        accs = [
            [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(4)]
            for _ in range(d.head_tiles)
        ]
        for kt2 in range_constexpr(nkt):
            rb0 = sl["raw"] + (kt2 * 16 + 4 * g) * (REC // 4)
            vv = fx.Vector.filled(16, 0.0, fx.Float32)  # [dt][j]
            if is_rope:
                rv = []
                for j in range_constexpr(4):
                    rw = fx.Vector(
                        fx.ptr_load(
                            rb0 + j * (REC // 4) + (NOPE + 8 * r16) // 4,
                            result_type=fx.Vector.make_type(2, fx.Int32),
                        )
                    )
                    rv.append([bf_lo(rw[0]), bf_hi(rw[0]), bf_lo(rw[1]), bf_hi(rw[1])])
                vv = fx.Vector.from_elements(
                    [rv[j][dt] for dt in range(4) for j in range(4)], fx.Float32
                )
            else:
                nv = []
                for j in range_constexpr(4):
                    rb = rb0 + j * (REC // 4)
                    w = fx.Int32(fx.ptr_load(rb + (dbase + 4 * r16) // 4))
                    scb = fx.Int32(fx.ptr_load(rb + DATA // 4 + wave // 4))
                    scl = pow2((scb >> sh) & 0xFF)
                    nv.append(
                        _fp8_pair_f32(w, scl, False) + _fp8_pair_f32(w, scl, True)
                    )
                vv = fx.Vector.from_elements(
                    [nv[j][dt] for dt in range(4) for j in range(4)], fx.Float32
                )
            for ht2 in range_constexpr(d.head_tiles):
                pb = fx.Vector(
                    fx.ptr_load(
                        sl["p"]
                        + (ht2 * HEAD_TILE + r16) * p_row(d, s)
                        + (kt2 * 16 + 4 * g) // 2,
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                ).bitcast(fx.Int16)
                for dt in range_constexpr(4):
                    av = (
                        fx.Vector.from_elements(
                            [vv[dt * 4 + j] for j in range(4)], fx.Float32
                        )
                        .to(fx.BFloat16)
                        .bitcast(fx.Int16)
                    )
                    accs[ht2][dt] = fx.Vector(
                        rocdl.mfma_f32_16x16x16bf16_1k(
                            T.vec(4, T.f32), [av, pb, accs[ht2][dt]]
                        )
                    )
        # partial[t][ch][head][dim] bf16: lane (r16, g) holds C rows 4 g + i ->
        # dims 64 w + 16 g + 4 i + dt of head ht2 * 16 + r16: 32 contiguous
        # bytes, two 16 B stores (4 lanes fill a 128 B line)
        for ht2 in range_constexpr(d.head_tiles):
            hh = ht2 * HEAD_TILE + r16
            words = []
            for i in range_constexpr(4):
                v4 = fx.Vector.from_elements(
                    [accs[ht2][dt][i] for dt in range(4)], fx.Float32
                )
                w2 = v4.to(fx.BFloat16).bitcast(fx.Int32)
                words += [w2[0], w2[1]]
            off = (((t * nchunk + ch) * d.heads + hh) * HEAD_DIM + dbase + 16 * g) // 2
            for hlf in range_constexpr(2):
                bo.buffer_store(
                    fx.Vector.from_elements(words[4 * hlf : 4 * hlf + 4], fx.Int32),
                    rsrc(c["part"]),
                    off + 4 * hlf,
                    cache_modifier=CM_DEV,
                )
    publish(c, c["srdy"], t * nchunk + ch)


# ---------------------------------------------------------------- combine

COMB_HEADS = WAVES  # heads a combine unit: a wave each


@traced
def stage_combine(c, unit):
    """(token, 8 heads) ``unit``, a wave a head, 8 dims a lane: the chunks
    merged with the sink folded in, bf16, inverse RoPE, MXFP8 of each 32 -> XO,
    published (a flag a (token, head group))."""
    lane, wave, d, s = c["lane"], c["wave"], c["d"], c["S"]
    a = c["args"]
    hg_n = d.heads // COMB_HEADS
    t = unit // hg_n
    hg = unit % hg_n
    h = hg * COMB_HEADS + wave
    nchunk = c["nchunk"]
    ck = chunk_keys(d, s)
    lens = key_lens(c, t)
    pos = ld_i32(a["pos"], 2 * t)
    sink = ld_f32(a["sink"], h) * fx.Float32(LOG2E)
    d0 = lane * 8  # this lane's 8 dims
    rope_lane = d0 >= NOPE
    jj = fx.max((d0 - NOPE) // 2, 0)  # 4 pairs from here
    cs = [
        rope_lane.select(ld_f32(a["cos_sin"], pos * ROPE + jj + i), fx.Float32(1.0))
        for i in range(4)
    ]
    sn = [
        rope_lane.select(
            ld_f32(a["cos_sin"], pos * ROPE + HALF + jj + i), fx.Float32(0.0)
        )
        for i in range(4)
    ]
    # every chunk's flag (a chunk without listed keys publishes at once)
    await_flags(c, c["srdy"], nchunk, lambda u: t * nchunk + u)
    m = fx.Float32(NEG)
    ms, ls, vs = [], [], []
    for cc in range_constexpr(nchunk):
        live = chunk_live(cc * ck, lens)
        stv = fx.Vector(
            bo.buffer_load(
                rsrc(c["pstat"]),
                ((t * nchunk + cc) * d.heads + h) * 2,
                vec_width=2,
                dtype=T.f32,
            )
        )
        pv = fx.Vector(
            bo.buffer_load(
                rsrc(c["part"]),
                (((t * nchunk + cc) * d.heads + h) * HEAD_DIM + d0) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        mc = fx.Float32(live.select(stv[0], fx.Float32(NEG)))
        ms.append(mc)
        ls.append(fx.Float32(live.select(stv[1], fx.Float32(0.0))))
        vals = []
        for e in range_constexpr(4):
            vals += [bf_lo(pv[e]), bf_hi(pv[e])]
        vs.append([fx.Float32(live.select(v, fx.Float32(0.0))) for v in vals])
        m = fx.max(m, mc)
    m_final = fx.max(m, sink)
    lsum = hw_exp2(sink - m_final)
    tot = [fx.Float32(0.0)] * 8
    for cc in range_constexpr(nchunk):
        wgt = hw_exp2(ms[cc] - m_final)
        lsum = lsum + ls[cc] * wgt
        tot = [tot[e] + vs[cc][e] * wgt for e in range(8)]
    inv = fx.Float32(1.0) / lsum
    o = [bf16_round(tot[e] * inv) for e in range(8)]
    # inverse GPT-J RoPE of each pair: e' = e cos + o sin, o' = o cos - e sin
    y = []
    for i in range_constexpr(4):
        e, od = o[2 * i], o[2 * i + 1]
        y += [e * cs[i] + od * sn[i], od * cs[i] - e * sn[i]]
    store_xo(c, t, h * HEAD_DIM + d0, y)
    publish(c, c["crdy"], t * hg_n + hg)


@traced
def store_xo(c, t, col, y):
    """A lane's 8 attention output values at column ``col`` of token t's row -> vLLM's MXFP8 of each 32 (4
    lanes) -> XO words / XOS codes, wo_a's input."""
    d, lane = c["d"], c["lane"]
    amax = abs(y[0])
    for v in y[1:]:
        amax = fx.max(amax, abs(v))
    amax = butterfly(amax, (1, 2), fx.max)  # 32 dims = 4 lanes
    code = mx_code(amax)
    mul = mx_mul(code)
    w0 = fp8_pack4(*[clamp_fp8(y[i] * mul) for i in range(4)])
    w1 = fp8_pack4(*[clamp_fp8(y[4 + i] * mul) for i in range(4)])
    bo.buffer_store(
        fx.Vector.from_elements([w0, w1], fx.Int32),
        rsrc(c["xo"]),
        t * (d.q_rows // 4) + col // 4,
        cache_modifier=CM_DEV,
    )
    if lane % 4 == 0:
        bo.buffer_store(
            code,
            rsrc(c["xos"]),
            t * (d.q_rows // 32) + col // 32,
            cache_modifier=CM_DEV,
        )


# ---------------------------------------------------------------- wo_a / wo_b


@traced
def stage_woa(c, task, ops, t0, n):
    """Rows 32 task .. of wo_a -> bf16 -> vLLM's MXFP8 of the 32 -> X8B."""
    tid, red, d, lane = c["tid"], c["red"], c["d"], c["lane"]
    split = WOA_SPLIT_K
    gemv_mfmas(c, d.group_k, c["xl"], c["xsl"], red, ops, split, tiled=True, rows=n)
    gpu.barrier()
    mine = tid < WOA_ROWS * n
    tl = fx.min(tid // WOA_ROWS, n - 1)
    t = plus(t0, tl)
    row = tid % WOA_ROWS
    tile = row // ROWS
    y = bf16_round(
        row_sum(red, row % ROWS, tl, [tile * split + w for w in range(split)])
    )
    amax = butterfly(abs(y), (1, 2, 4, 8, 16), fx.max)
    code = mx_code(amax)
    q = clamp_fp8(y * mx_mul(code))
    w = fp8_pack4(
        q,
        *[
            lane_gather(q.bitcast(fx.Int32), fx.min(lane + k, 63)).bitcast(fx.Float32)
            for k in (1, 2, 3)
        ],
    )
    if mine & (row % 4 == 0):
        bo.buffer_store(
            w,
            rsrc(c["x8b"]),
            t * (d.o_rows // 4) + (task * WOA_ROWS + row) // 4,
            cache_modifier=CM_DEV,
        )
    if mine & (row == 0):
        bo.buffer_store(
            code, rsrc(c["x8bs"]), t * (d.o_rows // 32) + task, cache_modifier=CM_DEV
        )
    gpu.barrier()


def woa_loads(c, task):
    a = c["args"]
    return gemv_loads(
        c,
        a["woa"],
        a["woa_s"],
        c["d"].group_k,
        task * WOA_ROWS,
        WOA_SPLIT_K,
        tiled=True,
    )


def wob_loads(c, task, pred=None):
    a = c["args"]
    return gemv_loads(c, a["wob"], a["wob_s"], c["d"].o_rows, task * ROWS, pred=pred)


def _wob_store(a, t, col, v0, v1):
    word = (
        fx.Vector.from_elements([v0, v1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    bo.buffer_store(word, rsrc(a["out"]), (t * a["out_stride"] + col) // 2)


@traced
def stage_wob(c, task, ops, task2, ops2, has2, t0, n):
    """Rows 16 task .. (and 16 task2 .., ``has2``) of wo_b -> bf16 -> out, or
    (``c["wob_out"]``, the mono layer) the caller's hook with each bf16 pair."""
    tid, red = c["tid"], c["red"]
    o_rows = c["d"].o_rows
    a = c["args"]
    gemv_mfmas(c, o_rows, c["xl"], c["xsl"], red, ops, rows=n)
    gemv_mfmas(c, o_rows, c["xl"], c["xsl"], red + RED, ops2, rows=n)
    gpu.barrier()
    per = ROWS * n // 2  # bf16 pairs a task
    k = tid // per
    emit = c.get("wob_out") or (lambda t, col, v0, v1: _wob_store(a, t, col, v0, v1))
    if (tid < 2 * per) & ((k == 0) | has2):
        u = tid % per
        rp = u % (ROWS // 2)
        tl = u // (ROWS // 2)
        t = plus(t0, tl)
        rk = red + k * RED
        col = (k == 0).select(task, task2) * ROWS + 2 * rp
        v0 = row_sum(rk, 2 * rp, tl)
        v1 = row_sum(rk, 2 * rp + 1, tl)
        emit(t, col, v0, v1)
    gpu.barrier()


# ---------------------------------------------------------------- kernel


def back_plan(key):
    """(nchunk, n_split, n_comb, COMB0, WOA0, WOB0, early_wob) of a build."""
    s, d = key.tokens, Dims(key.tp)
    ck = chunk_keys(d, s)
    nchunk = cdiv(KEYS, ck)
    n_split = s * nchunk
    n_comb = s * (d.heads // COMB_HEADS)
    COMB0 = n_split % BLOCKS
    assert d.woa_tasks <= BLOCKS
    assert BLOCKS < d.wob_tasks <= 2 * BLOCKS
    # the GEMVs past the attention units, so CTAs without attention work have
    # their weights in flight from the start
    WOA0 = (n_split + n_comb) % BLOCKS
    WOB0 = (WOA0 + d.woa_tasks) % BLOCKS
    # wo_b's weights before the wo_a stage only while the wo_a CTAs have no
    # attention work (loads complete in issue order: else they would hold up
    # wo_a's activation rows)
    early_wob = n_split + n_comb + d.woa_tasks <= BLOCKS
    return nchunk, n_split, n_comb, COMB0, WOA0, WOB0, early_wob


def back_lds_members(s: int, d: Dims) -> dict:
    """K2's attention LDS structs: the split stage's and the GEMVs' (a union's
    members: one at a time)."""
    rows = tile_rows(s)
    ck = chunk_keys(d, s)

    @fx.struct
    class SplitLds:
        q: fx.Array[fx.Int32, d.heads * Q_ROW, 16]
        raw: fx.Array[fx.Int32, ck * REC // 4, 16]
        p: fx.Array[fx.Int32, d.heads * p_row(d, s), 16]
        st: fx.Array[fx.Float32, 2 * d.heads * (WAVES // head_groups(d, s)), 16]

    @fx.struct
    class GemvLds:
        red: fx.Array[fx.Float32, 2 * RED, 16]
        xl: fx.Array[fx.Int32, rows * lds_row(d.group_k // 4), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(d.group_k // 32), 16]

    return {"split": SplitLds, "gemv": GemvLds}


def back_context(key, lds, args: dict, scratch, tag, layout) -> dict:
    """The attention back's context: LDS views (``lds``: a dict of
    ``back_lds_members``' views), mailbox (tag ``tag``), scratch regions
    (``layout``: name -> (byte offset, bytes)), arguments by name."""
    s, d = key.tokens, Dims(key.tp)
    tid = fx.thread_idx.x
    spl, gml = lds["split"], lds["gemv"]
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": gml.red.ptr,
        "xl": gml.xl.ptr,
        "xsl": gml.xsl.ptr,
        "d": d,
        "mb": Mailbox(tag),
        "nchunk": back_plan(key)[0],
        "sl": {"q": spl.q.ptr, "raw": spl.raw.ptr, "p": spl.p.ptr, "st": spl.st.ptr},
        "args": args,
    }
    for region, (off, _) in layout.items():
        c[region] = scratch + fx.Int64(off)
    return c


@traced
def run_back(c, key, bid):
    """The attention back's stages on this CTA: split, combine, wo_a, wo_b."""
    s = key.tokens
    dd = Dims(key.tp)
    # plain ints: a Python object read inside traced control flow is carried
    woa_tasks, groups, heads, group_k = dd.woa_tasks, dd.groups, dd.heads, dd.group_k
    q_rows, o_rows, wob_tasks = dd.q_rows, dd.o_rows, dd.wob_tasks
    a = c["args"]
    tiles = token_tiles(s)
    _, n_split, n_comb, COMB0, WOA0, WOB0, early_wob = back_plan(key)
    for unit in range(first_task(bid, 0), n_split, BLOCKS):
        stage_split(c, unit)
    for unit in range(first_task(bid, COMB0), n_comb, BLOCKS):
        stage_combine(c, unit)
    gpu.barrier()
    # the GEMV weights, every CTA right after its attention work: its wo_a
    # task (if any), then its one or two wo_b tasks
    woa_task = first_task(bid, WOA0)
    has_woa = woa_task < woa_tasks
    woa_t = fx.min(woa_task, woa_tasks - 1)
    ops = gemv_loads(
        c,
        a["woa"],
        a["woa_s"],
        group_k,
        woa_t * WOA_ROWS,
        WOA_SPLIT_K,
        tiled=True,
        pred=has_woa,
    )
    wob_task = first_task(bid, WOB0)
    wob_second = wob_task + BLOCKS
    has_second = wob_second < wob_tasks

    def wob_weights():
        return (
            wob_loads(c, wob_task),
            wob_loads(c, fx.min(wob_second, wob_tasks - 1), pred=has_second),
        )

    if const_expr(early_wob):
        ops1, ops2 = wob_weights()
    # wo_a: every token tile of its group's XO rows, the next tile's in flight
    if has_woa:
        grp = woa_t // (woa_tasks // groups)
        await_flags(c, c["crdy"], s, lambda u: u * (heads // COMB_HEADS) + grp)
        xo = (
            c["xo"],
            c["xos"],
            q_rows // 4,
            grp * group_k // 4,
            group_k // 4,
            q_rows // 32,
            grp * group_k // 32,
            group_k // 32,
        )
        regs = rows_loads(c, *xo, *tiles[0])
        for ti in range_constexpr(len(tiles)):
            t0, n = tiles[ti]
            rows_store(c, regs, group_k // 4, group_k // 32, c["xl"], c["xsl"], n)
            if const_expr(ti + 1 < len(tiles)):
                regs = rows_loads(c, *xo, *tiles[ti + 1])
            stage_woa(c, woa_t, ops, t0, n)
        publish(c, c["ardy"], woa_t)
    # wo_b
    if const_expr(not early_wob):
        ops1, ops2 = wob_weights()
    await_flags(c, c["ardy"], woa_tasks, lambda u: u)
    x8b = (
        c["x8b"],
        c["x8bs"],
        o_rows // 4,
        0,
        o_rows // 4,
        o_rows // 32,
        0,
        o_rows // 32,
    )
    regs = rows_loads(c, *x8b, *tiles[0])
    for ti in range_constexpr(len(tiles)):
        t0, n = tiles[ti]
        rows_store(c, regs, o_rows // 4, o_rows // 32, c["xl"], c["xsl"], n)
        if const_expr(ti + 1 < len(tiles)):
            regs = rows_loads(c, *x8b, *tiles[ti + 1])
        stage_wob(
            c,
            wob_task,
            ops1,
            fx.min(wob_second, wob_tasks - 1),
            ops2,
            has_second,
            t0,
            n,
        )
