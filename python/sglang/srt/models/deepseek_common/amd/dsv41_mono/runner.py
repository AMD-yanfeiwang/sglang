# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host runner and buffers of the mono decode launches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .attention.plan import HEAD_DIM, KEYS, TOPK, WINDOW, Dims
from .common.plan import BLOCKS
from .layer import (
    EPOCH_WORDS,
    HIDDEN,
    MAX_TOKENS,
    MonoBuild,
    build_mono_ffn,
    build_mono_k2,
    peer_half_bytes,
    scratch_bytes,
)
from .stages.dims import Dims as MoeDims
from .stages.moe_shape import EXPERTS

__all__ = ["BLOCKS", "MAX_TOKENS", "AttnWeights", "DSV41MonoLayer", "MonoLayerWeights"]

HC = 4
RECORD = 584  # a V4 KV record: 576 data bytes, then 8 scale bytes
_LOG2E = 1.4426950408889634


def _ensure_writable_flydsl_cache() -> None:
    """Point FlyDSL at a writable cache (aiter sets a read-only one)."""
    cur = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if cur and os.access(cur, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


def _check_tensors(owner, tag: str, want: dict) -> None:
    """Raise unless each named tensor has the expected shape and dtype and is contiguous."""
    for name, (shape, dtype) in want.items():
        t = getattr(owner, name)
        if not (
            tuple(t.shape) == shape
            and t.dtype == dtype
            and t.is_contiguous()
            and t.is_cuda
        ):
            raise RuntimeError(
                f"{tag} {name}: {tuple(t.shape)} {t.dtype}, want {shape} {dtype}"
            )


def check_rows(tag: str, rows: int, device, want: dict) -> None:
    """Raise unless each tensor is contiguous on ``device`` with ``rows`` rows of (row size, dtype)."""
    for name, (t, size, dtype) in want.items():
        if not (
            t.dtype == dtype
            and t.numel() == rows * size
            and t.is_contiguous()
            and t.device == device
        ):
            raise RuntimeError(
                f"{tag} {name}: {tuple(t.shape)} {t.dtype} {t.device}, want {rows} x {size} {dtype}"
            )


def no_capture(what: str) -> None:
    """Raise inside a CUDA graph capture: builds and allocations run eagerly."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"DSv4.1 mono decode: {what} first reached inside a CUDA graph capture"
        )


@dataclass
class AttnWeights:
    """One layer's attention tensors on this rank, in vLLM's loaded layout."""

    layer_id: int
    wqkv: torch.Tensor  # [1792, 5120] e4m3: [wq_a; wkv]
    wqkv_scale: torch.Tensor  # [56, 160] uint8 E8M0 (32 x 32 blocks)
    q_norm: torch.Tensor  # [1280] bf16
    kv_norm: torch.Tensor  # [512] bf16
    wq_b: torch.Tensor  # [H * 512, 1280] e4m3
    wq_b_scale: torch.Tensor  # [H * 16, 40] uint8
    wo_a: torch.Tensor  # [G * 1024, 4096] e4m3
    wo_a_scale: torch.Tensor  # [G * 32, 128] uint8
    wo_b: torch.Tensor  # [5120, G * 1024] e4m3
    wo_b_scale: torch.Tensor  # [160, G * 32] uint8
    attn_sink: torch.Tensor  # [>= H] f32
    cos_sin: torch.Tensor  # [positions, 64] f32 (the layer's rope table)
    ratio: int  # 0 / 1 / 2: the layer's compress ratio

    def check(self, d: Dims) -> None:
        """Raise unless every tensor has the shape and dtype the launches read."""
        h, g = d.heads, d.groups
        want = {
            "wqkv": ((1792, HIDDEN), torch.float8_e4m3fn),
            "wqkv_scale": ((56, HIDDEN // 32), torch.uint8),
            "wq_b": ((h * HEAD_DIM, 1280), torch.float8_e4m3fn),
            "wq_b_scale": ((h * HEAD_DIM // 32, 40), torch.uint8),
            "wo_a": ((g * 1024, d.group_k), torch.float8_e4m3fn),
            "wo_a_scale": ((g * 32, d.group_k // 32), torch.uint8),
            "wo_b": ((HIDDEN, g * 1024), torch.float8_e4m3fn),
            "wo_b_scale": ((HIDDEN // 32, g * 32), torch.uint8),
        }
        _check_tensors(self, f"layer {self.layer_id}", want)
        cs, sink = self.cos_sin, self.attn_sink
        if not (
            cs.dtype == torch.float32
            and cs.dim() == 2
            and cs.shape[1] == 64
            and cs.is_contiguous()
            and cs.is_cuda
        ):
            raise RuntimeError(
                f"layer {self.layer_id} cos_sin: {tuple(cs.shape)} {cs.dtype}, want [positions, 64] f32"
            )
        if not (
            sink.dtype == torch.float32
            and sink.numel() >= h
            and sink.is_contiguous()
            and sink.is_cuda
        ):
            raise RuntimeError(
                f"layer {self.layer_id} attn_sink: {tuple(sink.shape)} {sink.dtype}, want >= {h} f32"
            )
        if self.ratio not in (0, 1, 2):
            raise RuntimeError(f"layer {self.layer_id}: compress ratio {self.ratio}")


@dataclass
class MonoLayerWeights:
    """One layer's FFN-launch tensors on this rank, in SGLang's loaded layout."""

    hc_ffn_fn: torch.Tensor  # [24, 4 * 5120] f32
    hc_ffn_scale: torch.Tensor  # [3] f32
    hc_ffn_base: torch.Tensor  # [24] f32
    ffn_norm: torch.Tensor  # [5120] bf16
    gate_w: torch.Tensor  # [384, 5120] bf16
    bias: torch.Tensor  # [384] f32 (e_score_correction_bias)
    w13: torch.Tensor  # [384, 2 inter, 2560] fp4x2, aiter (16, 16) shuffle
    w13_s: torch.Tensor  # its scales, aiter shuffle_scale order
    w2: torch.Tensor  # [384, 5120, inter / 2] fp4x2
    w2_s: torch.Tensor
    sgu: torch.Tensor  # shared gate_up [2 inter, 5120] e4m3, row-major
    sgu_s: torch.Tensor  # [2 inter / 32, 160] E8M0
    sw2: torch.Tensor  # shared down [5120, inter] e4m3
    sw2_s: torch.Tensor

    def check(self, tp: int) -> None:
        """One rank's MoE tensors in SGLang's loaded layout."""
        d, fp4, e4m3, u8 = (
            MoeDims(tp),
            torch.float4_e2m1fn_x2,
            torch.float8_e4m3fn,
            torch.uint8,
        )
        want = {
            "gate_w": ((EXPERTS, HIDDEN), torch.bfloat16),
            "bias": ((EXPERTS,), torch.float32),
            "w13": ((EXPERTS, 2 * d.inter, HIDDEN // 2), fp4),
            "w13_s": ((EXPERTS * 2 * d.inter, HIDDEN // 32), u8),
            "w2": ((EXPERTS, HIDDEN, d.inter // 2), fp4),
            "w2_s": ((EXPERTS * HIDDEN, d.down_scale_cols), u8),
            "sgu": ((2 * d.sh_inter, HIDDEN), e4m3),
            "sgu_s": ((2 * d.sh_inter // 32, HIDDEN // 32), u8),
            "sw2": ((HIDDEN, d.sh_inter), e4m3),
            "sw2_s": ((HIDDEN // 32, d.sh_inter // 32), u8),
        }
        _check_tensors(self, "MoE", want)


class DSV41MonoLayer:
    """The mono decode runner of one TP rank (``group``: a gloo group for the peer handle exchange)."""

    def __init__(self, tp: int, rank: int, group, device: torch.device | str = "cuda"):
        _ensure_writable_flydsl_cache()
        from .common.peer_memory import PeerBuffer

        self.tp, self.rank = tp, rank
        self.d = Dims(tp)
        dev = torch.device(device)
        if dev.index is None:
            dev = torch.device(dev.type, torch.cuda.current_device())
        self.device = dev
        self._scratch: dict[int, torch.Tensor] = {}
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        self.epoch = torch.zeros(EPOCH_WORDS, dtype=torch.int32, device=dev)
        self.q = torch.zeros(
            MAX_TOKENS, self.d.heads, HEAD_DIM, dtype=torch.bfloat16, device=dev
        )
        # K1's outputs: the attention seam's residual and mixes
        self.res_mid = torch.zeros(
            MAX_TOKENS, HC, HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self.post_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        self.comb_a = torch.zeros(MAX_TOKENS, HC, HC, dtype=torch.float32, device=dev)
        self.pre_a = torch.zeros(MAX_TOKENS, HC, dtype=torch.float32, device=dev)
        # the keys K2 looked up, as vLLM's key table: KT [MAX_TOKENS, KEYS] (window keys, then compressed),
        # KLEN = key counts, then compressed counts
        self.kt = torch.zeros(MAX_TOKENS * KEYS, dtype=torch.int32, device=dev)
        self.klen = torch.zeros(2 * MAX_TOKENS, dtype=torch.int32, device=dev)
        # a layer without compressed keys: an empty list a row
        self._no_comp = torch.full(
            (MAX_TOKENS, TOPK), -1, dtype=torch.int32, device=dev
        )
        self._no_comp_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32, device=dev)
        # the record an absent key reads (zero scores masked, zero values)
        self._zero_rec = torch.zeros(1024, dtype=torch.uint8, device=dev)
        # the QK scale 1 / sqrt(512) times log2(e) (the softmax runs in base 2), as f32 bits
        self._qk_scale = (
            torch.tensor([HEAD_DIM**-0.5 * _LOG2E], dtype=torch.float32)
            .view(torch.int32)
            .item()
        )
        self.peer = PeerBuffer(2 * peer_half_bytes(tp), group, rank, tp, dev)
        self.peer.bytes.zero_()
        self._kernels: dict = {}

    def scratch(self, tokens: int) -> torch.Tensor:
        """Return this width's scratch, allocated before graph capture."""
        buf = self._scratch.get(tokens)
        if buf is None:
            no_capture(f"step width {tokens}")
            buf = torch.zeros(
                scratch_bytes(tokens, self.tp), dtype=torch.uint8, device=self.device
            )
            self._scratch[tokens] = buf
        return buf

    @staticmethod
    def supports(tokens: int) -> bool:
        return 1 <= tokens <= MAX_TOKENS

    def ffn_kernel(self, tokens: int):
        key = (tokens, "ffn")
        if key not in self._kernels:
            no_capture(f"FFN launch width {tokens}")
            self._kernels[key] = build_mono_ffn(MonoBuild(tokens, self.tp))
        return self._kernels[key]

    def k2_kernel(self, tokens: int):
        """K2 for ``tokens`` rows: every compress ratio's (the ratio reaches it only through the lists and the cache)."""
        key = (tokens, "k2")
        if key not in self._kernels:
            no_capture(f"K2 width {tokens}")
            self._kernels[key] = build_mono_k2(MonoBuild(tokens, self.tp))
        return self._kernels[key]

    @staticmethod
    def _outs(M: int, residual: torch.Tensor) -> tuple:
        dev = residual.device
        return (
            torch.empty(M, HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.empty_like(residual),
            torch.empty(M, HC, 1, dtype=torch.float32, device=dev),
            torch.empty(M, HC, HC, dtype=torch.float32, device=dev),
            torch.empty(M, HC, dtype=torch.float32, device=dev),
        )

    def ffn(
        self,
        w: MonoLayerWeights,
        part: torch.Tensor,  # [M, 5120] bf16: this rank's unreduced wo_b output
        residual: torch.Tensor,  # [M, 4, 5120] bf16, after the attention seam
        post_mix: torch.Tensor,  # [M, 4, 1] f32, the attention seam's
        res_mix: torch.Tensor,  # [M, 4, 4] f32
        pre_mix: torch.Tensor,  # [M, 4] f32
        outs: tuple | None = None,
    ):
        """Run the FFN launch: attention TP reduce, FFN seam, MoE and all-reduce."""
        M = part.shape[0]
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "FFN",
            M,
            self.device,
            {
                "part": (part, HIDDEN, bf),
                "residual": (residual, HC * HIDDEN, bf),
                "post": (post_mix, HC, f32),
                "comb": (res_mix, HC * HC, f32),
                "pre": (pre_mix, HC, f32),
            },
        )
        if outs is None:
            outs = self._outs(M, residual)
        out, res_out, post_out, comb_out, pre_out = outs
        check_rows(
            "FFN",
            M,
            self.device,
            {
                "out": (out, HIDDEN, bf),
                "res_out": (res_out, HC * HIDDEN, bf),
                "post_out": (post_out, HC, f32),
                "comb_out": (comb_out, HC * HC, f32),
                "pre_out": (pre_out, HC, f32),
            },
        )
        self.ffn_kernel(M)(
            part.data_ptr(),
            residual.data_ptr(),
            post_mix.data_ptr(),
            res_mix.data_ptr(),
            pre_mix.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return outs

    def _check_k2_inputs(self, M, lists, lens, caches, positions):
        """Raise unless K2's key lists, counts, caches and positions are laid out as its raw loads read them."""

        def bad(name, t, want):
            raise RuntimeError(
                f"K2 {name}: {tuple(t.shape)} {t.dtype} strides {t.stride()} {t.device}, want {want}"
            )

        for name, (t, width) in lists.items():
            if not (
                t.dtype == torch.int32
                and t.dim() == 2
                and t.shape[0] >= M
                and t.shape[1] == width
                and t.is_contiguous()
                and t.device == self.device
            ):
                bad(name, t, f"[>= {M}, {width}] contiguous int32")
        for name, (t, dtype) in {**lens, "positions": (positions, torch.int64)}.items():
            if not (
                t.dtype == dtype
                and t.is_contiguous()
                and t.numel() >= M
                and t.device == self.device
            ):
                bad(name, t, f">= {M} contiguous {dtype}")
        for name, t in caches.items():
            if not (
                t.dtype == torch.uint8
                and t.dim() == 3
                and t.shape[2] == RECORD
                and t.stride(0) >= t.shape[1] * RECORD
                and t.stride(1) == RECORD
                and t.stride(2) == 1
                and t.device == self.device
            ):
                bad(name, t, f"[pages, rows, {RECORD}] uint8 records")

    def k2(
        self,
        w: MonoLayerWeights,
        a: AttnWeights,
        rows: int,
        positions: torch.Tensor,  # [>= M] int64
        swa_cache: torch.Tensor,  # [pages, rows, 584] uint8: the SWA pool's records
        swa_indices: torch.Tensor,  # [>= M, 128] int32: window slots, -1 none
        swa_lens: torch.Tensor,  # [>= M] int32: their count
        comp_cache: torch.Tensor | None = None,  # the compressed pool's (ratio 1 / 2)
        comp_indices: torch.Tensor | None = None,  # [>= M, 512] int32, -1 none
        comp_lens: torch.Tensor | None = None,  # [>= M] int32
        outs: tuple | None = None,
    ):
        """Run K2 on the ``rows`` rows of K1's outputs held here (q, res_mid, post_a, comb_a, pre_a): the attention
        over each token's listed keys, wo_a, wo_b, their TP reduce, the FFN seam, the MoE and its all-reduce.
        Return (MoE output, residual, post, comb, pre)."""
        M = rows
        if not self.supports(M):
            raise RuntimeError(f"DSv4.1 mono decode: {M} rows")
        caches = {"swa_cache": swa_cache}
        if a.ratio:
            if comp_cache is None or comp_indices is None or comp_lens is None:
                raise RuntimeError(
                    f"K2 ratio {a.ratio} needs the compressed cache, its slots and their counts"
                )
            caches["comp_cache"] = comp_cache
            comp, comp_stride, comp_block = (
                comp_cache,
                comp_cache.stride(0),
                comp_cache.shape[1],
            )
        else:
            comp, comp_stride, comp_block = self._zero_rec, 0, 1
            comp_indices, comp_lens = self._no_comp, self._no_comp_lens
        self._check_k2_inputs(
            M,
            {
                "swa_indices": (swa_indices, WINDOW),
                "comp_indices": (comp_indices, TOPK),
            },
            {
                "swa_lens": (swa_lens, torch.int32),
                "comp_lens": (comp_lens, torch.int32),
            },
            caches,
            positions,
        )
        if outs is None:
            outs = self._outs(M, self.res_mid[:M])
        out, res_out, post_out, comb_out, pre_out = outs
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "K2",
            M,
            self.device,
            {
                "out": (out, HIDDEN, bf),
                "res_out": (res_out, HC * HIDDEN, bf),
                "post_out": (post_out, HC, f32),
                "comb_out": (comb_out, HC * HC, f32),
                "pre_out": (pre_out, HC, f32),
            },
        )
        self.k2_kernel(M)(
            self.q.data_ptr(),
            swa_cache.data_ptr(),
            swa_cache.stride(0),
            swa_cache.shape[1],
            comp.data_ptr(),
            comp_stride,
            comp_block,
            swa_indices.data_ptr(),
            swa_lens.data_ptr(),
            comp_indices.data_ptr(),
            comp_lens.data_ptr(),
            self.kt.data_ptr(),
            self.klen.data_ptr(),
            positions.data_ptr(),
            a.attn_sink.data_ptr(),
            self._qk_scale,
            a.cos_sin.data_ptr(),
            a.wo_a.data_ptr(),
            a.wo_a_scale.data_ptr(),
            a.wo_b.data_ptr(),
            a.wo_b_scale.data_ptr(),
            self._zero_rec.data_ptr(),
            self.res_mid.data_ptr(),
            self.post_a.data_ptr(),
            self.comb_a.data_ptr(),
            self.pre_a.data_ptr(),
            w.hc_ffn_fn.data_ptr(),
            w.hc_ffn_scale.data_ptr(),
            w.hc_ffn_base.data_ptr(),
            w.ffn_norm.data_ptr(),
            res_out.data_ptr(),
            post_out.data_ptr(),
            comb_out.data_ptr(),
            pre_out.data_ptr(),
            w.gate_w.data_ptr(),
            w.bias.data_ptr(),
            w.w13.data_ptr(),
            w.w13_s.data_ptr(),
            w.w2.data_ptr(),
            w.w2_s.data_ptr(),
            w.sgu.data_ptr(),
            w.sgu_s.data_ptr(),
            w.sw2.data_ptr(),
            w.sw2_s.data_ptr(),
            out.data_ptr(),
            self.scratch(M).data_ptr(),
            self.peer.local,
            self.peer.addresses.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return outs
