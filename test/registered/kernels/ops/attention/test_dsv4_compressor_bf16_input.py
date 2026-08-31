from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import (
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.deepseek_v4.common import (
    make_legacy_context,
    make_state_pool,
    to_seq_extend,
)

register_amd_ci(est_time=300, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DeepSeek V4 compressor BF16-input tests require a GPU.",
)

_SENTINEL = 8192.25


@dataclass(frozen=True)
class _Shape:
    name: str
    ratio: int
    head_dim: int

    @property
    def input_dim(self) -> int:
        return self.head_dim * (4 if self.ratio == 4 else 2)

    @property
    def ape_rows(self) -> int:
        return 8 if self.ratio == 4 else 128


_SHAPES = (
    _Shape("c4_core", 4, 512),
    _Shape("c4_indexer", 4, 128),
    _Shape("c128_core", 128, 512),
)
_GENERIC_SHAPES = _SHAPES + (
    _Shape("c4_core_d256", 4, 256),
    _Shape("c128_core_d256", 128, 256),
)
_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
_STATE_DTYPES = (torch.float32, torch.bfloat16)


def _special_fp32_ape(shape: _Shape) -> torch.Tensor:
    one_up = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    values = torch.tensor(
        [
            0.1,
            -0.1,
            one_up,
            -one_up,
            2.0**-120,
            -(2.0**-120),
            123.4567,
            -87.6543,
            0.0,
            -0.0,
        ],
        dtype=torch.float32,
        device=get_device(),
    )
    count = shape.ape_rows * shape.head_dim
    repeats = (count + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:count].view(shape.ape_rows, shape.head_dim)


def _assert_bitwise(lhs: torch.Tensor, rhs: torch.Tensor, label: str) -> None:
    if torch.equal(lhs, rhs):
        return
    max_diff = (lhs.float() - rhs.float()).abs().max().item()
    raise AssertionError(f"{label} is not bitwise equal; max_abs_diff={max_diff}")


def _store_core_cache(
    compressed: torch.Tensor,
    plan,
    *,
    shape: _Shape,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    out_loc: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    compress_norm_rope_store(
        compressed,
        plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=out_loc,
        kvcache=cache.view(torch.uint8),
        page_size=1,
        bf16_store=True,
    )


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda shape: shape.name)
def test_bf16_projected_input_matches_fp32_decode_cycle(shape: _Shape) -> None:
    """Exercise non-boundaries, boundaries, and the first ring-wrap write."""
    steps = 129
    torch.manual_seed(20260829 + shape.input_dim)
    projected_bf16 = torch.randn(
        (steps, shape.input_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    ape = _special_fp32_ape(shape)
    context = make_legacy_context(
        bs=1,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        head_dim=shape.head_dim,
    )
    baseline_state = make_state_pool(context.num_pages, shape.ratio, shape.head_dim)
    candidate_state = baseline_state.clone()

    num_events = steps // shape.ratio
    baseline_cache = None
    candidate_cache = None
    norm_weight = None
    freqs_cis = None
    if shape.head_dim == 512:
        baseline_cache = torch.full(
            (num_events, shape.head_dim),
            _SENTINEL,
            dtype=torch.bfloat16,
            device=get_device(),
        )
        candidate_cache = baseline_cache.clone()
        norm_weight = torch.randn(
            shape.head_dim, dtype=torch.float32, device=get_device()
        )
        freqs_cis = precompute_freqs_cis(64, steps + 1, 0, 10000, 1, 32, 1).to(
            get_device()
        )

    event_id = 0
    for step in range(1, steps + 1):
        seq_lens = torch.tensor([step], dtype=torch.int64, device=get_device())
        plan = context.make_decode_plan(seq_lens)
        baseline_out = torch.full(
            (1, shape.head_dim),
            _SENTINEL,
            dtype=torch.float32,
            device=get_device(),
        )
        candidate_out = baseline_out.clone()
        projected = projected_bf16[step - 1 : step]

        compress_forward(
            baseline_state,
            projected.float(),
            ape,
            plan,
            head_dim=shape.head_dim,
            compress_ratio=shape.ratio,  # type: ignore[arg-type]
            out=baseline_out,
        )
        compress_forward(
            candidate_state,
            projected,
            ape,
            plan,
            head_dim=shape.head_dim,
            compress_ratio=shape.ratio,  # type: ignore[arg-type]
            out=candidate_out,
        )
        torch.cuda.synchronize()

        _assert_bitwise(baseline_state, candidate_state, f"{shape.name} state@{step}")
        _assert_bitwise(baseline_out, candidate_out, f"{shape.name} output@{step}")
        if step % shape.ratio:
            assert torch.all(baseline_out == _SENTINEL)
            continue

        assert baseline_out.dtype == candidate_out.dtype == torch.float32
        if baseline_cache is not None:
            assert candidate_cache is not None
            assert norm_weight is not None
            assert freqs_cis is not None
            out_loc = torch.tensor([event_id], dtype=torch.int64, device=get_device())
            _store_core_cache(
                baseline_out,
                plan,
                shape=shape,
                norm_weight=norm_weight,
                freqs_cis=freqs_cis,
                out_loc=out_loc,
                cache=baseline_cache,
            )
            _store_core_cache(
                candidate_out,
                plan,
                shape=shape,
                norm_weight=norm_weight,
                freqs_cis=freqs_cis,
                out_loc=out_loc,
                cache=candidate_cache,
            )
            torch.cuda.synchronize()
            _assert_bitwise(
                baseline_cache,
                candidate_cache,
                f"{shape.name} norm-rope cache@{step}",
            )
        event_id += 1

    if baseline_cache is None:
        return
    assert candidate_cache is not None
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
        _sparse_attn_v4_paged_decode_triton,
    )

    query = torch.randn(
        (1, 16, shape.head_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    indices = torch.arange(num_events, dtype=torch.int32, device=get_device())
    indptr = torch.tensor([0, num_events], dtype=torch.int32, device=get_device())
    sink = torch.randn(16, dtype=torch.float32, device=get_device())
    baseline_attention = _sparse_attn_v4_paged_decode_triton(
        query,
        baseline_cache,
        indices,
        indptr,
        sink,
        shape.head_dim**-0.5,
        kv_splits=1,
    )
    candidate_attention = _sparse_attn_v4_paged_decode_triton(
        query,
        candidate_cache,
        indices,
        indptr,
        sink,
        shape.head_dim**-0.5,
        kv_splits=1,
    )
    torch.cuda.synchronize()
    _assert_bitwise(
        baseline_attention,
        candidate_attention,
        f"{shape.name} downstream attention",
    )


@pytest.mark.parametrize("shape", _GENERIC_SHAPES, ids=lambda shape: shape.name)
@pytest.mark.parametrize("batch_size", _BATCH_SIZES, ids=lambda bs: f"bs{bs}")
@pytest.mark.parametrize(
    "state_dtype",
    _STATE_DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
def test_bf16_projected_input_matches_fp32_batched_decode(
    shape: _Shape,
    batch_size: int,
    state_dtype: torch.dtype,
) -> None:
    """Cover mixed boundary states for serving batch sizes and both state dtypes."""
    torch.manual_seed(20260830 + batch_size + shape.input_dim)
    context = make_legacy_context(
        bs=batch_size,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        head_dim=shape.head_dim,
    )
    initial_state = torch.randn(
        (context.num_pages, shape.ratio, shape.input_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    ).to(state_dtype)
    baseline_state = initial_state.clone()
    candidate_state = initial_state.clone()
    projected = torch.randn(
        (batch_size, shape.input_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    seq_pattern = (
        shape.ratio,
        1,
        shape.ratio - 1,
        shape.ratio + 1,
        2 * shape.ratio,
        2 * shape.ratio + 1,
        129,
    )
    seq_lens = torch.tensor(
        [seq_pattern[i % len(seq_pattern)] for i in range(batch_size)],
        dtype=torch.int64,
        device=get_device(),
    )
    plan = context.make_decode_plan(seq_lens)
    baseline_out = torch.full(
        (batch_size, shape.head_dim),
        _SENTINEL,
        dtype=torch.float32,
        device=get_device(),
    )
    candidate_out = baseline_out.clone()
    ape = _special_fp32_ape(shape)

    compress_forward(
        baseline_state,
        projected.float(),
        ape,
        plan,
        head_dim=shape.head_dim,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        out=baseline_out,
    )
    compress_forward(
        candidate_state,
        projected,
        ape,
        plan,
        head_dim=shape.head_dim,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        out=candidate_out,
    )
    torch.cuda.synchronize()

    label = f"{shape.name}/bs{batch_size}/{state_dtype}"
    _assert_bitwise(baseline_state, candidate_state, f"{label} state")
    _assert_bitwise(baseline_out, candidate_out, f"{label} output")


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda shape: shape.name)
def test_true_fp32_history_is_not_a_valid_bf16_input_oracle(shape: _Shape) -> None:
    """Document why heterogeneous/imported FP32 state must remain fail-closed."""
    torch.manual_seed(20260905 + shape.input_dim)
    context = make_legacy_context(
        bs=1,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        head_dim=shape.head_dim,
    )
    baseline_state = torch.randn(
        (context.num_pages, shape.ratio, shape.input_dim),
        dtype=torch.float32,
        device=get_device(),
    )
    assert not torch.equal(baseline_state, baseline_state.bfloat16().float())
    candidate_state = baseline_state.clone()
    projected = torch.randn(
        (1, shape.input_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    seq_len = 8 if shape.ratio == 4 else 128
    plan = context.make_decode_plan(
        torch.tensor([seq_len], dtype=torch.int64, device=get_device())
    )
    ape = _special_fp32_ape(shape)
    baseline_out = torch.empty(
        (1, shape.head_dim), dtype=torch.float32, device=get_device()
    )
    candidate_out = torch.empty_like(baseline_out)

    compress_forward(
        baseline_state,
        projected.float(),
        ape,
        plan,
        head_dim=shape.head_dim,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        out=baseline_out,
    )
    compress_forward(
        candidate_state,
        projected,
        ape,
        plan,
        head_dim=shape.head_dim,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        out=candidate_out,
    )
    torch.cuda.synchronize()

    assert not torch.equal(baseline_out, candidate_out)


@pytest.mark.parametrize("shape", _GENERIC_SHAPES, ids=lambda shape: shape.name)
@pytest.mark.parametrize("batch_size", (1, 4), ids=lambda bs: f"bs{bs}")
@pytest.mark.parametrize(
    "state_dtype",
    _STATE_DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
def test_bf16_projected_input_matches_fp32_prefill(
    shape: _Shape,
    batch_size: int,
    state_dtype: torch.dtype,
) -> None:
    """Validate the generic prefill kernel even though production routing is decode-only."""
    torch.manual_seed(20260831 + batch_size + shape.input_dim)
    context = make_legacy_context(
        bs=batch_size,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        head_dim=shape.head_dim,
    )
    seq_extend_pairs = [
        (shape.ratio * (1 + i % 2), shape.ratio * (1 + i % 2))
        for i in range(batch_size)
    ]
    seq_lens, extend_lens, num_q_tokens = to_seq_extend(seq_extend_pairs)
    plan = context.make_prefill_plan(seq_lens, extend_lens, num_q_tokens)
    baseline_state = torch.zeros(
        (context.num_pages, shape.ratio, shape.input_dim),
        dtype=state_dtype,
        device=get_device(),
    )
    candidate_state = baseline_state.clone()
    projected = torch.randn(
        (num_q_tokens, shape.input_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    ape = _special_fp32_ape(shape)

    baseline_out = compress_forward(
        baseline_state,
        projected.float(),
        ape,
        plan,
        head_dim=shape.head_dim,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
    )
    candidate_out = compress_forward(
        candidate_state,
        projected,
        ape,
        plan,
        head_dim=shape.head_dim,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
    )
    torch.cuda.synchronize()

    label = f"{shape.name}/prefill-bs{batch_size}/{state_dtype}"
    _assert_bitwise(baseline_state, candidate_state, f"{label} state")
    _assert_bitwise(baseline_out, candidate_out, f"{label} output")


def _capture_graph(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


@pytest.mark.parametrize("shape", _GENERIC_SHAPES, ids=lambda shape: shape.name)
@pytest.mark.parametrize("batch_size", (1, 8, 32, 64), ids=lambda bs: f"bs{bs}")
@pytest.mark.parametrize(
    "state_dtype",
    _STATE_DTYPES,
    ids=lambda dtype: str(dtype).removeprefix("torch."),
)
def test_bf16_projected_input_graph_replay_with_mutated_inputs(
    shape: _Shape,
    batch_size: int,
    state_dtype: torch.dtype,
) -> None:
    context = make_legacy_context(
        bs=batch_size,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        head_dim=shape.head_dim,
    )
    initial_plan = context.make_decode_plan(
        torch.ones(batch_size, dtype=torch.int64, device=get_device())
    )
    baseline_plan = type(initial_plan)(
        initial_plan.compress_ratio, initial_plan.plan_d.clone()
    )
    candidate_plan = type(initial_plan)(
        initial_plan.compress_ratio, initial_plan.plan_d.clone()
    )
    baseline_state = make_state_pool(context.num_pages, shape.ratio, shape.head_dim).to(
        state_dtype
    )
    candidate_state = baseline_state.clone()
    baseline_source = torch.zeros(
        (batch_size, shape.input_dim),
        dtype=torch.bfloat16,
        device=get_device(),
    )
    candidate_source = baseline_source.clone()
    baseline_out = torch.full(
        (batch_size, shape.head_dim),
        _SENTINEL,
        dtype=torch.float32,
        device=get_device(),
    )
    candidate_out = baseline_out.clone()
    ape = _special_fp32_ape(shape)

    def baseline_fn() -> None:
        compress_forward(
            baseline_state,
            baseline_source.float(),
            ape,
            baseline_plan,
            head_dim=shape.head_dim,
            compress_ratio=shape.ratio,  # type: ignore[arg-type]
            out=baseline_out,
        )

    def candidate_fn() -> None:
        compress_forward(
            candidate_state,
            candidate_source,
            ape,
            candidate_plan,
            head_dim=shape.head_dim,
            compress_ratio=shape.ratio,  # type: ignore[arg-type]
            out=candidate_out,
        )

    baseline_graph = _capture_graph(baseline_fn)
    candidate_graph = _capture_graph(candidate_fn)
    baseline_state.zero_()
    candidate_state.zero_()

    allocated_before = torch.cuda.memory_allocated()
    candidate_graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated_before

    replay_steps = (1, 3, 4, 8, 9) if shape.ratio == 4 else (1, 64, 127, 128, 129)
    generator = torch.Generator(device=get_device()).manual_seed(
        20260829 + shape.input_dim
    )
    for step in replay_steps:
        row = torch.randn(
            (batch_size, shape.input_dim),
            dtype=torch.bfloat16,
            device=get_device(),
            generator=generator,
        )
        baseline_source.copy_(row)
        candidate_source.copy_(row)
        plan = context.make_decode_plan(
            torch.full(
                (batch_size,),
                step,
                dtype=torch.int64,
                device=get_device(),
            )
        )
        baseline_plan.copy_(plan)
        candidate_plan.copy_(plan)
        baseline_out.fill_(_SENTINEL)
        candidate_out.fill_(_SENTINEL)

        baseline_graph.replay()
        candidate_graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(
            baseline_state,
            candidate_state,
            f"{shape.name}/bs{batch_size}/{state_dtype} graph state@{step}",
        )
        _assert_bitwise(
            baseline_out,
            candidate_out,
            f"{shape.name}/bs{batch_size}/{state_dtype} graph output@{step}",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
