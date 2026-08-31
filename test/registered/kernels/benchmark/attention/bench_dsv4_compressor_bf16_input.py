"""Benchmark DeepSeek-V4 AITER projection plus C4/C128 compression.

Run on one MI355X with ``SGLANG_USE_AITER=1``. Both measured CUDA/HIP graphs
contain the complete projection and compression sequence; the baseline graph
also contains the standalone BF16-to-FP32 projection cast.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.dsv4 import compress_forward, linear_bf16_fp32
from sglang.kernels.ops.attention.dsv4.gemm import (
    aiter_bf16_projection_available,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.deepseek_v4.common import (
    make_legacy_context,
    make_state_pool,
)

register_amd_ci(est_time=30, stage="jit-kernel-benchmark", runner_config="amd")

_MODEL_PROJECTION_COUNT = 91


@dataclass(frozen=True)
class _Shape:
    name: str
    ratio: int
    head_dim: int
    output_dim: int
    layer_count: int

    @property
    def input_dim(self) -> int:
        return self.head_dim * (4 if self.ratio == 4 else 2)

    @property
    def ape_rows(self) -> int:
        return 8 if self.ratio == 4 else 128


_SHAPES = (
    _Shape("C4 core", 4, 512, 2048, 30),
    _Shape("C4 indexer", 4, 128, 512, 30),
    _Shape("C128 core", 128, 512, 1024, 31),
)
assert sum(shape.layer_count for shape in _SHAPES) == _MODEL_PROJECTION_COUNT


@dataclass
class _Arm:
    state: torch.Tensor
    out: torch.Tensor
    run: Callable[[], None]
    graph: torch.cuda.CUDAGraph | None = None


@dataclass(frozen=True)
class _Result:
    shape: _Shape
    hidden_size: int
    batch_size: int
    state_dtype: torch.dtype
    baseline_eager_us: float
    candidate_eager_us: float
    baseline_graph_us: float
    candidate_graph_us: float
    projection_max_diff: float
    output_max_diff: float
    state_max_diff: float

    @property
    def graph_saving_us(self) -> float:
        return self.baseline_graph_us - self.candidate_graph_us


def _require_gfx950() -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("This benchmark requires ROCm on an MI355X/gfx950 GPU.")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if arch.split(":", 1)[0] != "gfx950":
        raise RuntimeError(f"Expected gfx950, got {arch!r}.")
    if not aiter_bf16_projection_available():
        raise RuntimeError("Set SGLANG_USE_AITER=1 before starting Python.")


def _time_eager_us(fn: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        fn()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph_us(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _make_arm(
    shape: _Shape,
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    ape: torch.Tensor,
    plan,
    initial_state: torch.Tensor,
    keep_projection_bf16: bool,
) -> _Arm:
    state = initial_state.clone()
    out = torch.empty((x.shape[0], shape.head_dim), dtype=torch.float32, device="cuda")

    def run() -> None:
        projected = linear_bf16_fp32(
            x,
            weight,
            allow_aiter_bf16_output=keep_projection_bf16,
        )
        compress_forward(
            state,
            projected,
            ape,
            plan,
            head_dim=shape.head_dim,
            compress_ratio=shape.ratio,  # type: ignore[arg-type]
            out=out,
        )

    return _Arm(state=state, out=out, run=run)


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


def _benchmark_shape(
    shape: _Shape,
    *,
    hidden_size: int,
    batch_size: int,
    state_dtype: torch.dtype,
    warmup: int,
    iterations: int,
    seed: int,
) -> _Result:
    torch.manual_seed(seed + hidden_size + batch_size + shape.output_dim)
    x = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(
        (shape.output_dim, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    baseline_projection = linear_bf16_fp32(x, weight)
    candidate_projection = linear_bf16_fp32(x, weight, allow_aiter_bf16_output=True)
    if baseline_projection.dtype != torch.float32:
        raise RuntimeError(
            f"{shape.name} baseline projection is {baseline_projection.dtype}, "
            "expected FP32."
        )
    if candidate_projection.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{shape.name} candidate did not receive AITER's BF16 projection."
        )
    projection_max_diff = _max_abs_diff(baseline_projection, candidate_projection)
    ape = torch.randn(
        (shape.ape_rows, shape.head_dim), dtype=torch.float32, device="cuda"
    )
    context = make_legacy_context(
        bs=batch_size,
        compress_ratio=shape.ratio,  # type: ignore[arg-type]
        head_dim=shape.head_dim,
    )
    plan = context.make_decode_plan(
        torch.full(
            (batch_size,),
            shape.ratio,
            dtype=torch.int64,
            device="cuda",
        )
    )
    initial_state = make_state_pool(context.num_pages, shape.ratio, shape.head_dim).to(
        state_dtype
    )
    initial_state.copy_(
        torch.randn(
            initial_state.shape,
            dtype=torch.bfloat16,
            device=initial_state.device,
        ).to(state_dtype)
    )

    baseline = _make_arm(
        shape,
        x=x,
        weight=weight,
        ape=ape,
        plan=plan,
        initial_state=initial_state,
        keep_projection_bf16=False,
    )
    candidate = _make_arm(
        shape,
        x=x,
        weight=weight,
        ape=ape,
        plan=plan,
        initial_state=initial_state,
        keep_projection_bf16=True,
    )

    baseline.run()
    candidate.run()
    torch.cuda.synchronize()
    baseline_eager_us = _time_eager_us(baseline.run, warmup, iterations)
    candidate_eager_us = _time_eager_us(candidate.run, warmup, iterations)

    baseline.graph = _capture(baseline.run)
    candidate.graph = _capture(candidate.run)
    baseline_graph_us = _time_graph_us(baseline.graph, warmup, iterations)
    candidate_graph_us = _time_graph_us(candidate.graph, warmup, iterations)

    # Mutate the graph input in place and compare one fresh replay. This checks
    # that both captured paths read the stable input address rather than a
    # capture-time value.
    x.copy_(torch.randn_like(x))
    baseline.state.copy_(initial_state)
    candidate.state.copy_(initial_state)
    baseline.graph.replay()
    candidate.graph.replay()
    torch.cuda.synchronize()
    output_max_diff = _max_abs_diff(baseline.out, candidate.out)
    state_max_diff = _max_abs_diff(baseline.state, candidate.state)

    return _Result(
        shape=shape,
        hidden_size=hidden_size,
        batch_size=batch_size,
        state_dtype=state_dtype,
        baseline_eager_us=baseline_eager_us,
        candidate_eager_us=candidate_eager_us,
        baseline_graph_us=baseline_graph_us,
        candidate_graph_us=candidate_graph_us,
        projection_max_diff=projection_max_diff,
        output_max_diff=output_max_diff,
        state_max_diff=state_max_diff,
    )


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_state_dtypes(value: str) -> tuple[torch.dtype, ...]:
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        values = tuple(
            aliases[item.strip().lower()] for item in value.split(",") if item.strip()
        )
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            "state dtypes must be fp32/float32 or bf16/bfloat16"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one state dtype is required")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--hidden-sizes", type=_parse_csv_ints, default=(7168,))
    parser.add_argument("--batch-sizes", type=_parse_csv_ints, default=(1,))
    parser.add_argument(
        "--state-dtypes",
        type=_parse_state_dtypes,
        default=(torch.float32,),
    )
    parser.add_argument("--min-average-saving-us", type=float, default=1.7)
    parser.add_argument("--min-model-saving-ms", type=float, default=0.15)
    parser.add_argument(
        "--no-enforce-thresholds",
        action="store_true",
        help="Report measurements without failing the benchmark threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _require_gfx950()
    results = [
        _benchmark_shape(
            shape,
            hidden_size=hidden_size,
            batch_size=batch_size,
            state_dtype=state_dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        )
        for hidden_size in args.hidden_sizes
        for batch_size in args.batch_sizes
        for state_dtype in args.state_dtypes
        for shape in _SHAPES
    ]

    print("DeepSeek-V4 projection + compressor (microseconds/invocation)")
    for result in results:
        print(
            f"K={result.hidden_size:<4d} BS={result.batch_size:<2d} "
            f"state={str(result.state_dtype).removeprefix('torch.'):<8s} "
            f"{result.shape.name:11s} "
            f"eager={result.baseline_eager_us:.3f}->{result.candidate_eager_us:.3f} "
            f"graph={result.baseline_graph_us:.3f}->{result.candidate_graph_us:.3f} "
            f"save={result.graph_saving_us:.3f} "
            f"projection_diff={result.projection_max_diff:.9g} "
            f"out_diff={result.output_max_diff:.9g} "
            f"state_diff={result.state_max_diff:.9g}"
        )

    missed_thresholds = []
    for hidden_size in args.hidden_sizes:
        for batch_size in args.batch_sizes:
            for state_dtype in args.state_dtypes:
                case_results = [
                    result
                    for result in results
                    if result.hidden_size == hidden_size
                    and result.batch_size == batch_size
                    and result.state_dtype == state_dtype
                ]
                model_saving_us = sum(
                    result.graph_saving_us * result.shape.layer_count
                    for result in case_results
                )
                average_saving_us = model_saving_us / _MODEL_PROJECTION_COUNT
                model_saving_ms = model_saving_us / 1000.0
                per_token_saving_ms = model_saving_ms / batch_size
                print(
                    f"K={hidden_size} BS={batch_size} "
                    f"state={str(state_dtype).removeprefix('torch.')} "
                    f"weighted estimate={model_saving_ms:.6f} ms/step, "
                    f"{per_token_saving_ms:.6f} ms/token, "
                    f"{average_saving_us:.3f} us/invocation average"
                )
                if (
                    average_saving_us < args.min_average_saving_us
                    or model_saving_ms < args.min_model_saving_ms
                ):
                    missed_thresholds.append(
                        (hidden_size, batch_size, state_dtype, model_saving_ms)
                    )

    # The timed arms launch AITER's split-K projection independently. Its BF16
    # result can vary with reduction scheduling, and compressor softmax
    # amplifies that pre-existing input difference. The GPU unit tests enforce
    # bitwise semantic parity by feeding the same BF16 projection to baseline
    # `.float()` and candidate kernels over a complete 128-step cycle; retain
    # these end-to-end values as projection-variance diagnostics only.
    if not args.no_enforce_thresholds and missed_thresholds:
        raise RuntimeError(
            "Candidate missed the pre-full-model gate for: "
            + ", ".join(
                f"K={hidden_size}/BS={batch_size}/{state_dtype}={saving_ms:.6f}ms"
                for hidden_size, batch_size, state_dtype, saving_ms in missed_thresholds
            )
        )


if __name__ == "__main__":
    main()
