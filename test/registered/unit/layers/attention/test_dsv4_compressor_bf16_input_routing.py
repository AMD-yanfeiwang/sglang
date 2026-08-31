from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

sys.modules.setdefault("sgl_kernel", ModuleType("sgl_kernel"))

from sglang.kernels.ops.attention.dsv4 import compress as compress_module
from sglang.kernels.ops.attention.dsv4 import gemm as gemm_module
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import compressor_bf16 as routing_module

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.bfloat16,
        is_cuda: bool = True,
        contiguous: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.is_cuda = is_cuda
        self.device = torch.device(device)
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous


@pytest.mark.parametrize(
    ("ratio", "is_indexer", "head_dim", "output_dim", "rotate"),
    [
        (4, False, 512, 2048, False),
        (4, True, 128, 512, True),
        (128, False, 512, 1024, False),
    ],
)
@pytest.mark.parametrize("batch_size", (1, 2, 8, 64))
@pytest.mark.parametrize("hidden_size", (4096, 5120, 7168))
def test_valid_compressor_projection_shapes_are_eligible(
    ratio: int,
    is_indexer: bool,
    head_dim: int,
    output_dim: int,
    rotate: bool,
    batch_size: int,
    hidden_size: int,
) -> None:
    x = _FakeTensor((batch_size, hidden_size))
    weight = _FakeTensor((output_dim, hidden_size))

    assert routing_module.can_use_compressor_bf16_projection(
        x,
        weight,
        enabled=True,
        is_decode=True,
        ratio=ratio,
        is_indexer=is_indexer,
        head_dim=head_dim,
        rotate=rotate,
        rope_head_dim=64,
    )


@pytest.mark.parametrize(
    ("ratio", "head_dim", "output_dim"),
    [
        (4, 256, 1024),
        (128, 256, 512),
    ],
)
def test_raw_kernel_only_head_shapes_do_not_route_in_production(
    ratio: int,
    head_dim: int,
    output_dim: int,
) -> None:
    assert not routing_module.can_use_compressor_bf16_projection(
        _FakeTensor((8, 4096)),
        _FakeTensor((output_dim, 4096)),
        enabled=True,
        is_decode=True,
        ratio=ratio,
        is_indexer=False,
        head_dim=head_dim,
        rotate=False,
        rope_head_dim=64,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"enabled": False},
        {"is_decode": False},
        {"x": _FakeTensor((7168,))},
        {"x": _FakeTensor((0, 7168))},
        {"x": _FakeTensor((1, 7168), dtype=torch.float32)},
        {"x": _FakeTensor((1, 7168), is_cuda=False)},
        {"x": _FakeTensor((1, 7168), contiguous=False)},
        {"x": _FakeTensor((1, 7168), device="cuda:1")},
        {"weight": _FakeTensor((2048, 4096))},
        {"weight": _FakeTensor((1024, 7168))},
        {"weight": _FakeTensor((2048, 7168), dtype=torch.float32)},
        {"ratio": 128},
        {"ratio": 8},
        {"is_indexer": True},
        {"rotate": True},
        {"rope_head_dim": 32},
        {
            "head_dim": 320,
            "weight": _FakeTensor((1280, 7168)),
        },
    ],
)
def test_projection_shape_fallbacks(override: dict) -> None:
    kwargs = {
        "x": _FakeTensor((1, 7168)),
        "weight": _FakeTensor((2048, 7168)),
        "enabled": True,
        "is_decode": True,
        "ratio": 4,
        "is_indexer": False,
        "head_dim": 512,
        "rotate": False,
        "rope_head_dim": 64,
    }
    kwargs.update(override)
    x = kwargs.pop("x")
    weight = kwargs.pop("weight")

    assert not routing_module.can_use_compressor_bf16_projection(x, weight, **kwargs)


@pytest.mark.parametrize(
    ("flag", "aiter_available", "is_gfx950", "expected"),
    [
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_platform_gate_is_strict(
    flag: bool,
    aiter_available: bool,
    is_gfx950: bool,
    expected: bool,
) -> None:
    assert (
        routing_module.compressor_bf16_input_platform_enabled(
            flag=flag,
            aiter_available=aiter_available,
            is_gfx950=is_gfx950,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("state_dtype", "expected"),
    [
        (torch.float32, True),
        (torch.bfloat16, True),
        (torch.float16, False),
    ],
)
def test_backend_gate_accepts_supported_state_dtypes(
    state_dtype: torch.dtype,
    expected: bool,
) -> None:
    assert routing_module.compressor_backend_allows_bf16_input(state_dtype) is expected


@pytest.mark.parametrize(
    (
        "ratio",
        "use_online_c128",
        "use_online_c128_mtp",
        "is_disaggregated",
        "expected",
    ),
    [
        (4, True, True, False, True),
        (128, False, False, False, True),
        (128, True, False, False, False),
        (128, False, True, False, False),
        (4, False, False, True, False),
        (128, False, False, True, False),
    ],
)
def test_mode_gate_rejects_online_c128_and_disaggregation(
    ratio: int,
    use_online_c128: bool,
    use_online_c128_mtp: bool,
    is_disaggregated: bool,
    expected: bool,
) -> None:
    assert (
        routing_module.compressor_mode_allows_bf16_input(
            ratio,
            use_online_c128=use_online_c128,
            use_online_c128_mtp=use_online_c128_mtp,
            is_disaggregated=is_disaggregated,
        )
        is expected
    )


def test_flag_defaults_off() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("SGLANG_DSV4_COMPRESSOR_BF16_INPUT", None)
        assert envs.SGLANG_DSV4_COMPRESSOR_BF16_INPUT.get() is False


def test_linear_default_keeps_fp32_and_opt_in_returns_aiter_bf16() -> None:
    x = torch.randn(1, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    aiter_output = torch.randn(1, 4, dtype=torch.bfloat16)
    tgemm = SimpleNamespace(mm=Mock(return_value=aiter_output))

    with (
        patch.object(gemm_module, "_use_aiter", True),
        patch.object(gemm_module, "tgemm", tgemm, create=True),
    ):
        baseline = gemm_module.linear_bf16_fp32(x, weight)
        candidate = gemm_module.linear_bf16_fp32(
            x, weight, allow_aiter_bf16_output=True
        )

    assert baseline.dtype == torch.float32
    assert candidate is aiter_output


def test_linear_aiter_off_ignores_bf16_opt_in() -> None:
    x = torch.randn(1, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    expected = torch.randn(1, 4, dtype=torch.float32)

    with (
        patch.object(gemm_module, "_use_aiter", False),
        patch.object(
            gemm_module,
            "_linear_bf16_fp32_cublas",
            return_value=expected,
        ) as fallback,
    ):
        result = gemm_module.linear_bf16_fp32(x, weight, allow_aiter_bf16_output=True)

    assert result is expected
    fallback.assert_called_once_with(x, weight)


def test_compress_routes_bf16_input_fp32_bias_and_fp32_output_independently() -> None:
    state = torch.zeros((2, 4, 2048), dtype=torch.float32)
    projected = torch.zeros((1, 2048), dtype=torch.bfloat16)
    one_up = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    special_bias = torch.tensor([0.1, -0.1, one_up, -0.0], dtype=torch.float32).repeat(
        8 * 512 // 4
    )
    ape = special_bias.view(8, 512)
    plan = compress_module.CompressorDecodePlan(
        4, torch.zeros((1, 16), dtype=torch.uint8)
    )
    module = SimpleNamespace(decode=Mock(), prefill=Mock())

    with patch.object(
        compress_module, "_jit_compress_module", return_value=module
    ) as load_module:
        out = compress_module.compress_forward(
            state,
            projected,
            ape,
            plan,
            head_dim=512,
            compress_ratio=4,
        )

    assert out.dtype == torch.float32
    load_module.assert_called_once_with(
        512,
        torch.float32,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        4,
    )
    assert module.decode.call_args.args[3] is ape


def test_compress_rejects_non_fp32_output_workspace() -> None:
    state = torch.zeros((2, 4, 2048), dtype=torch.float32)
    projected = torch.zeros((1, 2048), dtype=torch.bfloat16)
    ape = torch.zeros((8, 512), dtype=torch.float32)
    plan = compress_module.CompressorDecodePlan(
        4, torch.zeros((1, 16), dtype=torch.uint8)
    )

    with pytest.raises(AssertionError, match="must stay FP32"):
        compress_module.compress_forward(
            state,
            projected,
            ape,
            plan,
            head_dim=512,
            compress_ratio=4,
            out=torch.empty((1, 512), dtype=torch.bfloat16),
        )
