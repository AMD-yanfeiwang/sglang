from __future__ import annotations

import torch

_PRODUCTION_VARIANTS = {
    # (ratio, is_indexer, head_dim, rotate, rope_head_dim)
    (4, False, 512, False, 64),
    (4, True, 128, True, 64),
    (128, False, 512, False, 64),
}


def compressor_bf16_input_platform_enabled(
    *, flag: bool, aiter_available: bool, is_gfx950: bool
) -> bool:
    return flag and aiter_available and is_gfx950


def compressor_mode_allows_bf16_input(
    ratio: int,
    *,
    use_online_c128: bool,
    use_online_c128_mtp: bool,
    is_disaggregated: bool,
) -> bool:
    if is_disaggregated:
        return False
    return ratio != 128 or not (use_online_c128 or use_online_c128_mtp)


def compressor_backend_allows_bf16_input(
    state_dtype: torch.dtype,
) -> bool:
    return state_dtype in (torch.float32, torch.bfloat16)


def can_use_compressor_bf16_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    enabled: bool,
    is_decode: bool,
    ratio: int,
    is_indexer: bool,
    head_dim: int,
    rotate: bool,
    rope_head_dim: int,
) -> bool:
    if not enabled or not is_decode:
        return False
    if x.ndim != 2 or weight.ndim != 2 or x.shape[0] == 0:
        return False
    if (ratio, is_indexer, head_dim, rotate, rope_head_dim) not in (
        _PRODUCTION_VARIANTS
    ):
        return False
    expected_output_dim = (4 if ratio == 4 else 2) * head_dim
    hidden_dim = x.shape[1]
    return (
        hidden_dim > 0
        and weight.shape == (expected_output_dim, hidden_dim)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
    )
