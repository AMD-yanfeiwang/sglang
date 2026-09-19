import pytest

from sglang.srt.speculative.ragged_verify import (
    build_ragged_capture_token_buckets,
)


def test_cp8_dspark_uses_dense_aligned_token_tiers():
    buckets = build_ragged_capture_token_buckets(
        request_buckets=[8, 16, 24, 32],
        num_tokens_per_req=7,
        token_alignment=8,
    )

    assert buckets == list(range(8, 225, 8))


def test_unaligned_runtime_keeps_request_derived_tiers():
    buckets = build_ragged_capture_token_buckets(
        request_buckets=[1, 2, 4, 8],
        num_tokens_per_req=7,
        token_alignment=1,
    )

    assert buckets == [7, 14, 28, 56]


@pytest.mark.parametrize(
    ("request_buckets", "num_tokens_per_req"),
    [([], 7), ([1], 0)],
)
def test_invalid_ragged_capture_geometry_is_rejected(
    request_buckets, num_tokens_per_req
):
    with pytest.raises(ValueError):
        build_ragged_capture_token_buckets(
            request_buckets=request_buckets,
            num_tokens_per_req=num_tokens_per_req,
            token_alignment=8,
        )
