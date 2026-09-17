import unittest

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.quantization.fp8_utils import (
    shuffle_aiter_fp8_weight,
    unshuffle_aiter_fp8_weight,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _context(rank: int, size: int) -> CpDecodeAttnTpContext:
    context = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
    context.decode_tp_rank = rank
    context.decode_tp_size = size
    context.use_decode_attn_tp = False
    context._slice_cache = {}
    return context


class TestCpDecodeAttnTpSlicing(CustomTestCase):
    def test_plain_row_weight_shards_reconstruct_matmul(self):
        tp_size = 4
        weight = torch.arange(12 * 16, dtype=torch.float32).reshape(12, 16) / 100
        inputs = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16) / 10

        partials = []
        for rank in range(tp_size):
            context = _context(rank, tp_size)
            weight_shard = context._slice(weight, dim=1)
            input_shard = context._slice(inputs, dim=1)
            self.assertTrue(weight_shard.is_contiguous())
            partials.append(input_shard @ weight_shard.t())

        torch.testing.assert_close(sum(partials), inputs @ weight.t())

    def test_plain_column_weight_shards_reconstruct_matmul(self):
        tp_size = 4
        weight = torch.arange(16 * 12, dtype=torch.float32).reshape(16, 12) / 100
        inputs = torch.arange(3 * 12, dtype=torch.float32).reshape(3, 12) / 10

        outputs = []
        for rank in range(tp_size):
            context = _context(rank, tp_size)
            outputs.append(inputs @ context._slice(weight, dim=0).t())

        torch.testing.assert_close(torch.cat(outputs, dim=1), inputs @ weight.t())

    def test_activate_restores_and_reuses_cached_pointer(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16),
            requires_grad=False,
        )
        context = _context(rank=2, size=4)
        original = layer.weight.detach().clone()
        original_pointer = layer.weight.data_ptr()

        context._activate(layer, "weight", dim=1)
        shard_pointer = layer.weight.data_ptr()
        torch.testing.assert_close(layer.weight, original[:, 8:12])

        context._restore(layer, "weight")
        self.assertEqual(layer.weight.data_ptr(), original_pointer)
        torch.testing.assert_close(layer.weight, original)

        context._activate(layer, "weight", dim=1)
        self.assertEqual(layer.weight.data_ptr(), shard_pointer)
        context._restore(layer, "weight")
        self.assertEqual(layer.weight.data_ptr(), original_pointer)

    def test_bpreshuffled_activate_uses_logical_slice_and_cached_pointer(self):
        logical_weight = torch.arange(32 * 128, dtype=torch.int64).to(torch.uint8)
        logical_weight = logical_weight.reshape(32, 128)
        shuffled_weight = shuffle_aiter_fp8_weight(logical_weight)
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(shuffled_weight, requires_grad=False)
        layer.aiter_bpreshuffled = True
        context = _context(rank=2, size=4)
        original_pointer = layer.weight.data_ptr()

        context._activate(layer, "weight", dim=1)
        shard_pointer = layer.weight.data_ptr()
        torch.testing.assert_close(
            unshuffle_aiter_fp8_weight(layer.weight), logical_weight[:, 64:96]
        )

        context._restore(layer, "weight")
        self.assertEqual(layer.weight.data_ptr(), original_pointer)
        torch.testing.assert_close(layer.weight, shuffled_weight)

        context._activate(layer, "weight", dim=1)
        self.assertEqual(layer.weight.data_ptr(), shard_pointer)
        context._restore(layer, "weight")
        self.assertEqual(layer.weight.data_ptr(), original_pointer)

    def test_bpreshuffled_row_weight_and_scales_reconstruct_matmul(self):
        tp_size = 4
        block_size = 128
        n, k = 256, 512
        logical_weight = (
            torch.arange(n * k, dtype=torch.int64).remainder(7).to(torch.uint8)
        ).reshape(n, k)
        shuffled_weight = shuffle_aiter_fp8_weight(logical_weight)
        weight_scale = (
            torch.arange((n // block_size) * (k // block_size), dtype=torch.float32)
            .reshape(n // block_size, k // block_size)
            .add_(1)
            .div_(8)
        )
        dequant_weight = logical_weight.float() * weight_scale.repeat_interleave(
            block_size, 0
        ).repeat_interleave(block_size, 1)
        inputs = torch.arange(3 * k, dtype=torch.float32).reshape(3, k) / 100

        naive_logical_shards = []
        partials = []
        logical_shards = []
        scale_shards = []
        for rank in range(tp_size):
            context = _context(rank, tp_size)
            naive_shard = context._slice(shuffled_weight, dim=1)
            naive_logical_shards.append(unshuffle_aiter_fp8_weight(naive_shard))

            shuffled_shard = context._slice_aiter_bpreshuffled_weight(
                shuffled_weight, dim=1
            )
            logical_shard = unshuffle_aiter_fp8_weight(shuffled_shard)
            scale_shard = context._slice(weight_scale, dim=1)
            input_shard = context._slice(inputs, dim=1)

            expected_weight = context._slice(logical_weight, dim=1)
            expected_scale = context._slice(weight_scale, dim=1)
            torch.testing.assert_close(logical_shard, expected_weight)
            torch.testing.assert_close(scale_shard, expected_scale)

            local_dequant = logical_shard.float() * scale_shard.repeat_interleave(
                block_size, 0
            ).repeat_interleave(block_size, 1)
            partials.append(input_shard @ local_dequant.t())
            logical_shards.append(logical_shard)
            scale_shards.append(scale_shard)

        self.assertFalse(
            torch.equal(torch.cat(naive_logical_shards, dim=1), logical_weight),
            "the fixture must detect naive physical dim-1 slicing",
        )
        torch.testing.assert_close(torch.cat(logical_shards, dim=1), logical_weight)
        torch.testing.assert_close(torch.cat(scale_shards, dim=1), weight_scale)
        torch.testing.assert_close(sum(partials), inputs @ dequant_weight.t())


if __name__ == "__main__":
    unittest.main()
