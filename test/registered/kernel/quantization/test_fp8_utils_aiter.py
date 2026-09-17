# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import unittest

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.quantization.fp8_utils import (
    shuffle_aiter_fp8_weight,
    unshuffle_aiter_fp8_weight,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(is_hip(), "requires ROCm AITER")
class TestAiterFp8Utils(CustomTestCase):
    def test_unshuffle_weight_round_trip(self):
        from aiter.ops.shuffle import shuffle_weight

        for shape in ((32, 64), (2, 32, 64)):
            with self.subTest(shape=shape):
                logical = (
                    torch.arange(
                        torch.Size(shape).numel(), device="cuda", dtype=torch.float32
                    )
                    .remainder(7)
                    .to(torch.float8_e4m3fn)
                    .reshape(shape)
                )
                shuffled = shuffle_weight(logical, layout=(16, 16))

                torch.testing.assert_close(
                    unshuffle_aiter_fp8_weight(shuffled), logical
                )

    def test_torch_shuffle_matches_aiter(self):
        from aiter.ops.shuffle import shuffle_weight

        for shape in ((32, 64), (2, 32, 64)):
            with self.subTest(shape=shape):
                logical = (
                    torch.arange(
                        torch.Size(shape).numel(), device="cuda", dtype=torch.float32
                    )
                    .remainder(7)
                    .to(torch.float8_e4m3fn)
                    .reshape(shape)
                )

                torch.testing.assert_close(
                    shuffle_aiter_fp8_weight(logical),
                    shuffle_weight(logical, layout=(16, 16)),
                )

    def test_cp_column_physical_slices_keep_aiter_layout(self):
        from aiter.ops.shuffle import shuffle_weight

        tp_size = 4
        n, k = 64, 128
        logical_weight = (
            torch.arange(n * k, device="cuda", dtype=torch.float32)
            .remainder(7)
            .to(torch.float8_e4m3fn)
            .reshape(n, k)
        )
        shuffled_weight = shuffle_weight(logical_weight, layout=(16, 16))

        for rank in range(tp_size):
            context = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
            context.decode_tp_rank = rank
            context.decode_tp_size = tp_size
            physical_slice = context._slice(shuffled_weight, dim=0)
            logical_slice = context._slice(logical_weight, dim=0)

            torch.testing.assert_close(
                physical_slice,
                shuffle_weight(logical_slice, layout=(16, 16)),
            )

    def test_cp_row_shards_reconstruct_logical_computation(self):
        from aiter.ops.shuffle import shuffle_weight

        tp_size = 4
        block_size = 128
        n, k = 256, 512
        logical_weight = (
            torch.arange(n * k, device="cuda", dtype=torch.float32).remainder(7) - 3
        ).to(torch.float8_e4m3fn)
        logical_weight = logical_weight.reshape(n, k)
        shuffled_weight = shuffle_weight(logical_weight, layout=(16, 16))
        weight_scale = (
            torch.arange(
                (n // block_size) * (k // block_size),
                device="cuda",
                dtype=torch.float32,
            )
            .reshape(n // block_size, k // block_size)
            .add_(1)
            .div_(8)
        )
        inputs = (
            torch.arange(3 * k, device="cuda", dtype=torch.float32).remainder(11) - 5
        ).reshape(3, k)

        dequant_weight = logical_weight.float() * weight_scale.repeat_interleave(
            block_size, 0
        ).repeat_interleave(block_size, 1)
        naive_logical_shards = []
        partials = []
        logical_shards = []
        scale_shards = []
        for rank in range(tp_size):
            context = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
            context.decode_tp_rank = rank
            context.decode_tp_size = tp_size

            naive_shard = context._slice(shuffled_weight, dim=1)
            naive_logical_shards.append(unshuffle_aiter_fp8_weight(naive_shard))

            shuffled_shard = context._slice_aiter_bpreshuffled_weight(
                shuffled_weight, dim=1
            )
            logical_shard = unshuffle_aiter_fp8_weight(shuffled_shard)
            scale_shard = context._slice(weight_scale, dim=1)
            input_shard = context._slice(inputs, dim=1)
            expected_weight = context._slice(logical_weight, dim=1)

            torch.testing.assert_close(logical_shard, expected_weight)
            torch.testing.assert_close(
                shuffled_shard,
                shuffle_weight(expected_weight, layout=(16, 16)),
            )

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
        torch.testing.assert_close(
            sum(partials), inputs @ dequant_weight.t(), rtol=1e-5, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
