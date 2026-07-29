import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4 import indexer as indexer_module
from sglang.srt.layers.attention.dsv4.indexer import (
    C4IndexerBackendMixin,
    transform_raw_c4_indices_to_page_indices,
    transform_raw_c4_indices_to_page_indices_torch,
    transform_raw_c4_indices_to_page_indices_triton,
)
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=25, suite="stage-b-test-1-gpu-small-amd-mi35x")
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4CSAIndexCacheTransformCPU(CustomTestCase):
    def test_wrapper_falls_back_to_torch_on_cpu(self):
        raw_indices = torch.tensor([[0, 63, 64, -1], [5, 128, 256, 1024]])
        seq_lens = torch.tensor([128, 300])
        page_table = torch.tensor([[10, 11, 12, 13], [20, 21, -1, 23]])
        expected = torch.empty_like(raw_indices)
        actual = torch.empty_like(raw_indices)

        transform_raw_c4_indices_to_page_indices_torch(
            raw_indices, seq_lens, page_table, expected, page_size=64
        )
        transform_raw_c4_indices_to_page_indices(
            raw_indices, seq_lens, page_table, actual, page_size=64
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_page_table_returns_invalid_indices(self):
        raw_indices = torch.tensor([[0, 63, 64], [1, 2, 3]], dtype=torch.int32)
        seq_lens = torch.tensor([128, 64], dtype=torch.int32)
        page_table = torch.empty((2, 0), dtype=torch.int32)
        actual = torch.empty_like(raw_indices)

        transform_raw_c4_indices_to_page_indices(
            raw_indices, seq_lens, page_table, actual, page_size=64
        )

        torch.testing.assert_close(actual, torch.full_like(actual, -1))

    def test_input_validation(self):
        raw_indices = torch.zeros((2, 4), dtype=torch.int32)
        seq_lens = torch.ones(2, dtype=torch.int32)
        page_table = torch.zeros((2, 2), dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "same shape"):
            transform_raw_c4_indices_to_page_indices_torch(
                raw_indices,
                seq_lens,
                page_table,
                torch.empty((2, 3), dtype=torch.int32),
                page_size=64,
            )
        with self.assertRaisesRegex(ValueError, "positive power of two"):
            transform_raw_c4_indices_to_page_indices_torch(
                raw_indices,
                seq_lens,
                page_table,
                torch.empty_like(raw_indices),
                page_size=96,
            )
        with self.assertRaisesRegex(ValueError, "seq_lens must have shape"):
            transform_raw_c4_indices_to_page_indices_torch(
                raw_indices,
                torch.ones((2, 2), dtype=torch.int32),
                page_table,
                torch.empty_like(raw_indices),
                page_size=64,
            )

    def test_hisparse_decode_skips_redundant_page_translation(self):
        backend = C4IndexerBackendMixin()
        backend._update_hisparse_c4_sparse_indices = mock.Mock()
        raw_indices = torch.tensor([[0, 1, -1]], dtype=torch.int32)
        c4_sparse_page_indices = torch.empty_like(raw_indices)
        token_to_kv_pool = types.SimpleNamespace(
            layer_mapping={
                7: types.SimpleNamespace(compress_layer_id=3),
            }
        )
        core_metadata = types.SimpleNamespace(c4_sparse_raw_indices=None)

        with (
            mock.patch.object(
                indexer_module, "maybe_capture_indexer_topk", return_value=raw_indices
            ),
            mock.patch.object(
                indexer_module, "transform_raw_c4_indices_to_page_indices"
            ) as transform,
        ):
            result = backend._forward_c4_indexer_skip_topk(
                c4_indexer=types.SimpleNamespace(layer_id=7),
                forward_batch=types.SimpleNamespace(),
                token_to_kv_pool=token_to_kv_pool,
                indexer_metadata=types.SimpleNamespace(c4_page_size=64),
                core_metadata=core_metadata,
                c4_seq_lens=torch.tensor([3], dtype=torch.int32),
                page_table=torch.tensor([[0]], dtype=torch.int32),
                c4_sparse_page_indices=c4_sparse_page_indices,
                prev_topk_indices=raw_indices,
                return_topk_indices=True,
                hisparse_coordinator=mock.Mock(),
                hisparse_decode=True,
            )

        transform.assert_not_called()
        backend._update_hisparse_c4_sparse_indices.assert_called_once()
        self.assertIs(result, raw_indices)


@unittest.skipIf(not torch.cuda.is_available(), "A CUDA or ROCm GPU is required")
class TestDSV4CSAIndexCacheTransform(CustomTestCase):
    def _assert_triton_matches_torch(
        self,
        raw_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        page_size: int,
    ):
        device = torch.device("cuda")
        raw_indices = raw_indices.to(device=device, dtype=torch.int32)
        page_table = page_table.to(device=device, dtype=torch.int32)
        seq_lens = seq_lens.to(device=device, dtype=torch.int32)
        expected = torch.empty_like(raw_indices)
        actual = torch.empty_like(raw_indices)
        wrapped = torch.empty_like(raw_indices)

        transform_raw_c4_indices_to_page_indices_torch(
            raw_indices, seq_lens, page_table, expected, page_size
        )
        transform_raw_c4_indices_to_page_indices_triton(
            raw_indices, seq_lens, page_table, actual, page_size
        )
        transform_raw_c4_indices_to_page_indices(
            raw_indices, seq_lens, page_table, wrapped, page_size
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(wrapped, expected, rtol=0, atol=0)

    def test_c4_index_transform_triton_matches_torch_basic_cases(self):
        raw_indices = torch.tensor(
            [
                [0, 63, 64, 127, 128, -1, 512],
                [7, 65, 129, 256, 300, 999, -4],
                [1, 2, 3, 4, 5, 6, 7],
            ]
        )
        page_table = torch.tensor(
            [
                [100, 101, 102, -1, 104, 105, 106, 107],
                [200, -1, 202, 203, 204, 205, 206, 207],
                [300, 301, 302, 303, 304, 305, 306, 307],
            ]
        )
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([256, 320, 4]), page_table, page_size=64
        )
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([[256], [320], [4]]), page_table, page_size=64
        )

    def test_c4_index_transform_triton_matches_torch_different_page_size(self):
        raw_indices = torch.tensor(
            [
                [0, 127, 128, 255, 256, -1, 1024],
                [3, 129, 257, 384, 500, 999, -8],
            ]
        )
        page_table = torch.tensor(
            [
                [10, 11, 12, -1, 14, 15, 16, 17],
                [20, -1, 22, 23, 24, 25, 26, 27],
            ]
        )
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([512, 640]), page_table, page_size=128
        )

    def test_c4_index_transform_triton_matches_torch_large_odd_topk(self):
        batch_size = 3
        topk = 513
        page_size = 64
        raw_indices = (
            torch.arange(batch_size * topk, dtype=torch.int32).reshape(batch_size, topk)
            % 700
        )
        raw_indices[0, 17] = -1
        raw_indices[1, 257] = 4096
        raw_indices[2, 512] = 63
        page_table = (
            torch.arange(batch_size * 16, dtype=torch.int32).reshape(batch_size, 16)
            + 100
        )
        page_table[1, 3] = -1
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([640, 768, 256]), page_table, page_size
        )

    def test_c4_index_transform_triton_matches_torch_non_contiguous_inputs(self):
        raw_base = torch.tensor(
            [
                [0, 999, 64, 999, 128, 999, 192, 999],
                [3, 999, 67, 999, 131, 999, 195, 999],
            ]
        )
        page_table_base = torch.tensor(
            [
                [10, 999, 11, 999, 12, 999, 13, 999],
                [20, 999, 21, 999, -1, 999, 23, 999],
            ]
        )
        raw_indices = raw_base[:, ::2]
        page_table = page_table_base[:, ::2]
        self.assertFalse(raw_indices.is_contiguous())
        self.assertFalse(page_table.is_contiguous())
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([256, 256]), page_table, page_size=64
        )

    def test_c4_index_transform_triton_handles_empty_topk(self):
        raw_indices = torch.empty((2, 0), dtype=torch.int32)
        page_table = torch.tensor([[1, 2], [3, 4]])
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([64, 64]), page_table, page_size=64
        )

    def test_c4_index_transform_handles_empty_page_table(self):
        raw_indices = torch.tensor([[0, 63, 64], [1, 2, 3]], dtype=torch.int32)
        page_table = torch.empty((2, 0), dtype=torch.int32)
        self._assert_triton_matches_torch(
            raw_indices, torch.tensor([128, 64]), page_table, page_size=64
        )

    def test_gpu_wrapper_dispatches_to_triton(self):
        raw_indices = torch.tensor([[0, 63, 64, -1]], device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor([128], device="cuda", dtype=torch.int32)
        page_table = torch.tensor([[10, 11]], device="cuda", dtype=torch.int32)
        actual = torch.empty_like(raw_indices)

        with (
            mock.patch.object(
                indexer_module,
                "transform_raw_c4_indices_to_page_indices_triton",
                wraps=transform_raw_c4_indices_to_page_indices_triton,
            ) as triton_impl,
            mock.patch.object(
                indexer_module,
                "transform_raw_c4_indices_to_page_indices_torch",
                wraps=transform_raw_c4_indices_to_page_indices_torch,
            ) as torch_impl,
        ):
            indexer_module.transform_raw_c4_indices_to_page_indices(
                raw_indices, seq_lens, page_table, actual, page_size=64
            )

        triton_impl.assert_called_once()
        torch_impl.assert_not_called()
        torch.testing.assert_close(
            actual,
            torch.tensor([[640, 703, 704, -1]], device="cuda", dtype=torch.int32),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
