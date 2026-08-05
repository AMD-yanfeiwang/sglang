# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0

"""The unified_kv DSV4 buffers must be allocated at a dtype the caller chose.

They used to be hardcoded ``torch.bfloat16`` while ``DeepSeekV4TokenToKVPool``
was handed ``--kv-cache-dtype`` and passed it to every *other* sub-pool, so an
fp8 deployment silently got 1024-byte KV rows where the flag (and the memory
planner in ``pool_configurator``) implies 584.
"""

import ast
import contextlib
import inspect
import textwrap
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    UNIFIED_KV_DTYPES,
    UNIFIED_KV_FALLBACK_DTYPE,
    DeepSeekV4UnifiedKVPool,
    resolve_unified_kv_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2)


class _NoopMemorySaverAdapter:
    @contextlib.contextmanager
    def region(self, _tag):
        yield


def _make_pool(*, dtype: torch.dtype) -> DeepSeekV4UnifiedKVPool:
    """A two-layer (one c4, one c128) pool small enough to build on CPU."""
    return DeepSeekV4UnifiedKVPool(
        stage_ratios=[4, 128],
        num_slots=2,
        num_blocks=2,
        page_size=256,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        dtype=dtype,
        device="cpu",
        memory_saver_adapter=_NoopMemorySaverAdapter(),
        custom_mem_pool=None,
        swa_ring_size=128,
    )


class TestResolveUnifiedKvDtype(unittest.TestCase):
    def test_supported_dtypes_pass_through(self):
        for dtype in UNIFIED_KV_DTYPES:
            self.assertIs(resolve_unified_kv_dtype(dtype), dtype)

    def test_fp8_falls_back_and_says_so(self):
        for dtype in _FP8_DTYPES:
            with self.assertLogs(
                "sglang.srt.mem_cache.deepseek_v4_memory_pool", level="WARNING"
            ) as logs:
                resolved = resolve_unified_kv_dtype(dtype)
            self.assertIs(resolved, UNIFIED_KV_FALLBACK_DTYPE)
            # The operator has to be able to tell from the log that the flag
            # they passed is not the layout they got.
            self.assertIn(str(dtype), "\n".join(logs.output))

    def test_fallback_is_itself_supported(self):
        self.assertIn(UNIFIED_KV_FALLBACK_DTYPE, UNIFIED_KV_DTYPES)


class TestUnifiedKvPoolDtype(unittest.TestCase):
    def test_buffers_use_the_requested_dtype(self):
        # fp16 rather than bf16: bf16 is the fallback, so a reintroduced
        # hardcode would still pass a bf16-only assertion.
        pool = _make_pool(dtype=torch.float16)
        self.assertIs(pool.dtype, torch.float16)
        self.assertEqual(len(pool.kv_buffer), 2)
        for buf in pool.kv_buffer:
            self.assertIs(buf.dtype, torch.float16)

    def test_bf16_still_works(self):
        pool = _make_pool(dtype=torch.bfloat16)
        for buf in pool.kv_buffer:
            self.assertIs(buf.dtype, torch.bfloat16)

    def test_row_bytes_track_the_dtype(self):
        # get_buf_infos() feeds the PD-disaggregation registration, so the
        # advertised item length has to follow the real element size.
        head_dim = 448 + 64
        for dtype, want_row_bytes in ((torch.bfloat16, 2), (torch.float16, 2)):
            _, _, item_lens = _make_pool(dtype=dtype).get_buf_infos()
            for item_len in item_lens:
                self.assertEqual(item_len, head_dim * want_row_bytes)

    def test_unsupported_dtype_is_rejected_at_construction(self):
        # Callers must resolve first; a pool that silently substitutes would be
        # the original bug wearing a parameter.
        for dtype in _FP8_DTYPES + (torch.float32,):
            with self.assertRaises(ValueError):
                _make_pool(dtype=dtype)


class TestConstructionSiteRoutesThroughResolve(unittest.TestCase):
    """Guard the caller, not just the callee.

    ``DeepSeekV4TokenToKVPool`` builds the unified pool inside a branch that
    needs a GPU-sized allocation to exercise, so pin the wiring at the source
    level instead: the ``dtype=`` argument must be a ``resolve_unified_kv_dtype``
    call on the pool's own ``dtype`` parameter.
    """

    @staticmethod
    def _unified_pool_call() -> ast.Call:
        import sglang.srt.mem_cache.deepseek_v4_memory_pool as module

        tree = ast.parse(inspect.getsource(module))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "DeepSeekV4UnifiedKVPool"
        ]
        assert len(calls) == 1, f"expected exactly one construction site, got {calls}"
        return calls[0]

    def test_dtype_argument_is_resolved_not_hardcoded(self):
        call = self._unified_pool_call()
        dtype_args = [kw.value for kw in call.keywords if kw.arg == "dtype"]
        self.assertEqual(len(dtype_args), 1, "dtype must be passed by keyword")
        (dtype_arg,) = dtype_args
        self.assertIsInstance(
            dtype_arg, ast.Call, f"dtype={ast.dump(dtype_arg)} is not resolved"
        )
        self.assertEqual(dtype_arg.func.id, "resolve_unified_kv_dtype")
        self.assertEqual(len(dtype_arg.args), 1)
        self.assertEqual(dtype_arg.args[0].id, "dtype")

    def test_pool_init_has_no_hardcoded_dtype_literal(self):
        source = textwrap.dedent(inspect.getsource(DeepSeekV4UnifiedKVPool.__init__))
        tree = ast.parse(source)
        zeros_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "zeros"
        ]
        self.assertTrue(zeros_calls, "expected the buffer allocation to still be here")
        for node in zeros_calls:
            (dtype_arg,) = [kw.value for kw in node.keywords if kw.arg == "dtype"]
            self.assertIsInstance(
                dtype_arg,
                ast.Attribute,
                "buffer dtype must come from self.dtype",
            )
            self.assertEqual(ast.unparse(dtype_arg), "self.dtype")


class TestKernelPremise(unittest.TestCase):
    """Why the fallback exists at all.

    If the unified_kv kernels ever grow an fp8 path for this buffer, this test
    fails and ``UNIFIED_KV_DTYPES`` should be widened rather than the fallback
    quietly kept.
    """

    def test_prefill_kernel_still_requires_kv_dtype_to_match_q(self):
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import paged_prefill

        source = inspect.getsource(paged_prefill)
        self.assertIn("unified_kv.dtype != q.dtype", source)
        self.assertIn("expects fp16/bf16 q", source)


if __name__ == "__main__":
    unittest.main()
