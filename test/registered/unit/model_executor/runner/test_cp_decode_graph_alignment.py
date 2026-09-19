from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.utils import common


def _exec_config(batch_sizes):
    return SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=list(batch_sizes))
            ),
            torch_compile_max_bs=32,
        ),
        overlap=SimpleNamespace(enable_two_batch_overlap=False),
    )


def test_cp_alignment_can_be_disabled_for_decode_attention_tp():
    parallel = SimpleNamespace(
        attn_tp_size=1,
        attn_cp_size=8,
        enable_cp_decode_attn_tp=True,
    )
    with (
        patch.object(common, "get_exec", return_value=_exec_config(range(1, 9))),
        patch.object(common, "get_parallel", return_value=parallel),
        patch.object(common, "require_gathered_buffer", return_value=True),
    ):
        assert not common.should_align_attn_cp_decode_graph()
        assert common.get_cuda_graph_batch_size_alignment() == 8
        assert common.get_cuda_graph_batch_size_alignment(align_attn_cp=False) == 1
        assert common.get_cuda_graph_max_batch_size(2) == 8
        assert common.get_cuda_graph_max_batch_size(2, align_attn_cp=False) == 2


def test_cp_alignment_remains_for_regular_context_parallel():
    parallel = SimpleNamespace(
        attn_tp_size=1,
        attn_cp_size=8,
        enable_cp_decode_attn_tp=False,
    )
    with patch.object(common, "get_parallel", return_value=parallel):
        assert common.should_align_attn_cp_decode_graph()
