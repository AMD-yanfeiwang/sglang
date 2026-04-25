"""DWDP (Distributed Weight Data Parallelism) for MoE models.

DWDP distributes MoE expert weights across GPUs within a node while keeping
attention weights fully replicated.  Instead of collective synchronization
(AllReduce / AllGather), DWDP uses asynchronous peer-to-peer prefetches to
pull remote expert weights before they are needed, eliminating synchronization
barriers from the critical path.
"""

from sglang.srt.layers.moe.dwdp.dwdp_manager import (
    DwdpExpertLayout,
    DwdpManager,
    NvFp4WeightView,
    get_global_dwdp_manager,
    set_global_dwdp_manager,
)
from sglang.srt.layers.moe.dwdp.prefetch_buffer import DwdpPrefetchBuffer

__all__ = [
    "DwdpExpertLayout",
    "DwdpManager",
    "DwdpPrefetchBuffer",
    "NvFp4WeightView",
    "get_global_dwdp_manager",
    "set_global_dwdp_manager",
]
