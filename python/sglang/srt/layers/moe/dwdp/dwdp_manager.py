"""DWDP Manager: expert layout, IPC handles, weight views, lifecycle.

This module implements the core DWDP infrastructure:
- DwdpExpertLayout: expert-to-rank mapping with overlap support
- DwdpLayerHandleCollector: per-layer CUDA IPC handle management
- NvFp4WeightView: multi-B weight tensor bundle for CuteDSL
- DwdpManager: global singleton orchestrating all DWDP components
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_global_dwdp_manager: Optional["DwdpManager"] = None


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


def set_global_dwdp_manager(manager: Optional["DwdpManager"]):
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def enable_dwdp() -> bool:
    return _global_dwdp_manager is not None


# ---------------------------------------------------------------------------
# DwdpExpertLayout
# ---------------------------------------------------------------------------
@dataclass
class DwdpExpertLayout:
    """Expert-to-rank mapping with support for overlapping allocation."""

    num_routed_experts: int
    dwdp_size: int
    dwdp_rank: int
    num_experts_per_worker: int

    def __post_init__(self):
        assert self.num_experts_per_worker >= self.num_routed_experts // self.dwdp_size
        assert self.num_experts_per_worker <= self.num_routed_experts
        # Number of experts to prefetch from each peer
        if self.dwdp_size > 1:
            self.num_prefetch_experts = math.ceil(
                (self.num_routed_experts - self.num_experts_per_worker)
                / (self.dwdp_size - 1)
            )
        else:
            self.num_prefetch_experts = 0

        # Local expert range
        self.local_expert_start = min(
            self.num_prefetch_experts * self.dwdp_rank,
            self.num_routed_experts - self.num_experts_per_worker,
        )
        self.local_expert_end = (
            self.local_expert_start + self.num_experts_per_worker
        )

        # Build peer expert ranges
        self.peer_expert_ranges: Dict[int, Tuple[int, int]] = {}
        for rank in range(self.dwdp_size):
            start = min(
                self.num_prefetch_experts * rank,
                self.num_routed_experts - self.num_experts_per_worker,
            )
            end = start + self.num_experts_per_worker
            self.peer_expert_ranges[rank] = (start, end)

    def get_prefetch_src_offset(self, peer_rank: int) -> int:
        """Get offset (in number of experts) into peer's local tensor for prefetch."""
        peer_start, peer_end = self.peer_expert_ranges[peer_rank]
        if self.dwdp_rank < peer_rank:
            # Need tail of peer's experts
            prefetch_start = peer_end - self.num_prefetch_experts
        else:
            # Need head of peer's experts
            prefetch_start = peer_start
        return prefetch_start - peer_start


# ---------------------------------------------------------------------------
# NvFp4WeightView  (Multi-B weight tensor bundle)
# ---------------------------------------------------------------------------
@dataclass
@dataclass
class DwdpWeightView:
    """Generic multi-B weight tensor bundle for DWDP forward.

     maps param_name -> List[Tensor] in rank order.
    The caller (quant scheme apply method) decides which keys to use.
    """
    weights: Dict[str, List[torch.Tensor]]
    expert_size_per_partition: int  # num_experts_per_worker
    slot_start: int  # local_expert_start (global expert ID offset)


# Keep alias for backward compatibility
NvFp4WeightView = DwdpWeightView


# ---------------------------------------------------------------------------
# DwdpLayerHandleCollector
# ---------------------------------------------------------------------------


class DwdpLayerHandleCollector:
    """Per-layer CUDA IPC handle management for DWDP weight exchange."""

    def __init__(self, num_moe_layers: int, dwdp_size: int, dwdp_rank: int):
        self.num_moe_layers = num_moe_layers
        self.dwdp_size = dwdp_size
        self.dwdp_rank = dwdp_rank

        # Local weight tensors: layer_id -> {param_name: tensor}
        self.local_weights: Dict[int, Dict[str, torch.Tensor]] = {}

        # Peer tensor base pointers: (peer_rank, layer_id, param_name) -> int
        self.peer_base_ptrs: Dict[Tuple[int, int, str], int] = {}

        # IPC handles opened on this rank (for cleanup)
        self._opened_handles: List[int] = []

    def register_layer_weights(self, layer_id: int, **kwargs):
        """Register weight tensors for a MoE layer.
        
        Accepts arbitrary param_name=tensor pairs. Common names:
        - w13_weight, w2_weight: packed expert weights
        - w13_weight_sf, w2_weight_sf: block-scale factors
        - w1_alpha, w2_alpha: per-expert GEMM scales (nvidia FP4)
        """
        self.local_weights[layer_id] = {
            k: v for k, v in kwargs.items() if v is not None
        }

    def exchange_ipc_handles(self, dwdp_group):
        """Exchange CUDA IPC handles across DWDP group for all registered layers."""
        try:
            from cuda import cuda as cuda_driver
            from cuda import cudart
        except ImportError:
            logger.warning(
                "cuda-python not available; falling back to hipIPC via ctypes"
            )
            self._exchange_ipc_handles_hip(dwdp_group)
            return

        for layer_id in sorted(self.local_weights.keys()):
            local_handles = {}
            for param_name, tensor in self.local_weights[layer_id].items():
                err, handle = cudart.cudaIpcGetMemHandle(tensor.data_ptr())
                if err != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(
                        f"cudaIpcGetMemHandle failed for layer {layer_id} "
                        f"{param_name}: {err}"
                    )
                err, alloc_base, alloc_size = cuda_driver.cuMemGetAddressRange(
                    tensor.data_ptr()
                )
                offset = tensor.data_ptr() - int(alloc_base)
                handle_bytes = bytes(handle)
                local_handles[param_name] = (handle_bytes, offset)

            # AllGather handles across DWDP group
            all_handles = [None] * self.dwdp_size
            dist.all_gather_object(all_handles, local_handles, group=dwdp_group)

            # Open peer handles
            for peer_rank in range(self.dwdp_size):
                if peer_rank == self.dwdp_rank:
                    continue
                for param_name, (handle_bytes, offset) in all_handles[
                    peer_rank
                ].items():
                    handle = cudart.cudaIpcMemHandle_t()
                    handle_array = (ctypes.c_char * len(handle_bytes)).from_buffer_copy(
                        handle_bytes
                    )
                    ctypes.memmove(handle.reserved, handle_array, len(handle_bytes))

                    err, base_ptr = cudart.cudaIpcOpenMemHandle(
                        handle,
                        cudart.cudaIpcMemLazyEnablePeerAccess,
                    )
                    if err != cudart.cudaError_t.cudaSuccess:
                        raise RuntimeError(
                            f"cudaIpcOpenMemHandle failed for peer {peer_rank} "
                            f"layer {layer_id} {param_name}: {err}"
                        )
                    self._opened_handles.append(int(base_ptr))
                    actual_ptr = int(base_ptr) + offset
                    self.peer_base_ptrs[(peer_rank, layer_id, param_name)] = actual_ptr

    def _exchange_ipc_handles_hip(self, dwdp_group):
        """Fallback IPC handle exchange using hipIPC via ctypes for AMD/ROCm."""
        import ctypes

        try:
            hip = ctypes.CDLL("libamdhip64.so")
        except OSError:
            raise RuntimeError("Cannot load libamdhip64.so for hipIPC")

        # hipIpcMemHandle_t is 64 bytes
        IPC_HANDLE_SIZE = 64

        class hipIpcMemHandle_t(ctypes.Structure):
            _fields_ = [("reserved", ctypes.c_char * IPC_HANDLE_SIZE)]

        for layer_id in sorted(self.local_weights.keys()):
            local_handles = {}
            for param_name, tensor in self.local_weights[layer_id].items():
                handle = hipIpcMemHandle_t()
                ret = hip.hipIpcGetMemHandle(
                    ctypes.byref(handle), ctypes.c_void_p(tensor.data_ptr())
                )
                if ret != 0:
                    raise RuntimeError(
                        f"hipIpcGetMemHandle failed for layer {layer_id} "
                        f"{param_name}: error {ret}"
                    )
                # For HIP, we need to get the allocation base to compute offset
                # Use hipPointerGetAttributes
                class hipPointerAttribute_t(ctypes.Structure):
                    _fields_ = [
                        ("memoryType", ctypes.c_int),  # enum
                        ("device", ctypes.c_int),
                        ("devicePointer", ctypes.c_void_p),
                        ("hostPointer", ctypes.c_void_p),
                        ("isManaged", ctypes.c_int),
                        ("allocationFlags", ctypes.c_uint),
                    ]

                # For simplicity, use offset = 0 with direct tensor data_ptr
                # PyTorch on ROCm typically gives us the exact allocation start
                # for tensors created via torch.empty
                handle_bytes = bytes(handle.reserved)
                offset = 0  # We'll handle offset via tensor slicing
                local_handles[param_name] = (
                    handle_bytes,
                    offset,
                    tensor.shape,
                    str(tensor.dtype),
                )

            all_handles = [None] * self.dwdp_size
            dist.all_gather_object(all_handles, local_handles, group=dwdp_group)

            for peer_rank in range(self.dwdp_size):
                if peer_rank == self.dwdp_rank:
                    continue
                for param_name, (
                    handle_bytes,
                    offset,
                    shape,
                    dtype_str,
                ) in all_handles[peer_rank].items():
                    handle = hipIpcMemHandle_t()
                    ctypes.memmove(handle.reserved, handle_bytes, IPC_HANDLE_SIZE)

                    dev_ptr = ctypes.c_void_p()
                    ret = hip.hipIpcOpenMemHandle(
                        ctypes.byref(dev_ptr),
                        handle,
                        ctypes.c_uint(1),  # hipIpcMemLazyEnablePeerAccess
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"hipIpcOpenMemHandle failed for peer {peer_rank} "
                            f"layer {layer_id} {param_name}: error {ret}"
                        )
                    base_ptr = dev_ptr.value
                    self._opened_handles.append(base_ptr)
                    actual_ptr = base_ptr + offset
                    self.peer_base_ptrs[(peer_rank, layer_id, param_name)] = actual_ptr

    def cleanup(self):
        """Close all opened IPC handles."""
        try:
            from cuda import cudart

            for ptr in self._opened_handles:
                cudart.cudaIpcCloseMemHandle(ptr)
        except ImportError:
            import ctypes

            try:
                hip = ctypes.CDLL("libamdhip64.so")
                for ptr in self._opened_handles:
                    hip.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))
            except OSError:
                pass
        self._opened_handles.clear()
        self.peer_base_ptrs.clear()


# ---------------------------------------------------------------------------
# DwdpManager
# ---------------------------------------------------------------------------
class DwdpManager:
    """Global DWDP manager orchestrating layout, IPC, prefetch, and weight views."""

    def __init__(
        self,
        num_routed_experts: int,
        dwdp_size: int,
        dwdp_rank: int,
        num_moe_layers: int,
        first_moe_layer_id: int,
        num_experts_per_worker: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.dwdp_size = dwdp_size
        self.dwdp_rank = dwdp_rank
        self.num_moe_layers = num_moe_layers
        self.first_moe_layer_id = first_moe_layer_id
        self.device = device or torch.device(f"cuda:{torch.cuda.current_device()}")

        if num_experts_per_worker is None:
            num_experts_per_worker = num_routed_experts // dwdp_size

        self.layout = DwdpExpertLayout(
            num_routed_experts=num_routed_experts,
            dwdp_size=dwdp_size,
            dwdp_rank=dwdp_rank,
            num_experts_per_worker=num_experts_per_worker,
        )

        self.handle_collector = DwdpLayerHandleCollector(
            num_moe_layers=num_moe_layers,
            dwdp_size=dwdp_size,
            dwdp_rank=dwdp_rank,
        )

        self.prefetch_buffer: Optional[Any] = None  # Set after init_prefetch_buffers
        self._weight_views_cache: Dict[int, NvFp4WeightView] = {}

        logger.info(
            f"DwdpManager initialized: rank={dwdp_rank}/{dwdp_size}, "
            f"experts_per_worker={num_experts_per_worker}, "
            f"prefetch_experts={self.layout.num_prefetch_experts}, "
            f"local_range=[{self.layout.local_expert_start}, {self.layout.local_expert_end}), "
            f"moe_layers={num_moe_layers}, first_moe_layer={first_moe_layer_id}"
        )

    def register_layer_weights(self, layer_id: int, **kwargs):
        """Register weight tensors for a MoE layer."""
        self.handle_collector.register_layer_weights(layer_id, **kwargs)

    def exchange_ipc_handles(self, dwdp_group):
        """Exchange IPC handles across DWDP group."""
        self.handle_collector.exchange_ipc_handles(dwdp_group)
        logger.info(
            f"DWDP rank {self.dwdp_rank}: IPC handles exchanged for "
            f"{len(self.handle_collector.local_weights)} MoE layers"
        )

    def init_prefetch_buffers(self):
        """Initialize double-buffered prefetch system."""
        from sglang.srt.layers.moe.dwdp.prefetch_buffer import DwdpPrefetchBuffer

        if not self.handle_collector.local_weights:
            logger.warning(
                f"DWDP rank {self.dwdp_rank}: No MoE layers registered, skipping prefetch buffer init"
            )
            return

        # Collect param shapes and dtypes from the first registered layer
        first_layer_id = min(self.handle_collector.local_weights.keys())
        param_shapes = {}
        param_dtypes = {}
        for param_name, tensor in self.handle_collector.local_weights[
            first_layer_id
        ].items():
            # Shape per expert (tensor shape is [num_experts_per_worker, ...])
            if tensor.ndim > 1:
                param_shapes[param_name] = tensor.shape[1:]
            else:
                # Scalar per expert (e.g., alphas shape [num_experts])
                param_shapes[param_name] = ()
            param_dtypes[param_name] = tensor.dtype

        self.prefetch_buffer = DwdpPrefetchBuffer(
            layout=self.layout,
            num_moe_layers=self.num_moe_layers,
            param_shapes=param_shapes,
            param_dtypes=param_dtypes,
            device=self.device,
        )
        logger.info(
            f"DWDP rank {self.dwdp_rank}: prefetch buffers initialized "
            f"(2 x {self.layout.num_prefetch_experts} experts per peer)"
        )

    def initialize_compute_events(self):
        """Pre-record compute events for the first buffer slots."""
        if self.prefetch_buffer is not None:
            self.prefetch_buffer.initialize_compute_events()

    def prefetch_first_layers(self):
        """Trigger initial prefetch for the first 2 MoE layers.

        Called at the start of forward_extend(), before Dense layers.
        The prefetch runs on the prefetch stream while Dense layers compute
        on the default stream, providing overlap.
        """
        if self.prefetch_buffer is None:
            return

        # Prefetch first two MoE layers (indices 0 and 1)
        sorted_layers = sorted(self.handle_collector.local_weights.keys())
        for i in range(min(2, len(sorted_layers))):
            layer_id = sorted_layers[i]
            moe_layer_idx = i
            self.prefetch_buffer.prefetch_layer(
                moe_layer_idx=moe_layer_idx,
                layer_id=layer_id,
                handle_collector=self.handle_collector,
                wait_compute_layer_idx=None,  # No prior compute to wait for
            )

    def wait_prefetch(self, layer_id: int):
        """Wait for prefetch of the given layer to complete on default stream."""
        if self.prefetch_buffer is None:
            return
        moe_layer_idx = self._layer_id_to_moe_idx(layer_id)
        self.prefetch_buffer.wait_prefetch(moe_layer_idx)

    def record_compute_and_prefetch_next(self, layer_id: int):
        """Record compute done and trigger prefetch for layer+2."""
        if self.prefetch_buffer is None:
            return
        moe_layer_idx = self._layer_id_to_moe_idx(layer_id)
        self.prefetch_buffer.record_compute_done(moe_layer_idx)

        # Trigger prefetch for moe_layer_idx + 2
        next_moe_idx = moe_layer_idx + 2
        if next_moe_idx < self.num_moe_layers:
            sorted_layers = sorted(self.handle_collector.local_weights.keys())
            next_layer_id = sorted_layers[next_moe_idx]
            self.prefetch_buffer.prefetch_layer(
                moe_layer_idx=next_moe_idx,
                layer_id=next_layer_id,
                handle_collector=self.handle_collector,
                wait_compute_layer_idx=moe_layer_idx,
            )

    def get_weight_view(self, layer_id: int) -> "DwdpWeightView":
        """Get multi-B weight view for a MoE layer (prefetched + local).

        Returns a DwdpWeightView with per-param-name List[Tensor] in rank order.
        The caller (quant apply method) decides how to use them (concat, etc.).
        """
        moe_layer_idx = self._layer_id_to_moe_idx(layer_id)
        local_weights = self.handle_collector.local_weights[layer_id]
        buf_idx = moe_layer_idx % 2
        param_names = list(local_weights.keys())

        # Build weight lists in rank order for each param
        result = {}  # param_name -> List[Tensor]
        for pname in param_names:
            plist = []
            for rank in range(self.dwdp_size):
                if rank == self.dwdp_rank:
                    plist.append(local_weights[pname])
                else:
                    plist.append(self.prefetch_buffer.buffers[buf_idx][pname][rank])
            result[pname] = plist

        return DwdpWeightView(
            weights=result,
            expert_size_per_partition=self.layout.num_experts_per_worker,
            slot_start=self.layout.local_expert_start,
        )

    def _layer_id_to_moe_idx(self, layer_id: int) -> int:
        """Convert absolute layer_id to MoE layer index (0-based)."""
        sorted_layers = sorted(self.handle_collector.local_weights.keys())
        return sorted_layers.index(layer_id)

    def cleanup(self):
        """Release all DWDP resources."""
        if self.prefetch_buffer is not None:
            self.prefetch_buffer.cleanup()
            self.prefetch_buffer = None
        self.handle_collector.cleanup()
        self._weight_views_cache.clear()
        logger.info(f"DWDP rank {self.dwdp_rank}: cleaned up")
