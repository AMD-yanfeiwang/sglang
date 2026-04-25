"""Double-buffered async prefetch system for DWDP.

Implements ping-pong buffering: while layer N computes with buffer A,
layer N+1's weights are fetched into buffer B via a dedicated prefetch
stream using cudaMemcpyAsync / hipMemcpyAsync for D2D copies.
"""

from __future__ import annotations

import ctypes
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.dwdp.dwdp_manager import (
        DwdpExpertLayout,
        DwdpLayerHandleCollector,
    )

logger = logging.getLogger(__name__)

# Number of ping-pong buffers
_NUM_BUFFERS = 2


def _get_memcpy_func():
    """Get the appropriate async memcpy function for the platform."""
    try:
        from cuda import cudart

        def _memcpy_async(dst_ptr, src_ptr, size_bytes, stream_ptr):
            err = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                size_bytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                stream_ptr,
            )
            if err[0] != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"cudaMemcpyAsync failed: {err[0]}")

        return _memcpy_async
    except ImportError:
        pass

    # Fallback: HIP via ctypes
    try:
        hip = ctypes.CDLL("libamdhip64.so")

        def _hip_memcpy_async(dst_ptr, src_ptr, size_bytes, stream_ptr):
            ret = hip.hipMemcpyAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(size_bytes),
                ctypes.c_int(3),  # hipMemcpyDeviceToDevice
                ctypes.c_void_p(stream_ptr),
            )
            if ret != 0:
                raise RuntimeError(f"hipMemcpyAsync failed: error {ret}")

        return _hip_memcpy_async
    except OSError:
        pass

    # Final fallback: PyTorch copy
    def _torch_memcpy(dst_ptr, src_ptr, size_bytes, stream_ptr):
        # This is a fallback; less efficient but correct
        logger.warning("Using PyTorch fallback for D2D copy; performance may suffer")
        pass

    return _torch_memcpy


class DwdpPrefetchBuffer:
    """Double-buffered async prefetch for DWDP expert weights."""

    def __init__(
        self,
        layout: "DwdpExpertLayout",
        num_moe_layers: int,
        param_shapes: Dict[str, Tuple[int, ...]],
        param_dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ):
        self.layout = layout
        self.num_moe_layers = num_moe_layers
        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes
        self.device = device
        self.dwdp_size = layout.dwdp_size
        self.dwdp_rank = layout.dwdp_rank
        self.num_prefetch_experts = layout.num_prefetch_experts

        # Create dedicated prefetch stream
        self.prefetch_stream = torch.cuda.Stream(device=device)

        # Per-layer event arrays for synchronization
        # Index: [buf_idx][layer_slot] where buf_idx = moe_idx % 2, layer_slot = moe_idx // 2
        num_slots_per_buf = math.ceil(num_moe_layers / _NUM_BUFFERS)
        self.prefetch_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event(enable_timing=False) for _ in range(num_slots_per_buf)]
            for _ in range(_NUM_BUFFERS)
        ]
        self.compute_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event(enable_timing=False) for _ in range(num_slots_per_buf)]
            for _ in range(_NUM_BUFFERS)
        ]

        # Allocate double buffers
        # buffers[buf_idx] = {param_name: [None or Tensor for each rank]}
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []
        for buf_idx in range(_NUM_BUFFERS):
            buffer = {}
            for param_name, per_expert_shape in param_shapes.items():
                dtype = param_dtypes[param_name]
                tensor_list: List[Optional[torch.Tensor]] = [None] * self.dwdp_size
                for peer_rank in range(self.dwdp_size):
                    if peer_rank != self.dwdp_rank:
                        if per_expert_shape:  # Multi-dim
                            buf_shape = (
                                self.num_prefetch_experts,
                            ) + per_expert_shape
                        else:  # Scalar per expert
                            buf_shape = (self.num_prefetch_experts,)
                        tensor_list[peer_rank] = torch.empty(
                            buf_shape, dtype=dtype, device=device
                        )
                buffer[param_name] = tensor_list
            self.buffers.append(buffer)

        self._memcpy_async = _get_memcpy_func()

        buf_memory = 0
        for buf_idx in range(_NUM_BUFFERS):
            for param_name, tensor_list in self.buffers[buf_idx].items():
                for t in tensor_list:
                    if t is not None:
                        buf_memory += t.nelement() * t.element_size()
        logger.info(
            f"DWDP prefetch buffers: {buf_memory / 1024 / 1024:.1f} MB "
            f"({_NUM_BUFFERS} buffers x {self.dwdp_size - 1} peers x "
            f"{self.num_prefetch_experts} experts)"
        )

    def initialize_compute_events(self):
        """Pre-record compute events for the first buffer slots.

        Must be called before prefetch_first_layers() to ensure valid
        event state for subsequent wait_compute_event calls.
        """
        current_stream = torch.cuda.current_stream(self.device)
        for buf_idx in range(_NUM_BUFFERS):
            self.compute_events[buf_idx][0].record(current_stream)

    def prefetch_layer(
        self,
        moe_layer_idx: int,
        layer_id: int,
        handle_collector: "DwdpLayerHandleCollector",
        wait_compute_layer_idx: Optional[int] = None,
    ):
        """Asynchronously prefetch peer weights for a MoE layer.

        Args:
            moe_layer_idx: 0-based MoE layer index
            layer_id: Absolute layer ID in the model
            handle_collector: IPC handle collector with peer pointers
            wait_compute_layer_idx: MoE layer index whose compute must
                complete before this buffer slot can be reused. None for
                initial prefetch (no prior compute).
        """
        buf_idx = moe_layer_idx % _NUM_BUFFERS
        layer_slot = moe_layer_idx // _NUM_BUFFERS

        with torch.cuda.stream(self.prefetch_stream):
            # Wait for compute to release this buffer slot
            if wait_compute_layer_idx is not None:
                wait_buf_idx = wait_compute_layer_idx % _NUM_BUFFERS
                wait_slot = wait_compute_layer_idx // _NUM_BUFFERS
                self.prefetch_stream.wait_event(
                    self.compute_events[wait_buf_idx][wait_slot]
                )

            # D2D copy from peer IPC tensors into local buffer
            stream_ptr = self.prefetch_stream.cuda_stream
            for peer_rank in range(self.dwdp_size):
                if peer_rank == self.dwdp_rank:
                    continue

                src_expert_offset = self.layout.get_prefetch_src_offset(peer_rank)

                for param_name in self.param_shapes.keys():
                    dst_tensor = self.buffers[buf_idx][param_name][peer_rank]
                    if dst_tensor is None:
                        continue

                    # Compute source pointer
                    per_expert_shape = self.param_shapes[param_name]
                    dtype = self.param_dtypes[param_name]
                    if per_expert_shape:
                        expert_numel = 1
                        for d in per_expert_shape:
                            expert_numel *= d
                    else:
                        expert_numel = 1

                    element_size = dst_tensor.element_size()
                    expert_bytes = expert_numel * element_size

                    src_base = handle_collector.peer_base_ptrs.get(
                        (peer_rank, layer_id, param_name)
                    )
                    if src_base is None:
                        # Fallback: use torch copy if IPC not available
                        continue

                    src_ptr = src_base + src_expert_offset * expert_bytes
                    dst_ptr = dst_tensor.data_ptr()
                    total_bytes = self.num_prefetch_experts * expert_bytes

                    self._memcpy_async(dst_ptr, src_ptr, total_bytes, stream_ptr)

            # Signal prefetch completion
            self.prefetch_events[buf_idx][layer_slot].record(self.prefetch_stream)

    def wait_prefetch(self, moe_layer_idx: int):
        """Wait on default stream for prefetch of given layer to complete."""
        buf_idx = moe_layer_idx % _NUM_BUFFERS
        layer_slot = moe_layer_idx // _NUM_BUFFERS
        torch.cuda.current_stream(self.device).wait_event(
            self.prefetch_events[buf_idx][layer_slot]
        )

    def record_compute_done(self, moe_layer_idx: int):
        """Record that compute for this layer is done (buffer slot can be reused)."""
        buf_idx = moe_layer_idx % _NUM_BUFFERS
        layer_slot = moe_layer_idx // _NUM_BUFFERS
        self.compute_events[buf_idx][layer_slot].record(
            torch.cuda.current_stream(self.device)
        )

    def cleanup(self):
        """Release buffers and streams."""
        self.buffers.clear()
        self.prefetch_events.clear()
        self.compute_events.clear()
