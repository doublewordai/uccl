"""One-node compact IPC path for small FP8 routed expert batches.

Dispatch and combine alternate on the current CUDA stream. A supplied Buffer
can retain normal LL service when compact memory occupies a disjoint suffix;
its caller must reserve that suffix beyond the normal LL size hint.
Expert computation writes BF16 local weighted partials into ``partial``.
"""

import socket
import torch
import torch.distributed as dist
from .buffer import Buffer


class CompactIPC:
    def __init__(
        self,
        group,
        capacity,
        hidden,
        topk,
        experts,
        *,
        buffer: Buffer | None = None,
        workspace_offset: int = 0,
    ):
        self.world = dist.get_world_size(group)
        self.capacity, self.hidden, self.topk = capacity, hidden, topk
        if not (1 <= self.world <= 4 and 1 <= capacity <= 32 and 1 <= topk <= 8):
            raise ValueError(
                "Compact IPC requires 1-4 ranks, capacity 1-32 and topk 1-8"
            )
        if hidden not in (2048, 2560, 4096, 5120, 6144, 7168, 8192) or experts <= 0:
            raise ValueError("Unsupported compact IPC hidden size or expert count")
        if workspace_offset < 0 or workspace_offset % 256:
            raise ValueError(
                "Compact workspace offset must be nonnegative and 256-byte aligned"
            )
        if buffer is not None and (
            workspace_offset == 0
            or buffer.group is not group
            or not buffer.low_latency_mode
        ):
            raise ValueError(
                "A shared LL buffer requires a positive suffix offset and the same group"
            )
        if buffer is None and workspace_offset != 0:
            raise ValueError("A standalone compact buffer uses offset zero")
        geometry = (
            socket.gethostname(),
            capacity,
            hidden,
            topk,
            experts,
            workspace_offset,
        )
        peers = [None] * self.world
        dist.all_gather_object(peers, geometry, group=group)
        if not all(peer == geometry for peer in peers):
            raise ValueError(
                f"Compact IPC requires matching geometry on one host: {peers}"
            )
        self._owns_transport = buffer is None
        self.transport = (
            buffer
            if buffer is not None
            else Buffer(
                group,
                num_nvl_bytes=Buffer.get_dispatch_config(
                    self.world
                ).get_nvl_buffer_size_hint(hidden * 2, self.world),
                num_rdma_bytes=32 * 1024 * 1024,
                low_latency_mode=True,
                num_qps_per_rank=1,
                allow_nvlink_for_low_latency_mode=True,
                explicitly_destroy=True,
                is_intranode=True,
            )
        )
        if self.transport.scratch.device.type != "cuda":
            if self._owns_transport:
                self.transport.destroy()
            raise RuntimeError("Compact IPC requires device-resident transport memory")
        offset = self.transport.runtime.compact_configure(
            capacity,
            hidden,
            topk,
            experts,
            torch.cuda.current_stream().cuda_stream,
            workspace_offset,
        )
        self.partial = (
            self.transport.scratch.narrow(0, offset, self.world * capacity * hidden * 2)
            .view(torch.bfloat16)
            .view(self.world * capacity, hidden)
        )
        device = self.partial.device
        self.q = torch.empty(
            self.world * capacity, hidden, dtype=torch.float8_e4m3fn, device=device
        )
        self.scales = torch.empty(
            self.world * capacity, hidden // 128, dtype=torch.float32, device=device
        )
        self.ids = torch.empty(
            self.world * capacity, topk, dtype=torch.int64, device=device
        )
        self.weights = torch.empty(
            self.world * capacity, topk, dtype=torch.float32, device=device
        )
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def dispatch(self, x, ids, weights):
        if not (
            x.dtype == torch.bfloat16
            and x.ndim == 2
            and x.shape[1] == self.hidden
            and len(x) <= self.capacity
        ):
            raise ValueError("Expected BF16 input [tokens <= capacity, hidden]")
        if not (
            ids.shape == weights.shape == (len(x), self.topk)
            and ids.dtype == torch.int64
            and weights.dtype == torch.float32
        ):
            raise ValueError("Expected int64 routes and FP32 weights [tokens, topk]")
        if not all(
            value.device == self.partial.device and value.is_contiguous()
            for value in (x, ids, weights)
        ):
            raise ValueError("Inputs must be contiguous tensors on the buffer device")
        self.transport.runtime.compact_dispatch(
            x.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
            len(x),
            self.q.data_ptr(),
            self.scales.data_ptr(),
            self.ids.data_ptr(),
            self.weights.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        return self.q, self.scales, self.ids, self.weights

    def combine(self, out):
        if not (
            out.dtype == torch.bfloat16
            and out.ndim == 2
            and out.shape[1] == self.hidden
            and len(out) <= self.capacity
            and out.device == self.partial.device
            and out.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous BF16 output [tokens <= capacity, hidden] on the buffer device"
            )
        self.transport.runtime.compact_combine(
            out.data_ptr(), len(out), torch.cuda.current_stream().cuda_stream
        )
        return out

    def destroy(self):
        if self._owns_transport:
            self.transport.destroy()
