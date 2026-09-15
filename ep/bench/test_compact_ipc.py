"""Independent FP8/metadata/reduction oracle for the compact one-node protocol."""

import os, time
from datetime import timedelta
import torch
import torch.distributed as dist
from uccl.ep_api.compact import CompactIPC
from uccl.ep_api import Buffer


def per_token_group_quant_fp8(x, group_size, use_ue8m0=False):
    # Independent eager reference: FP32 amax and division, then FP8 rounding.
    values = x.float().reshape(x.shape[0], x.shape[1] // group_size, group_size)
    # Tensor/scalar FP32 division may be lowered to reciprocal multiplication.
    # Evaluate in FP64, then round explicitly to FP32 for the wire contract.
    scales = (values.abs().amax(-1).clamp_min(1e-10).double() / 448.0).float()
    quantized = (
        (values.double() / scales.double().unsqueeze(-1))
        .float()
        .to(torch.float8_e4m3fn)
    )
    return quantized.reshape(x.shape), scales


rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
torch.set_num_threads(4)
dist.init_process_group("gloo", timeout=timedelta(seconds=120))
H, K, E, C = 7168, 6, 384, 32
shared = os.environ.get("UCCL_TEST_COMPACT_SHARED_LL") == "1"
ll_buffer = None
if shared:
    ll_bytes = Buffer.get_low_latency_rdma_size_hint(C, H, world, E)
    offset = (ll_bytes + 255) // 256 * 256
    ll_buffer = Buffer(
        dist.group.WORLD,
        Buffer.get_dispatch_config(world).get_nvl_buffer_size_hint(H * 2, world),
        offset + 32 * 1024 * 1024,
        low_latency_mode=True,
        num_qps_per_rank=E // world,
        explicitly_destroy=True,
        is_intranode=True,
    )
    transport = CompactIPC(
        dist.group.WORLD, C, H, K, E, buffer=ll_buffer, workspace_offset=offset
    )
else:
    transport = CompactIPC(dist.group.WORLD, C, H, K, E)
device = transport.partial.device


def check_ll():
    if ll_buffer is None:
        return
    values = torch.full((3, H), rank + 1, dtype=torch.bfloat16, device=device)
    routes = torch.arange(K, device=device, dtype=torch.int64).expand(3, K).contiguous()
    route_weights = torch.ones(3, K, device=device)
    received, counts, handle, _, _ = ll_buffer.low_latency_dispatch(
        values,
        routes,
        C,
        E,
        use_fp8=False,
        async_finish=False,
        return_recv_hook=False,
    )
    output, _, _ = ll_buffer.low_latency_combine(
        received,
        routes,
        route_weights,
        handle,
        async_finish=False,
        return_recv_hook=False,
    )
    torch.testing.assert_close(
        output, torch.full_like(values, (rank + 1) * K), rtol=0, atol=0
    )


check_ll()
graphs = []
for pattern in range(4):
    sizes = (
        [C] * world
        if pattern == 0
        else [0] * world
        if pattern == 1
        else [0 if r % 2 == 0 else 7 + r for r in range(world)]
        if pattern == 2
        else [1 + r for r in range(world)]
    )
    T = sizes[rank]
    gen = torch.Generator().manual_seed(12700 + rank + 17 * pattern)
    x = torch.randn(T, H, generator=gen, dtype=torch.bfloat16).to(device)
    ids = torch.randint(E, (T, K), generator=gen, dtype=torch.int64).to(device)
    if T:
        ids[:, 1] = ids[:, 0]
        ids[::3, -1] = -1
    weights = torch.randn(T, K, generator=gen, dtype=torch.float32).to(device)
    out = torch.empty_like(x)

    def forward(x=x, ids=ids, weights=weights, out=out):
        q, s, ri, rw = transport.dispatch(x, ids, weights)
        decoded = (q.float().view(world * C, H // 128, 128) * s.unsqueeze(-1)).reshape(
            world * C, H
        )
        transport.partial.copy_((decoded * ((rank + 1) / 16)).bfloat16())
        transport.combine(out)
        return q, s, ri, rw, decoded

    for _ in range(2):
        held = forward()
        torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        held1 = forward()
        held2 = forward()
    cases = []
    for trial, factor in enumerate([1.0, 0.0, 1e-8, -0.75]):
        trial_x = (torch.randn(T, H, generator=gen) * factor).bfloat16().to(device)
        trial_ids = torch.randint(E, (T, K), generator=gen, dtype=torch.int64).to(
            device
        )
        if T:
            trial_ids[:, 1] = trial_ids[:, 0]
            trial_ids[::2, -1] = -1
        trial_weights = torch.randn(T, K, generator=gen, dtype=torch.float32).to(device)
        gathered = [None] * world
        dist.all_gather_object(
            gathered, (trial_x.cpu(), trial_ids.cpu(), trial_weights.cpu())
        )
        oracle_x = torch.zeros(world * C, H, device=device, dtype=torch.bfloat16)
        oracle_ids = torch.full((world * C, K), -1, device=device, dtype=torch.int64)
        oracle_weights = torch.zeros(world * C, K, device=device)
        mask = torch.zeros(world * C, device=device, dtype=torch.bool)
        for source, (sx, si, sw) in enumerate(gathered):
            n = len(sx)
            at = source * C
            oracle_x[at : at + n] = sx.to(device)
            oracle_ids[at : at + n] = si.to(device)
            oracle_weights[at : at + n] = sw.to(device)
            mask[at : at + n] = True
        oq, osc = per_token_group_quant_fp8(oracle_x, 128, use_ue8m0=False)
        osc[~mask] = 1
        decoded = (
            oq.float().view(world * C, H // 128, 128) * osc.unsqueeze(-1)
        ).reshape(world * C, H)
        summed = torch.zeros_like(decoded)
        for source in range(world):
            summed.add_((decoded * ((source + 1) / 16)).bfloat16().float())
        expected = summed[rank * C : rank * C + T].bfloat16()
        cases.append(
            (
                trial_x,
                trial_ids,
                trial_weights,
                oq.view(torch.uint8),
                osc,
                oracle_ids,
                oracle_weights,
                expected,
            )
        )
    graphs.append((graph, x, ids, weights, out, cases, held1, held2))

for replay in range(128):
    if replay % 16 == 0:
        check_ll()
    for index, (graph, x, ids, weights, out, cases, held1, held2) in enumerate(graphs):
        case = cases[(replay + index) % len(cases)]
        x.copy_(case[0])
        ids.copy_(case[1])
        weights.copy_(case[2])
        if replay % 16 == 0:
            time.sleep(rank * 0.0002)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(transport.q.view(torch.uint8), case[3]):
            mismatch = transport.q.view(torch.uint8) != case[3]
            print(
                "FP8_MISMATCH",
                rank,
                replay,
                index,
                mismatch.sum().item(),
                "scale_max_diff",
                (transport.scales - case[4]).abs().max().item(),
                flush=True,
            )
        assert torch.equal(transport.q.view(torch.uint8), case[3]), (
            "FP8 bytes",
            rank,
            replay,
            index,
        )
        assert torch.equal(transport.scales, case[4]), ("scales", rank, replay, index)
        assert torch.equal(transport.ids, case[5]), ("routes", rank, replay, index)
        assert torch.equal(transport.weights, case[6]), ("weights", rank, replay, index)
        assert torch.equal(out, case[7]), ("sum", rank, replay, index)
    if rank == 0 and replay % 16 == 0:
        print("REPLAY", replay, flush=True)
dist.barrier()
if rank == 0:
    print(
        "PASS compact IPC: 1024 captured roundtrips, exact FP8/scales/routes/weights/BF16 sums; changing values, duplicates, empty/ragged ranks, no inter-replay barrier",
        flush=True,
    )
transport.destroy()
if ll_buffer is not None:
    ll_buffer.destroy()
dist.destroy_process_group()
