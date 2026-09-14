"""Node-shared LL regression with changing precision/capacity and exact routes.

Capture two round trips so both LL ping-pong buffers participate in every replay.
Run on >=2 nodes with a coalesced build; covers BF16 and optional FP8 dispatch.
"""

import argparse
import gc
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from buffer import Buffer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--topk", type=int, default=6)
    p.add_argument("--fp8", action="store_true")
    p.add_argument("--duplicates", action="store_true")
    p.add_argument("--replays", type=int, default=24)
    p.add_argument("--unbarriered", action="store_true")
    p.add_argument("--recv-hook", action="store_true")
    a = p.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    assert world >= local_world * 2 and world % local_world == 0
    G = a.experts // world
    T = a.tokens
    H = a.hidden
    K = a.topk
    assert a.experts % world == 0 and K <= G <= 64
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    buffer = Buffer(
        dist.group.WORLD,
        num_nvl_bytes=Buffer.get_dispatch_config(world).get_nvl_buffer_size_hint(
            H * 2, world
        ),
        num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(T, H, world, a.experts),
        low_latency_mode=True,
        num_qps_per_rank=G,
        allow_nvlink_for_low_latency_mode=True,
        explicitly_destroy=True,
    )
    x = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(T, K, device="cuda", dtype=torch.int64)
    weights = torch.full((T, K), 1 / K, device="cuda")
    factors = (
        1
        + torch.arange(rank * G, (rank + 1) * G, device="cuda", dtype=torch.float32)
        / a.experts
    )[:, None, None]

    def routes(r, trial):
        rows = torch.arange(T)[:, None]
        slots = torch.arange(K)[None, :]
        target = (
            (r // local_world + 1) % (world // local_world)
        ) * local_world + r % local_world
        mode = trial % 5
        if mode in (0, 2, 4):
            choices = (target * G + (G - K) + slots).expand(T, K).clone()
        else:
            choices = (r * G + rows * 3 + slots * G) % a.experts
        if mode == 4 and a.duplicates:
            choices.fill_(-1)
            choices[: T // K, :] = target * G + G - 1
        if mode == 2 and r % 3 == trial % 3:
            choices.fill_(-1)
        if mode == 3:
            choices[(r * 13 + trial * 7) % (T + 1) :] = -1
        return choices

    def inputs(trial):
        values = (
            (
                torch.arange(T, device="cuda")[:, None] * 7
                + torch.arange(H, device="cuda")[None, :] * 3
                + rank * 11
                + trial * 13
            )
            % 61
            - 30
        ) / 8
        x.copy_(values)
        ids.copy_(routes(rank, trial))

    def roundtrip(xx=x, ii=ids, ww=weights, fp8=None, capacity=T):
        fp8 = a.fp8 if fp8 is None else fp8
        rx, counts, handle, event, hook = buffer.low_latency_dispatch(
            xx,
            ii,
            capacity,
            a.experts,
            use_fp8=fp8,
            round_scale=False,
            use_ue8m0=False,
            async_finish=not a.recv_hook,
            return_recv_hook=a.recv_hook,
        )
        if a.recv_hook:
            hook()
        else:
            event.current_stream_wait()
        if fp8:
            q, scale = rx
            decoded = (
                q.float().view(G, world * capacity, H // 128, 128)
                * scale.float().unsqueeze(-1)
            ).reshape(G, world * capacity, H)
        else:
            decoded = rx.float()
        supplied = (decoded * factors).to(torch.bfloat16)
        y, event, hook = buffer.low_latency_combine(
            supplied,
            ii,
            ww,
            handle,
            use_logfmt=False,
            async_finish=not a.recv_hook,
            return_recv_hook=a.recv_hook,
        )
        if a.recv_hook:
            hook()
        else:
            event.current_stream_wait()
        return y, counts, handle, rx, supplied, fp8

    worst = 0.0

    def check(result, trial, actual=T, actual_by_rank=None):
        nonlocal worst
        y, counts, handle, _, _, fp8 = result
        expected = (
            (
                x[:actual].float()[:, None, :]
                * (1 + ids[:actual].clamp(min=0).float() / a.experts)[:, :, None]
            )
            .bfloat16()
            .float()
        )
        expected = (
            expected * weights[:actual, :, None] * (ids[:actual] >= 0)[:, :, None]
        ).sum(1)
        assert bool(torch.isfinite(y).all()), ("nonfinite", rank, trial)
        # Power-of-two uniform weights permit an exact BF16 round-trip check.
        if not fp8 and K & (K - 1) == 0:
            assert torch.equal(y, expected.bfloat16()), (rank, trial, "BF16 bytes")
        rel = float((y.float() - expected).norm() / expected.norm().clamp(min=1e-12))
        worst = max(worst, rel)
        assert rel < (0.045 if fp8 else 0.012), (rank, trial, rel)
        src, layout = handle[:2]
        src = src.cpu()
        layout = layout.cpu()
        counts = counts.cpu()
        all_ids = [
            routes(r, trial)[: actual_by_rank[r] if actual_by_rank is not None else T]
            for r in range(world)
        ]
        for e in range(G):
            total = 0
            for r in range(world):
                packed = int(layout[e, r])
                count = packed & 0xFFFFFFFF
                begin = (packed >> 32) & 0xFFFFFFFF
                expected_tokens = (all_ids[r] == rank * G + e).nonzero()[:, 0]
                assert count == len(expected_tokens), (
                    trial,
                    rank,
                    e,
                    r,
                    count,
                    len(expected_tokens),
                )
                assert torch.equal(
                    src[e, begin : begin + count].sort().values,
                    expected_tokens.sort().values,
                ), (trial, rank, e, r, "source indices")
                total += count
            assert int(counts[e]) == total, (trial, rank, e, "total")

    for trial in range(10):
        inputs(trial)
        result = roundtrip(fp8=bool((trial // 2) % 2))
        torch.cuda.synchronize()
        check(result, trial)
        del result
    inputs(10)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [roundtrip(fp8=fp8) for fp8 in [False, False, True, True]]
    torch.cuda.synchronize()
    for trial in range(10, 10 + a.replays):
        inputs(trial)
        if not a.unbarriered:
            dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        for result in captured:
            check(result, trial)
        if rank == 0 and (trial - 10) % 8 == 0:
            print("REPLAY", trial - 10, flush=True)
    dist.barrier()
    del graph, captured, result
    gc.collect()
    torch.cuda.synchronize()
    actual_by_rank = [0 if r % 2 == 0 else T // 2 for r in range(world)]
    inputs(100)
    actual = actual_by_rank[rank]
    result = roundtrip(x[:actual], ids[:actual], weights[:actual])
    torch.cuda.synchronize()
    check(result, 100, actual, actual_by_rank)
    del result
    # Reuse both ping-pong buffers across different capacity/precision layouts.
    actual = T // 2
    for i, capacity in enumerate([T, T // 2, T // 2, T]):
        trial = 200 + 5 * i  # distinct hot experts, not duplicate-capacity stress
        inputs(trial)
        result = roundtrip(
            x[:actual],
            ids[:actual],
            weights[:actual],
            fp8=bool(i % 2),
            capacity=capacity,
        )
        torch.cuda.synchronize()
        check(result, trial, actual, [actual] * world)
        del result
    worst_tensor = torch.tensor(worst, dtype=torch.float64)
    dist.all_reduce(worst_tensor, op=dist.ReduceOp.MAX)
    worst = float(worst_tensor)
    dist.barrier()
    buffer.destroy()
    if rank == 0:
        print(
            f"PASS 10 eager calls + {a.replays * 4} captured calls + true empty/ragged inputs + changing precision/capacity, exact routing metadata, max global relL2 {worst:.6f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
