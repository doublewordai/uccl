"""LL batching regression: writes over 16 MiB and over 8191 staged messages.

Run with torchrun on at least two nodes, using a build with
PER_EXPERT_BATCHING=1 or both LANE_E_*_COALESCE options enabled.
Defaults use 2048 tokens, H=6144, top-k=8: each expert batch is 24 MiB,
and the coalesced destination batch has 16384 messages. Every iteration
changes inputs; expert outputs are regenerated after each dispatch.
"""
import argparse
import gc
import os

import torch
import torch.distributed as dist

from buffer import Buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    local_world = int(os.environ["LOCAL_WORLD_SIZE"])
    assert world >= 2 * local_world and world % local_world == 0
    assert args.experts % world == 0
    groups = args.experts // world
    assert args.topk <= groups
    torch.cuda.set_device(local)
    dist.init_process_group("gloo")
    torch.manual_seed(1234 + rank)
    x = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    target = (rank + local_world) % world
    ids = (torch.arange(args.topk, device="cuda") + target * groups).expand(args.tokens, -1).contiguous()
    weights = torch.full((args.tokens, args.topk), 1 / args.topk, device="cuda")
    buffer = Buffer(dist.group.WORLD,
                    # Coalesced low-latency builds keep per-node reader acknowledgements in the NVLink buffer.
                    num_nvl_bytes=Buffer.get_dispatch_config(world).get_nvl_buffer_size_hint(args.hidden * 2, world),
                    num_rdma_bytes=Buffer.get_low_latency_rdma_size_hint(args.tokens, args.hidden, world, args.experts),
                    low_latency_mode=True, num_qps_per_rank=groups,
                    allow_nvlink_for_low_latency_mode=True, explicitly_destroy=True)
    worst = 0.0
    for iteration in range(args.iters):
        x.neg_()
        received, counts, handle, event, _ = buffer.low_latency_dispatch(
            x, ids, args.tokens, args.experts, use_fp8=False,
            async_finish=True, return_recv_hook=False)
        event.current_stream_wait()
        supplied = torch.empty_like(received)
        for expert in range(groups):
            supplied[expert].copy_(received[expert].float() * (1 + (rank * groups + expert) / args.experts))
        output, event, _ = buffer.low_latency_combine(
            supplied, ids, weights, handle, use_logfmt=False,
            async_finish=True, return_recv_hook=False)
        event.current_stream_wait()
        expected = x.float() * (weights * (1 + ids.float() / args.experts)).sum(1)[:, None]
        error = ((output.float() - expected).norm() / expected.norm()).cpu()
        assert bool(torch.isfinite(output).all())
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        worst = max(worst, float(error))
        assert float(error) < 0.012, (iteration, float(error))
        del received, counts, handle, event, supplied, output, expected, error
    dist.barrier()
    gc.collect()
    torch.cuda.synchronize()
    buffer.destroy()
    if rank == 0:
        print(f"PASS: {args.iters} changing-input iterations, max relative L2 {worst:.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
