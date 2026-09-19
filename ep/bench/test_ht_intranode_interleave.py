"""Intranode HT: a graph-replayed worst-token dispatch followed, without a
device sync, by a host-synced eager dispatch on the same Buffer.

The eager dispatch must size its receive buffer from its own routing. The
replayed notify kernel is delayed behind queued GPU work so that it runs after
the eager call has reset the host-mapped receive counter.

Run on one node: torchrun --nproc_per_node=4 test_ht_intranode_interleave.py
"""

import argparse
import faulthandler
import gc
import os

import torch
import torch.distributed as dist

from buffer import Buffer
from utils import init_dist_under_torchrun


def make_inputs(seed, rank, num_tokens, hidden, num_experts, num_topk):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed * 4096 + rank)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, generator=g)
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, generator=g)
    topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)[1]
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32)
    return x, topk_idx, topk_weights


def dispatch(buffer, x, topk_idx, topk_weights, num_experts, worst, config):
    ntpr, _, ntpe, in_rank, _ = buffer.get_dispatch_layout(topk_idx, num_experts)
    recv_x, _, _, counts, handle, _ = buffer.dispatch(
        x,
        num_tokens_per_rank=ntpr,
        is_token_in_rank=in_rank,
        num_tokens_per_expert=ntpe,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_worst_tokens=worst,
        config=config,
    )
    return recv_x, counts, handle, in_rank


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-tokens", type=int, default=64)
    parser.add_argument("--eager-tokens", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--delay-matmuls", type=int, default=40)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank, num_ranks, group = init_dist_under_torchrun(
        local_rank, int(os.environ["LOCAL_WORLD_SIZE"])
    )
    buffer = Buffer(group, int(2e9), 0, low_latency_mode=False, explicitly_destroy=True)
    config = Buffer.get_dispatch_config(num_ranks)
    worst = args.graph_tokens * num_ranks

    xs, tis, tws = make_inputs(
        1, rank, args.graph_tokens, args.hidden, args.num_experts, args.num_topk
    )
    for _ in range(2):
        dispatch(buffer, xs, tis, tws, args.num_experts, worst, config)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recv_st, _, _, _ = dispatch(
            buffer, xs, tis, tws, args.num_experts, worst, config
        )
    torch.cuda.synchronize()
    dist.barrier(group)

    blocker = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    wrong = 0
    first = None
    for trial in range(args.trials):
        x, ti, tw = make_inputs(
            100 + trial, rank, args.graph_tokens, args.hidden,
            args.num_experts, args.num_topk,
        )
        xs.copy_(x)
        tis.copy_(ti)
        tws.copy_(tw)
        xe, tie, twe = make_inputs(
            1000 + trial, rank, args.eager_tokens, args.hidden,
            args.num_experts, args.num_topk,
        )
        # What this rank must receive from the eager dispatch.
        ntpr, _, _, in_rank, _ = buffer.get_dispatch_layout(tie, args.num_experts)
        sent = ntpr.clone()
        gathered = [torch.empty_like(sent) for _ in range(num_ranks)]
        dist.all_gather(gathered, sent, group=group)
        expected = int(sum(g[rank].item() for g in gathered))
        torch.cuda.synchronize()
        dist.barrier(group)

        for _ in range(args.delay_matmuls):
            blocker @ blocker
        graph.replay()
        recv_x, _, _, _ = dispatch(
            buffer, xe, tie, twe, args.num_experts, 0, config
        )
        got = recv_x.size(0)
        if got != expected:
            wrong += 1
            if first is None:
                first = (trial, got, expected)
                # Report before synchronizing: a short receive buffer can
                # fault the dispatch kernels that are still running.
                print(
                    f"[rank {rank}] trial={trial} eager dispatch sized for "
                    f"{got} tokens, routing sends {expected}",
                    flush=True,
                )
        torch.cuda.synchronize()
    total = torch.tensor([wrong], device="cuda")
    dist.all_reduce(total, group=group)
    print(
        f"[rank {rank}] wrong_recv_counts={wrong}/{args.trials} first={first}",
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier(group)
    del graph
    ok = total.item() == 0
    if rank == 0:
        print(
            "INTERLEAVE PASS" if ok else f"INTERLEAVE FAIL total_wrong={total.item()}",
            flush=True,
        )
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
