# Node-shared low-latency dispatch

This opt-in extension sends one input payload per token and remote destination node.
The node's GPUs read their experts' rows through the gateway GPU's existing IPC
mapping. Expert computation and weighted combine are unchanged.

Enable the existing coalesced/indirect path and the additional default-off switch:

```
LANE_E_DESTRANK_COALESCE=1 LANE_E_DISPATCH_DEDUP=1
LANE_E_DISPATCH_INDIRECT=1 LANE_E_COMBINE_COALESCE=1
LANE_E_COMBINE_INDIRECT=1 LANE_E_DISPATCH_NODE=1
```

All ranks must use the same build. The tested topology uses four GH200 GPUs per
node (`NUM_MAX_NVL_PEERS=4`) and Slingshot/CXI. Remote dispatch supports top-k at
most 8 and at most 256 experts per node. The header contains a list of byte-sized
node-local expert IDs; repeated expert IDs remain explicit. Local-node routes
retain the existing direct IPC path.

This mode requires a nonzero NVLink allocation. For the Python Buffer API, use
`num_nvl_bytes=Buffer.get_dispatch_config(world).get_nvl_buffer_size_hint(H*2, world)`
as in the standalone regression test. The new option defaults to zero and does
not change the default buffer allocation or dispatch behavior.

## Readiness and buffer reuse

The gateway is the destination GPU with the same node-local index as the source
GPU. After acquiring the NIC arrival flag, it publishes the immutable packet
count and a generation with system-scope release stores. Other local GPUs acquire
that generation before reading the payload.

A small suffix of the NVLink allocation holds controls for each source and both
ping-pong buffers. It is independent of token capacity, payload precision and
hidden dimension. After a grid barrier joins its payload readers, each GPU
acknowledges the generation; the gateway cannot finish until all local readers
have acknowledged. Generations advance on the GPU, including under graph replay,
and do not use a clear-to-zero handshake. The existing collective ordering and
buffer-reuse contract still applies; arbitrary overlapping dispatches are not
supported by this mechanism.

## Direct combine packing

`LANE_E_COMBINE_DIRECT_SEND=1` is another default-off option, requiring
`LANE_E_COMBINE_COALESCE=1`. It packs remote BF16 results directly from the supplied
expert-output tensor into the coalesced send stage. Local routes still copy that
tensor through IPC. The redundant full per-expert combine send arena and its
payload copy are removed; the separate coalesced send stage is retained.

The native C++ entry point rejects `zero_copy=true` and `use_logfmt=true` in this
mode: those paths require the removed arena or its LogFMT transformation. The
Python adapter's existing staged-tensor compatibility path still works and is
included in the graph regression using `--zero-copy`.

At DeepSeek EP16 with 2048 tokens per rank, the layout change saves
14,965,211,136 bytes (13.94 GiB) per GPU. Without it, the node-only benchmark failed
while allocating `packed_recv_x`: 10.50 GiB requested, 2.04 GiB free after allocator
retry. With direct packing it completes and passes the unchanged full-output
reference. The first qualified run took 23.43 ms, still slower than the strongest
HT and AgRs controls. This allocation improvement is not a claim of winning that
performance case. The 2048-token rows and the final 128-token ablation with a node-sharing control below use this additional option. The small-batch ablation is nearly flat; the allocation saving is the main demonstrated benefit.

## Measured scope

Measurements use the existing complete-MoE harness, original frozen full-output
reference, and deterministic expert weights. They include dispatch quantization,
communication, both GEMMs, SwiGLU and weighted BF16 combine. Timings are eager
wall latency, with GPU synchronization and a maximum across ranks, excluding the
outer rank barrier: 12 warmups and 80 measured iterations per run. They are not
model-serving throughput or full-model validation.

<!-- measurements -->
Full-layer p50 latency in milliseconds; each cell shows both runs:

| Model | GPUs | Tokens/rank | UCCL control | Control | Candidate | AgRs |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| DeepSeek | 8 | 128 | Per-GPU dedup | 1.934 / 1.916 | 1.850 / 1.888 | 1.720 / 1.715 |
| DeepSeek | 8 | 512 | Per-GPU dedup | 5.397 / 5.406 | 4.994 / 4.989 | 4.252 / 4.262 |
| GLM | 8 | 128 | Per-GPU dedup | 1.199 / 1.188 | 1.151 / 1.145 | 1.439 / 1.441 |
| DeepSeek | 16 | 128 | Per-GPU dedup | 2.169 / 2.089 | 1.959 / 1.997 | 1.843 / 1.844 |
| DeepSeek | 16 | 512 | HT | 6.445 / 6.450 | 6.231 / 6.290 | 5.356 / 5.349 |
| GLM | 16 | 128 | Per-GPU dedup | 1.501 / 1.477 | 1.507 / 1.386 | 1.743 / 1.743 |
| GLM | 16 | 512 | HT | 4.569 / 4.598 | 4.478 / 4.609 | 5.225 / 5.210 |
| DeepSeek | 16 | 2048 | HT | 21.865 / 21.866 | 23.180 / 22.752 | 19.228 / 19.205 |
| GLM | 16 | 2048 | HT | 15.140 / 15.137 | 20.995 / 21.121 | 19.541 / 19.540 |
| DeepSeek | 16 | 128 | Node sharing | 1.970 / 1.999 | 1.965 / 1.959 | 1.843 / 1.844 |
<!-- end measurements -->

DeepSeek EP8 at 512 tokens improves 7.59% over per-GPU deduplication, but remains slower than both AgRs and the previously qualified HT path. DeepSeek EP16 at 512 tokens improves about 2.9% over HT, while AgRs remains faster. GLM EP16 at 128 and 512 tokens does not show a consistent improvement; the largest cases remain slower than the strongest controls.

Paired comparisons and larger-topology qualification are recorded in the data
files below. Node sharing improves selected cases over per-GPU deduplication;
DeepSeek still has cases where AgRs is faster. This is not a claim that UCCL or
megakernel wins the complete matrix.

## Correctness

`../test_ll_node_dispatch.py` checks 10 changing-input eager calls, 96 captured
calls spanning both ping-pong buffers and both BF16/FP8 payloads, true empty and
ragged ranks, capacity changes, duplicate routes, and exact expert/source-token
metadata. Hot-reader cases route remote traffic onto only one local GPU, including
node-local expert ID 255. Uniform power-of-two weights make BF16 results exact;
FP8 retains the existing relative-L2 limit of 0.045. Both event-wait and recv-hook
paths have been exercised; direct combine packing also passes with the Python
staged-buffer compatibility path. The node-enabled integration build passes on 8 and
16 GPUs; the default-off integration build also passes on 8 GPUs. The final
standalone PR build with node sharing and direct combine passes on 16 GPUs,
and its all-default-off build passes on 8 GPUs.

Example under the existing distributed launcher with 16 ranks:

```
python ep/bench/test_ll_node_dispatch.py --tokens 128 --hidden 7168 --experts 1024 --topk 8 --duplicates --unbarriered --recv-hook
```

Full-MoE reference limits remain relative L2 <0.02 for DeepSeek and <0.075 for
GLM. All measured outputs must be finite. Cross-run bitwise equality is not
assumed; the data retain the observed errors and reference/weight identities.

- [Run summary](2026-09-14-node-dispatch.csv)
- [Timing samples and provenance](2026-09-14-node-dispatch.json)

All 200 saved-output comparisons pass the unchanged paired relative-L2 limit of 0.0001; unchanged-control pairs are included. They are not all bitwise equal. The standalone PR uses the same GCC 13 compiler-selection overlay as the earlier coalescing qualification because its upstream base hardcodes the system compiler. The integration branch already includes the separate compiler-selection fix and large-LL-write splitting fix.

- [Build and graph checks](2026-09-14-node-dispatch-graphs.json)
- [Paired output comparisons](2026-09-14-node-dispatch-paired-outputs.json)
- [Allocation failure and layout saving](2026-09-14-node-dispatch-memory.json)
