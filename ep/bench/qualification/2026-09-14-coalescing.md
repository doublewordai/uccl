# Coalesced low-latency transport qualification

This optional path sends one dispatch payload per token and remote GPU, then lets
multiple local experts consume that staged payload. Dispatch and combine can also
read their receive staging directly instead of copying every payload through a
second receive arena. The expert arithmetic and weighted combine are unchanged.

All five build switches default to zero:

```
LANE_E_DESTRANK_COALESCE=1 LANE_E_COMBINE_COALESCE=1
LANE_E_DISPATCH_DEDUP=1 LANE_E_DISPATCH_INDIRECT=1 LANE_E_COMBINE_INDIRECT=1
```

Dispatch coalescing and `PER_EXPERT_BATCHING` are mutually exclusive. Remote
coalescing supports top-k at most 8 and at most 64 experts per GPU. Every rank
must use the same build: deduplication changes the staged dispatch header.
Single-node dispatch does not use the remote expert mask. Relative to the existing coalesced path, buffer
capacity and staging lifetime are retained; indirection replaces payload copies
with message indices in the old receive slots. There are no extra asynchronous
consumers of the staging arena.

## Complete-MoE result

Two Isambard nodes, eight GH200 GPUs (four per node), Slingshot/CXI. DeepSeek-V4
geometry: hidden 7168, intermediate 3072, 384 experts, top-k 6, MXFP4 expert
weights, BF16 dispatch/combine. Weights use the existing deterministic complete
MoE reference recipe. These are full-layer latency measurements, **not** model
serving throughput or full-model validation.

At 128 tokens per rank, the uninstrumented, alternating A/B/B/A run gave:

| Variant | First p50 | Second p50 | Mean of run p50s |
| --- | ---: | ---: | ---: |
| Existing coalesced UCCL | 2078.247 us | 2092.119 us | 2085.183 us |
| Dedup + both indirect reads | 1936.603 us | 1928.988 us | 1932.795 us |

That is **7.31% lower complete-MoE latency**. Each run includes dispatch
quantization/communication, both expert GEMMs and SwiGLU, and the weighted BF16
combine. It uses 12 warmups and 80 measured iterations, with device synchronization
and max-over-ranks wall latency; the outer rank barrier is outside timing.
No CUDA-graph timing was substituted for the original eager benchmark.

AgRs remains faster at this shape (1736.272 us in the phase investigation).
At 32 tokens per rank, a separate alternating comparison found no consistent
improvement: the control run p50s were 1186.640/1182.992 us, the new path
1181.711/1191.390 us, and AgRs 1167.969/1170.495 us. This patch does not establish
superiority over AgRs or the strongest complete-MoE backend across the matrix.

At 512 tokens per rank, the new LL path remained slower than both controls:
5377.057/5436.159 us versus UCCL HT 3995.877/4021.813 us and AgRs
4231.407/4251.919 us. Those alternating runs are also included.

The phase-instrumented ablations isolate the mechanism. Deduplication reduced
rank-0 dispatch from 434–453 us to 359–374 us while expert computation stayed
around 1.2 ms. Removing the intermediate dispatch copy further reduced dispatch
to 315–318 us, although compute variability reduced the net whole-layer gain.
Removing the combine copy added another modest whole-layer gain. All phase runs
and their uninstrumented confirmation are retained in the data files.

The performance candidate is `049c8024`, on the integration base `8b4623fb`.
The standalone PR is based on `upstream-base` and includes the previously
integrated coalescing prerequisite so its diff is self-contained. The PR's SM90
device code matches the qualified integration code; subsequent host-side guards
reject unsupported remote expert counts and conflicting build switches.

## Correctness and reproducibility

Both the enabled and all-disabled standalone builds compiled with GCC 13.2 and
CUDA 13.0.88. Both passed the exact BF16 graph regression; the enabled build also
passed FP8. [Build and graph evidence](2026-09-14-coalescing-graphs.json) records
the extension/test hashes, command arguments and results.

`../test_ll_token_dedup.py` checks ten eager round trips, a graph containing two
round trips (both ping-pong buffers), 24 changing-input replays, and genuine
zero-token/ragged inputs. Tests cover the high mask bit 63, invalid routes,
duplicate expert IDs bounded by the existing per-expert capacity, exact expert
counts and source-token indices. `--unbarriered` omits inter-rank barriers between
graph replays. BF16 with power-of-two uniform routing weights is checked exactly
against the expected BF16 result. FP8 uses the same mathematical input reference,
with a global observed relative-L2 maximum of 0.025362 (limit 0.045).

Run the script under the existing distributed launcher with eight ranks:

```
python ep/bench/test_ll_token_dedup.py --hidden 7168 --experts 512 --topk 8 --duplicates --unbarriered
python ep/bench/test_ll_token_dedup.py --hidden 6144 --experts 512 --topk 8 --fp8 --duplicates --unbarriered
```

All complete-MoE runs retain the original full-output reference check: relative
L2 below 0.02 for DeepSeek (0.075 for GLM). The published JSON includes observed
errors, reference identity, first-expert weight hashes, extension/harness hashes and
all max-rank timing samples. The original complete-MoE harness and captured
routing inputs are campaign artifacts; the included standalone test reproduces
the transport correctness checks, not those full-layer timings.

There are small cross-run differences in complete-MoE outputs. Across 72 paired
rank outputs, relative L2 was below 0.0001; unchanged-control pairs showed the
same scale of variation. Do not interpret this as bitwise full-layer equivalence
or a one-ULP guarantee, especially near zero. See the paired-output file for
maximum absolute differences, ULP distances and differing-element counts.

- [Run summary](2026-09-14-coalescing.csv)
- [Timing samples and provenance](2026-09-14-coalescing.json)
- [Paired output checks](2026-09-14-coalescing-paired-outputs.json)
