# GH200 MoE build and Python interface

`uccl.ep_api` is the installed Python interface. It exports `Buffer`, `Config`,
`EventHandle`, and `EventOverlap`. The old `ep/bench/buffer.py` import remains
available for benchmark compatibility; an inference engine should use the
installed package.

The Doubleword integration branch carries the four-GPU-per-node CXI profile.
Build its exact source revision in an environment with Python 3.12+, PyTorch,
CUDA 13, nanobind 2.10.2, setuptools, wheel, libfabric/CXI, libibverbs, libnl,
and NUMA headers/libraries:

```sh
export CUDA_HOME=/usr/local/cuda
export LIBFABRIC_HOME=/path/to/libfabric
# Set CPATH/LIBRARY_PATH if development headers/libraries use other prefixes.
ep/scripts/build-gh200-moe-wheel.sh /path/to/wheels
python3 -m pip install /path/to/wheels/uccl-*.whl
```

The script cleans prior objects, builds the GH200 profile, and packages its
native extension. Ordinary root-package `pip install` packages an existing
extension; it does not compile this profile. All ranks must use the same source
revision and build flags because the coalescing flags change the wire format.
The profile depends on the coalesced/node-shared dispatch and direct-combine
patches in the integration stack. It is not a generic build for other devices.

At runtime select `UCCL_EP_TRANSPORT=cxi` and configure the site's libfabric and
CUDA library paths. The native profile fixes four GPUs per node and supports
the qualified EP4/8/16 geometries. vLLM owns the shape-selection policy.

```python
from uccl.ep_api import Buffer
from uccl.ep_api.compact import CompactIPC
```

`CompactIPC` handles small batches on one host. A standalone instance owns its
LL registration. To share an existing LL registration, reserve a 256-byte-aligned
suffix beyond `Buffer.get_low_latency_rdma_size_hint(...)` and pass that buffer
and `workspace_offset` to `CompactIPC`. The ordinary LL arena must fit before
the offset. Reserve 32 MiB for the compact suffix. Compute writes the weighted
local BF16 result directly into `compact.partial`; `compact.combine(out)` then
sums those partials across peers. Dispatch and combine alternate on the current
CUDA stream. The shared buffer remains owned by its creator.

The independent communication oracle is runnable against an installed wheel:

```sh
python3 -m torch.distributed.run --nproc_per_node=4 \
  --master-addr=127.0.0.1 --master-port=29591 ep/bench/test_compact_ipc.py
UCCL_TEST_COMPACT_SHARED_LL=1 python3 -m torch.distributed.run \
  --nproc_per_node=4 --master-addr=127.0.0.1 --master-port=29592 \
  ep/bench/test_compact_ipc.py
```

Both modes check 1,024 captured round trips with changing inputs, scales,
routes, weights, empty/ragged batches, and negative/duplicate routes.
