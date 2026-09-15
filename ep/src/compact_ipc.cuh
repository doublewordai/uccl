template <typename T>
__device__ __forceinline__ T* compact_peer(T* p, void** ipc,
                                         int rank, int peer, int peers) {
  auto remote = uccl::get_ipc_p2p_ptr(reinterpret_cast<uint64_t>(p),
                                     ipc, rank, peer, peers, 0);
  EP_DEVICE_ASSERT(remote != 0);
  return reinterpret_cast<T*>(remote);
}

template <int kHidden>
__global__ __launch_bounds__(256, 1) void compact_dispatch_kernel(
    uccl::CompactIPCLayout workspace, void const* x, int64_t const* ids,
    float const* weights, void* output_q, float* output_scales,
    int64_t* output_ids, float* output_weights, int tokens, int capacity,
    int topk, int experts, int rank, int ranks, int peers, void** ipc) {
  constexpr int kThreads = 256, kScales = kHidden / 128;
  int const tid = threadIdx.x;
  if (blockIdx.x == 0 && tid < ranks) workspace.combine_ready[tid] = 0;
  for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
    if (tid < topk) {
      int64_t id = __ldg(ids + token * topk + tid);
      EP_DEVICE_ASSERT(id >= -1 && id < experts);
      workspace.ids[token * topk + tid] = id;
      workspace.weights[token * topk + tid] = __ldg(weights + token * topk + tid);
    }
    for (int col = tid; col < kHidden / 8; col += kThreads) {
      int4 value = __ldg(static_cast<int4 const*>(x) + token * (kHidden / 8) + col);
      auto const* bf = reinterpret_cast<nv_bfloat16 const*>(&value);
      float f[8], amax = 1e-10f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        f[j] = static_cast<float>(bf[j]);
        amax = fmaxf(amax, fabsf(f[j]));
      }
      amax = warp_reduce_max<16>(amax);
      float scale = __fdiv_rn(amax, 448.0f);
      if (tid % 16 == 0) workspace.scales[token * kScales + col / 16] = scale;
      int2 quantized;
      auto* q2 = reinterpret_cast<__nv_fp8x2_storage_t*>(&quantized);
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        float2 pair = {__fdiv_rn(f[j], scale), __fdiv_rn(f[j + 1], scale)};
        q2[j / 2] = __nv_cvt_float2_to_fp8x2(pair, __NV_SATFINITE, __NV_E4M3);
      }
      reinterpret_cast<int2*>(workspace.q)[token * (kHidden / 8) + col] = quantized;
    }
  }
  __threadfence_system();
  cg::this_grid().sync();
  if (blockIdx.x == 0 && tid < ranks)
    st_release_sys_global(compact_peer(workspace.dispatch_ready + rank, ipc, rank, tid, peers), tokens + 1);
  __shared__ int source_tokens[NUM_MAX_NVL_PEERS];
  if (tid < ranks) {
    int value;
    while ((value = ld_acquire_sys_global(workspace.dispatch_ready + tid)) == 0) {}
    source_tokens[tid] = value - 1;
    EP_DEVICE_ASSERT(value > 0 && value <= capacity + 1);
  }
  __syncthreads();
  for (int row = blockIdx.x; row < ranks * capacity; row += gridDim.x) {
    int const source = row / capacity, token = row % capacity;
    bool const valid = token < source_tokens[source];
    auto* q = compact_peer(workspace.q, ipc, rank, source, peers);
    auto* scales = compact_peer(workspace.scales, ipc, rank, source, peers);
    auto* route_ids = compact_peer(workspace.ids, ipc, rank, source, peers);
    auto* route_weights = compact_peer(workspace.weights, ipc, rank, source, peers);
    for (int col = tid; col < kHidden / 16; col += kThreads) {
      int4 value = valid ? ld_nc_global(reinterpret_cast<int4*>(q) + token * (kHidden / 16) + col) : make_int4(0, 0, 0, 0);
      static_cast<int4*>(output_q)[row * (kHidden / 16) + col] = value;
    }
    if (tid < kScales)
      output_scales[row * kScales + tid] = valid ? ld_nc_global(scales + token * kScales + tid) : 1.0f;
    if (tid < topk) {
      output_ids[row * topk + tid] = valid ? ld_nc_global(route_ids + token * topk + tid) : -1;
      output_weights[row * topk + tid] = valid ? ld_nc_global(route_weights + token * topk + tid) : 0.0f;
    }
  }
}

template <int kHidden>
__global__ __launch_bounds__(256, 1) void compact_combine_kernel(
    uccl::CompactIPCLayout workspace, void* output, int tokens, int capacity,
    int rank, int ranks, int peers, void** ipc) {
  constexpr int kThreads = 256, kVectors = kHidden / 8;
  int const tid = threadIdx.x;
  // The preceding expert kernel writes directly into partial on this stream.
  // Publish only after cleanup, then pull the rank's contiguous output slice.
  if (blockIdx.x == 0) {
    if (tid < ranks) workspace.dispatch_ready[tid] = 0;
    __threadfence_system();
    __syncthreads();
    if (tid < ranks)
      st_release_sys_global(compact_peer(workspace.combine_ready + rank, ipc, rank, tid, peers), 1);
  }
  if (tid < ranks)
    while (ld_acquire_sys_global(workspace.combine_ready + tid) == 0) {}
  __syncthreads();
  for (int index = blockIdx.x * kThreads + tid; index < tokens * kVectors; index += gridDim.x * kThreads) {
    float sums[8] = {};
    for (int source = 0; source < ranks; ++source) {
      auto* partial = compact_peer(static_cast<int4*>(workspace.partial), ipc, rank, source, peers);
      int4 value = ld_nc_global(partial + rank * capacity * kVectors + index);
      auto const* bf = reinterpret_cast<nv_bfloat16 const*>(&value);
#pragma unroll
      for (int j = 0; j < 8; ++j) sums[j] += static_cast<float>(bf[j]);
    }
    int4 value;
    auto* bf = reinterpret_cast<nv_bfloat16*>(&value);
#pragma unroll
    for (int j = 0; j < 8; ++j) bf[j] = static_cast<nv_bfloat16>(sums[j]);
    static_cast<int4*>(output)[index] = value;
  }
}

void compact_dispatch(uccl::CompactIPCLayout workspace, void const* x,
                      int64_t const* ids, float const* weights, void* output_q,
                      float* output_scales, int64_t* output_ids, float* output_weights,
                      int tokens, int capacity, int hidden, int topk, int experts,
                      int rank, int ranks, int peers, void** ipc, cudaStream_t stream) {
  SETUP_LAUNCH_CONFIG(ranks * capacity, 256, stream);
#define COMPACT_DISPATCH_CASE(hidden_size) \
  LAUNCH_KERNEL(&cfg, compact_dispatch_kernel<hidden_size>, workspace, x, ids, weights, \
    output_q, output_scales, output_ids, output_weights, tokens, capacity, topk, experts, \
    rank, ranks, peers, ipc); break
  SWITCH_HIDDEN(COMPACT_DISPATCH_CASE);
#undef COMPACT_DISPATCH_CASE
}

void compact_combine(uccl::CompactIPCLayout workspace, void* output,
                     int tokens, int capacity, int hidden, int rank, int ranks,
                     int peers, void** ipc, cudaStream_t stream) {
  int blocks = std::max(1, std::min(128, (tokens * (hidden / 8) + 255) / 256));
  SETUP_LAUNCH_CONFIG(blocks, 256, stream);
#define COMPACT_COMBINE_CASE(hidden_size) \
  LAUNCH_KERNEL(&cfg, compact_combine_kernel<hidden_size>, workspace, output, tokens, capacity, \
    rank, ranks, peers, ipc); break
  SWITCH_HIDDEN(COMPACT_COMBINE_CASE);
#undef COMPACT_COMBINE_CASE
}
