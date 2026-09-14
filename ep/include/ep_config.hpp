#pragma once
#include "ep_configs.cuh"
#include "ep_util.hpp"
#include "internode.cuh"
#include <cstddef>
#include <cstdint>
#include <utility>

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

namespace uccl {

template <typename dtype_t>
dtype_t ceil_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align(dtype_t a, dtype_t b) {
  return ceil_div<dtype_t>(a, b) * b;
}

// thirdparty/DeepEP/csrc/config.hpp
struct Config {
  int num_sms;
  int num_max_nvl_chunked_send_tokens;
  int num_max_nvl_chunked_recv_tokens;
  int num_max_rdma_chunked_send_tokens;
  int num_max_rdma_chunked_recv_tokens;

  Config(int num_sms, int num_max_nvl_chunked_send_tokens,
         int num_max_nvl_chunked_recv_tokens,
         int num_max_rdma_chunked_send_tokens,
         int num_max_rdma_chunked_recv_tokens)
      : num_sms(num_sms),
        num_max_nvl_chunked_send_tokens(num_max_nvl_chunked_send_tokens),
        num_max_nvl_chunked_recv_tokens(num_max_nvl_chunked_recv_tokens),
        num_max_rdma_chunked_send_tokens(num_max_rdma_chunked_send_tokens),
        num_max_rdma_chunked_recv_tokens(num_max_rdma_chunked_recv_tokens) {
    EP_HOST_ASSERT(num_sms >= 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens > 0 and
                   num_max_nvl_chunked_recv_tokens > 0);
    EP_HOST_ASSERT(num_max_nvl_chunked_send_tokens <
                   num_max_nvl_chunked_recv_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens > 0 and
                   num_max_rdma_chunked_recv_tokens > 0);

    // Ceil up RDMA buffer size
    this->num_max_rdma_chunked_recv_tokens = align<int>(
        num_max_rdma_chunked_recv_tokens, num_max_rdma_chunked_send_tokens);
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <
                   num_max_rdma_chunked_recv_tokens);
    // NOTES: this assertion is related to RDMA lazy head update, we must ensure
    // senders always have space to push
    EP_HOST_ASSERT(num_max_rdma_chunked_send_tokens <=
                   num_max_rdma_chunked_recv_tokens / 2);
  }

  size_t get_nvl_buffer_size_hint(size_t hidden_bytes, int num_ranks) const {
    // Below are some assumptions
    // TODO: add assertions
    constexpr int kNumMaxTopK = 128;
    constexpr int kNumMaxScales = 128;
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or
                   num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(num_ranks <= NUM_MAX_NVL_PEERS or num_sms % 2 == 0);
    auto const num_rdma_ranks = std::max(num_ranks / NUM_MAX_NVL_PEERS, 1);
    auto const num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
    int const num_channels = num_sms / 2;

    size_t num_bytes = 0;
    num_bytes +=
        num_channels * num_nvl_ranks * (2 * num_rdma_ranks + 3) * sizeof(int);
    num_bytes += num_channels * num_nvl_ranks *
                 num_max_nvl_chunked_recv_tokens * hidden_bytes;
#ifndef DISABLE_NVSHMEM
    num_bytes += num_channels * num_nvl_ranks *
                 num_max_nvl_chunked_recv_tokens *
                 uccl::internode::get_source_meta_bytes();
#endif
    num_bytes += num_channels * num_nvl_ranks *
                 num_max_nvl_chunked_recv_tokens * kNumMaxTopK *
                 sizeof(int64_t);
    num_bytes += num_channels * num_nvl_ranks *
                 num_max_nvl_chunked_recv_tokens * kNumMaxTopK * sizeof(float);
    num_bytes += num_channels * num_nvl_ranks *
                 num_max_nvl_chunked_recv_tokens * kNumMaxScales *
                 sizeof(float);
    num_bytes = ((num_bytes + 127) / 128) * 128;
    return num_bytes;
  }

  size_t get_rdma_buffer_size_hint(int64_t hidden_bytes, int num_ranks) const {
#ifndef DISABLE_NVSHMEM
    // Legacy mode
    if (num_ranks <= NUM_MAX_NVL_PEERS) return 0;

    // Below are some assumptions
    // TODO: add assertions
    constexpr int kNumMaxTopK = 128;
    constexpr int kNumMaxScales = 128;
    EP_HOST_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0);
    EP_HOST_ASSERT(num_sms % 2 == 0);
    int const num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
    int const num_channels = num_sms / 2;

    size_t num_bytes = 0;
    num_bytes += num_channels * num_rdma_ranks * (NUM_MAX_NVL_PEERS * 2 + 2) *
                 2 * sizeof(int);
    num_bytes += num_channels * num_rdma_ranks *
                 num_max_rdma_chunked_recv_tokens * hidden_bytes * 2;
    num_bytes += num_channels * num_rdma_ranks *
                 num_max_rdma_chunked_recv_tokens *
                 uccl::internode::get_source_meta_bytes() * 2;
    num_bytes += num_channels * num_rdma_ranks *
                 num_max_rdma_chunked_recv_tokens * kNumMaxTopK *
                 sizeof(int64_t) * 2;
    num_bytes += num_channels * num_rdma_ranks *
                 num_max_rdma_chunked_recv_tokens * kNumMaxTopK *
                 sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks *
                 num_max_rdma_chunked_recv_tokens * kNumMaxScales *
                 sizeof(float) * 2;
    num_bytes += num_channels * num_rdma_ranks *
                 num_max_rdma_chunked_recv_tokens * sizeof(int4) * 2;
    num_bytes = ((num_bytes + 127) / 128) * 128;
    return num_bytes;
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disable during compilation");
#endif
  }
};

// thirdparty/DeepEP/csrc/config.hpp
struct LowLatencyBuffer {
  int num_clean_int = 0;

  void* dispatch_rdma_send_buffer = nullptr;
  void* dispatch_rdma_recv_data_buffer = nullptr;
  int* dispatch_rdma_recv_count_buffer = nullptr;
  // Internode signaling buffer used by RDMA atomics (must be 64-bit / 8B
  // aligned).
  int64_t* dispatch_rdma_recv_count_buffer_internode = nullptr;

  void* combine_rdma_send_buffer = nullptr;
  void* combine_rdma_recv_data_buffer = nullptr;
  int* combine_rdma_recv_flag_buffer = nullptr;
  // Internode signaling buffer used by RDMA atomics (must be 64-bit / 8B
  // aligned).
  int64_t* combine_rdma_recv_flag_buffer_internode = nullptr;

  void* combine_rdma_send_buffer_data_start = nullptr;
  size_t num_bytes_per_combine_msg = 0;

  // Lane E dest-rank coalescing (LANE_E_DESTRANK_COALESCE). All default null
  // and unused unless that path is compiled in.
  //  - dispatch_rdma_x_stage: per-dst-rank contiguous send staging, lives
  //    inside the dispatch send buffer just past the per-token temp region.
  //  - dispatch_recv_stage: symmetric per-src-rank contiguous recv staging,
  //    the target of the coalesced put (one large write per remote dst rank).
  //  - dispatch_stage_flag_internode: per-src-rank arrival flag, an RDMA-atomic
  //    target carved out of the internode signaling region just past the
  //    per-expert counts.
  void* dispatch_rdma_x_stage = nullptr;
  void* dispatch_recv_stage = nullptr;
  int64_t* dispatch_stage_flag_internode = nullptr;

  // Lane E combine-side coalescing (LANE_E_COMBINE_COALESCE). Same shape as the
  // dispatch staging but for the bf16 combine message, with a 16-byte
  // (global_expert, src_idx) header per staged token so the recv scatter can
  // place it at rdma_recv_x[global_expert*maxtok + src_idx].
  void* combine_send_stage = nullptr;
  void* combine_recv_stage = nullptr;
  int64_t* combine_stage_flag_internode = nullptr;

  // TODO(MaoZiming)
  std::tuple<int*, int64_t*, int> clean_meta() {
    EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer ==
                   combine_rdma_recv_flag_buffer);
    EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer_internode ==
                   combine_rdma_recv_flag_buffer_internode);
    return std::make_tuple(dispatch_rdma_recv_count_buffer,
                           dispatch_rdma_recv_count_buffer_internode,
                           num_clean_int);
  }
};

struct LowLatencyLayout {
  size_t total_bytes = 0;
  LowLatencyBuffer buffers[2];

  template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*,
            typename in_ptr_t = void*>
  out_ptr_t advance(in_ptr_t const& ptr, size_t count) {
    return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) +
                                       count);
  }

  LowLatencyLayout(void* rdma_buffer, int num_max_dispatch_tokens_per_rank,
                   int hidden, int num_ranks, int num_experts,
                   void* atomic_buffer_ptr = nullptr) {
    int const num_scales = hidden / 128;

    // Dispatch and combine layout:
    //  - 2 symmetric odd/even send buffer
    //  - 2 symmetric odd/even receive buffers
    //  - 2 symmetric odd/even signaling buffers

    // Message sizes
    // NOTES: you should add a control `int4` for combine messages if you want
    // to do data transformation
    EP_HOST_ASSERT(num_scales * sizeof(float) <= static_cast<size_t>(hidden));
    size_t num_bytes_per_dispatch_msg =
        sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16),
                                hidden + num_scales * sizeof(float));
    size_t num_bytes_per_combine_msg = hidden * sizeof(nv_bfloat16);

#ifdef LANE_E_DESTRANK_COALESCE
    // Dest-rank coalescing staging sizes. A rank stages, per destination rank,
    // one message per token per local expert of that destination, bounded by
    // the top-k fan-out. GLM decode uses k=8; a per-token cap of 8 covers it
    // (dispatch launch asserts num_topk <= 8 when this path is active).
    int const num_local_experts_ll = num_experts / num_ranks;
    size_t const coalesce_max_msgs_per_dst =
        static_cast<size_t>(num_max_dispatch_tokens_per_rank) *
        static_cast<size_t>(std::min(num_local_experts_ll, 8));
    size_t const coalesce_stage_bytes_per_rank =
        coalesce_max_msgs_per_dst * num_bytes_per_dispatch_msg;
    size_t const coalesce_stage_region_bytes =
        static_cast<size_t>(num_ranks) * coalesce_stage_bytes_per_rank;
#endif

#ifdef LANE_E_COMBINE_COALESCE
    // Combine staging: bf16 message + a 16-byte (global_expert, src_idx) header.
    int const num_local_experts_cmb = num_experts / num_ranks;
    size_t const cmb_max_msgs_per_dst =
        static_cast<size_t>(num_max_dispatch_tokens_per_rank) *
        static_cast<size_t>(std::min(num_local_experts_cmb, 8));
    size_t const cmb_stage_entry_bytes =
        sizeof(int4) + num_bytes_per_combine_msg;
    size_t const cmb_stage_bytes_per_rank =
        cmb_max_msgs_per_dst * cmb_stage_entry_bytes;
    size_t const cmb_stage_region_bytes =
        static_cast<size_t>(num_ranks) * cmb_stage_bytes_per_rank;
#endif

    // Send buffer
#if defined(LANE_E_DESTRANK_COALESCE)
    // Per-token temp region (FP8 cast target) followed by the per-dst-rank
    // send staging that the coalesced put reads from.
    size_t dispatch_send_buffer_bytes =
        static_cast<size_t>(num_max_dispatch_tokens_per_rank) *
            num_bytes_per_dispatch_msg +
        coalesce_stage_region_bytes;
#elif defined(PER_EXPERT_BATCHING)
    // Buffer layout for RDMA sends, used by the batched RDMA-send path in the
    // dispatch-LL kernel.
    // clang-format off
    // ┌──────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
    // │ Temp buffer (offset 0)                   │ Per-expert RDMA batch buffer (offset num_max_token)      │
    // │ rdma_x[token_idx]                        │ rdma_x[num_max_token + expert * num_max_token + slot]    │
    // │ Size: num_max_token * msg_size           │ Size: num_experts * num_max_token * msg_size             │
    // └──────────────────────────────────────────┴──────────────────────────────────────────────────────────┘
    // clang-format on
    // Flow: (optional FP8 cast) -> temp buffer -> copy to per-expert batch
    // buffer -> batched RDMA send
    // TODO: Support per-GPU destination batching in this path.
    size_t dispatch_send_buffer_bytes = (num_experts + 1) *
                                        num_max_dispatch_tokens_per_rank *
                                        num_bytes_per_dispatch_msg;
#else
    // Send buffer
    // Uses only per-token temporary send buffer.
    size_t dispatch_send_buffer_bytes =
        num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
#endif
    size_t combine_send_buffer_bytes = num_experts *
                                       num_max_dispatch_tokens_per_rank *
                                       num_bytes_per_combine_msg;
    size_t send_buffer_bytes =
        std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
    EP_HOST_ASSERT(send_buffer_bytes % sizeof(int4) == 0);
    total_bytes += send_buffer_bytes * 2;

    // Symmetric receive buffers
    // TODO: optimize memory usages
    size_t dispatch_recv_data_buffer_bytes = num_experts *
                                             num_max_dispatch_tokens_per_rank *
                                             num_bytes_per_dispatch_msg;
    size_t combine_recv_buffer_bytes = num_experts *
                                       num_max_dispatch_tokens_per_rank *
                                       num_bytes_per_combine_msg;
    size_t recv_buffer_bytes =
        std::max(dispatch_recv_data_buffer_bytes, combine_recv_buffer_bytes);
    EP_HOST_ASSERT(recv_buffer_bytes % sizeof(int4) == 0);
    total_bytes += recv_buffer_bytes * 2;

#ifdef LANE_E_DESTRANK_COALESCE
    // Symmetric per-src-rank recv staging (target of the coalesced put), placed
    // after the send/recv buffers. Same size as the send staging region.
    size_t const recv_stage_region_bytes = coalesce_stage_region_bytes;
    EP_HOST_ASSERT(recv_stage_region_bytes % sizeof(int4) == 0);
    total_bytes += recv_stage_region_bytes * 2;
#endif

#ifdef LANE_E_COMBINE_COALESCE
    // Combine send + recv staging (both symmetric, double-buffered).
    EP_HOST_ASSERT(cmb_stage_region_bytes % sizeof(int4) == 0);
    total_bytes += cmb_stage_region_bytes * 2;  // send stage
    total_bytes += cmb_stage_region_bytes * 2;  // recv stage
#endif

    // Number of extra per-src arrival-flag slots appended after the per-expert
    // internode counts: num_ranks for dispatch coalescing, another num_ranks for
    // combine coalescing (separate slots — dispatch and combine reuse the same
    // buffer index within a step, so their flags must not alias).
    size_t coalesce_extra_flag_slots = 0;
#ifdef LANE_E_DESTRANK_COALESCE
    coalesce_extra_flag_slots += num_ranks;
#endif
#ifdef LANE_E_COMBINE_COALESCE
    coalesce_extra_flag_slots += num_ranks;
#endif

    // Symmetric signaling buffers
#if defined(LANE_E_DESTRANK_COALESCE) || defined(LANE_E_COMBINE_COALESCE)
    // Coalesce keeps the per-expert internode counts (written locally by the
    // recv scatter) and appends per-src arrival flags. The int signaling
    // buffer's element count drives the clean loop that zeroes both int and
    // int64 signaling regions, so size it to cover counts + all flags.
    size_t dispatch_recv_count_buffer_bytes =
        (static_cast<size_t>(num_experts) + coalesce_extra_flag_slots) *
        sizeof(int);
#elif defined(PER_EXPERT_BATCHING)
    // Batched dispatch needs one count per (dst_rank, src_rank).
    size_t dispatch_recv_count_buffer_bytes =
        static_cast<size_t>(num_ranks * num_ranks) * sizeof(int);
#else
    // Default dispatch path tracks one counter per expert.
    size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
#endif
    size_t combine_recv_flag_buffer_bytes = num_experts * sizeof(int);
    size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes,
                                             combine_recv_flag_buffer_bytes);
    size_t signaling_buffer_bytes_aligned =
        align<size_t>(signaling_buffer_bytes, 128);
    total_bytes += signaling_buffer_bytes_aligned * 2;

    // Internode signaling buffers (for RDMA atomics): use 64-bit slots.
#if defined(LANE_E_DESTRANK_COALESCE) || defined(LANE_E_COMBINE_COALESCE)
    // Per-expert internode counts/flags followed by the per-src arrival flags,
    // contiguous so the clean loop covers both.
    size_t dispatch_recv_count_buffer_bytes_internode =
        (static_cast<size_t>(num_experts) + coalesce_extra_flag_slots) *
        sizeof(int64_t);
#elif defined(PER_EXPERT_BATCHING)
    // Batched dispatch needs one count per (dst_rank, src_rank).
    size_t dispatch_recv_count_buffer_bytes_internode =
        static_cast<size_t>(num_ranks * num_ranks) * sizeof(int64_t);
#else
    // Legacy dispatch tracks one count per expert.
    size_t dispatch_recv_count_buffer_bytes_internode =
        num_experts * sizeof(int64_t);
#endif
    size_t combine_recv_flag_buffer_bytes_internode =
        num_experts * sizeof(int64_t);
    size_t signaling_buffer_bytes_internode =
        std::max(dispatch_recv_count_buffer_bytes_internode,
                 combine_recv_flag_buffer_bytes_internode);
    size_t signaling_buffer_bytes_internode_aligned =
        align<size_t>(signaling_buffer_bytes_internode, 128);
    // These internode signaling buffers live inside `atomic_buffer_ptr` (not
    // rdma_buffer). If they overflow `kAtomicBufferSize`, kernels will spin
    // forever waiting for flags.
    if (atomic_buffer_ptr != nullptr) {
      EP_HOST_ASSERT(2 * signaling_buffer_bytes_internode_aligned <=
                     static_cast<size_t>(kAtomicBufferSize));
    }

    // Assign pointers
    // NOTES: we still leave some space for distinguishing dispatch/combine
    // buffer, so you may see some parameters are duplicated
    for (int i = 0; i < 2; ++i) {
      buffers[i] = {
          static_cast<int>(signaling_buffer_bytes / sizeof(int)),
          advance(rdma_buffer,
                  signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 +
                                   send_buffer_bytes * 2 +
                                   recv_buffer_bytes * i),
          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
          reinterpret_cast<int64_t*>(
              (uint8_t*)atomic_buffer_ptr +
              i * signaling_buffer_bytes_internode_aligned), /* dispatch_rdma_recv_count_buffer_internode
                                                              */

          advance(rdma_buffer,
                  signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 +
                                   send_buffer_bytes * 2 +
                                   recv_buffer_bytes * i),
          advance<int*>(rdma_buffer, signaling_buffer_bytes_aligned * i),
          reinterpret_cast<int64_t*>(
              (uint8_t*)atomic_buffer_ptr +
              i * signaling_buffer_bytes_internode_aligned), /* combine_rdma_recv_flag_buffer_internode
                                                              */

          advance(rdma_buffer,
                  signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * i),
          num_bytes_per_combine_msg};
    }

#ifdef LANE_E_DESTRANK_COALESCE
    // Coalesce staging pointers (appended after the base LowLatencyBuffer
    // fields). Send staging lives inside the send buffer past the per-token
    // temp; recv staging is its own symmetric region after the recv buffers;
    // the per-src arrival flags sit just past the per-expert internode counts.
    for (int i = 0; i < 2; ++i) {
      buffers[i].dispatch_rdma_x_stage =
          advance(buffers[i].dispatch_rdma_send_buffer,
                  static_cast<size_t>(num_max_dispatch_tokens_per_rank) *
                      num_bytes_per_dispatch_msg);
      buffers[i].dispatch_recv_stage =
          advance(rdma_buffer, signaling_buffer_bytes_aligned * 2 +
                                   send_buffer_bytes * 2 + recv_buffer_bytes * 2 +
                                   recv_stage_region_bytes * i);
      buffers[i].dispatch_stage_flag_internode =
          buffers[i].dispatch_rdma_recv_count_buffer_internode + num_experts;
    }
#endif

#ifdef LANE_E_COMBINE_COALESCE
    // Combine staging sits after the dispatch recv-stage region (if present).
    size_t combine_region_base =
        signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2 +
        recv_buffer_bytes * 2;
#ifdef LANE_E_DESTRANK_COALESCE
    combine_region_base += recv_stage_region_bytes * 2;
#endif
    // Combine flags follow the per-expert counts and the dispatch flags (if any).
    size_t combine_flag_offset = num_experts;
#ifdef LANE_E_DESTRANK_COALESCE
    combine_flag_offset += num_ranks;
#endif
    for (int i = 0; i < 2; ++i) {
      buffers[i].combine_send_stage =
          advance(rdma_buffer, combine_region_base + cmb_stage_region_bytes * i);
      buffers[i].combine_recv_stage =
          advance(rdma_buffer, combine_region_base +
                                   cmb_stage_region_bytes * 2 +
                                   cmb_stage_region_bytes * i);
      buffers[i].combine_stage_flag_internode =
          buffers[i].dispatch_rdma_recv_count_buffer_internode +
          combine_flag_offset;
    }
#endif
  }
};

size_t get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank,
                                      int hidden, int num_ranks,
                                      int num_experts) {
  auto num_bytes = LowLatencyLayout(nullptr, num_max_dispatch_tokens_per_rank,
                                    hidden, num_ranks, num_experts, nullptr)
                       .total_bytes;
  return ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES) /
          NUM_BUFFER_ALIGNMENT_BYTES) *
         NUM_BUFFER_ALIGNMENT_BYTES;
}

}  // namespace uccl