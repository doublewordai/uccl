#pragma once
#include <cstddef>
#include <cstdint>

namespace uccl {
// Exclusive one-node protocol workspace. Dispatch clears combine readiness;
// combine clears dispatch readiness. Each phase waits for all ranks before
// returning, so the other phase retires readers before a workspace is reused.
struct CompactIPCLayout {
  int* dispatch_ready;
  int* combine_ready;
  uint8_t* q;
  float* scales;
  int64_t* ids;
  float* weights;
  void* partial;
  std::size_t partial_offset, bytes;
  CompactIPCLayout(void* base, int capacity, int hidden, int topk, int ranks) {
    auto* p = static_cast<uint8_t*>(base);
    dispatch_ready = reinterpret_cast<int*>(p);
    combine_ready = dispatch_ready + ranks;
    std::size_t offset = 256;
    q = p + offset; offset += capacity * hidden;
    scales = reinterpret_cast<float*>(p + offset);
    offset += capacity * (hidden / 128) * sizeof(float);
    ids = reinterpret_cast<int64_t*>(p + offset);
    offset += capacity * topk * sizeof(int64_t);
    weights = reinterpret_cast<float*>(p + offset);
    offset += capacity * topk * sizeof(float);
    partial_offset = (offset + 255) / 256 * 256;
    partial = p + partial_offset;
    bytes = partial_offset + ranks * capacity * hidden * 2;
  }
};
}  // namespace uccl
