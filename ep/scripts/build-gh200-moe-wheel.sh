#!/usr/bin/env bash
# Build the qualified four-GPU-per-node GH200/CXI MoE profile as a wheel.
set -euo pipefail
: "${LIBFABRIC_HOME:?Set LIBFABRIC_HOME to the libfabric include/lib prefix}"
uccl_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
uccl_python=${UCCL_PYTHON:-python3}
uccl_wheel_dir=${1:-"$uccl_root/dist"}
# Make does not track command-line flag changes. Start with clean objects so
# a prior generic build cannot silently produce a different wire protocol.
make -C "$uccl_root/ep" clean PYTHON="$uccl_python"
make -C "$uccl_root/ep" -j "${UCCL_BUILD_JOBS:-8}" \
  PYTHON="$uccl_python" CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
  CXX="${CXX:-g++}" USE_LIBFABRIC_CXI=1 LIBFABRIC_HOME="$LIBFABRIC_HOME" \
  SM=90 NUM_MAX_NVL_PEERS=4 \
  LANE_E_DESTRANK_COALESCE=1 LANE_E_DISPATCH_DEDUP=1 \
  LANE_E_DISPATCH_INDIRECT=1 LANE_E_COMBINE_COALESCE=1 \
  LANE_E_COMBINE_INDIRECT=1 LANE_E_DISPATCH_NODE=1 \
  LANE_E_COMBINE_DIRECT_SEND=1 UCCL_FP8_EXACT_QUANT=1
cp "$uccl_root/ep/ep.abi3.so" "$uccl_root/uccl/"
"$uccl_python" -m pip wheel --no-deps --no-build-isolation \
  "$uccl_root" --wheel-dir "$uccl_wheel_dir"
