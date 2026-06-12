#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdio>

// Stall introspection for the silent-proxy-wedge hunt (2026-06-12 wedge:
// one rank's count RDMA never arrives, all ranks stall at the dispatch
// barrier, no host- or device-side print identifies the blocked site).
//
// Every potentially-unbounded wait loop in the proxy holds a StallReporter
// on the stack and calls tick(...) once per spin. The reporter is silent
// until the loop has spun for kFirstReportMs, then prints one line with
// caller-supplied context and re-prints every kRepeatMs. Cost while not
// stalled: one steady_clock read per spin, in loops that already
// cpu_relax()/sched_yield().
class StallReporter {
 public:
  StallReporter(char const* site, int rank, int thread_idx, int peer = -1)
      : site_(site),
        rank_(rank),
        thread_idx_(thread_idx),
        peer_(peer),
        start_(std::chrono::steady_clock::now()),
        next_report_(start_ + std::chrono::milliseconds(kFirstReportMs)) {}

  // fmt/args: caller context (outstanding counts, queue depths, ...).
  // Returns true if a report was emitted on this tick.
  bool tick(char const* fmt = nullptr, ...) {
    auto const now = std::chrono::steady_clock::now();
    if (now < next_report_) return false;
    next_report_ = now + std::chrono::milliseconds(kRepeatMs);
    auto const stalled_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start_)
            .count();
    char ctx[512] = {0};
    if (fmt) {
      va_list ap;
      va_start(ap, fmt);
      vsnprintf(ctx, sizeof(ctx), fmt, ap);
      va_end(ap);
    }
    fprintf(stderr,
            "[proxy-stall site=%s rank=%d thread=%d peer=%d stalled_ms=%lld] "
            "%s\n",
            site_, rank_, thread_idx_, peer_,
            static_cast<long long>(stalled_ms), ctx);
    return true;
  }

  static constexpr int kFirstReportMs = 2000;
  static constexpr int kRepeatMs = 5000;

 private:
  char const* site_;
  int rank_;
  int thread_idx_;
  int peer_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point next_report_;
};
