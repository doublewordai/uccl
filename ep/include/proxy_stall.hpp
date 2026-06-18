#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

class StallReporter {
 public:
  StallReporter(char const* site, int rank, int thread_idx, int peer = -1)
      : site_(site),
        rank_(rank),
        thread_idx_(thread_idx),
        peer_(peer),
        enabled_(enabled()),
        start_(std::chrono::steady_clock::now()),
        next_report_(start_ + std::chrono::milliseconds(kFirstReportMs)) {}

  bool tick(char const* fmt = nullptr, ...) {
    if (!enabled_) return false;
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

    std::fprintf(stderr,
                 "[proxy-stall site=%s rank=%d thread=%d peer=%d "
                 "stalled_ms=%lld] %s\n",
                 site_, rank_, thread_idx_, peer_,
                 static_cast<long long>(stalled_ms), ctx);
    return true;
  }

  static bool enabled() {
    static bool const value = [] {
      char const* env = std::getenv("UCCL_PROXY_STALL_MONITOR");
      return !(env && std::strcmp(env, "0") == 0);
    }();
    return value;
  }

  static constexpr int kFirstReportMs = 2000;
  static constexpr int kRepeatMs = 5000;

 private:
  char const* site_;
  int rank_;
  int thread_idx_;
  int peer_;
  bool enabled_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point next_report_;
};
