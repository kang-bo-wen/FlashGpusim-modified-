// Copyright (c) 2026
// Hierarchical Roofline traffic counters for FlashGPU-Sim.
#ifndef ROOFLINE_STATS_H
#define ROOFLINE_STATS_H

#include <atomic>
#include <cstdio>

class mem_fetch;

namespace roofline_stats {

inline std::atomic<unsigned long long> l1_read_bytes{0};
inline std::atomic<unsigned long long> l1_write_bytes{0};
inline std::atomic<unsigned long long> l2_read_bytes{0};
inline std::atomic<unsigned long long> l2_write_bytes{0};
inline std::atomic<unsigned long long> l2_writeback_bytes{0};
inline std::atomic<unsigned long long> hbm_read_bytes{0};
inline std::atomic<unsigned long long> hbm_write_bytes{0};
inline std::atomic<unsigned long long> hbm_writeback_bytes{0};

inline void reset() {
  l1_read_bytes.store(0, std::memory_order_relaxed);
  l1_write_bytes.store(0, std::memory_order_relaxed);
  l2_read_bytes.store(0, std::memory_order_relaxed);
  l2_write_bytes.store(0, std::memory_order_relaxed);
  l2_writeback_bytes.store(0, std::memory_order_relaxed);
  hbm_read_bytes.store(0, std::memory_order_relaxed);
  hbm_write_bytes.store(0, std::memory_order_relaxed);
  hbm_writeback_bytes.store(0, std::memory_order_relaxed);
}

inline void add_l1(bool is_write, unsigned bytes) {
  if (is_write)
    l1_write_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else
    l1_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void add_l2(bool is_write, bool is_writeback, unsigned bytes) {
  if (is_writeback)
    l2_writeback_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else if (is_write)
    l2_write_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else
    l2_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void add_hbm(bool is_write, bool is_writeback, unsigned bytes) {
  if (is_writeback)
    hbm_writeback_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else if (is_write)
    hbm_write_bytes.fetch_add(bytes, std::memory_order_relaxed);
  else
    hbm_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void print(FILE *out) {
  const auto l1r = l1_read_bytes.load(std::memory_order_relaxed);
  const auto l1w = l1_write_bytes.load(std::memory_order_relaxed);
  const auto l2r = l2_read_bytes.load(std::memory_order_relaxed);
  const auto l2w = l2_write_bytes.load(std::memory_order_relaxed);
  const auto l2wb = l2_writeback_bytes.load(std::memory_order_relaxed);
  const auto hbr = hbm_read_bytes.load(std::memory_order_relaxed);
  const auto hbw = hbm_write_bytes.load(std::memory_order_relaxed);
  const auto hbwb = hbm_writeback_bytes.load(std::memory_order_relaxed);
  std::fprintf(out, "roofline_l1_read_bytes = %llu\n", l1r);
  std::fprintf(out, "roofline_l1_write_bytes = %llu\n", l1w);
  std::fprintf(out, "roofline_l1_total_bytes = %llu\n", l1r + l1w);
  std::fprintf(out, "roofline_l2_read_bytes = %llu\n", l2r);
  std::fprintf(out, "roofline_l2_write_bytes = %llu\n", l2w);
  std::fprintf(out, "roofline_l2_writeback_bytes = %llu\n", l2wb);
  std::fprintf(out, "roofline_l2_total_bytes = %llu\n", l2r + l2w + l2wb);
  std::fprintf(out, "roofline_hbm_read_bytes = %llu\n", hbr);
  std::fprintf(out, "roofline_hbm_write_bytes = %llu\n", hbw);
  std::fprintf(out, "roofline_hbm_writeback_bytes = %llu\n", hbwb);
  std::fprintf(out, "roofline_hbm_total_bytes = %llu\n", hbr + hbw + hbwb);
}

}  // namespace roofline_stats

#endif
