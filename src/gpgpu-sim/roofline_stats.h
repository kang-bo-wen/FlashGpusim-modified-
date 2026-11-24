// Copyright (c) 2026
// Hierarchical Roofline traffic counters for FlashGPU-Sim.
#ifndef ROOFLINE_STATS_H
#define ROOFLINE_STATS_H

#include <atomic>
#include <cstdio>

namespace roofline_stats {

using counter = std::atomic<unsigned long long>;
inline counter l1d_request_read_bytes{0}, l1d_request_write_bytes{0};
inline counter shared_bank_read_bytes{0}, shared_bank_write_bytes{0};
inline counter l2_data_read_hit_bytes{0}, l2_data_write_hit_bytes{0};
inline counter l2_data_miss_read_bytes{0};
inline counter l2_data_fill_bytes{0}, l2_data_dirty_eviction_bytes{0};
inline counter l2_fill_port_bytes{0};
inline counter dram_command_read_bytes{0}, dram_command_write_bytes{0};
inline counter dram_command_writeback_bytes{0};

inline void add(counter &value, unsigned long long bytes) {
  value.fetch_add(bytes, std::memory_order_relaxed);
}
inline unsigned long long get(const counter &value) {
  return value.load(std::memory_order_relaxed);
}

inline void reset() {
  counter *all[] = {&l1d_request_read_bytes, &l1d_request_write_bytes,
                    &shared_bank_read_bytes, &shared_bank_write_bytes,
                    &l2_data_read_hit_bytes, &l2_data_write_hit_bytes,
                    &l2_data_miss_read_bytes,
                    &l2_data_fill_bytes, &l2_data_dirty_eviction_bytes,
                    &l2_fill_port_bytes, &dram_command_read_bytes,
                    &dram_command_write_bytes, &dram_command_writeback_bytes};
  for (counter *value : all) value->store(0, std::memory_order_relaxed);
}

inline void add_l1d_request(bool write, unsigned bytes) {
  add(write ? l1d_request_write_bytes : l1d_request_read_bytes, bytes);
}
inline void add_shared_bank(bool write, unsigned bytes) {
  add(write ? shared_bank_write_bytes : shared_bank_read_bytes, bytes);
}
enum l2_data_work {
  L2_READ_HIT, L2_WRITE_HIT, L2_MISS_READ, L2_FILL, L2_DIRTY_EVICTION
};
inline void add_l2_data(l2_data_work work, unsigned sectors) {
  counter *target = &l2_data_read_hit_bytes;
  if (work == L2_WRITE_HIT) target = &l2_data_write_hit_bytes;
  if (work == L2_MISS_READ) target = &l2_data_miss_read_bytes;
  if (work == L2_FILL) target = &l2_data_fill_bytes;
  if (work == L2_DIRTY_EVICTION) target = &l2_data_dirty_eviction_bytes;
  add(*target, 32ull * sectors);
}
inline void add_l2_fill_port(unsigned sectors) {
  add(l2_fill_port_bytes, 32ull * sectors);
}
inline void add_dram_command(bool write, bool writeback, unsigned bytes) {
  if (writeback) add(dram_command_writeback_bytes, bytes);
  else add(write ? dram_command_write_bytes : dram_command_read_bytes, bytes);
}

inline void print(FILE *out) {
  const auto l1r=get(l1d_request_read_bytes), l1w=get(l1d_request_write_bytes);
  const auto sr=get(shared_bank_read_bytes), sw=get(shared_bank_write_bytes);
  const auto l2rh=get(l2_data_read_hit_bytes), l2wh=get(l2_data_write_hit_bytes);
  const auto l2h=l2rh+l2wh, l2m=get(l2_data_miss_read_bytes);
  const auto l2f=get(l2_data_fill_bytes), l2e=get(l2_data_dirty_eviction_bytes);
  const auto dr=get(dram_command_read_bytes), dw=get(dram_command_write_bytes);
  const auto db=get(dram_command_writeback_bytes);
  std::fprintf(out, "roofline_l1d_request_read_bytes = %llu\n", l1r);
  std::fprintf(out, "roofline_l1d_request_write_bytes = %llu\n", l1w);
  std::fprintf(out, "roofline_l1d_request_total_bytes = %llu\n", l1r+l1w);
  std::fprintf(out, "roofline_shared_bank_read_bytes = %llu\n", sr);
  std::fprintf(out, "roofline_shared_bank_write_bytes = %llu\n", sw);
  std::fprintf(out, "roofline_shared_bank_total_bytes = %llu\n", sr+sw);
  std::fprintf(out, "roofline_l2_data_read_hit_bytes = %llu\n", l2rh);
  std::fprintf(out, "roofline_l2_data_write_hit_bytes = %llu\n", l2wh);
  std::fprintf(out, "roofline_l2_data_hit_bytes = %llu\n", l2h);
  std::fprintf(out, "roofline_l2_data_miss_read_bytes = %llu\n", l2m);
  std::fprintf(out, "roofline_l2_data_fill_bytes = %llu\n", l2f);
  std::fprintf(out, "roofline_l2_data_dirty_eviction_bytes = %llu\n", l2e);
  std::fprintf(out, "roofline_l2_data_port_total_bytes = %llu\n", l2h+l2m+l2f+l2e);
  std::fprintf(out, "roofline_l2_fill_port_total_bytes = %llu\n", get(l2_fill_port_bytes));
  std::fprintf(out, "roofline_hbm_command_read_bytes = %llu\n", dr);
  std::fprintf(out, "roofline_hbm_command_write_bytes = %llu\n", dw);
  std::fprintf(out, "roofline_hbm_command_writeback_bytes = %llu\n", db);
  std::fprintf(out, "roofline_hbm_command_total_bytes = %llu\n", dr+dw+db);
  std::fprintf(out, "roofline_l1_total_bytes = %llu\n", l1r+l1w+sr+sw);
  std::fprintf(out, "roofline_l2_total_bytes = %llu\n", l2h+l2m+l2f+l2e);
  std::fprintf(out, "roofline_hbm_total_bytes = %llu\n", dr+dw+db);
}

}  // namespace roofline_stats

#endif
