# Hierarchical Physical-Traffic Instrumentation

This document describes the FlashGPU-Sim changes used to report physical traffic at the first-level memory path, L2, and HBM. The counters support hierarchical Roofline analysis and bandwidth-efficiency studies.

## Reported Statistics

The final simulator statistics include:

```text
roofline_l1_read_bytes
roofline_l1_write_bytes
roofline_l1_total_bytes
roofline_l2_read_bytes
roofline_l2_write_bytes
roofline_l2_writeback_bytes
roofline_l2_total_bytes
roofline_hbm_read_bytes
roofline_hbm_write_bytes
roofline_hbm_writeback_bytes
roofline_hbm_total_bytes
```

The current `L1` label is a combined **L1D + Shared** physical-traffic counter:

- Global/local memory requests are counted when an accepted processor-side request enters L1D.
- Shared-memory traffic is counted as physical bank/word transactions.
- Cache fills are not counted as new L1 processor requests.

L2 counts accepted requests presented to the L2 interface. HBM counts requests entering the DRAM request queue.

## Files Changed

### Counter storage and output

`src/gpgpu-sim/roofline_stats.h` defines thread-safe atomic counters and the helper functions:

```text
roofline_stats::add_l1(...)
roofline_stats::add_l2(...)
roofline_stats::add_hbm(...)
roofline_stats::print(...)
roofline_stats::reset(...)
```

The file is included by:

```text
src/gpgpu-sim/shader.cc
src/gpgpu-sim/l2cache.cc
src/gpgpu-sim/dram.cc
```

`roofline_stats::print()` is called from the shader statistics output path in `shader.cc`, so the counters appear in the standard simulation log.

### Shared-memory physical bytes

`warp_inst_t` in `src/abstract_hardware_model.h` contains:

```cpp
unsigned m_smem_physical_bytes;
```

The value is initialized to zero with the other per-instruction shared-memory fields.

During shared-memory access generation in `src/abstract_hardware_model.cc`, the simulator already groups accesses by bank and 32-bit word. The instrumentation sums the number of unique `(bank, word)` transactions and converts them to bytes:

```text
shared physical bytes = unique (bank, 32-bit word) pairs × WORD_SIZE
```

For the current configurations, `WORD_SIZE` is four bytes. This counts broadcast/coalescing behavior correctly and exposes bank-conflict traffic without multiplying every active lane by its nominal operand size.

### L1D and Shared hook

In `src/gpgpu-sim/shader.cc`:

- `ldst_unit::shared_cycle(...)` charges `m_smem_physical_bytes` once per shared-memory instruction. The existing `m_smem_exposed` flag prevents repeated charging while the instruction remains in the service pipeline.
- `ldst_unit::L1_latency_queue_cycle()` charges `mem_fetch::get_data_size()` after the L1D access is accepted. Requests returning `RESERVATION_FAIL` are not counted until accepted.

The resulting L1 statistic is intentionally named `L1D + Shared` in analysis figures. It is not a pure L1D-only counter.

### L2 hook

In `src/gpgpu-sim/l2cache.cc`, `memory_sub_partition::process_l2_access_result(...)` records `mem_fetch::get_data_size()` when the L2 access is accepted. `L1_WRBK_ACC` and `L2_WRBK_ACC` are classified as writeback traffic. Reservation failures are excluded until a later successful attempt.

### HBM hook

In `src/gpgpu-sim/dram.cc`, `dram_t::push(mem_fetch *data)` records traffic as the request enters the DRAM request queue. The access is classified as read, write, or writeback from the `mem_fetch` access type.

This point measures physical traffic delivered to the modeled HBM system, rather than CUDA-level requested bytes.

## Thread Safety

FlashGPU-Sim can simulate SMs with OpenMP host threads. All traffic counters therefore use `std::atomic<unsigned long long>` and relaxed atomic increments. Relaxed ordering is sufficient because the counters only require an exact final sum; they do not synchronize simulator state.

## Counter Lifecycle

The counters are process-global. The included tutorials launch each measured kernel variant in a separate process, so every log begins from zero.

If multiple independent kernels must be measured inside one process, call:

```cpp
roofline_stats::reset();
```

before the measurement interval. Do not reset while simulator worker threads are actively updating the counters.

## Converting Bytes to Bandwidth

For an SM core clock of 1.5 GHz:

```text
simulated time (seconds) = gpu_tot_sim_cycle / 1.5e9
physical bandwidth (GB/s) = physical bytes × 1.5 / gpu_tot_sim_cycle
```

The factor above produces decimal GB/s. Divide by 1000 for decimal TB/s.

For operator efficiency, also report complete useful bandwidth:

```text
useful bandwidth = complete algorithmic bytes / simulated time
```

Useful bandwidth and physical hierarchy bandwidth answer different questions:

- Useful bandwidth measures how quickly the operator completes required data movement.
- Physical bandwidth measures how much traffic a hierarchy level actually processes, including transaction amplification.

## Important Interpretation Rules

1. **Kernel-end dirty data:** FlashGPU-Sim does not force all dirty L2 lines to HBM at kernel completion. Physical HBM writeback bytes can therefore be lower than complete output bytes.
2. **Non-coalesced traffic:** A naive transpose can generate much more L1/L2 physical write traffic than its algorithmic output size. This is expected and is useful for identifying transaction amplification.
3. **L1D + Shared:** The combined counter does not have a strict unified hardware peak. Treat its structural ceiling as a reference unless L1D and Shared are instrumented separately.
4. **Cold-cache runs:** Use `-gpgpu_perf_sim_memcpy 0` when host-to-device copies would otherwise prewarm L2 before the measured kernel.
5. **Separate processes:** Use separate simulation processes for variants unless an explicit counter reset is added around the measurement interval.

## Validation Checks

Recommended validation workloads:

- Coalesced copy: L1 and L2 traffic should match complete input plus output traffic, subject to kernel-end HBM writeback behavior.
- Naive transpose: L1/L2 writes should show transaction amplification while HBM writeback remains cache-line based.
- Padded tiled transpose: L2 traffic should return close to complete algorithmic input/output traffic.
- Shared-memory loop: the measured shared bytes should match unique bank/word transactions multiplied by four bytes.

For every test, verify:

```text
L1D + Shared bytes >= L2 bytes >= HBM bytes
```

This hierarchy is a useful sanity check, but exceptions can occur for specialized access paths or when comparing counters with different measurement intervals.

## Included Tutorials

- `tutorials/memory-bound-ops/` contains V0–V3 copy/transpose kernels, retained results, and the hierarchy comparison figure.
- `tutorials/bandwidth-benchmarks/` contains reusable L1D+Shared, L2, and HBM sustained-throughput microbenchmarks.

See the README in each directory for build commands, benchmark parameters, retained results, and limitations.
