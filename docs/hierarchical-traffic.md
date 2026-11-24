# Hierarchical Physical-Traffic Instrumentation

This document describes the FlashGPU-Sim changes used to report physical traffic at the first-level memory path, L2, and HBM. The counters support hierarchical Roofline analysis and bandwidth-efficiency studies.

## Reported Statistics

The final simulator statistics include:

```text
roofline_l1d_request_read_bytes
roofline_l1d_request_write_bytes
roofline_l1d_request_total_bytes
roofline_shared_bank_read_bytes
roofline_shared_bank_write_bytes
roofline_shared_bank_total_bytes
roofline_l2_data_read_hit_bytes
roofline_l2_data_write_hit_bytes
roofline_l2_data_hit_bytes
roofline_l2_data_miss_read_bytes
roofline_l2_data_fill_bytes
roofline_l2_data_dirty_eviction_bytes
roofline_l2_data_port_total_bytes
roofline_l2_fill_port_total_bytes
roofline_hbm_command_read_bytes
roofline_hbm_command_write_bytes
roofline_hbm_command_writeback_bytes
roofline_hbm_command_total_bytes
```

The instrumentation exposes separate observation points rather than combining
unlike resources:

- `roofline_l1d_request_*` counts accepted processor-side L1D requests.
- `roofline_shared_bank_*` counts unique shared bank/word service bytes.
- `roofline_l2_data_read_hit_bytes` and `roofline_l2_data_write_hit_bytes`
  separate hit service by the original request direction; their sum is the
  compatibility `roofline_l2_data_hit_bytes`. Other `roofline_l2_data_*`
  fields count miss-read, fill, and dirty-eviction sectors served by the port.
- `roofline_l2_fill_port_total_bytes` counts sectors served by the fill port.
- `roofline_hbm_command_*` counts bytes transferred by issued DRAM column commands.

The compatibility `roofline_l1_total_bytes`, `roofline_l2_total_bytes`, and
`roofline_hbm_total_bytes` aliases remain for old scripts, but new analysis
should consume the explicit fields above.

## Files Changed

### Counter storage and output

`src/gpgpu-sim/roofline_stats.h` defines thread-safe atomic counters and the helper functions:

```text
roofline_stats::add_l1d_request(...)
roofline_stats::add_shared_bank(...)
roofline_stats::add_l2_data(...)
roofline_stats::add_l2_fill_port(...)
roofline_stats::add_dram_command(...)
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

### L1D and Shared hooks

In `src/gpgpu-sim/shader.cc`:

- `ldst_unit::shared_cycle(...)` charges `m_smem_physical_bytes` once per shared-memory instruction. The existing `m_smem_exposed` flag prevents repeated charging while the instruction remains in the service pipeline.
- `ldst_unit::L1_latency_queue_cycle()` charges `mem_fetch::get_data_size()` after the L1D access is accepted. Requests returning `RESERVATION_FAIL` are not counted until accepted.

The two resources are printed separately and must not share a utilization denominator.

### L2 hook

The multi-issue L2 port model records 32-byte sectors at the exact calls that
consume hit, miss-read, fill, and dirty-eviction service. Request admission is
not counted as data-array traffic.

### HBM hook

In `src/gpgpu-sim/dram.cc`, `dram_t::issue_col_command(...)` records one
`dram_atom_size` transfer for each issued read or write column command.

This point matches the modeled DRAM data-command service rather than queue payload.

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
3. **Separate resources:** L1D requests, shared banks, L2 data/fill ports, and HBM commands require separate ceilings.
4. **Cold-cache runs:** Use `-gpgpu_perf_sim_memcpy 0` when host-to-device copies would otherwise prewarm L2 before the measured kernel.
5. **Separate processes:** Use separate simulation processes for variants unless an explicit counter reset is added around the measurement interval.

## Validation Checks

Recommended validation workloads:

- Coalesced copy: L1 and L2 traffic should match complete input plus output traffic, subject to kernel-end HBM writeback behavior.
- Naive transpose: L1/L2 writes should show transaction amplification while HBM writeback remains cache-line based.
- Padded tiled transpose: L2 traffic should return close to complete algorithmic input/output traffic.
- Shared-memory loop: the measured shared bytes should match unique bank/word transactions multiplied by four bytes.

There is no universal byte inequality across these counters: L1D requests,
shared bank service, L2 hit/fill service, and HBM commands observe different
events and may amplify or bypass one another. Instead, verify the accounting
identities at each observation point:

```text
L1D request total = L1D request read + L1D request write
L2 data-port total = read hit + write hit + miss read + fill + dirty eviction
HBM command total = read + write + writeback
```

For a controlled workload, also compare the relevant algorithmic bytes with the
specific physical event it is intended to generate. Do not add Shared bytes to
L1D bytes or compare unlike counters using one common ceiling.

## Included Tutorials

- `tutorials/bandwidth-benchmarks/` contains the maintained theoretical
  derivation, saturation sweeps, plateau checks, and reproducibility guide.

See the README in each directory for build commands, benchmark parameters, retained results, and limitations.
