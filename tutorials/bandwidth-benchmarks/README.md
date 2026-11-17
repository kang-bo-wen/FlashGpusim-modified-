
# Hierarchy Bandwidth Benchmarks

This directory contains reusable microbenchmarks for measuring sustained hierarchy throughput on FlashGPU-Sim. Future bandwidth microbenchmarks should be added here rather than to individual operator tutorials.

## Build

```bash
make -C tutorials/bandwidth-benchmarks
```

The included `cold_config` uses `-gpgpu_perf_sim_memcpy 0`. Initialize FlashGPU-Sim and use separate processes/directories for each test.

## Benchmarks

```bash
./hierarchy_bandwidth --level l1 --blocks 1056 --iterations 512
./hierarchy_bandwidth --level l2 --elements 4194304 --iterations 32 --blocks 1056
./hierarchy_bandwidth --level hbm --elements 67108864 --blocks 4096
```

## Retained Results

| Benchmark | Measured throughput | Structural theoretical peak | Ratio |
|---|---:|---:|---:|
| L1D + Shared read-modify-write | 12.057 TB/s | 25.344 TB/s | 47.6% |
| L2 read-dominant workload | 4.796 TB/s total physical traffic | 8.704 TB/s | 55.1% |
| HBM Float4 copy | 2.555 TB/s physical HBM traffic | 3.350 TB/s | 76.3% |

Raw extracted counters are in `hierarchy_peak_results.txt`; the compact summary is `measured_peaks.csv`.

## Important Limitations

- `L1D + Shared` is a combined physical counter. The L1 benchmark is a shared-memory read-modify-write workload, not an isolated L1D peak test.
- The L2 workload is read-dominant and also generates auxiliary writes. Its value is benchmark throughput, not a proven absolute L2 limit.
- The HBM benchmark is a 1:1 copy workload and is the most directly comparable sustainable ceiling for copy and transpose kernels.
- HBM physical bytes exclude dirty output still resident in L2 at kernel completion. Report complete useful bandwidth separately when evaluating operators.
