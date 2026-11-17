# Matrix Transpose Bandwidth Tutorial


This tutorial evaluates complete effective bandwidth and physical hierarchy traffic for four `8192 x 8192` FP32 kernels on the `SM90_H100` FlashGPU-Sim configuration.

## Kernels

- V0: `float4` grid-stride copy baseline.
- V1: naive transpose with coalesced reads and strided writes.
- V2: `32 x 32` shared-memory tiled transpose with padding.
- V3: `float4` shared-memory tiled transpose.

## Build

```bash
make -C tutorials/memory-bound-ops
```

## Run

Use a cold configuration containing `-gpgpu_perf_sim_memcpy 0`. Run each version in a separate directory/process:

```bash
./transpose --version 0 --rows 8192 --cols 8192
./transpose --version 1 --rows 8192 --cols 8192
./transpose --version 2 --rows 8192 --cols 8192
./transpose --version 3 --rows 8192 --cols 8192
```

The matrix input and output are each 256 MiB. Complete algorithmic traffic is therefore 512 MiB.

```text
Useful effective bandwidth = 512 MiB / simulated kernel time
Physical hierarchy bandwidth = hierarchy counter bytes / simulated kernel time
Simulated kernel time = cycles / 1.5 GHz
```

## Results

The retained simulator summary is `transpose_results_r8192_c8192.txt`. Raw counters used by the plotting script are stored in `transpose_bandwidth_raw_r8192_c8192.csv`. The generated presentation figure is `transpose_bandwidth_rigorous_r8192_c8192.svg`. Regenerate it with:

```bash
python3 tutorials/memory-bound-ops/plot_transpose_bandwidth_rigorous.py
```

Useful effective bandwidth:

| Kernel | Cycles | Effective bandwidth |
|---|---:|---:|
| V0 Float4 Copy | 298,581 | 2.697 TB/s |
| V1 Naive Transpose | 886,719 | 0.908 TB/s |
| V2 Tiled Transpose | 488,039 | 1.650 TB/s |
| V3 Float4 Tiled | 374,126 | 2.153 TB/s |

All four kernels passed exact correctness checking.

## Interpretation

- V1 creates an 8x global-write transaction amplification, so its L1/L2 physical bandwidth is high while its useful bandwidth is low.
- V2 and V3 include shared-memory physical traffic in the combined `L1D + Shared` counter.
- HBM counters are kernel-scope. Dirty output still resident in L2 is not counted until eviction, so useful bandwidth and physical HBM bandwidth are intentionally shown separately.
- Red dashed lines are measured benchmark throughput from `tutorials/bandwidth-benchmarks`. L1/L2 lines are workload-specific benchmark throughput rather than proven absolute hardware limits; the HBM copy line is the most directly comparable sustainable ceiling.
