# SM90 H100 Memory-Hierarchy Bandwidth Ceilings

This document records the calculation process and assumptions used for the memory-hierarchy bandwidth reference lines in:

```text
tutorials/memory-bound-ops/plot_transpose_bandwidth_rigorous.py
```

The hardware topology and clock parameters are taken from:

```text
configs/SM90_H100/gpgpusim.config
```

The resulting values are:

| Level | Reference bandwidth | Classification |
|---|---:|---|
| L1D + Shared | 25.344 TB/s | Structural reference ceiling |
| L2 | 8.704 TB/s | Structural reference ceiling |
| HBM | 3.34976 TB/s | Configuration-derived theoretical bandwidth |

> **Important:** The HBM value can be derived entirely from explicit configuration parameters. The L1D + Shared and L2 calculations additionally use assumed per-cycle transfer widths that are not directly declared as throughput parameters in `gpgpusim.config`. They should therefore be described as structural reference ceilings rather than NVIDIA-published hardware peaks.

## 1. Unit conversion

For `N` parallel units, a transfer width of `W` bytes per cycle per unit, and an operating frequency of `f` GHz:

```text
Bandwidth = N × W × f
```

The numerical result is in decimal GB/s because:

```text
1 GHz = 10^9 cycles/s
1 GB   = 10^9 bytes
```

Therefore:

```text
N × W bytes/cycle × f × 10^9 cycles/s
--------------------------------------- = N × W × f GB/s
              10^9 bytes/GB
```

Divide the GB/s result by 1000 to obtain decimal TB/s.

## 2. L1D + Shared structural reference ceiling

### 2.1 Parameters from the SM90 H100 configuration

```text
-gpgpu_n_clusters 132
-gpgpu_n_cores_per_cluster 1
-gpgpu_clock_domains 1500:1700:1700:2617
```

The clock-domain order is documented in the configuration as:

```text
<Core Clock>:<Interconnect Clock>:<L2 Clock>:<DRAM Clock>
```

The number of modeled SMs is therefore:

```text
132 clusters × 1 core/cluster = 132 SMs
```

The core frequency is:

```text
1500 MHz = 1.5 GHz
```

### 2.2 Additional structural assumption

The calculation assumes a transfer width of:

```text
128 bytes/cycle/SM
```

This is an external structural-throughput assumption used by the plotting analysis. It is not directly declared as an L1D or Shared throughput parameter in `gpgpusim.config`.

In particular, the `128` in the following cache configuration describes cache geometry, including the cache-line size; it does not by itself prove a throughput of 128 bytes per cycle:

```text
-gpgpu_cache:dl1 S:4:128:256,L:T:m:L:L,A:384:48,32:0,32
```

### 2.3 Calculation

Total structural transfer width across all SMs:

```text
132 SMs × 128 bytes/cycle/SM
= 16,896 bytes/cycle
```

Multiply by the 1.5 GHz core clock:

```text
16,896 bytes/cycle × 1.5 × 10^9 cycles/s
= 25,344 × 10^9 bytes/s
= 25,344 GB/s
= 25.344 TB/s
```

Thus:

```text
L1D + Shared reference ceiling = 25.344 TB/s
```

The corresponding plotting constant is:

```python
l1_reference_gbs = 25344.0
```

### 2.4 Interpretation limitation

The instrumented statistic combines:

```text
L1D global/local request traffic + Shared-memory physical traffic
```

L1D and Shared memory do not necessarily consume one identical, unified physical data path. Consequently, 25.344 TB/s is a normalization reference for the combined counter, not a strict unified hardware peak.

## 3. L2 structural reference ceiling

### 3.1 Parameters from the SM90 H100 configuration

```text
-gpgpu_n_mem 80
-gpgpu_n_sub_partition_per_mchannel 2
-gpgpu_clock_domains 1500:1700:1700:2617
```

The total number of L2 subpartitions is:

```text
80 memory channels × 2 subpartitions/channel
= 160 L2 subpartitions
```

The L2 frequency is:

```text
1700 MHz = 1.7 GHz
```

### 3.2 Additional structural assumption

The calculation assumes a transfer width of:

```text
32 bytes/cycle/L2 subpartition
```

This transfer width is an analysis assumption. It is not directly specified as an L2 throughput parameter in `gpgpusim.config`.

### 3.3 Calculation

Total structural transfer width across the L2 subpartitions:

```text
160 subpartitions × 32 bytes/cycle/subpartition
= 5,120 bytes/cycle
```

Multiply by the 1.7 GHz L2 clock:

```text
5,120 bytes/cycle × 1.7 × 10^9 cycles/s
= 8,704 × 10^9 bytes/s
= 8,704 GB/s
= 8.704 TB/s
```

Thus:

```text
L2 reference ceiling = 8.704 TB/s
```

The corresponding plotting constant is:

```python
l2_theoretical_gbs = 8704.0
```

### 3.4 Interpretation limitation

The configuration directly provides the channel count, subpartition count, and L2 clock. However, the 32-byte-per-cycle transfer width is an additional assumption. The result should therefore be described as:

```text
An L2 structural reference ceiling based on the SM90_H100 topology
and an assumed 32 bytes/cycle/subpartition transfer width.
```

## 4. HBM configuration-derived theoretical bandwidth

### 4.1 Parameters from the SM90 H100 configuration

```text
-gpgpu_n_mem 80
-gpgpu_clock_domains 1500:1700:1700:2617
-gpgpu_dram_buswidth 8
-dram_data_command_freq_ratio 2
```

The relevant values are:

```text
Memory channels                 = 80
DRAM bus width per channel      = 8 bytes
Data-command frequency ratio    = 2
DRAM clock                      = 2617 MHz = 2.617 GHz
```

The effective transfer width per channel per DRAM clock is:

```text
8 bytes × 2 = 16 bytes/cycle/channel
```

### 4.2 Calculation

Total transfer width across all memory channels:

```text
80 channels × 16 bytes/cycle/channel
= 1,280 bytes/cycle
```

Multiply by the 2.617 GHz DRAM clock:

```text
1,280 bytes/cycle × 2.617 × 10^9 cycles/s
= 3,349.76 × 10^9 bytes/s
= 3,349.76 GB/s
= 3.34976 TB/s
```

Equivalently, in one expression:

```text
80 × 8 bytes × 2 × 2.617 GHz
= 3,349.76 GB/s
= 3.34976 TB/s
```

Thus:

```text
HBM theoretical bandwidth = 3.34976 TB/s ≈ 3.35 TB/s
```

The corresponding plotting constant is:

```python
hbm_theoretical_gbs = 3349.76
```

Unlike the L1D + Shared and L2 calculations, every factor in this HBM formula is explicitly represented in the simulator configuration.

## 5. Summary of sources and assumptions

| Quantity | Value | Source |
|---|---:|---|
| Number of clusters | 132 | `-gpgpu_n_clusters` |
| Cores per cluster | 1 | `-gpgpu_n_cores_per_cluster` |
| Number of SMs | 132 | Clusters × cores per cluster |
| Core clock | 1.5 GHz | First value in `-gpgpu_clock_domains` |
| L1D + Shared transfer width | 128 B/cycle/SM | Additional structural assumption |
| Memory channels | 80 | `-gpgpu_n_mem` |
| L2 subpartitions per channel | 2 | `-gpgpu_n_sub_partition_per_mchannel` |
| Total L2 subpartitions | 160 | Channels × subpartitions per channel |
| L2 clock | 1.7 GHz | Third value in `-gpgpu_clock_domains` |
| L2 transfer width | 32 B/cycle/subpartition | Additional structural assumption |
| DRAM clock | 2.617 GHz | Fourth value in `-gpgpu_clock_domains` |
| DRAM bus width | 8 B/channel | `-gpgpu_dram_buswidth` |
| Data frequency ratio | 2 | `-dram_data_command_freq_ratio` |

## 6. Recommended terminology

Use the following descriptions in figures, reports, and discussions:

- **L1D + Shared:** SM90_H100 structural reference ceiling.
- **L2:** SM90_H100 structural reference ceiling.
- **HBM:** SM90_H100 configuration-derived theoretical bandwidth.

Avoid describing all three values as NVIDIA-published theoretical peaks. Only the HBM calculation is fully determined by explicit simulator configuration parameters; the L1D + Shared and L2 values include additional per-cycle transfer-width assumptions.

## 7. Relation to measured bandwidth

The theoretical/reference values above are separate from measured benchmark throughput. For a simulation using a 1.5 GHz core clock, measured physical bandwidth is calculated from the recorded traffic and total simulated core cycles as:

```text
simulated time = gpu_tot_sim_cycle / (1.5 × 10^9)
```

```text
measured bandwidth (GB/s)
= physical bytes / simulated time / 10^9
= physical bytes × 1.5 / gpu_tot_sim_cycle
```

The theoretical/reference ceilings are used as normalization baselines, while measured bandwidth describes the throughput actually sustained by a particular simulated workload.
