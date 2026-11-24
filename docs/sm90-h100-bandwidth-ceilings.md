# SM90 H100 Memory-Hierarchy Bandwidth Ceilings

This document records the calculation process used by the maintained
`tutorials/bandwidth-benchmarks/` suite.

The hardware topology and clock parameters are taken from:

```text
configs/SM90_H100/gpgpusim.config
```

The resulting values are:

| Level | Reference bandwidth | Classification |
|---|---:|---|
| L1D request admission | 25.344 TB/s | Model-derived request-port ceiling |
| Shared memory | Unbounded in release config | `-gpgpu_shmem_bytes_per_cycle 0` |
| L2 data port | 8.704 TB/s | Sweep-port configuration ceiling |
| L2 fill port | 8.704 TB/s | Sweep-port configuration ceiling |
| HBM DRAM commands | 3.34976 TB/s | Model-derived command-service ceiling |

> **Important:** These are simulator-model ceilings, not NVIDIA-published
> hardware specifications. L2 uses the explicit sector-port settings appended by
> `tutorials/bandwidth-benchmarks/common.sh`; the SM90_H100 release preset
> itself retains the legacy L2 port path.

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

## 2. L1D request-admission ceiling

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

### 2.2 Model service width

The L1 latency queue accepts at most one request per configured bank per core
tick. SM90_H100 configures four banks, and sector-coalesced requests carry 32
bytes. Therefore the modeled request-admission width is:

```text
4 banks × 32 bytes = 128 bytes/core tick/SM
```

This result comes from `-gpgpu_l1_banks 4`, the 32-byte sector size, and the
four-bank admission loop in `ldst_unit::process_memory_access_queue_l1cache`.
It does not come from the `128` cache-line-size field in:

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
L1D request-admission ceiling = 25.344 TB/s
```

The corresponding plotting constant is:

```python
l1_reference_gbs = 25344.0
```

### 2.4 Interpretation limitation

This ceiling applies only to `roofline_l1d_request_total_bytes`. Shared memory
is a separate resource. In the release configuration
`-gpgpu_shmem_bytes_per_cycle 0` means its byte service is unlimited, so the
model does not currently define a finite shared-memory byte-bandwidth ceiling.

## 3. L2 explicit sector-port ceilings

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

### 3.2 Benchmark configuration

The maintained benchmark scripts append the following explicit port settings:

```text
-gpgpu_l2_multi_issue_port_model 1
-gpgpu_l2_data_port_sectors_per_cycle 1
-gpgpu_l2_data_port_cycle_period 1
-gpgpu_l2_fill_port_sectors_per_cycle 1
-gpgpu_l2_shared_data_array 0
```

One sector is 32 bytes. Thus both independent ports have an explicit width of
32 bytes per L2 tick per subpartition.

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
L2 data-port ceiling = 8.704 TB/s
```

The corresponding plotting constant is:

```python
l2_theoretical_gbs = 8704.0
```

### 3.4 Interpretation limitation

The result should therefore be described as:

```text
The L2 data-port ceiling (and independently the fill-port ceiling) for the
explicit benchmark port configuration.
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

Each issued column command transfers `dram_atom_size`:

```text
BL × busW × chips/controller = 4 × 8 × 1 = 32 bytes/command
```

### 4.2 Calculation

The minimum same-direction column-command spacing is `tCCD = 2` DRAM ticks.
The aggregate command-service width is therefore:

```text
80 channels × 32 bytes/command ÷ 2 ticks/command
= 1,280 bytes/DRAM tick
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
80 × 32 bytes ÷ 2 × 2.617 GHz
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

Every factor is explicitly represented in the simulator configuration and in
the `dram_t::issue_col_command()` service rule. The older expression
`80 × busW × data_command_freq_ratio × clock` happens to give the same number
for this preset, but it does not state the actual modeled event and should not
be used as the general derivation.

## 5. Summary of sources and assumptions

| Quantity | Value | Source |
|---|---:|---|
| Number of clusters | 132 | `-gpgpu_n_clusters` |
| Cores per cluster | 1 | `-gpgpu_n_cores_per_cluster` |
| Number of SMs | 132 | Clusters × cores per cluster |
| Core clock | 1.5 GHz | First value in `-gpgpu_clock_domains` |
| L1D request width | 4 × 32 B/core tick/SM | L1 bank admission model |
| Memory channels | 80 | `-gpgpu_n_mem` |
| L2 subpartitions per channel | 2 | `-gpgpu_n_sub_partition_per_mchannel` |
| Total L2 subpartitions | 160 | Channels × subpartitions per channel |
| L2 clock | 1.7 GHz | Third value in `-gpgpu_clock_domains` |
| L2 data/fill width | 1 × 32 B/L2 tick/subpartition | Sweep port settings |
| DRAM clock | 2.617 GHz | Fourth value in `-gpgpu_clock_domains` |
| DRAM bus width | 8 B/channel | `-gpgpu_dram_buswidth` |
| DRAM atom size | 32 B/column command | `BL × busW × chips/controller` |
| Column-command spacing | 2 DRAM ticks | `tCCD` |

## 6. Recommended terminology

Use the following descriptions in figures, reports, and discussions:

- **L1D:** SM90_H100 modeled request-admission ceiling.
- **Shared:** no finite modeled byte ceiling while its budget is zero/unlimited.
- **L2:** benchmark data-port or fill-port ceiling; name the port.
- **HBM:** SM90_H100 modeled DRAM command-service ceiling.

Do not describe these values as NVIDIA-published theoretical peaks. They are
ceilings of the configured simulator service rules. In particular, the L2
number applies to the explicit benchmark port configuration rather than the legacy
L2 path in the release preset.

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
