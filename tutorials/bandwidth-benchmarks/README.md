# FlashGPU-Sim 分层存储带宽基准

完整阶段性结果和分析结论统一维护在
[`report.md`](report.md)。

本目录用于在明确的统计口径和生效配置下，推导 FlashGPU-Sim 存储层级的模型理论上限，并通过可复现的参数扫描寻找逼近该上限的持续带宽。这里测量的是**模拟器模型的服务能力**，不是 NVIDIA 官方硬件峰值。

## 1. 基本原则

普通算子同时受到指令吞吐、依赖链、占用率、缓存未命中、互连和 DRAM 时序影响，不能用于证明某个存储层级已经饱和。本基准遵循：

1. 每个测试只针对一个明确观察点。
2. 理论峰值只从生效配置和实际服务规则推导。
3. 实测使用多参数扫描，不挑选单个手工参数。
4. 最高点附近形成平台后，才称为“当前扫描范围内最大持续带宽”。
5. 即使形成平台，也不宣称数学意义上的全局极限。

## 2. 流量插桩

计数器定义在 `src/gpgpu-sim/roofline_stats.h`，使用 relaxed atomic 保证多 OpenMP 主机线程下最终求和正确。

### 2.1 L1D 请求流量

```text
roofline_l1d_request_read_bytes
roofline_l1d_request_write_bytes
roofline_l1d_request_total_bytes
```

插桩位于 `ldst_unit::L1_latency_queue_cycle()`。只有请求真正被 L1D 接受且不是 `RESERVATION_FAIL` 时计数，retry 不重复收费。它表示处理器侧合并后的 request payload，不包括 L2 fill，也不是 L1D data array 的双向总流量。

### 2.2 Shared bank-word 流量

```text
roofline_shared_bank_read_bytes
roofline_shared_bank_write_bytes
roofline_shared_bank_total_bytes
```

普通 shared 指令在 `warp_inst_t::generate_mem_accesses()` 中按 subwarp 生成 `(bank, 32-bit word)` 集合。每个唯一 bank-word 计 4 bytes；broadcast 只计一次，不同 word 的 bank conflict 分别计数。`ldst_unit::shared_cycle()` 在首次服务时收费一次。

该口径目前只保证覆盖普通本地 `ld.shared`/`st.shared` 路径。固定 dispatch 的
matrix 指令、SSTARR、TMA landing 和 DSM 使用各自的服务路径，不能在未增加
对应插桩前混入本计数器。

SM90_H100 的 `-gpgpu_shmem_bytes_per_cycle 0` 表示预算无限，因此当前模型没有有限 Shared byte-bandwidth 峰值。可以验证字节数，但不能计算有意义的 utilization。

### 2.3 L2 data/fill port 流量

```text
roofline_l2_data_hit_bytes
roofline_l2_data_read_hit_bytes
roofline_l2_data_write_hit_bytes
roofline_l2_data_miss_read_bytes
roofline_l2_data_fill_bytes
roofline_l2_data_dirty_eviction_bytes
roofline_l2_data_port_total_bytes
roofline_l2_fill_port_total_bytes
```

插桩位于 `l2_multi_issue_ports::accept_data()` 和 `accept_fill()`。每个被端口实际服务的 sector 计 32 bytes。hit 按原请求方向拆为 read-hit 和 write-hit；兼容字段 `roofline_l2_data_hit_bytes` 是二者之和。请求进入 L2、tag lookup 或 MSHR merge 不计作 data-port 字节。若 `-gpgpu_l2_shared_data_array 0`，data 与 fill 是独立资源，必须分别报告。

### 2.4 HBM DRAM-command 流量

```text
roofline_hbm_command_read_bytes
roofline_hbm_command_write_bytes
roofline_hbm_command_writeback_bytes
roofline_hbm_command_total_bytes
```

插桩位于 `dram_t::issue_col_command()`。每次真正发出 read/write column command 时计一个 `dram_atom_size`。未在 kernel 结束前形成 DRAM command 的脏 cache line 不会被计入。

## 3. 理论峰值

运行：

```bash
python3 tutorials/bandwidth-benchmarks/theoretical.py <生效的 gpgpusim.config>
```

脚本读取同名配置项的最后一个值，因此可识别 sweep 脚本追加的覆盖项。单位为 decimal GB/s。

### 3.1 L1D request ceiling

L1 latency queue 每个 core tick 最多向每个 bank 接受一个 32-byte sector：

```text
BW_L1D = SM 数 × L1 bank 数 × 32 B × core clock
       = 132 × 4 × 32 × 1.5 GHz
       = 25,344 GB/s = 25.344 TB/s
```

它是请求入口上限，不是 L1D+Shared 统一峰值。

### 3.2 L2 data/fill port ceiling

三个层级的 `run.sh` 通过公共 `common.sh` 显式启用：

```text
-gpgpu_l2_multi_issue_port_model 1
-gpgpu_l2_data_port_sectors_per_cycle 1
-gpgpu_l2_data_port_cycle_period 1
-gpgpu_l2_fill_port_sectors_per_cycle 1
-gpgpu_l2_shared_data_array 0
```

因此两个独立端口分别为：

```text
BW_L2 = 160 subpartitions × 1 sector/tick × 32 B × 1.7 GHz
      = 8,704 GB/s = 8.704 TB/s
```

该数值只属于这个显式端口配置。SM90_H100 发布配置默认使用 legacy L2 路径，不能无条件引用 8.704 TB/s。

8.704 TB/s 是 160 个 L2 slice 同时满载时的内部结构上限。普通 CUDA load
还受到 SM 请求生成路径约束。当前 132 SM、每 SM 每 core tick 一个 32-byte
sector，因此 kernel-visible 上限为：

```text
min(8.704 TB/s, 132 × 32 B × 1.5 GHz) = 6.336 TB/s
```

Roofline 的端到端参考值应使用 6.336 TB/s；8.704 TB/s 单独保留为 L2
internal data-port structural ceiling。

这里的 `1 sector/SM/core tick` 不是独立硬件常数，而是当前生效配置和 legacy
请求路径共同产生的结果：

1. `gpgpu_ldst_request_width=0`，进入 legacy 请求路径；
2. legacy 路径每次 `ldst_unit::cycle()` 只对 access queue 执行一次
   `inject_one()`；
3. `gpgpu_mem_unit_ports=1`，所以每个 core tick 每个 SM 只调用一次该 LD/ST
   service cycle；
4. `SECTOR_SIZE=32`，且本 `.cg.v4` 微基准的一个 accepted child 对应一个
   32-byte sector。

若修改 request width、memory-unit ports、合并方式或单个 child 的 sector 数，该
producer-side 上限必须重新计算。

### 3.3 HBM command-service ceiling

```text
dram_atom_size = BL × busW × chips/controller
               = 4 × 8 × 1 = 32 B/command

BW_HBM = channel 数 × dram_atom_size / tCCD × DRAM clock
       = 80 × 32 / 2 × 2.617 GHz
       = 3,349.76 GB/s = 3.34976 TB/s
```

该公式与 `dram_t::issue_col_command()` 的实际服务事件一致。

## 4. 微基准设计

### 4.1 L1D hot-set

每个 CTA 使用私有 16 KiB slice，并通过 `ld.global.ca.v4.f32` 反复读取。第一次迭代负责装入，后续迭代主要产生 L1D hit。扫描 block 数和 iteration 数，用于判断 SM/LSU 并发是否足够，以及 compulsory miss 是否已经可忽略。

### 4.2 L2 resident-set

使用 `ld.global.cg.v4.f32` 绕过 L1D，反复扫描 4/8/16 MiB 工作集。工作集大于单个 L1、小于 H100 50 MiB L2。扫描工作集、block 数和复用次数，用于排除容量 miss、并发不足和 warm-up 偏差。

### 4.3 HBM read stream

使用大于 L2 的 64/128/256 MiB 只读连续流，通过 `ld.global.cg.v4.f32` 生成合并请求。只读设计避免 copy benchmark 中 output dirty line 停留在 L2，造成算法写字节与实际 DRAM command 字节不一致。扫描 block 数验证 memory-level parallelism 是否充分。

## 5. 分层工作目录

三个层级完全分开运行和保存结果：

```text
l1d/   L1D hot-set read-request 测试
l2/    L2 resident-set read-hit 测试
hbm/   HBM streaming read-command 测试
```

每个目录都有独立的 `run.sh`、`analyze.sh`、`README.md` 和
`runs-<CONFIG_NAME>/`。执行某个层级时不会创建或修改另外两个层级的结果目录。
公共 CUDA kernel、理论值计算和差分分析器保留在父目录，避免复制三份实现。

## 6. 构建与运行

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
tutorials/bandwidth-benchmarks/l1d/run.sh
tutorials/bandwidth-benchmarks/l2/run.sh
tutorials/bandwidth-benchmarks/hbm/run.sh
```

每个层级的输出目录均为自己的 `runs-SM90_H100/`。每个参数点保存完整生效
配置、`command.txt` 和 `simulation.log`。汇总文件包括：

- `theoretical.csv`：从生效配置自动推导的模型峰值；
- `steady-state.csv`：相邻工作量点的差分 bytes、cycles、GB/s 和理论利用率。

本目录目前只提供纯读测试：L1 使用 read request bytes，L2 使用 read-hit served
bytes，HBM 使用 read column-command bytes。checksum 输出和其他辅助写流量不计入
目标读带宽。读写比例与 HBM 双向带宽将在后续独立专题中研究，不与本目录的
纯读标定混合。

## 7. 能否保证测到真正极限

### 7.1 推荐：三点差分稳态带宽

周期级模拟不推荐用大规模参数网格寻找单个极值。更简单且更可解释的方法是固定
并发度，只增加有效工作量，并用相邻两点的增量计算稳态带宽：

```text
BW_steady = (Bytes_large - Bytes_small) / (Time_large - Time_small)
```

启动、冷缓存和收尾开销在差分中大部分被抵消。使用三个工作量点会得到两段
独立斜率；两段差异不超过 3% 时，认为已经进入稳态。运行：

```bash
tutorials/bandwidth-benchmarks/l1d/analyze.sh
tutorials/bandwidth-benchmarks/l2/analyze.sh
tutorials/bandwidth-benchmarks/hbm/analyze.sh
```

当前读测试固定使用 L1 的 64/128/256 次复用、L2 的 8/16/32 次复用和 HBM
的 64/128/256 MiB 流。若两段斜率不一致，只需增加一个更大工作量点，无需重新
搜索 blocks 或运行完整笛卡尔积。

### 7.2 极值扫描的定位

有限微基准不能严格证明全局极限。本套方法能保证：

1. 理论公式和实测计数观察同一个服务事件；
2. 配置、命令和日志完整可复现；
3. 扫描并发度、复用次数和工作集大小；
4. 汇总脚本比较最优与次优点，默认差距不超过 3% 才标记“形成平台”；
5. 未形成平台时明确要求扩大扫描。

即使形成平台，严谨表述仍是：

> 在给定 FlashGPU-Sim 版本、配置和参数扫描范围内观测到的最大持续带宽。

若要增强证据，应继续增加地址映射、bank/row 分布和扫描边界，并交叉检查端口
stall counter、L2 hit/miss 组成和 DRAM command 利用率。
