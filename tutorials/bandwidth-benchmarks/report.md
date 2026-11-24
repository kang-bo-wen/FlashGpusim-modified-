# FlashGPU-Sim 分层存储带宽测试报告

> 文档状态：纯读阶段定稿
> 最近更新：2026-11-18
> 当前完成：L1D、L2、HBM 纯读稳态带宽
> 当前范围：仅保留 L1D、L2、HBM 纯读可复现测试

## 1. 报告目的

本报告汇总 FlashGPU-Sim 在 `SM90_H100` 配置下的分层存储带宽标定结果，
用于后续 global-memory hierarchy Roofline 分析。报告关注模拟器模型中可被普通
CUDA kernel 持续利用的带宽，而不是直接照搬 NVIDIA 产品规格，也不把一次有限
参数扫描得到的最高点称为数学意义上的全局极值。

HBM 读写比例和双向带宽暂不属于本轮标定范围。后续如开展该专题，应另建独立
测试和报告，避免把纯读标定、缓存写策略与 HBM 双向命令调度混为一个指标。

## 2. 测试环境与范围

当前结果来自：

```text
FlashGPU-Sim build: flashgpu-sim_git-commit-857fa854-modified_38.0
配置: SM90_H100
SM 数量: 132
Core clock: 1.5 GHz
L2 clock: 1.7 GHz
HBM/DRAM clock: 2.617 GHz
L2 subpartition/slice 数量: 80 channel × 2 = 160
DRAM channel 数量: 80
统计单位: decimal GB/s 和 TB/s
```

本轮只测量纯读方向：

- L1D：持续 read request；
- L2：持续 resident-set read hit；
- HBM：持续 streaming read command。

本轮明确排除 Shared-memory bandwidth roof。Shared 指令造成的执行时间在普通
算子测试中仍会反映到 kernel cycles，但 Shared bytes 不加入 L1D arithmetic
intensity，也不使用 L1D 带宽作为 Shared 的上限。

## 3. 流量统计口径

### 3.1 L1D

纯读测试使用：

```text
roofline_l1d_request_read_bytes
```

计数点位于 L1D 真正接受 processor-side request 的位置。`RESERVATION_FAIL`
不会提前收费，retry 成功后只计一次。该流量是合并后的 request payload，不包括
Shared traffic，也不包括 L2 fill。

### 3.2 L2

纯读测试使用：

```text
roofline_l2_data_read_hit_bytes
```

只统计 L2 data port 实际服务的 hit sector，每个 sector 为 32 bytes。测试工作集
常驻 L2，首次装入产生的 miss/fill 通过多工作量差分大部分抵消。辅助 checksum
写入和 dirty eviction 不计入目标 read-hit 带宽。

### 3.3 HBM

纯读测试使用：

```text
roofline_hbm_command_read_bytes
```

每次 DRAM 真正发出 read column command 时计一个 `dram_atom_size`。该口径与
DRAM command-service 理论公式一致。checksum 写回不计入纯读带宽。

## 4. 理论上限

### 4.1 L1D request-admission 上限

```text
132 SM × 4 L1 bank × 32 B × 1.5 GHz
= 25,344 GB/s
= 25.344 TB/s
```

这是当前模型 L1D read-request admission 的结构上限。它来自实现中的 4 个 bank
服务槽，不应直接解释为真实 H100 芯片公开规格中的物理 L1D 数据阵列带宽。

### 4.2 L2 internal data-port 上限

显式 L2 multi-issue port 配置为每个 slice 每个 L2 tick 服务一个 32-byte sector：

```text
160 slice × 1 sector/tick × 32 B × 1.7 GHz
= 8,704 GB/s
= 8.704 TB/s
```

该值要求 160 个 slice 在每个 L2 tick 都有可服务请求，是 L2 内部 data-port 的
结构上限，不等于普通 CUDA kernel 的端到端可达上限。

### 4.3 L2 kernel-visible 上限

当前普通 bypass-L1 load 路径使用 legacy 单-sector 推进方式，132 个 SM 的请求
生成上限为：

```text
132 SM × 1 sector/core tick × 32 B × 1.5 GHz
= 6,336 GB/s
= 6.336 TB/s
```

因此当前 kernel-visible 理论上限为：

```text
min(8.704 TB/s internal data port, 6.336 TB/s SM request path)
= 6.336 TB/s
```

`1 sector/SM/core tick` 的来源不是额外假设，而是：

1. `gpgpu_ldst_request_width` 未被配置覆盖，默认值为 0；
2. request width 为 0 时，legacy 分支在每次 `ldst_unit::cycle()` 中只执行一次
   `inject_one()`，并从 access queue 弹出一个 child；
3. `gpgpu_mem_unit_ports` 默认值为 1，`ldst_unit::clock_multiplier()` 因而返回 1，
   每个 SM 每个 core tick 只运行一次该 service cycle；
4. `SECTOR_SIZE=32`，且本 `.cg.v4` 测试日志中 accepted request child 与
   accepted data sector 一一对应。

因此该值应理解为当前微基准、当前合并行为和当前 legacy LD/ST 路径的
producer-side reference ceiling，不是所有 kernel 永久不变的硬件常数。

后续 Roofline 应使用 6.336 TB/s 作为理论端到端参考；8.704 TB/s 单独报告为
内部端口结构上限。

### 4.4 HBM command-service 上限

```text
dram_atom_size = burst length 4 × bus width 8 × 1 chip = 32 B

80 channel × 32 B/command ÷ tCCD 2 × 2.617 GHz
= 3,349.76 GB/s
= 3.34976 TB/s
```

这是读写 column command 共享的聚合结构上限。本轮纯读测试只使用 read command
bytes；该数值不代表任意读写比例下都能获得相同的聚合持续带宽。

## 5. 稳态测量方法

早期尝试使用较大的参数网格寻找最高点，但周期精确模拟运行时间长，而且有限
网格的最大值容易受到启动、冷缓存、收尾阶段和扫描边界影响。最终采用固定并发、
增加有效工作量的多点差分法：

```text
BW_steady = (Bytes_large - Bytes_small)
            / (Time_large - Time_small)
```

使用三个或四个单调增加的工作量点，得到多段独立斜率。最后两段差异不超过
3% 时，认为测试已经进入稳态。最终报告最后一段斜率，而不是任意单点的
`bytes / total_time`。

这种方法的优点是：

- 大部分启动和收尾固定开销在差分中抵消；
- 冷缓存首次装入的影响随工作量增加而被隔离；
- 不需要搜索大量 blocks/iterations 组合；
- 可以用相邻斜率的一致性直接验证收敛。

## 6. 纯读测试结果

### 6.1 总结

| 层级 | 最终稳态读带宽 | 理论参考上限 | 利用率 | 最后两段差异 | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| L1D request | 25.115 TB/s | 25.344 TB/s | 99.10% | 0.491% | 接近理论峰值，基本饱和 |
| L2 read hit | 4.092 TB/s | 6.336 TB/s kernel-visible | 64.58% | 1.870% | 稳态中高利用率，未完全饱和 |
| HBM read command | 2.721 TB/s | 3.350 TB/s | 81.23% | 0.460% | 稳态高利用率，较充分饱和 |

### 6.2 L1D 纯读

固定参数：

```text
blocks = 528
iterations = 64, 128, 256
每个 CTA 私有 16 KiB hot set
load = ld.global.ca.v4.f32
```

差分结果：

| 差分区间 | 稳态斜率 | 理论利用率 |
| --- | ---: | ---: |
| i64 → i128 | 25,239.247 GB/s | 99.587% |
| i128 → i256 | 25,115.213 GB/s | 99.097% |

最后两段只相差 0.491%，说明稳定斜率已经形成。最终采用：

```text
L1D sustained read-request bandwidth = 25,115.213 GB/s
```

结论：L1D request-admission 路径可以被当前微基准基本打满。最后一段等价于
每个 SM 每个 core tick 接受 3.9639 个 32-byte sector，而模型槽位上限为 4 个。
这证明微基准饱和了 FlashGPU-Sim 的请求接受模型，但不独立证明真实 H100 的
物理 L1D 带宽为 25.115 TB/s。

L1D 能接近 100% 的原因比较明确：

1. 测试使用规则、连续且对齐的 `float4` 访问，合并后的 32-byte sector 能较均匀地
   分布到 4 个 L1D bank；模型代码每个 core tick 最多为 4 个 bank 各接受一个请求；
2. `528 blocks / 132 SM = 4 blocks/SM`，并发 warp 足以持续向 L1D latency queue
   供给请求并隐藏 L1D latency；
3. 每个 CTA 只反复访问私有 16 KiB hot set，完成第一次装入后，新增访问几乎都是
   L1D hit，不需要继续经过 L2/HBM；
4. 三个运行点的 L1D miss 数均为 270,864，没有随 iterations 增长。差分区间新增的
   流量因此基本都是稳定 hit，冷启动开销被抵消；
5. 理论上限和计数器都对应 L1D bank request-admission 事件，微基准又专门按该
   服务槽设计，因此在模拟器内部接近结构上限是预期现象。

最后一点也意味着该结果具有“模型内同口径”的性质：它严格说明请求接受槽被
打满，但不能直接外推为真实芯片数据阵列、返回网络或应用程序都能持续获得同样
的 25.115 TB/s。

### 6.3 L2 纯读命中

固定参数：

```text
elements = 4,194,304 float = 16 MiB
blocks = 2,112
iterations = 8, 16, 32, 64
load = ld.global.cg.v4.f32
```

16 MiB 工作集大于单个 L1D、低于约 50 MiB L2 容量。反复读取使主要新增
流量成为 L2 hit。

差分结果：

| 差分区间 | Read-hit ΔBytes | ΔCycles | 稳态斜率 | 相对 6.336 TB/s | 相对 8.704 TB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| i8 → i16 | 133,954,944 | 49,902 | 4,026.540 GB/s | 63.55% | 46.26% |
| i16 → i32 | 268,033,920 | 96,425 | 4,169.571 GB/s | 65.81% | 47.90% |
| i32 → i64 | 536,315,264 | 196,616 | 4,091.594 GB/s | 64.58% | 47.01% |

最后两段只相差 1.865%，因此 L2 读命中稳态带宽已经收敛。最终采用：

```text
L2 sustained read-hit bandwidth = 4,091.594 GB/s
```

早期汇总使用兼容字段 `roofline_l2_data_hit_bytes`，它同时包含 read-hit 和
checksum/local-store 产生的 write-hit，因此曾得到 5.092 TB/s。拆分
`roofline_l2_data_read_hit_bytes` 与 `roofline_l2_data_write_hit_bytes` 后，利用旧日志
中的 `GLOBAL_ACC_R[HIT] × 32 B` 重算，纯读结果修正为 4.092 TB/s。旧值不再用于
Roofline。

L2 结果需要分两层理解。首先，8.704 TB/s 是 160 个 slice 的内部 data-port
结构上限，并不是这个 CUDA load 路径可以产生的流量。当前 `.cg` 请求绕过 L1D，
legacy LD/ST 路径每个 SM 每个 core tick 最多推进一个 32-byte sector，因此在到达
L2 之前已经受到 6.336 TB/s producer-side 上限约束。4.092 TB/s 应优先和
6.336 TB/s 比较，其利用率为 64.58%；相对内部 8.704 TB/s 则为 47.01%。

在 6.336 TB/s 路径上限之下仍有约 35.4% 差距，主要来自持续供给和负载均衡：

1. `1 sector/SM/core tick` 只是请求队列始终非空、互连始终可接受时的最大推进率；
   实际 kernel 还需要发射 load、执行累加、等待 scoreboard，并处理响应返回；
2. 132 个 SM 的请求经过地址映射进入 160 个 L2 slice，瞬时分布、队列占用和响应
   回压不可能始终完全均匀，会造成部分 slice 有请求而部分 slice 出现空槽；
3. 单个 warp 的合并访问会被拆成多个 sector child 顺序推进，warp 调度与 LD/ST
   pipeline 之间可能形成供给气泡；
4. CTA 启动和结束阶段也不能维持全局满载，但差分法已经抵消了其中大部分固定
   开销，因此它不是 35.4% 差距的唯一解释。

首次装入产生的 L2 miss 在各 iteration 点中基本属于固定开销，已被相邻点差分
大幅抵消，所以不再把冷 miss 列为稳态差距的主要原因。若要把上述因素进一步
定量拆分，需要增加逐周期的 L2 slice occupancy、LD/ST request stall、互连
backpressure 和 response-retirement 计数器；当前结论是结构解释，而不是对
每一项损失比例的精确归因。

因此当前严谨的普通 CUDA load 路径稳定持续读命中带宽应为 4.092 TB/s；
5.092 TB/s 是混入辅助写命中后的旧兼容口径，不能再作为读带宽或 Roofline
带宽使用。4.092 TB/s 同样不是 L2 内部端口的绝对峰值。

### 6.4 HBM 纯读

固定参数：

```text
blocks = 1,056
stream size = 64, 128, 256, 512 MiB
load = ld.global.cg.v4.f32
```

工作集显著大于 L2，实测日志中 streaming 访问的 L2 miss 接近 100%，因此新增
流量真正到达 HBM command-service 路径。

差分结果：

| 差分区间 | ΔBytes | ΔCycles | 稳态斜率 | 理论利用率 |
| --- | ---: | ---: | ---: | ---: |
| 64 → 128 MiB | 67,108,864 | 39,107 | 2,574.048 GB/s | 76.84% |
| 128 → 256 MiB | 134,217,728 | 73,652 | 2,733.484 GB/s | 81.60% |
| 256 → 512 MiB | 268,435,456 | 147,985 | 2,720.905 GB/s | 81.23% |

最后两段只相差 0.460%，说明扩大到 512 MiB 后带宽不再继续上升。最终采用：

```text
HBM sustained read-command bandwidth = 2,720.905 GB/s
```

HBM 的 3.34976 TB/s 理论值假定 80 个 channel 都能长期每隔 `tCCD=2` 个 DRAM
tick 发出一个 32-byte column command，即始终存在已完成行准备、满足全部 timing
约束且可立即调度的请求。实测 81.23% 等价于全通道平均约每 2.46 个 DRAM tick
发出一个 column command，而不是理想的每 2 tick 一个。

剩余约 18.8% 的差距主要可能来自：

1. 顺序地址经映射后在 channel、bank 和 bank group 上仍存在瞬时负载不均；
2. 行切换需要 ACT/PRE，并受 `tRCD`、`tRP`、`tRAS`、`tRRD` 等约束，不能每次都只
   支付 `tCCD`；
3. bank-group timing、FR-FCFS 调度和有限队列会在 column command 之间引入空槽；
4. SM 请求生成、L2 miss/fill、片上互连和返回路径可能无法在每个 DRAM tick 都为
   80 个 channel 同时提供可发射请求。

最后两段只相差 0.460%，说明把流长度从 256 MiB 增加到 512 MiB 已不能继续提高
带宽，因此差距不是测试流过短造成的。当前日志证明了 command 利用率和稳态性，
但还不能精确量化以上四类原因各占多少；这需要额外的 channel busy、row-hit、
timing stall 和队列占用统计。

## 7. Roofline 使用建议

在当前只完成纯读标定的阶段，纯读或读主导算子的经验 Roofline 可以使用：

```text
L1D read roof = 25.115 TB/s
L2 read roof  = 4.092 TB/s
HBM read roof = 2.721 TB/s
```

各层 arithmetic intensity 的分母必须和带宽观察点一致：

```text
AI_L1D = FLOPs / roofline_l1d_request_total_bytes
AI_L2  = FLOPs / 对应 L2 data-port 实际服务字节
AI_HBM = FLOPs / roofline_hbm_command_total_bytes
```

对于包含写流量的真实算子，当前纯读 roof 只能作为统一的阶段性参照，不能被
解释成任意读写比例下的精确物理上限。本文当前研究目标是比较 HBM 参数变化对
算子性能和分层流量的相对影响，而不是标定所有读写比例的产品级峰值。

Shared memory 暂不建立 bandwidth roof，也不把 Shared bytes 加入 L1D bytes。
Shared 指令、同步、bank conflict 和 occupancy 的执行代价仍保留在 achieved
performance 的模拟周期中。

## 8. 复现方式

环境初始化：

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
```

分层运行：

```bash
tutorials/bandwidth-benchmarks/l1d/run.sh
tutorials/bandwidth-benchmarks/l2/run.sh
tutorials/bandwidth-benchmarks/hbm/run.sh
```

仅重新分析已有日志：

```bash
tutorials/bandwidth-benchmarks/l1d/analyze.sh
tutorials/bandwidth-benchmarks/l2/analyze.sh
tutorials/bandwidth-benchmarks/hbm/analyze.sh
```

有效结果分别保存在：

```text
l1d/runs-SM90_H100/
l2/runs-SM90_H100/
hbm/runs-SM90_H100/
```

每个测试点保留完整生效配置、运行命令和 `simulation.log`，汇总结果保存在各自
目录的 `steady-state.csv`。

## 9. 当前结论

1. L1D 纯读 request-admission 能力已经被微基准基本打满，实测达到理论值的
   99.10%。
2. L2 纯读命中稳定带宽为 4.092 TB/s。相对普通 CUDA kernel 可见路径上限
   6.336 TB/s，利用率为 64.58%；不能再用 8.704 TB/s 直接评价端到端效率。
3. HBM 纯读稳定带宽为 2.721 TB/s，相对 command-service 理论上限利用率为
   81.23%。256→512 MiB 斜率已经稳定，继续增加流长度预计不会显著提高结果。
4. 当前三个数字都是纯读经验 roof，不是通用读写聚合 roof。它们适合用作后续
   HBM 扩展实验的固定基线，以及判断瓶颈由 HBM 向 L2/请求生成路径迁移的位置。

## 10. 后续光互联/HBM 扩展研究建议

本课题不应继续把主要精力放在微基准峰值的小数点上，而应把当前三个纯读结果
固定为基线，研究提高单 GPU 外部存储服务能力后，真实算子的性能收益和瓶颈迁移。

需要特别注意：L2 resident read hit 使用 data-hit port，而 HBM miss 返回主要经过
L2 fill/return path。当前测试配置中 data port 与 fill port 是独立资源，因此不能仅用
`HBM read-command bandwidth > L2 read-hit bandwidth` 推断流式算子必然被 L2
限制。后续必须同时观测 L2 fill、memory-to-core interconnect 和 LD/ST response
retirement。

建议使用带宽缩放因子 `1x、1.5x、2x、3x、4x、8x`，并把两个因素分开：

1. bandwidth-only：提高 column-command 服务能力，保持访问延迟近似不变；
2. bandwidth+latency：在提高带宽的同时降低或重新设定外部存储延迟；
3. L2 fixed：第一阶段固定 L2 容量、端口和时钟，以观察真实瓶颈迁移；
4. L2 scaled：第二阶段再同步扩展 L2 fill/return 能力，判断需要怎样的片上配套。

每个算子、每个缩放点至少记录：kernel cycles、speedup、FLOPs、L1D request
bytes、L2 read-hit/fill bytes、HBM command bytes、L2 hit rate、DRAM/L2/ICNT 队列
与回压、LD/ST request/response 利用率。核心结果不是单一 GB/s，而是性能曲线何时
变平，以及变平后由哪个端口或 stall 原因接管瓶颈。
