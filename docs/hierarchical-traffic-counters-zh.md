# FlashGPU-Sim 分层流量计数器代码说明

本文解释 FlashGPU-Sim baseline 中新增的 L1D、Shared Memory、L2 和 HBM
分层物理流量计数器。内容包括每个日志字段的含义、底层变量、定义位置、计数
触发位置、计数单位、适用范围和使用限制。

本文对应当前 baseline 源码。文中行号用于定位当前版本；如果后续代码发生增删，
应以函数名和字段名为准重新确认行号。

## 1. 统计目标与观察点

这些计数器不尝试用一个数字表示整个存储层次，而是在实际服务事件发生的位置
分别观察流量：

```text
CUDA load/store
      |
      +--> L1D request accepted
      |       roofline_l1d_request_*
      |
      +--> Shared bank/word service
      |       roofline_shared_bank_*
      |
      +--> L2 data port / fill port
      |       roofline_l2_data_*
      |       roofline_l2_fill_port_total_bytes
      |
      +--> DRAM column command issued
              roofline_hbm_command_*
```

不同观察点记录的是不同硬件资源处理的物理工作，不能简单相加，也不存在普遍成立
的 `L1 >= L2 >= HBM` 关系。例如非合并写可能使 L1D 请求字节远大于算法写字节；
一次性冷读可能没有 L2 read hit，但会产生 L2 fill 和 HBM read command。

## 2. 计数器的公共实现

### 2.1 存储类型和线程安全

所有底层计数器定义在
[`src/gpgpu-sim/roofline_stats.h`](../src/gpgpu-sim/roofline_stats.h#L9) 的
`roofline_stats` 命名空间中。

[`roofline_stats.h:11`](../src/gpgpu-sim/roofline_stats.h#L11) 将计数器定义为：

```cpp
using counter = std::atomic<unsigned long long>;
```

计数使用
[`fetch_add(..., std::memory_order_relaxed)`](../src/gpgpu-sim/roofline_stats.h#L21)。
FlashGPU-Sim 可以由多个 OpenMP 主机线程并行模拟不同 SM，因此普通整数累加会产生
数据竞争。relaxed atomic 保证最终求和不丢失更新，同时不引入不需要的线程同步
顺序。

所有字段的基本单位都是 **byte**。L2 内部以 32-byte sector 计数，写入公共计数器
时乘以 32；HBM 每发出一个 column command，就增加一个 `dram_atom_size`。

### 2.2 生命周期

底层计数器是进程级 `inline` 全局变量，在进程启动时初始化为 0。重置函数定义在
[`roofline_stats.h:28-37`](../src/gpgpu-sim/roofline_stats.h#L28)：

```cpp
roofline_stats::reset();
```

当前普通测试不会自动在每个 kernel 前重置。现有带宽测试和 transpose 测试为每个
测量点启动一个独立进程，因此每份日志从 0 开始。如果一个进程连续运行预热 kernel、
辅助 kernel 和被测 kernel，日志字段将是这些 kernel 的累计值。此时必须：

1. 在测量区间前安全调用 `reset()`；或者
2. 在区间前后分别读取快照并做差分；或者
3. 继续让每个测量点使用独立进程。

不能在模拟器工作线程正在更新计数器时重置。

### 2.3 日志输出

所有 `roofline_*` 字段由
[`roofline_stats::print()`](../src/gpgpu-sim/roofline_stats.h#L64) 统一输出。
调用位置为
[`src/gpgpu-sim/shader.cc:5292`](../src/gpgpu-sim/shader.cc#L5292)，位于 L1D
统计输出分支内。因此当前配置需要启用 L1D，才能在标准 simulation log 中看到这些
字段。SM90_H100 baseline 的 L1D 已启用。

## 3. 字段总表

下表中的“独立变量”表示日志字段直接读取一个原子计数器；“派生字段”表示打印时
由其他计数器求和，没有同名的独立存储变量。

| 日志字段 | 类型 | 底层变量或公式 | 观察点 |
| --- | --- | --- | --- |
| `roofline_l1d_request_read_bytes` | 独立变量 | `l1d_request_read_bytes` | L1D 接受读请求 |
| `roofline_l1d_request_write_bytes` | 独立变量 | `l1d_request_write_bytes` | L1D 接受写请求 |
| `roofline_l1d_request_total_bytes` | 派生字段 | L1D read + write | L1D 请求总流量 |
| `roofline_shared_bank_read_bytes` | 独立变量 | `shared_bank_read_bytes` | Shared 读 bank/word 服务 |
| `roofline_shared_bank_write_bytes` | 独立变量 | `shared_bank_write_bytes` | Shared 写 bank/word 服务 |
| `roofline_shared_bank_total_bytes` | 派生字段 | Shared read + write | Shared 总服务流量 |
| `roofline_l2_data_read_hit_bytes` | 独立变量 | `l2_data_read_hit_bytes` | L2 data port 服务读命中 |
| `roofline_l2_data_write_hit_bytes` | 独立变量 | `l2_data_write_hit_bytes` | L2 data port 服务写命中 |
| `roofline_l2_data_hit_bytes` | 派生字段 | L2 read hit + write hit | 兼容用命中总量 |
| `roofline_l2_data_miss_read_bytes` | 独立变量 | `l2_data_miss_read_bytes` | fill 后经共享 data array 读出响应 |
| `roofline_l2_data_fill_bytes` | 独立变量 | `l2_data_fill_bytes` | fill 同时占用共享 data port |
| `roofline_l2_data_dirty_eviction_bytes` | 独立变量 | `l2_data_dirty_eviction_bytes` | data port 读出 dirty victim |
| `roofline_l2_data_port_total_bytes` | 派生字段 | 五类 L2 data work 之和 | L2 data port 总服务流量 |
| `roofline_l2_fill_port_total_bytes` | 独立变量 | `l2_fill_port_bytes` | L2 fill port 接受返回 sector |
| `roofline_hbm_command_read_bytes` | 独立变量 | `dram_command_read_bytes` | 发出 DRAM read column command |
| `roofline_hbm_command_write_bytes` | 独立变量 | `dram_command_write_bytes` | 发出非 L2 writeback 的 write command |
| `roofline_hbm_command_writeback_bytes` | 独立变量 | `dram_command_writeback_bytes` | 发出 L2 dirty writeback command |
| `roofline_hbm_command_total_bytes` | 派生字段 | HBM read + write + writeback | HBM column-command 总字节 |
| `roofline_l1_total_bytes` | 兼容字段 | L1D read/write + Shared read/write | 旧脚本兼容，不代表统一物理端口 |
| `roofline_l2_total_bytes` | 兼容字段 | 等于 L2 data-port total | 旧脚本兼容 |
| `roofline_hbm_total_bytes` | 兼容字段 | 等于 HBM command total | 旧脚本兼容 |

底层变量集中定义在
[`roofline_stats.h:12-19`](../src/gpgpu-sim/roofline_stats.h#L12)，日志公式集中在
[`roofline_stats.h:65-92`](../src/gpgpu-sim/roofline_stats.h#L65)。

## 4. L1D 请求计数器

### 4.1 `roofline_l1d_request_read_bytes`

**含义**

统计 L1D 接受的读请求数据大小。它描述 processor-side L1D request admission
流量，不是 L1D hit 返回字节，也不是所有 global load 的逻辑字节。

**定义位置**

- 底层变量：
  [`roofline_stats.h:12`](../src/gpgpu-sim/roofline_stats.h#L12)
- 读写选择辅助函数：
  [`roofline_stats.h:39-41`](../src/gpgpu-sim/roofline_stats.h#L39)
- 日志输出：
  [`roofline_stats.h:72`](../src/gpgpu-sim/roofline_stats.h#L72)

**何时计数**

SM90_H100 的 `-gpgpu_l1_latency 39` 使请求先进入每个 L1D bank 的 latency queue。
请求到达 queue 首部后，
[`ldst_unit::L1_latency_queue_cycle()`](../src/gpgpu-sim/shader.cc#L3788) 调用
`m_L1D->access()`。当返回状态不是 `RESERVATION_FAIL` 时，
[`shader.cc:3799-3802`](../src/gpgpu-sim/shader.cc#L3799) 执行：

```cpp
roofline_stats::add_l1d_request(mf_next->is_write(),
                                mf_next->get_data_size());
```

若 `mf_next->is_write()` 为 false，本次 `get_data_size()` 加入 read counter。

**为什么在这里计数**

请求已经被具体 L1D bank 接受，hit 和 miss 都会消耗一次请求接受槽。发生
`RESERVATION_FAIL` 的尝试不计数，等它以后真正被接受时再计，因此不会把重试重复
计算为物理服务流量。

**不计入的情况**

- 使用 cache operator 绕过 L1D 的请求；
- constant cache、texture cache 和 instruction cache 流量；
- L1D queue 尚未接受的 reservation-fail 尝试。

### 4.2 `roofline_l1d_request_write_bytes`

定义位置和触发点与 L1D read 相同。当
[`mf_next->is_write()`](../src/gpgpu-sim/shader.cc#L3800) 为 true 时，将
`get_data_size()` 加入
[`l1d_request_write_bytes`](../src/gpgpu-sim/roofline_stats.h#L12)。日志输出位于
[`roofline_stats.h:73`](../src/gpgpu-sim/roofline_stats.h#L73)。

这个字段特别适合观察非合并写放大。例如 naive transpose 中每个 4-byte 有效写
可能占用一个 32-byte sector，因此 L1D physical write request 可以达到算法写字节
的 8 倍。

### 4.3 `roofline_l1d_request_total_bytes`

这是打印时计算的派生字段，没有独立原子变量：

```text
L1D total = L1D read + L1D write
```

公式位于
[`roofline_stats.h:74`](../src/gpgpu-sim/roofline_stats.h#L74)。它适合表示 L1D
请求接受端口的总工作量，但不能与 Shared bytes 相加后继续使用 L1D 峰值作为分母。

## 5. Shared Memory bank/word 计数器

### 5.1 物理字节如何生成

每条普通 shared load/store 在生成访存事务时，代码按 bank 和 32-bit word 对活跃
lane 的地址分组。实现位于
[`warp_inst_t::generate_mem_accesses()`](../src/abstract_hardware_model.cc#L291)。

核心步骤位于
[`abstract_hardware_model.cc:356-388`](../src/abstract_hardware_model.cc#L356)：

1. 遍历 active lane 和向量元素；
2. 计算 shared bank；
3. 计算 32-bit word 地址；
4. 对每个唯一 `(bank, word)` 记一次物理 word transaction。

最终在
[`abstract_hardware_model.cc:454`](../src/abstract_hardware_model.cc#L454)
计算：

```text
m_smem_physical_bytes = unique(bank, word) 数量 × WORD_SIZE
```

当前 `WORD_SIZE` 为 4 bytes。同一 warp 的多个 lane 读取同一 word 时按一次 broadcast
事务计数；同一 bank 的不同 word 会形成多个事务并引入 bank-conflict 序列化。
字节数表示实际 word 服务量，冲突造成的额外时间由 shared timing 模型体现，不能
通过重复增加相同字节表示。

每条指令的临时字段定义在
[`abstract_hardware_model.h:1384-1388`](../src/abstract_hardware_model.h#L1384)，
并在指令初始化时清零，见
[`abstract_hardware_model.h:1100-1103`](../src/abstract_hardware_model.h#L1100)。

### 5.2 `roofline_shared_bank_read_bytes`

**定义位置**

- 底层变量：
  [`roofline_stats.h:13`](../src/gpgpu-sim/roofline_stats.h#L13)
- 读写选择函数：
  [`roofline_stats.h:42-44`](../src/gpgpu-sim/roofline_stats.h#L42)
- 日志输出：
  [`roofline_stats.h:75`](../src/gpgpu-sim/roofline_stats.h#L75)

**何时计数**

普通 shared 指令进入
[`ldst_unit::shared_cycle()`](../src/gpgpu-sim/shader.cc#L3590) 后，在开始
占用 shared service resource 时计数。具体调用位于
[`shader.cc:3610-3616`](../src/gpgpu-sim/shader.cc#L3610)：

```cpp
if (!inst.m_smem_exposed) {
  roofline_stats::add_shared_bank(inst.is_store(),
                                  inst.m_smem_physical_bytes);
  inst.m_smem_exposed = true;
}
```

当 `inst.is_store()` 为 false 时，物理字节加入 read counter。`m_smem_exposed` 保证
指令即使因为 shared latency 或 bank conflict 在 pipeline 中停留多个 cycle，也只
计数一次。

### 5.3 `roofline_shared_bank_write_bytes`

触发位置与 shared read 相同。当 `inst.is_store()` 为 true 时，
`m_smem_physical_bytes` 加入
[`shared_bank_write_bytes`](../src/gpgpu-sim/roofline_stats.h#L13)。日志输出位于
[`roofline_stats.h:76`](../src/gpgpu-sim/roofline_stats.h#L76)。

### 5.4 `roofline_shared_bank_total_bytes`

派生公式为：

```text
Shared total = Shared read + Shared write
```

输出位置为
[`roofline_stats.h:77`](../src/gpgpu-sim/roofline_stats.h#L77)。Shared 和 L1D 是
两套不同资源，不能使用同一个理论上限计算联合利用率。

### 5.5 当前覆盖边界

上述逻辑覆盖普通 LSU shared load/store，包括普通向量 shared 访问。本项目当前
不把 TMA 纳入统计完整性要求。DSM remote shared 访问被
[`!inst.is_dsm_remote()`](../src/gpgpu-sim/shader.cc#L3605) 明确排除，由 DSM
endpoint/fabric 处理。以后若算子使用 `ldmatrix`、`cp.async` 等特殊路径，应先用
受控微基准验证其 source 和 destination 是否落入现有观察点，不能直接假设与普通
shared load/store 完全相同。

## 6. L2 端口计数器

### 6.1 启用条件和基本单位

新增 L2 计数挂在 multi-issue L2 port model 上。核心类为
[`l2_multi_issue_ports`](../src/gpgpu-sim/l2cache.h#L448)。每个 L2 工作包是一个
32-byte sector，类注释位于
[`l2cache.h:444-447`](../src/gpgpu-sim/l2cache.h#L444)。

必须启用：

```text
-gpgpu_l2_multi_issue_port_model 1
```

现有带宽脚本在
[`tutorials/bandwidth-benchmarks/common.sh:30-36`](../tutorials/bandwidth-benchmarks/common.sh#L30)
显式追加该配置，transpose 脚本也在
[`tutorials/transpose-benchmarks/run.sh:35-41`](../tutorials/transpose-benchmarks/run.sh#L35)
追加相同配置。若直接使用未追加该选项的 legacy L2 路径，以下新增 L2 物理字节
计数不完整，不能使用。

所有 data-port 工作由
[`accept_data()`](../src/gpgpu-sim/l2cache.h#L512) 接受。只有真正得到本 cycle
端口 credit 的 sector 才计数。请求进入 L2、tag lookup、MSHR merge 和
`RESERVATION_FAIL` 都不是 data-port 服务，不增加这些字节。

### 6.2 `roofline_l2_data_read_hit_bytes`

**定义位置**

- 底层变量：
  [`roofline_stats.h:14`](../src/gpgpu-sim/roofline_stats.h#L14)
- L2 work 类型：
  [`roofline_stats.h:45-47`](../src/gpgpu-sim/roofline_stats.h#L45)
- sector 到 byte 的转换：
  [`roofline_stats.h:48-55`](../src/gpgpu-sim/roofline_stats.h#L48)
- 日志输出：
  [`roofline_stats.h:78`](../src/gpgpu-sim/roofline_stats.h#L78)

**何时计数**

请求在
[`service_l2_requests_multi_issue()`](../src/gpgpu-sim/l2cache.cc#L848) 中完成
tag/data-array 访问。当 `status == HIT` 时，
[`l2cache.cc:914-919`](../src/gpgpu-sim/l2cache.cc#L914) 根据
`mf->is_write()` 选择 read-hit 或 write-hit。读请求调用：

```cpp
accept_data(1, L2_MULTI_ISSUE_READ_HIT_DATA)
```

随后
[`l2cache.h:518-526`](../src/gpgpu-sim/l2cache.h#L518) 把接受的 sector 数乘以
32 bytes 加入 `l2_data_read_hit_bytes`。

它统计的是 data port 实际服务的读命中，不统计读请求数，也不统计读 miss 后的
fill。一次性冷流式算子的该字段可以为 0。

### 6.3 `roofline_l2_data_write_hit_bytes`

定义在
[`roofline_stats.h:14`](../src/gpgpu-sim/roofline_stats.h#L14)，输出位于
[`roofline_stats.h:79`](../src/gpgpu-sim/roofline_stats.h#L79)。L2 hit 请求满足
`mf->is_write()` 时，
[`l2cache.cc:915-918`](../src/gpgpu-sim/l2cache.cc#L915) 选择
`L2_MULTI_ISSUE_WRITE_HIT_DATA`，接受的每个 sector 增加 32 bytes。

该字段是 L2 data array 的写命中服务流量，不等于最终到达 HBM 的写流量。write-back
cache 中，写命中只把 L2 line 标记为 dirty；只有后续 eviction 才会产生 HBM
writeback。

### 6.4 `roofline_l2_data_hit_bytes`

兼容派生字段：

```text
L2 hit = L2 read hit + L2 write hit
```

公式和输出位于
[`roofline_stats.h:67-80`](../src/gpgpu-sim/roofline_stats.h#L67)。它曾被旧脚本
当成纯读 L2 流量，导致把辅助 store 的 write-hit 混入读带宽。新的分析必须优先
使用拆分后的 read-hit 和 write-hit 字段。

### 6.5 `roofline_l2_data_miss_read_bytes`

定义在
[`roofline_stats.h:15`](../src/gpgpu-sim/roofline_stats.h#L15)，输出位于
[`roofline_stats.h:81`](../src/gpgpu-sim/roofline_stats.h#L81)。

这个名字不能理解为“所有 L2 read miss 请求字节”。它表示 read miss 从 HBM fill
回来以后，为了向上层返回数据而再次占用 **共享 L2 data array** 的读服务。

触发逻辑位于
[`service_ready_l2_response()`](../src/gpgpu-sim/l2cache.cc#L716)。只有
`-gpgpu_l2_shared_data_array 1` 时，
[`l2cache.cc:724-732`](../src/gpgpu-sim/l2cache.cc#L724) 才调用：

```cpp
accept_data(1, L2_MULTI_ISSUE_MISS_READ)
```

当前带宽和 transpose 测试使用 `-gpgpu_l2_shared_data_array 0`，data 与 fill 是独立
资源，因此该字段通常为 0。这不表示没有 L2 miss；miss 返回流量应查看
`roofline_l2_fill_port_total_bytes`。

### 6.6 `roofline_l2_data_fill_bytes`

定义在
[`roofline_stats.h:16`](../src/gpgpu-sim/roofline_stats.h#L16)，输出位于
[`roofline_stats.h:82`](../src/gpgpu-sim/roofline_stats.h#L82)。

HBM 返回数据进入 L2 时总会消耗 fill-port credit，但只有
`m_fill_uses_data_port == true` 时，fill 还会同时消耗共享 data-port credit。
该条件和计数调用位于
[`l2cache.h:543-559`](../src/gpgpu-sim/l2cache.h#L543)：

```cpp
if (m_fill_uses_data_port) {
  roofline_stats::add_l2_data(roofline_stats::L2_FILL, accepted);
}
roofline_stats::add_l2_fill_port(accepted);
```

所以：

- `-gpgpu_l2_shared_data_array 1`：同一 fill 会同时出现在 data-fill 和 fill-port
  两个观察点，因为它确实占用两个约束；
- `-gpgpu_l2_shared_data_array 0`：`l2_data_fill_bytes` 为 0，只增加独立 fill-port
  字段。

不能把这两个观察点相加后称为唯一的 L2 总流量。

### 6.7 `roofline_l2_data_dirty_eviction_bytes`

定义在
[`roofline_stats.h:16`](../src/gpgpu-sim/roofline_stats.h#L16)，输出位于
[`roofline_stats.h:83`](../src/gpgpu-sim/roofline_stats.h#L83)。

L2 miss 选择到 dirty victim 后，victim 必须先从 L2 data array 读出，才能送往
HBM。每个实际使用 data-port credit 的 dirty sector 计 32 bytes。

已有 pending writeback 的服务位置为
[`l2cache.cc:851-863`](../src/gpgpu-sim/l2cache.cc#L851)；新访问生成 deferred
writeback 后的首次服务位置为
[`l2cache.cc:923-936`](../src/gpgpu-sim/l2cache.cc#L923)。两处都最终调用：

```cpp
accept_data(..., L2_MULTI_ISSUE_DIRTY_EVICTION)
```

该字段描述 L2 data port 读取 dirty victim 的工作量；真正发到 HBM 的写命令由
`roofline_hbm_command_writeback_bytes` 记录。

### 6.8 `roofline_l2_data_port_total_bytes`

派生公式位于
[`roofline_stats.h:84`](../src/gpgpu-sim/roofline_stats.h#L84)：

```text
L2 data-port total = read hit
                   + write hit
                   + miss read
                   + fill using shared data port
                   + dirty eviction
```

这是 data port 的总服务量，可以与相同 data-port 定义的结构上限比较。若 fill port
独立，不能把独立 fill bytes 再加进这个字段后仍使用 data-port 单端口上限。

### 6.9 `roofline_l2_fill_port_total_bytes`

底层变量定义在
[`roofline_stats.h:17`](../src/gpgpu-sim/roofline_stats.h#L17)，增加函数位于
[`roofline_stats.h:56-58`](../src/gpgpu-sim/roofline_stats.h#L56)，日志输出位于
[`roofline_stats.h:85`](../src/gpgpu-sim/roofline_stats.h#L85)。

HBM 返回请求在
[`service_dram_to_l2_multi_issue()`](../src/gpgpu-sim/l2cache.cc#L779) 中等待 fill
服务。pending fill 在
[`l2cache.cc:783-800`](../src/gpgpu-sim/l2cache.cc#L783) 调用 `service_fill()`，
最后由
[`l2cache.h:557-559`](../src/gpgpu-sim/l2cache.h#L557) 对每个接受的 sector 增加
32 bytes。

这个字段最适合解释冷流式访问。即使 `L2 read-hit = 0`，只要数据从 HBM 返回并
填入 L2，fill-port 字段就会增加。

## 7. HBM command 计数器

### 7.1 观察点和基本语义

HBM 插桩位于详细 DRAM timing model 的
[`dram_t::issue_col_command()`](../src/gpgpu-sim/dram.cc#L564)。它不是在请求进入
DRAM queue 时计数，而是在满足 row state、`tCCD`、bank-group timing、读写切换和
返回队列条件后，真正发出 read/write column command 时计数。

因此这些字段表示 **已发出的物理 DRAM data command bytes**。ACT/PRE 是行命令，
没有数据 payload，不计入字节。

底层 HBM 变量定义在
[`roofline_stats.h:18-19`](../src/gpgpu-sim/roofline_stats.h#L18)，分类辅助函数位于
[`roofline_stats.h:59-62`](../src/gpgpu-sim/roofline_stats.h#L59)。

### 7.2 `roofline_hbm_command_read_bytes`

当 bank 中当前请求是 READ，且所有 timing 和 queue 条件满足时，
[`dram.cc:571-586`](../src/gpgpu-sim/dram.cc#L571) 发出读 column command，并执行：

```cpp
roofline_stats::add_dram_command(false, false,
                                 m_config->dram_atom_size);
```

每个命令增加一个 `dram_atom_size`。SM90_H100 当前为 32 bytes。日志输出位于
[`roofline_stats.h:86`](../src/gpgpu-sim/roofline_stats.h#L86)。

该字段包含实际发出的 DRAM read command，包括普通 demand read 以及可能的
write-allocate read；当前没有进一步按来源拆分。

### 7.3 `roofline_hbm_command_write_bytes`

当 bank 中请求是 WRITE 且满足 timing 和 queue 条件时，
[`dram.cc:610-627`](../src/gpgpu-sim/dram.cc#L610) 发出写 column command。
若请求类型不是 `L2_WRBK_ACC`，`add_dram_command()` 将 `dram_atom_size` 加入
`dram_command_write_bytes`。日志输出位于
[`roofline_stats.h:87`](../src/gpgpu-sim/roofline_stats.h#L87)。

该字段用于记录非 L2 dirty-eviction writeback 的 DRAM 写命令，例如取决于 cache
policy 的直达写流量。它不是 CUDA store 指令的逻辑写字节。

### 7.4 `roofline_hbm_command_writeback_bytes`

同一个 WRITE command hook 根据
[`dram.cc:625-627`](../src/gpgpu-sim/dram.cc#L625) 判断：

```cpp
bk[j]->mrq->data->get_access_type() == L2_WRBK_ACC
```

若成立，字节加入
[`dram_command_writeback_bytes`](../src/gpgpu-sim/roofline_stats.h#L19)。日志输出
位于
[`roofline_stats.h:88`](../src/gpgpu-sim/roofline_stats.h#L88)。

它表示 L2 dirty victim 已经实际形成并发出的 HBM write command。kernel 结束时
FlashGPU-Sim 不会自动把所有 dirty L2 line 冲刷到 HBM，因此：

```text
HBM write + HBM writeback < 完整算法输出字节
```

可能完全合理。该字段完整描述测量区间内已发出的 writeback 命令，但不保证所有
算法输出已经落到 HBM。

### 7.5 `roofline_hbm_command_total_bytes`

派生公式位于
[`roofline_stats.h:89`](../src/gpgpu-sim/roofline_stats.h#L89)：

```text
HBM command total = read command + write command + writeback command
```

该字段可描述测量区间内 data command 的总物理 payload。当前纯读峰值只能直接作为
`hbm_command_read_bytes` 的同口径参考。读写混合 total 受 read/write turnaround、
调度策略和队列状态影响，不能直接除以纯读上限后称为严格的总带宽利用率。

### 7.6 simple DRAM 模型限制

HBM hook 位于详细 DRAM 的 `issue_col_command()`。如果配置启用
`-gpgpu_simple_dram_model 1`，请求改走
[`memory_partition_unit::simple_dram_model_cycle()`](../src/gpgpu-sim/l2cache.cc#L314)，
当前该路径没有调用 `add_dram_command()`。因此 simple DRAM 模式下这些 HBM
`roofline_*` 字段不完整，不能使用。

## 8. 兼容字段

### 8.1 `roofline_l1_total_bytes`

定义公式位于
[`roofline_stats.h:90`](../src/gpgpu-sim/roofline_stats.h#L90)：

```text
L1 compatibility total = L1D read + L1D write
                       + Shared read + Shared write
```

该字段只为旧脚本兼容。L1D request port 与 Shared bank service 是不同资源，没有
统一物理端口和统一峰值。新实验不应使用这个字段计算所谓“L1 总利用率”。

### 8.2 `roofline_l2_total_bytes`

输出位于
[`roofline_stats.h:91`](../src/gpgpu-sim/roofline_stats.h#L91)，当前等于
`roofline_l2_data_port_total_bytes`。它不包含独立 fill-port 字节。新分析应直接使用
显式 data/fill 字段，避免误解。

### 8.3 `roofline_hbm_total_bytes`

输出位于
[`roofline_stats.h:92`](../src/gpgpu-sim/roofline_stats.h#L92)，当前与
`roofline_hbm_command_total_bytes` 相同，仅为旧名称兼容。

## 9. 与 L2 原生端口统计的交叉验证

multi-issue L2 模型还维护了一套 sector 数量和 stall cycle 统计，并在
[`gpu-sim.cc:2398-2428`](../src/gpgpu-sim/gpu-sim.cc#L2398) 输出：

```text
L2_multi_issue_data_port_read_hit_sectors
L2_multi_issue_data_port_write_hit_sectors
L2_multi_issue_data_port_miss_read_sectors
L2_multi_issue_data_port_fill_sectors
L2_multi_issue_data_port_dirty_eviction_sectors
L2_multi_issue_fill_port_accepted_sectors
```

它们应与 `roofline_*_bytes` 满足：

```text
对应 roofline bytes = 对应 accepted sectors × 32
```

同时可结合以下 stall 字段判断端口是否成为瓶颈：

```text
L2_multi_issue_lookup_width_stall_cycles
L2_multi_issue_data_port_width_stall_cycles
L2_multi_issue_fill_port_width_stall_cycles
```

字节计数回答“处理了多少流量”，stall cycle 回答“有多少周期因端口宽度不足而
阻塞”，二者用途不同。

## 10. 带宽计算

设日志中的计数器值为 `bytes`，kernel 总周期为 `gpu_tot_sim_cycle`，生效配置的
core clock 为 `core_GHz`，则十进制带宽为：

```text
simulated_seconds = gpu_tot_sim_cycle / (core_GHz × 1e9)
bandwidth_GBps = bytes × core_GHz / gpu_tot_sim_cycle
```

必须从该运行目录中的实际 `gpgpusim.config` 读取时钟，不能对所有配置写死
1.5 GHz。若总周期包含固定 kernel launch latency，应明确报告的是 end-to-end
带宽还是扣除 launch 后的 kernel-body 带宽，不能混用。

算子有效带宽采用算法必须完成的逻辑字节：

```text
useful_bandwidth = algorithmic_bytes / simulated_seconds
```

有效带宽与分层物理带宽是两种不同指标：前者评价算子完成效率，后者解释每层实际
承担的流量和事务放大。

## 11. 必须满足的会计恒等式

每份完整日志应满足：

```text
roofline_l1d_request_total_bytes
  = roofline_l1d_request_read_bytes
  + roofline_l1d_request_write_bytes

roofline_shared_bank_total_bytes
  = roofline_shared_bank_read_bytes
  + roofline_shared_bank_write_bytes

roofline_l2_data_hit_bytes
  = roofline_l2_data_read_hit_bytes
  + roofline_l2_data_write_hit_bytes

roofline_l2_data_port_total_bytes
  = roofline_l2_data_read_hit_bytes
  + roofline_l2_data_write_hit_bytes
  + roofline_l2_data_miss_read_bytes
  + roofline_l2_data_fill_bytes
  + roofline_l2_data_dirty_eviction_bytes

roofline_hbm_command_total_bytes
  = roofline_hbm_command_read_bytes
  + roofline_hbm_command_write_bytes
  + roofline_hbm_command_writeback_bytes
```

这些恒等式只能验证计数器内部求和一致，不能单独证明插桩覆盖了所用配置的所有
执行路径。还应结合受控 workload 验证预期事件，例如：

- coalesced copy：L1D read/write 应接近完整输入/输出请求流量；
- naive transpose：L1D/L2 写请求应显示非合并事务放大；
- padded shared transpose：L1D 写放大应恢复接近 1 倍，并出现 Shared read/write；
- cold streaming read：L2 read-hit 可为 0，但 fill-port 和 HBM read 应增加；
- 超过 L2 容量的输出：应出现 L2 dirty eviction 和 HBM writeback。

## 12. 当前适用范围与使用规则

对于普通 global load/store 和普通 shared load/store 算子，例如当前 naive/shared/
vector transpose 以及普通实现的 Softmax，这套计数器可以作为分层读写分析基础，
但必须满足：

1. 使用 `-gpgpu_l2_multi_issue_port_model 1`；
2. 使用详细 DRAM model，而不是 simple DRAM model；
3. 明确 `-gpgpu_l2_shared_data_array` 是 0 还是 1；
4. 每个测量点使用独立进程，或对计数器做区间 reset/snapshot；
5. 冷启动测试使用 `-gpgpu_perf_sim_memcpy 0`，避免 host copy 预热 L2；
6. 分别报告 L2 data port 和 fill port，不创建虚构的统一 L2 带宽；
7. HBM writeback 只表示测量区间内实际发出的命令，不等于完整输出；
8. 不把兼容字段 `roofline_l1_total_bytes` 用作统一 L1 Roofline 分母。

当前报告按研究计划暂不讨论 TMA 计数完整性。以后若引入 `cp.async`、`ldmatrix`、
global atomic、多 kernel pipeline、DSM remote shared 或双 GPU 模型，需要针对新增
路径重新确认插桩，并将进程级全局计数器改为至少按 GPU 实例区分，否则两个 GPU
的流量会被直接聚合，无法得到可靠的 per-GPU 带宽。

## 13. 双 GPU 版本移植检查表

将这套计数器移植到双 GPU + 多 HBM 版本时，应逐项确认：

1. 每个 `gpgpu_sim` 实例是否拥有独立计数器，而不是共享进程级全局原子变量；
2. 日志是否同时输出 GPU ID、per-GPU bytes 和 aggregate bytes；
3. L1D hook 是否仍位于实际 request acceptance 点；
4. Shared hook 是否仍只计一次物理 bank/word service；
5. L2 是否仍使用相同 multi-issue port model，新增 remote-L2 路径是否有独立统计；
6. 每块 HBM 的 read/write/writeback command 是否可以按 GPU、channel 或 HBM 实例
   汇总；
7. GPU 间互连是否增加 request、response、payload、stall 和 backpressure 计数；
8. per-GPU 周期、并发 makespan 和聚合带宽的分母是否定义清楚；
9. 所有理论峰值是否从修改版生效配置重新推导，而不是沿用 baseline 常数；
10. 使用 copy、cold read、L2-resident read、naive transpose 和 shared transpose
    重新验证上述会计关系。

只有完成这些检查，baseline 与双 GPU 修改版的分层带宽差异才具有同口径可比性。
