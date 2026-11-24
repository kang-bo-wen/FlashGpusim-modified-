# 矩阵转置分层带宽测试

本目录比较三种单精度、out-of-place 矩阵转置实现，并用已经测得的
SM90_H100 纯读稳态带宽作为分层参考线。这里不建立 Roofline model：矩阵转置
几乎没有有意义的浮点计算，用算术强度和算力屋顶描述它没有额外价值。

## 三种 kernel

三种实现都以 `32 × 32` tile 为基本工作单元，每个 block 使用 256 个线程：

1. `naive`：全局读合并，但转置后的全局写跨行，产生非合并写事务；
2. `shared`：先把 tile 写入 `float tile[32][33]`，再交换坐标后合并写回；额外的
   一列 padding 用来消除转置读取 shared memory 时的 32-way bank conflict；
3. `vector`：同样使用带 padding 的 shared tile，但全局读和全局写都显式使用
   `float4`，每个线程每次搬运连续 16 bytes。

默认测试 `128 × 128`、`256 × 256` 和 `512 × 512`。它们分别产生 16、64 和
256 个 CTA，用来观察从 SM 未铺满到足以覆盖 132 个 SM 的扩展过程，同时避免把
一次模拟扩展成小时级的极值搜索。`vector` 要求行列均为 32 的整数倍，以保证
所有 `float4` 地址对齐且没有边界标量分支。

## 运行

```bash
cd ~/workspace/FlashGPU-Sim
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
tutorials/transpose-benchmarks/run.sh
```

运行时每个 kernel/规模使用独立进程和独立配置目录，终端通过 `tee` 实时显示当前
仿真输出。已完成且包含 `Test PASSED` 的点会自动跳过，所以中断后直接执行同一
命令即可续跑。

可以缩小或扩大规模，也可以只跑一个实现：

```bash
SIZES="512 1024 2048" tutorials/transpose-benchmarks/run.sh
SIZES="1024" VARIANTS="shared vector" tutorials/transpose-benchmarks/run.sh
```

精确重现下文提交的 18 个正式点：

```bash
SIZES="128 256 512 1024 2048 4096" tutorials/transpose-benchmarks/run.sh
```

原始日志保存在 `runs-SM90_H100/`，该目录不会提交 Git。汇总结果和图表为：

```text
results/transpose-results.csv
results/bandwidth-comparison.svg
```

## 当前 SM90_H100 结果

正式结果额外补充了 `1024 × 1024`、`2048 × 2048` 和 `4096 × 4096`，共 18
个完整运行点。最后一档的输入和输出合计 128 MiB，已经超过约 50 MiB L2 容量。

| 规模 | 原生有效带宽 | Shared 有效带宽 | Shared + float4 | Vector / 原生 |
| --- | ---: | ---: | ---: | ---: |
| 128² | 21.043 GB/s | 22.311 GB/s | 28.111 GB/s | 1.34× |
| 256² | 78.706 GB/s | 86.412 GB/s | 103.573 GB/s | 1.32× |
| 512² | 214.799 GB/s | 310.843 GB/s | 351.125 GB/s | 1.63× |
| 1024² | 648.671 GB/s | 879.248 GB/s | 1,066.710 GB/s | 1.64× |
| 2048² | 972.855 GB/s | 1,634.304 GB/s | 2,002.931 GB/s | 2.06× |
| 4096² | 1,127.785 GB/s | 1,757.114 GB/s | 2,158.998 GB/s | 1.91× |

主要结论：

1. 原生实现的 L1D physical write request 始终是算法写字节的 8 倍；shared 和
   vector 都是 1 倍。这是 shared tile 带来主要收益的直接证据；
2. vector 与普通 shared 的全局物理读写字节相同，但用更少的全局访存指令完成
   搬运，因此 4096² 下仍比普通 shared 快 22.9%；
3. 4096² 的 HBM read-command 带宽分别达到纯读参考的 20.7%、32.3% 和 39.7%。
   最快版本也没有打满 HBM，不能把它简单归类为只受 HBM 峰值限制；
4. 所有点的 L2 read-hit 都为 0，符合冷启动单次流式读取没有时间复用的访问模式。
   输入数据通过 L2 fill port 返回，不应伪装成 read-hit 流量；
5. 4096² 首次出现约 38.5 MiB（40.4 MB）的 HBM dirty writeback，三个版本写回量接近。原生
   实现的 8 倍写放大主要发生在片上请求路径，到了 HBM 已被 cache line 合并；
6. 模拟器不会在 kernel 结束时强制把所有 dirty L2 line 写回 HBM，所以约
   38.5 MiB 不是完整的 64 MiB 输出。HBM total bytes 不能当作完整算法读写字节。

小规模结果包含固定 kernel launch latency，并且没有铺满所有 SM，主要用来展示
扩展趋势；判断稳定的大规模差异应优先看 2048² 和 4096²。

## 时间和带宽计算

脚本从每个运行点的生效 `gpgpusim.config` 读取 core clock，并使用：

```text
kernel 时间 = gpu_tot_sim_cycle / core_clock
带宽 (GB/s) = 计数器字节数 × core_clock_GHz / gpu_tot_sim_cycle
算子有效带宽 = (输入矩阵字节数 + 输出矩阵字节数) / kernel 时间
```

有效带宽表示完成一次转置所要求的 1 次读和 1 次写，能直接比较三个 kernel 的
完成效率。它不是某一物理端口的流量，不能直接除以纯读上限声称端口利用率。

## 分层参考线

图中使用带宽基准的最后一段稳态结果：

| 层级和事件 | 纯读实测参考 |
| --- | ---: |
| L1D accepted read request | 25,115.213 GB/s |
| L2 data-port read hit | 4,091.594 GB/s |
| HBM read column command | 2,720.905 GB/s |

三幅分层图严格使用与参考线相同的读事件：

```text
L1D: roofline_l1d_request_read_bytes
L2 : roofline_l2_data_read_hit_bytes
HBM: roofline_hbm_command_read_bytes
```

因此图上的百分比是同口径的“相对纯读实测上限”，不是把读写总流量硬除以纯读
峰值。`transpose-results.csv` 仍保存 L1D 写、shared 读写、L2 write-hit、L2 fill
以及 HBM 写和 writeback，便于分析非合并写放大和 shared-memory 开销。

## L2 结果的解释

一次转置对每个输入元素只读一次，没有时间复用。禁用性能模拟 memcpy 预热后，
输入读大多是冷 miss，因此 L2 read-hit 柱可能很低。这不表示数据绕过了 L2：
miss 返回的数据由独立的 L2 fill port 接受，数值记录在
`l2_fill_port_GBps`。不能把 read-hit 与 fill 两个可并行资源直接相加，再除以单个
read-hit 上限称为利用率。

如果后续专门研究提升 HBM 后的瓶颈迁移，应同时观察 L2 data port、L2 fill port、
片上互连和 LD/ST response retirement。当前图的目的只是比较三个 transpose 实现
在原始 SM90_H100 配置下的完成速度和各纯读观察点，而不是宣称一个统一的“L2
总带宽”。

## 可复现性约束

- 三种实现、所有规模使用同一个可执行文件和相同的随机输入生成方式；
- 每次运行只启动一个被测 kernel，最终逐元素验证转置结果；
- `-gpgpu_perf_sim_memcpy 0` 避免 H2D copy 预热 L2；
- L2 multi-issue 端口配置与带宽基准相同；
- 分析脚本从运行目录读取实际时钟，不写死 1.5 GHz；
- 参考上限按 `CONFIG_NAME` 选择，不允许其他配置静默套用 H100 结果；
- 每条柱都来自 `Test PASSED` 的完整日志，未完成运行不会进入 CSV。
