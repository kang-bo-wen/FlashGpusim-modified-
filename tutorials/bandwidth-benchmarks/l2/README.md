# L2 持续读命中带宽

本目录只保存 L2 resident-set 测试与结果。默认固定 16 MiB 工作集和
`2112 blocks`，运行 8、16、32、64 次复用，通过相邻点差分估计 L2 data-port
read-hit 稳态带宽。工作集保持小于模型的 50 MiB L2 容量。

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
tutorials/bandwidth-benchmarks/l2/run.sh
```

只重新分析已有日志：

```bash
tutorials/bandwidth-benchmarks/l2/analyze.sh
```
