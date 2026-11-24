# HBM 持续读带宽

本目录只保存 HBM streaming-read 测试与结果。默认固定 `1056 blocks`，运行
64、128、256、512 MiB 连续流，通过相邻点差分估计 HBM read-command 稳态带宽。

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
tutorials/bandwidth-benchmarks/hbm/run.sh
```

只重新分析已有日志：

```bash
tutorials/bandwidth-benchmarks/hbm/analyze.sh
```
