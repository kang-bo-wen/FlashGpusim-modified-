# L1D 持续读带宽

本目录只保存 L1D 测试与结果。默认固定 `528 blocks`，运行 64、128、256 次
hot-set 复用，通过相邻点差分估计稳态 L1D read-request 带宽。

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment
tutorials/bandwidth-benchmarks/l1d/run.sh
```

只重新分析已有日志：

```bash
tutorials/bandwidth-benchmarks/l1d/analyze.sh
```

有效结果位于 `runs-SM90_H100/`。旧网格与未完成的 refinement 记录已经清理，
不参与后续分析。
