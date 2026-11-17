#!/usr/bin/env python3
"""Extract GEMM performance and hierarchical Roofline traffic counters."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

KV_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
DRAM_RE = re.compile(r"\bn_rd=(\d+)\b.*?\bn_write=(\d+)\b(?:.*?\bn_wr_bk=(\d+)\b)?")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?")

FIELDS = [
    "log", "variant", "m", "n", "k", "flops", "cycles",
    "core_clock_mhz", "simulated_gflops", "algorithmic_bytes",
    "algorithmic_intensity", "l1_accesses", "l1_misses",
    "l2_accesses", "l2_misses", "dram_reads", "dram_writes",
    "dram_writebacks", "l1_read_bytes", "l1_write_bytes", "l1_bytes",
    "l2_read_bytes", "l2_write_bytes", "l2_writeback_bytes", "l2_bytes",
    "hbm_read_bytes", "hbm_write_bytes", "hbm_writeback_bytes", "hbm_bytes",
    "l1_intensity", "l2_intensity", "hbm_intensity",
]

def number(text: str) -> float | None:
    match = NUMBER_RE.match(text)
    return float(match.group(0)) if match else None

def parse_clock(config_path: Path) -> float | None:
    if not config_path.exists():
        return None
    text = config_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^-gpgpu_clock_domains\s+([0-9.]+):", text, re.MULTILINE)
    return float(match.group(1)) if match else None

def parse_log(path: Path, cache_line: int, dram_burst: int) -> dict[str, object]:
    values: dict[str, float | str] = {}
    dram_reads = dram_writes = dram_writebacks = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = KV_RE.match(line)
        if match:
            key, raw = match.groups()
            parsed = number(raw)
            values[key] = parsed if parsed is not None else raw.strip()
        dram = DRAM_RE.search(line)
        if dram:
            dram_reads += int(dram.group(1))
            dram_writes += int(dram.group(2))
            dram_writebacks += int(dram.group(3) or 0)

    flops = float(values.get("algorithmic_flops", 0.0))
    cycles = float(values.get("gpu_tot_sim_cycle", values.get("gpu_sim_cycle", 0.0)))
    clock_mhz = parse_clock(path.parent / "gpgpusim.config")
    l1_accesses = float(values.get("L1D_total_cache_accesses", 0.0))
    l2_accesses = float(values.get("L2_total_cache_accesses", 0.0))
    l1_read_bytes = float(values.get("roofline_l1_read_bytes", 0.0))
    l1_write_bytes = float(values.get("roofline_l1_write_bytes", 0.0))
    l1_bytes = float(values.get("roofline_l1_total_bytes", l1_read_bytes + l1_write_bytes))
    l2_read_bytes = float(values.get("roofline_l2_read_bytes", 0.0))
    l2_write_bytes = float(values.get("roofline_l2_write_bytes", 0.0))
    l2_writeback_bytes = float(values.get("roofline_l2_writeback_bytes", 0.0))
    l2_bytes = float(values.get("roofline_l2_total_bytes", l2_read_bytes + l2_write_bytes + l2_writeback_bytes))
    hbm_read_bytes = float(values.get("roofline_hbm_read_bytes", 0.0))
    hbm_write_bytes = float(values.get("roofline_hbm_write_bytes", 0.0))
    hbm_writeback_bytes = float(values.get("roofline_hbm_writeback_bytes", 0.0))
    hbm_bytes = float(values.get("roofline_hbm_total_bytes", hbm_read_bytes + hbm_write_bytes + hbm_writeback_bytes))
    simulated_gflops = 0.0
    if cycles and clock_mhz:
        simulated_gflops = flops * clock_mhz * 1e6 / cycles / 1e9

    def intensity(byte_count: float) -> float:
        return flops / byte_count if byte_count else 0.0

    return {
        "log": str(path),
        "variant": values.get("kernel_variant", ""),
        "m": int(float(values.get("M", 0))),
        "n": int(float(values.get("N", 0))),
        "k": int(float(values.get("K", 0))),
        "flops": flops,
        "cycles": cycles,
        "core_clock_mhz": clock_mhz or "",
        "simulated_gflops": simulated_gflops,
        "algorithmic_bytes": values.get("algorithmic_bytes", 0.0),
        "algorithmic_intensity": values.get("algorithmic_intensity", 0.0),
        "l1_accesses": l1_accesses,
        "l1_misses": values.get("L1D_total_cache_misses", 0.0),
        "l2_accesses": l2_accesses,
        "l2_misses": values.get("L2_total_cache_misses", 0.0),
        "dram_reads": dram_reads,
        "dram_writes": dram_writes,
        "dram_writebacks": dram_writebacks,
        "l1_read_bytes": l1_read_bytes,
        "l1_write_bytes": l1_write_bytes,
        "l1_bytes": l1_bytes,
        "l2_read_bytes": l2_read_bytes,
        "l2_write_bytes": l2_write_bytes,
        "l2_writeback_bytes": l2_writeback_bytes,
        "l2_bytes": l2_bytes,
        "hbm_read_bytes": hbm_read_bytes,
        "hbm_write_bytes": hbm_write_bytes,
        "hbm_writeback_bytes": hbm_writeback_bytes,
        "hbm_bytes": hbm_bytes,
        "l1_intensity": intensity(l1_bytes),
        "l2_intensity": intensity(l2_bytes),
        "hbm_intensity": intensity(hbm_bytes),
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--cache-line", type=int, default=128)
    parser.add_argument("--dram-burst", type=int, default=32)
    args = parser.parse_args()
    rows = [parse_log(path, args.cache_line, args.dram_burst) for path in args.logs]
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
