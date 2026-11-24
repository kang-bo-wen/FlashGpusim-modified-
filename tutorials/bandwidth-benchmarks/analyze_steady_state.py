#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

LEVELS = {
    'l1d': ('L1D request', 'roofline_l1d_request_read_bytes', 'i'),
    'l2': ('L2 data port', 'roofline_l2_data_read_hit_bytes', 'i'),
    'hbm': ('HBM commands', 'roofline_hbm_command_read_bytes', 'e'),
}
PRIMARY_THEORY = {'l1d': 'L1D request', 'l2': 'L2 kernel path', 'hbm': 'HBM commands'}

def integer(text, key):
    values = re.findall(rf'^\s*{re.escape(key)}\s*=\s*([0-9]+)', text, re.M)
    return int(values[-1]) if values else None

def parameter(name, key):
    match = re.search(rf'(?:^|-){key}([0-9]+)(?:-|$)', name)
    return int(match.group(1)) if match else None

def legacy_l2_hit_bytes(text):
    access = 'GLOBAL_ACC_R'
    values = re.findall(
        rf'L2_cache_stats_breakdown\[{access}\]\[HIT\] = ([0-9]+)', text)
    return int(values[-1]) * 32 if values else None

def config_value(text, key):
    values = re.findall(rf'^-{re.escape(key)}\s+([^#\s]+)', text, re.M)
    return values[-1] if values else None

p = argparse.ArgumentParser(description='分析单个存储层级的三点/多点差分稳态读带宽')
p.add_argument('--level', choices=LEVELS, required=True)
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--theoretical', type=Path, required=True)
p.add_argument('--csv', type=Path, required=True)
p.add_argument('--stable-pct', type=float, default=3.0)
a = p.parse_args()
label, byte_key, work_key = LEVELS[a.level]
configs = sorted(a.run_dir.glob('*/gpgpusim.config'))
if not configs:
    raise SystemExit(f'{a.run_dir} 中没有生效的 gpgpusim.config')
clock_domains = config_value(configs[0].read_text(errors='replace'),
                             'gpgpu_clock_domains')
if clock_domains is None:
    raise SystemExit(f'{configs[0]} 中缺少 -gpgpu_clock_domains')
core_ghz = float(clock_domains.split(':')[0]) / 1000.0
with a.theoretical.open() as f:
    theory = {row['level']: float(row['GBps']) for row in csv.DictReader(f) if row['GBps']}
theoretical = theory[PRIMARY_THEORY[a.level]]
internal_theoretical = theory[label]
points = []
for log in a.run_dir.glob('*/simulation.log'):
    text = log.read_text(errors='replace')
    if 'Test PASSED' not in text:
        continue
    work = parameter(log.parent.name, work_key)
    byte_count = integer(text, byte_key)
    if byte_count is None and a.level == 'l2':
        byte_count = legacy_l2_hit_bytes(text)
    cycles = integer(text, 'gpu_tot_sim_cycle')
    if work is not None and byte_count is not None and cycles:
        points.append((work, byte_count, cycles, log.parent.name))
points.sort()
if len(points) < 3:
    raise SystemExit(f'{a.run_dir} 只有 {len(points)} 个完整点，至少需要 3 个')
slopes = []
for left, right in zip(points, points[1:]):
    delta_bytes = right[1] - left[1]
    delta_cycles = right[2] - left[2]
    if delta_bytes <= 0 or delta_cycles <= 0:
        raise SystemExit(f'点 {left[3]} -> {right[3]} 不满足单调增量')
    gbps = delta_bytes * core_ghz / delta_cycles
    slopes.append((left[3], right[3], delta_bytes, delta_cycles, gbps))
last_two = [row[4] for row in slopes[-2:]]
spread = abs(last_two[1] - last_two[0]) / max(last_two) * 100.0
stable = spread <= a.stable_pct
with a.csv.open('w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['interval', 'start_point', 'end_point', 'delta_bytes',
                     'delta_cycles', 'steady_slope_GBps', 'kernel_visible_theoretical_GBps',
                     'kernel_visible_utilization_pct', 'internal_port_theoretical_GBps',
                     'internal_port_utilization_pct'])
    for start, end, delta_bytes, delta_cycles, gbps in slopes:
        writer.writerow([f'{start} -> {end}', start, end, delta_bytes, delta_cycles,
                         f'{gbps:.6f}', f'{theoretical:.6f}',
                         f'{gbps / theoretical * 100.0:.3f}',
                         f'{internal_theoretical:.6f}',
                         f'{gbps / internal_theoretical * 100.0:.3f}'])
print(f'{label} steady read: {slopes[-1][4]:.3f} GB/s，kernel-visible 利用率 {slopes[-1][4] / theoretical * 100.0:.3f}%')
if internal_theoretical != theoretical:
    print(f'  L2 internal-port 利用率 {slopes[-1][4] / internal_theoretical * 100.0:.3f}%（结构上限 {internal_theoretical:.3f} GB/s）')
print(f'最后两段差异 {spread:.3f}%：{"稳定" if stable else "需要增加一个更大工作量点"}')
print(a.csv)
