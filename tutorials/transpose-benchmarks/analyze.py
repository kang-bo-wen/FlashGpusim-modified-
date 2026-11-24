#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

VARIANT_ORDER = {'naive': 0, 'shared': 1, 'vector': 2}
COUNTERS = [
    'roofline_l1d_request_read_bytes',
    'roofline_l1d_request_write_bytes',
    'roofline_l1d_request_total_bytes',
    'roofline_shared_bank_read_bytes',
    'roofline_shared_bank_write_bytes',
    'roofline_shared_bank_total_bytes',
    'roofline_l2_data_read_hit_bytes',
    'roofline_l2_data_write_hit_bytes',
    'roofline_l2_data_miss_read_bytes',
    'roofline_l2_data_fill_bytes',
    'roofline_l2_data_dirty_eviction_bytes',
    'roofline_l2_data_port_total_bytes',
    'roofline_l2_fill_port_total_bytes',
    'roofline_hbm_command_read_bytes',
    'roofline_hbm_command_write_bytes',
    'roofline_hbm_command_writeback_bytes',
    'roofline_hbm_command_total_bytes',
]


def last_integer(text, key):
    values = re.findall(rf'^\s*{re.escape(key)}\s*=\s*([0-9]+)', text, re.M)
    if not values:
        raise ValueError(f'missing integer field: {key}')
    return int(values[-1])


def last_string(text, key):
    values = re.findall(rf'^\s*{re.escape(key)}\s*=\s*([^\s]+)', text, re.M)
    if not values:
        raise ValueError(f'missing string field: {key}')
    return values[-1]


def config_value(text, key):
    values = re.findall(rf'^-{re.escape(key)}\s+([^#\s]+)', text, re.M)
    if not values:
        raise ValueError(f'missing config field: {key}')
    return values[-1]


def measured_reference(csv_path):
    with csv_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'empty reference file: {csv_path}')
    return float(rows[-1]['steady_slope_GBps'])


parser = argparse.ArgumentParser(
    description='Parse matrix-transpose hierarchy traffic and bandwidth')
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reference-config', default='SM90_H100')
args = parser.parse_args()

benchmark_root = Path(__file__).resolve().parents[1] / 'bandwidth-benchmarks'
references = {
    'l1d': measured_reference(
        benchmark_root / f'l1d/runs-{args.reference_config}/steady-state.csv'),
    'l2': measured_reference(
        benchmark_root / f'l2/runs-{args.reference_config}/steady-state.csv'),
    'hbm': measured_reference(
        benchmark_root / f'hbm/runs-{args.reference_config}/steady-state.csv'),
}

rows = []
for log_path in sorted(args.run_dir.glob('*/simulation.log')):
    text = log_path.read_text(errors='replace')
    if 'Test PASSED' not in text:
        print(f'[skip incomplete] {log_path.parent.name}')
        continue
    config_path = log_path.parent / 'gpgpusim.config'
    config = config_path.read_text(errors='replace')
    core_ghz = float(config_value(config, 'gpgpu_clock_domains').split(':')[0]) / 1000.0
    cycles = last_integer(text, 'gpu_tot_sim_cycle')
    if cycles <= 0:
        raise ValueError(f'non-positive cycle count in {log_path}')
    variant = last_string(text, 'kernel_variant')
    matrix_rows = last_integer(text, 'rows')
    matrix_cols = last_integer(text, 'cols')
    algorithmic_read = last_integer(text, 'algorithmic_read_bytes')
    algorithmic_write = last_integer(text, 'algorithmic_write_bytes')
    algorithmic_total = last_integer(text, 'algorithmic_total_bytes')
    counters = {name: last_integer(text, name) for name in COUNTERS}

    def gbps(byte_count):
        return byte_count * core_ghz / cycles

    row = {
        'variant': variant,
        'rows': matrix_rows,
        'cols': matrix_cols,
        'cycles': cycles,
        'core_GHz': f'{core_ghz:.6f}',
        'algorithmic_read_bytes': algorithmic_read,
        'algorithmic_write_bytes': algorithmic_write,
        'algorithmic_total_bytes': algorithmic_total,
        'useful_bandwidth_GBps': f'{gbps(algorithmic_total):.6f}',
    }
    row.update(counters)
    bandwidth_fields = {
        'l1d_read_GBps': counters['roofline_l1d_request_read_bytes'],
        'l1d_write_GBps': counters['roofline_l1d_request_write_bytes'],
        'l1d_total_GBps': counters['roofline_l1d_request_total_bytes'],
        'shared_read_GBps': counters['roofline_shared_bank_read_bytes'],
        'shared_write_GBps': counters['roofline_shared_bank_write_bytes'],
        'shared_total_GBps': counters['roofline_shared_bank_total_bytes'],
        'l2_read_hit_GBps': counters['roofline_l2_data_read_hit_bytes'],
        'l2_write_hit_GBps': counters['roofline_l2_data_write_hit_bytes'],
        'l2_data_port_total_GBps': counters['roofline_l2_data_port_total_bytes'],
        'l2_fill_port_GBps': counters['roofline_l2_fill_port_total_bytes'],
        'hbm_read_GBps': counters['roofline_hbm_command_read_bytes'],
        'hbm_write_GBps': counters['roofline_hbm_command_write_bytes'],
        'hbm_writeback_GBps': counters['roofline_hbm_command_writeback_bytes'],
        'hbm_total_GBps': counters['roofline_hbm_command_total_bytes'],
    }
    row.update({name: f'{gbps(value):.6f}'
                for name, value in bandwidth_fields.items()})
    row.update({
        'l1d_read_reference_GBps': f'{references["l1d"]:.6f}',
        'l1d_read_reference_pct':
            f'{gbps(counters["roofline_l1d_request_read_bytes"]) / references["l1d"] * 100:.3f}',
        'l2_read_hit_reference_GBps': f'{references["l2"]:.6f}',
        'l2_read_hit_reference_pct':
            f'{gbps(counters["roofline_l2_data_read_hit_bytes"]) / references["l2"] * 100:.3f}',
        'hbm_read_reference_GBps': f'{references["hbm"]:.6f}',
        'hbm_read_reference_pct':
            f'{gbps(counters["roofline_hbm_command_read_bytes"]) / references["hbm"] * 100:.3f}',
    })
    rows.append(row)

if not rows:
    raise SystemExit(f'no complete runs found under {args.run_dir}')
rows.sort(key=lambda row: (row['rows'], row['cols'],
                           VARIANT_ORDER.get(row['variant'], 99)))
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()),
                            lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

print(f'parsed {len(rows)} complete runs')
for row in rows:
    print(f"{row['rows']}x{row['cols']} {row['variant']:>6}: "
          f"cycles={row['cycles']}, useful={row['useful_bandwidth_GBps']} GB/s, "
          f"L1D/L2/HBM read-ref={row['l1d_read_reference_pct']}%/"
          f"{row['l2_read_hit_reference_pct']}%/{row['hbm_read_reference_pct']}%")
print(args.output)
