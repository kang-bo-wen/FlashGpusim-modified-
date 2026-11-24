#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

def last(text, key, default=None):
    values = re.findall(rf'^-{re.escape(key)}\s+([^#\s]+)', text, re.M)
    return values[-1] if values else default

p = argparse.ArgumentParser(description='从生效配置推导 FlashGPU-Sim 模型带宽上限')
p.add_argument('config', type=Path)
p.add_argument('--csv', type=Path)
a = p.parse_args(); text = a.config.read_text()
core, _, l2, dram = [float(x) / 1000 for x in last(text, 'gpgpu_clock_domains').split(':')]
sms = int(last(text, 'gpgpu_n_clusters')) * int(last(text, 'gpgpu_n_cores_per_cluster'))
sector = 32; l1_banks = int(last(text, 'gpgpu_l1_banks'))
channels = int(last(text, 'gpgpu_n_mem'))
subparts = int(last(text, 'gpgpu_n_sub_partition_per_mchannel'))
l2_width = int(last(text, 'gpgpu_l2_data_port_sectors_per_cycle', 0))
l2_period = int(last(text, 'gpgpu_l2_data_port_cycle_period', 1))
ldst_width = int(last(text, 'gpgpu_ldst_request_width', 0))
mem_unit_ports = int(last(text, 'gpgpu_mem_unit_ports', 1))
fill_width = int(last(text, 'gpgpu_l2_fill_port_sectors_per_cycle', 0))
bl = int(last(text, 'gpgpu_dram_burst_length')); bus = int(last(text, 'gpgpu_dram_buswidth'))
chips = int(last(text, 'gpgpu_n_mem_per_ctrlr')); atom = bl * bus * chips
timing = last(text, 'gpgpu_dram_timing_opt')
tccd = int(re.search(r'(?:^|:)CCD=([0-9]+)', timing).group(1))
shared_budget = int(last(text, 'gpgpu_shmem_bytes_per_cycle', 0))
l2_internal = channels*subparts*l2_width/l2_period*sector*l2
l2_sm_request = sms*(ldst_width if ldst_width else mem_unit_ports)*sector*core
rows = [
 {'level':'L1D request','GBps':sms*l1_banks*sector*core,'formula':f'{sms} SM × {l1_banks} bank × {sector} B × {core} GHz'},
 {'level':'Shared bank','GBps':'' if shared_budget == 0 else sms*shared_budget*core,'formula':'budget=0 表示模型无限，无法定义有限峰值' if shared_budget == 0 else f'{sms} SM × {shared_budget} B/tick × {core} GHz'},
 {'level':'L2 data port','GBps':l2_internal,'formula':f'{channels*subparts} slice × {l2_width}/{l2_period} sector/tick × {sector} B × {l2} GHz'},
 {'level':'L2 kernel path','GBps':min(l2_internal, l2_sm_request),'formula':f'min(L2 internal {l2_internal:.3f}, SM request {sms} × {ldst_width if ldst_width else mem_unit_ports} sector/tick × {sector} B × {core} GHz = {l2_sm_request:.3f})'},
 {'level':'L2 fill port','GBps':channels*subparts*fill_width*sector*l2,'formula':f'{channels*subparts} slice × {fill_width} sector/tick × {sector} B × {l2} GHz'},
 {'level':'HBM commands','GBps':channels*atom/tccd*dram,'formula':f'{channels} channel × {atom} B/command ÷ tCCD {tccd} × {dram} GHz'},
]
for row in rows:
    print(f"{row['level']}: {row['GBps'] if row['GBps'] != '' else '无有限值'} GB/s — {row['formula']}")
if a.csv:
    with a.csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['level','GBps','formula'],
                                lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
