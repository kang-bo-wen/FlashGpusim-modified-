#!/usr/bin/env python3
from pathlib import Path
import csv

root = Path(__file__).resolve().parent
output = root / 'transpose_bandwidth_rigorous_r8192_c8192.svg'
result_csv = root / 'transpose_bandwidth_raw_r8192_c8192.csv'
peak_csv = root.parent / 'bandwidth-benchmarks' / 'measured_peaks.csv'

core_ghz = 1.5
algorithmic_bytes = 536870912
# Structural reference ceilings for SM90_H100.
l1_reference_gbs = 25344.0
l2_theoretical_gbs = 8704.0
hbm_theoretical_gbs = 3349.76

peak_rows = {row['level']: row for row in csv.DictReader(peak_csv.open())}
l1_measured_gbs = float(peak_rows['L1D + Shared']['measured_GBps'])
l2_measured_gbs = float(peak_rows['L2']['measured_GBps'])
hbm_measured_gbs = float(peak_rows['HBM']['measured_GBps'])

colors = {
    'V0 Float4 Copy': '#2E86AB',
    'V1 Naive Transpose': '#D1495B',
    'V2 Tiled Transpose': '#F4A261',
    'V3 Float4 Tiled': '#2A9D8F',
}
raw = []
for row in csv.DictReader(result_csv.open()):
    raw.append((
        row['kernel'], int(row['cycles']), int(row['l1_bytes']),
        int(row['l2_bytes']), int(row['hbm_bytes']), colors[row['kernel']]
    ))

def bw(byte_count, cycles):
    return byte_count * core_ghz / cycles

rows = []
for name, cycles, l1_bytes, l2_bytes, hbm_bytes, color in raw:
    effective = bw(algorithmic_bytes, cycles)
    l1 = bw(l1_bytes, cycles)
    l2 = bw(l2_bytes, cycles)
    hbm = bw(hbm_bytes, cycles)
    rows.append({
        'kernel': name, 'cycles': cycles, 'color': color,
        'effective_gbs': effective, 'l1_gbs': l1, 'l2_gbs': l2, 'hbm_gbs': hbm,
        'effective_hbm_pct': effective / hbm_theoretical_gbs * 100,
        'l1_pct': l1 / l1_reference_gbs * 100,
        'l2_pct': l2 / l2_theoretical_gbs * 100,
        'hbm_pct': hbm / hbm_theoretical_gbs * 100,
    })

W, H = 1540, 920
left, right, top, bottom = 90, 35, 105, 145
gap = 28
panel_w = (W - left - right - 3 * gap) / 4
plot_top = 190
plot_bottom = H - bottom
plot_h = plot_bottom - plot_top

panels = [
    ('Useful Effective BW', hbm_theoretical_gbs, hbm_measured_gbs, 'effective_gbs', 'effective_hbm_pct', 'Complete 512 MiB / kernel time', 'Red dashed line: measured sustainable HBM peak'),
    ('L1D + Shared Physical', l1_reference_gbs, l1_measured_gbs, 'l1_gbs', 'l1_pct', 'Measured physical bytes / kernel time', 'Reference ceiling, not strict unified peak'),
    ('L2 Physical', l2_theoretical_gbs, l2_measured_gbs, 'l2_gbs', 'l2_pct', 'Measured L2 bytes / kernel time', 'Red dashed line: measured sustainable L2 peak'),
    ('HBM Physical', hbm_theoretical_gbs, hbm_measured_gbs, 'hbm_gbs', 'hbm_pct', 'Measured DRAM-queue bytes / kernel time', 'Dirty output remaining in L2 excluded'),
]

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<rect width="100%" height="100%" fill="white"/>',
    f'<text x="{W/2}" y="42" text-anchor="middle" font-family="Arial" font-size="27" font-weight="bold" fill="#111827">8192×8192 Transpose Bandwidth — Rigorous Hierarchy Comparison</text>',
    f'<text x="{W/2}" y="72" text-anchor="middle" font-family="Arial" font-size="14" fill="#475569">Useful bandwidth and physical hierarchy bandwidth are separated; all rates use the simulated 1.5 GHz kernel time.</text>',
]

for p, (title, ceiling, measured_peak, value_key, pct_key, subtitle, note) in enumerate(panels):
    x0 = left + p * (panel_w + gap)
    x1 = x0 + panel_w
    svg.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{panel_w:.1f}" height="{plot_bottom-top+55:.1f}" rx="9" fill="#f8fafc" stroke="#cbd5e1"/>')
    svg.append(f'<text x="{(x0+x1)/2:.1f}" y="{top+28}" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold" fill="#111827">{title}</text>')
    svg.append(f'<text x="{(x0+x1)/2:.1f}" y="{top+49}" text-anchor="middle" font-family="Arial" font-size="11" fill="#475569">{subtitle}</text>')
    svg.append(f'<text x="{(x0+x1)/2:.1f}" y="{top+67}" text-anchor="middle" font-family="Arial" font-size="11" fill="#64748b">Ceiling/reference: {ceiling/1000:.3f} TB/s</text>')
    for tick in range(0, 101, 20):
        y = plot_bottom - plot_h * tick / 100
        svg.append(f'<line x1="{x0+42:.1f}" y1="{y:.1f}" x2="{x1-12:.1f}" y2="{y:.1f}" stroke="#d9e2ec" stroke-dasharray="3 4"/>')
        svg.append(f'<text x="{x0+36:.1f}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10" fill="#64748b">{tick}%</text>')
    measured_pct = measured_peak / ceiling * 100
    measured_y = plot_bottom - plot_h * measured_pct / 100
    svg.append(f'<line x1="{x0+42:.1f}" y1="{measured_y:.1f}" x2="{x1-12:.1f}" y2="{measured_y:.1f}" stroke="#C00000" stroke-width="2.5" stroke-dasharray="9 6"/>')
    svg.append(f'<text x="{x1-14:.1f}" y="{measured_y-7:.1f}" text-anchor="end" font-family="Arial" font-size="10" font-weight="bold" fill="#C00000">Measured {measured_peak/1000:.3f} TB/s</text>')
    usable_w = panel_w - 62
    slot = usable_w / 4
    for i, row in enumerate(rows):
        pct = row[pct_key]
        val = row[value_key]
        bx = x0 + 48 + i * slot
        bar_w = slot * 0.62
        by = plot_bottom - plot_h * min(pct, 100) / 100
        svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{plot_bottom-by:.1f}" rx="3" fill="{row["color"]}"/>')
        svg.append(f'<text x="{bx+bar_w/2:.1f}" y="{max(plot_top+10,by-23):.1f}" text-anchor="middle" font-family="Arial" font-size="10" font-weight="bold" fill="#1f2937">{val/1000:.3f}</text>')
        svg.append(f'<text x="{bx+bar_w/2:.1f}" y="{max(plot_top+22,by-9):.1f}" text-anchor="middle" font-family="Arial" font-size="9" fill="#475569">{pct:.1f}%</text>')
        short = row['kernel'].replace(' Transpose','').replace('Float4','F4')
        svg.append(f'<text x="{bx+bar_w/2:.1f}" y="{plot_bottom+18:.1f}" text-anchor="middle" font-family="Arial" font-size="9" fill="#334155" transform="rotate(35 {bx+bar_w/2:.1f} {plot_bottom+18:.1f})">{short}</text>')
    svg.append(f'<text x="{(x0+x1)/2:.1f}" y="{plot_bottom+92:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#64748b">{note}</text>')

svg.append(f'<text x="{W/2}" y="{H-25}" text-anchor="middle" font-family="Arial" font-size="12" fill="#475569">TB/s labels are decimal. V1 L1/L2 physical bandwidth includes the 8× non-coalesced write amplification; V2/V3 L1D+Shared includes shared-memory transactions.</text>')
svg.append('</svg>')
output.write_text('\n'.join(svg))
print(output)
