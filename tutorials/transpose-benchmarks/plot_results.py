#!/usr/bin/env python3
import csv
import math
import sys
from html import escape
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit('Usage: plot_results.py RESULTS.csv OUTPUT.svg')
input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
with input_path.open(newline='') as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit(f'No rows in {input_path}')

variant_order = ['naive', 'shared', 'vector']
variant_labels = {
    'naive': '原生',
    'shared': 'Shared',
    'vector': 'Shared + float4',
}
colors = {
    'naive': '#D1495B',
    'shared': '#2E86AB',
    'vector': '#2A9D8F',
}
sizes = sorted({(int(row['rows']), int(row['cols'])) for row in rows})
by_key = {(int(row['rows']), int(row['cols']), row['variant']): row
          for row in rows}

width, height = 1600, 1020
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
    f'viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<text x="800" y="42" text-anchor="middle" font-family="Noto Sans CJK SC" '
    'font-size="28" font-weight="700" fill="#17202a">矩阵转置：有效带宽与分层纯读带宽</text>',
    '<text x="800" y="70" text-anchor="middle" font-family="Noto Sans CJK SC" '
    'font-size="14" fill="#52616b">柱高来自物理读计数器；红色虚线是同一计数口径的纯读实测上限</text>',
]

legend_x = 515
for index, variant in enumerate(variant_order):
    x = legend_x + index * 210
    svg.append(f'<rect x="{x}" y="88" width="20" height="14" rx="2" '
               f'fill="{colors[variant]}"/>')
    svg.append(f'<text x="{x + 28}" y="100" font-family="Noto Sans CJK SC" '
               f'font-size="13" fill="#334155">{escape(variant_labels[variant])}</text>')


def nice_limit(value):
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    step = 1 if scaled <= 5 else 2
    return math.ceil(scaled / step) * step * magnitude


def draw_panel(x0, y0, panel_width, panel_height, title, field, percent_field=None,
               reference_field=None, subtitle=''):
    svg.append(f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" '
               'rx="6" fill="#f8fafc" stroke="#cbd5e1"/>')
    svg.append(f'<text x="{x0 + panel_width / 2}" y="{y0 + 29}" text-anchor="middle" '
               f'font-family="Noto Sans CJK SC" font-size="18" font-weight="700" '
               f'fill="#1f2937">{escape(title)}</text>')
    if subtitle:
        svg.append(f'<text x="{x0 + panel_width / 2}" y="{y0 + 49}" text-anchor="middle" '
                   f'font-family="Noto Sans CJK SC" font-size="11" '
                   f'fill="#64748b">{escape(subtitle)}</text>')

    left, right, top, bottom = 70, 18, 65, 66
    plot_x = x0 + left
    plot_y = y0 + top
    plot_width = panel_width - left - right
    plot_height = panel_height - top - bottom
    values = [float(row[percent_field] if percent_field else row[field])
              for row in rows]
    if percent_field:
        y_limit = max(110.0, math.ceil(max(values) * 1.15 / 10.0) * 10.0)
        y_label = '相对纯读实测上限 (%)'
    else:
        y_limit = nice_limit(max(values) * 1.18)
        y_label = 'GB/s'

    for tick in range(6):
        value = y_limit * tick / 5
        y = plot_y + plot_height - plot_height * tick / 5
        svg.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_width}" '
                   f'y2="{y:.1f}" stroke="#dbe3ea" stroke-dasharray="3 4"/>')
        label = f'{value:.0f}'
        svg.append(f'<text x="{plot_x - 8}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-family="Noto Sans CJK SC" font-size="10" '
                   f'fill="#64748b">{label}</text>')
    svg.append(f'<text x="{x0 + 17}" y="{plot_y + plot_height / 2}" '
               f'text-anchor="middle" font-family="Noto Sans CJK SC" font-size="11" '
               f'fill="#52616b" transform="rotate(-90 {x0 + 17} {plot_y + plot_height / 2})">'
               f'{escape(y_label)}</text>')

    if percent_field:
        reference_y = plot_y + plot_height * (1.0 - 100.0 / y_limit)
        first = rows[0]
        reference = float(first[reference_field])
        svg.append(f'<line x1="{plot_x}" y1="{reference_y:.1f}" '
                   f'x2="{plot_x + plot_width}" y2="{reference_y:.1f}" '
                   'stroke="#c00000" stroke-width="2" stroke-dasharray="9 6"/>')
        svg.append(f'<text x="{plot_x + plot_width - 4}" y="{reference_y - 6:.1f}" '
                   f'text-anchor="end" font-family="Noto Sans CJK SC" font-size="10" '
                   f'font-weight="700" fill="#c00000">纯读上限 {reference / 1000:.3f} TB/s</text>')

    group_width = plot_width / len(sizes)
    bar_width = min(42.0, group_width / 5.0)
    for size_index, size in enumerate(sizes):
        center = plot_x + (size_index + 0.5) * group_width
        start = center - 1.5 * bar_width - 6
        for variant_index, variant in enumerate(variant_order):
            row = by_key.get((size[0], size[1], variant))
            if row is None:
                continue
            actual = float(row[field])
            plotted = float(row[percent_field]) if percent_field else actual
            bar_height = min(plot_height, plot_height * plotted / y_limit)
            x = start + variant_index * (bar_width + 6)
            y = plot_y + plot_height - bar_height
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                       f'height="{bar_height:.1f}" rx="2" fill="{colors[variant]}"/>')
            if percent_field:
                label = f'{actual / 1000:.2f}T'
                detail = f'{plotted:.1f}%'
            else:
                label = f'{actual:.0f}'
                detail = ''
            text_y = max(plot_y + 12, y - 5)
            svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{text_y:.1f}" '
                       f'text-anchor="middle" font-family="Noto Sans CJK SC" '
                       f'font-size="9" font-weight="700" fill="#334155">{label}</text>')
            if detail:
                svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{text_y + 11:.1f}" '
                           f'text-anchor="middle" font-family="Noto Sans CJK SC" '
                           f'font-size="8" fill="#64748b">{detail}</text>')
        size_label = f'{size[0]}×{size[1]}'
        svg.append(f'<text x="{center:.1f}" y="{plot_y + plot_height + 24}" '
                   f'text-anchor="middle" font-family="Noto Sans CJK SC" '
                   f'font-size="11" fill="#334155">{size_label}</text>')


draw_panel(55, 125, 735, 390, '算子有效带宽', 'useful_bandwidth_GBps',
           subtitle='2 × 矩阵字节数 / kernel 时间；不与纯读上限计算利用率')
draw_panel(810, 125, 735, 390, 'L1D request 纯读带宽', 'l1d_read_GBps',
           'l1d_read_reference_pct', 'l1d_read_reference_GBps',
           '读请求接受字节 / kernel 时间')
draw_panel(55, 535, 735, 390, 'L2 data-port read-hit 带宽', 'l2_read_hit_GBps',
           'l2_read_hit_reference_pct', 'l2_read_hit_reference_GBps',
           '只统计 L2 读命中；冷 miss 的 fill 单列在 CSV 中')
draw_panel(810, 535, 735, 390, 'HBM read-command 带宽', 'hbm_read_GBps',
           'hbm_read_reference_pct', 'hbm_read_reference_GBps',
           'DRAM 已发出的读 column command 字节 / kernel 时间')

svg.append('<text x="800" y="966" text-anchor="middle" '
           'font-family="Noto Sans CJK SC" font-size="12" fill="#52616b">'
           '说明：三条参考线均来自 SM90_H100 纯读稳态测试；写流量、shared 流量和 L2 fill 保留在 CSV 中，不混入纯读利用率。</text>')
svg.append('</svg>')
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text('\n'.join(svg), encoding='utf-8')
print(output_path)
