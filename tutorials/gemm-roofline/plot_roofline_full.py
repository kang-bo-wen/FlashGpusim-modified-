#!/usr/bin/env python3
import argparse, csv, html, math
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Generate a hierarchical GEMM Roofline SVG.')
parser.add_argument('--input', type=Path, default=root / 'physical_results_256.csv')
parser.add_argument('--output', type=Path, default=root / 'hierarchical_roofline_256.svg')
parser.add_argument('--title', default='Hierarchical Roofline — SM90 H100 GEMM 256³ (Cold L2)')
args = parser.parse_args()
rows = list(csv.DictReader(args.input.open()))

# Configuration-derived ceilings for SM90_H100.
CORE_GHZ = 1.5
L2_GHZ = 1.7
DRAM_GHZ = 2.617
SM_COUNT = 132
FP32_LANES_PER_SM = 128
L1_BANKS_PER_SM = 32
L2_SUBPARTITIONS = 80 * 2
HBM_CHANNELS = 80
HBM_BUS_BYTES = 8

cuda_compute_gflops = SM_COUNT * FP32_LANES_PER_SM * 2 * CORE_GHZ
# Contextual dense TF32 Tensor-Core model ceiling. The plotted SGEMM kernels
# use ordinary FP32 CUDA cores, so this line is architectural context only.
tensor_compute_gflops = SM_COUNT * 2048 * CORE_GHZ
l1_gbs = SM_COUNT * L1_BANKS_PER_SM * 4 * CORE_GHZ
l2_gbs = L2_SUBPARTITIONS * 32 * L2_GHZ
hbm_gbs = HBM_CHANNELS * HBM_BUS_BYTES * 2 * DRAM_GHZ

W, H = 1280, 820
L, R, T, B = 115, 55, 75, 100
xmin = 0.06
max_ai = max(float(row[key]) for row in rows for key in ('l1_intensity', 'l2_intensity', 'hbm_intensity'))
xmax = 100.0 if max_ai <= 90 else 200.0 if max_ai <= 180 else 500.0 if max_ai <= 450 else 1000.0
ymin, ymax = 50.0, 500000.0

def sx(v):
    return L + (math.log10(v)-math.log10(xmin))/(math.log10(xmax)-math.log10(xmin))*(W-L-R)
def sy(v):
    return H-B-(math.log10(v)-math.log10(ymin))/(math.log10(ymax)-math.log10(ymin))*(H-T-B)
def roof(ai, bw):
    # Extend hierarchy bandwidth roofs through the CUDA-Core ceiling and cap
    # them only at the contextual Tensor-Core ceiling.
    return min(tensor_compute_gflops, ai*bw)

svg=[f'<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\" viewBox=\"0 0 {W} {H}\">', '<rect width=\"100%\" height=\"100%\" fill=\"white\"/>']
for x in [0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]:
    if x > xmax: continue
    px=sx(x); svg.append(f'<line x1=\"{px:.1f}\" y1=\"{T}\" x2=\"{px:.1f}\" y2=\"{H-B}\" stroke=\"#e2e2e2\" stroke-dasharray=\"3 4\"/>'); svg.append(f'<text x=\"{px:.1f}\" y=\"{H-B+25}\" text-anchor=\"middle\" font-size=\"13\">{x:g}</text>')
for y in [50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000]:
    py=sy(y); svg.append(f'<line x1=\"{L}\" y1=\"{py:.1f}\" x2=\"{W-R}\" y2=\"{py:.1f}\" stroke=\"#e2e2e2\" stroke-dasharray=\"3 4\"/>'); svg.append(f'<text x=\"{L-12}\" y=\"{py+5:.1f}\" text-anchor=\"end\" font-size=\"13\">{y:g}</text>')
svg += [f'<line x1=\"{L}\" y1=\"{T}\" x2=\"{L}\" y2=\"{H-B}\" stroke=\"black\"/>',f'<line x1=\"{L}\" y1=\"{H-B}\" x2=\"{W-R}\" y2=\"{H-B}\" stroke=\"black\"/>',f'<text x=\"{W/2}\" y=\"34\" text-anchor=\"middle\" font-size=\"23\" font-weight=\"bold\">{html.escape(args.title)}</text>',f'<text x=\"{W/2}\" y=\"{H-30}\" text-anchor=\"middle\" font-size=\"16\">Arithmetic Intensity (FLOP / Byte, log scale)</text>',f'<text x=\"28\" y=\"{H/2}\" text-anchor=\"middle\" font-size=\"16\" transform=\"rotate(-90 28 {H/2})\">Performance (GFLOP/s, log scale)</text>']

ceilings=[('L1D + Shared reference',l1_gbs,'#1f77b4'),('L2',l2_gbs,'#ff7f0e'),('HBM',hbm_gbs,'#2ca02c')]
for name,bw,color in ceilings:
    pts=[]
    for i in range(161):
        x=10**(math.log10(xmin)+(math.log10(xmax)-math.log10(xmin))*i/160)
        pts.append(f'{sx(x):.1f},{sy(roof(x,bw)):.1f}')
    svg.append(f'<polyline points=\"{" ".join(pts)}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2.3\"/>')
    label_x=max(xmin*1.5,min(xmax/3,tensor_compute_gflops/bw/2))
    svg.append(f'<text x=\"{sx(label_x):.1f}\" y=\"{sy(roof(label_x,bw))-8:.1f}\" fill=\"{color}\" font-size=\"13\" font-weight=\"bold\">{name} {bw/1000:.2f} TB/s</text>')
py=sy(cuda_compute_gflops); svg.append(f'<line x1=\"{L}\" y1=\"{py:.1f}\" x2=\"{W-R}\" y2=\"{py:.1f}\" stroke=\"#b2182b\" stroke-width=\"2.5\"/>'); svg.append(f'<text x=\"{W-R-8}\" y=\"{py-9:.1f}\" text-anchor=\"end\" fill=\"#b2182b\" font-size=\"13\" font-weight=\"bold\">CUDA-Core FP32 peak {cuda_compute_gflops/1000:.2f} TFLOP/s</text>')
tpy=sy(tensor_compute_gflops); svg.append(f'<line x1=\"{L}\" y1=\"{tpy:.1f}\" x2=\"{W-R}\" y2=\"{tpy:.1f}\" stroke=\"#7b3294\" stroke-width=\"2.5\" stroke-dasharray=\"10 6\"/>'); svg.append(f'<text x=\"{W-R-8}\" y=\"{tpy-9:.1f}\" text-anchor=\"end\" fill=\"#7b3294\" font-size=\"13\" font-weight=\"bold\">Tensor-Core TF32 model peak {tensor_compute_gflops/1000:.2f} TFLOP/s</text>')

levels=[('L1','l1_intensity','#1f77b4','circle'),('L2','l2_intensity','#ff7f0e','square'),('HBM','hbm_intensity','#2ca02c','triangle')]
for row in rows:
    y=float(row['simulated_gflops']); py=sy(y); label=row['variant'].split('_')[0].upper(); pts=[]
    for _,key,_,_ in levels: pts.append((sx(float(row[key])),py))
    svg.append('<polyline points=\"'+' '.join(f'{x:.1f},{yy:.1f}' for x,yy in pts)+'\" fill=\"none\" stroke=\"#777\" stroke-width=\"1.2\"/>')
    for (level,key,color,shape),(x,yy) in zip(levels,pts):
        if shape=='circle': svg.append(f'<circle cx=\"{x:.1f}\" cy=\"{yy:.1f}\" r=\"6.5\" fill=\"{color}\" stroke=\"black\"/>')
        elif shape=='square': svg.append(f'<rect x=\"{x-6.5:.1f}\" y=\"{yy-6.5:.1f}\" width=\"13\" height=\"13\" fill=\"{color}\" stroke=\"black\"/>')
        else: svg.append(f'<polygon points=\"{x:.1f},{yy-8:.1f} {x-7.5:.1f},{yy+6.5:.1f} {x+7.5:.1f},{yy+6.5:.1f}\" fill=\"{color}\" stroke=\"black\"/>')
        svg.append(f'<text x=\"{x+8:.1f}\" y=\"{yy-9:.1f}\" font-size=\"11\">{html.escape(label)}-{level}</text>')

svg.append(f'<text x=\"{L}\" y=\"{H-19}\" font-size=\"11\" fill=\"#555\">L1D + Shared combines coalesced processor-side L1D requests and physical shared bank/word transactions; fills are excluded.</text>')
svg.append(f'<text x=\"{L}\" y=\"{H-6}\" font-size=\"11\" fill=\"#555\">HBM is kernel-scope physical traffic; dirty output remaining in L2 is excluded. TF32 Tensor roof is contextual; all plotted points are FP32 SIMT.</text>')
svg.append('</svg>')
args.output.write_text('\n'.join(svg))
print(f'Generated {args.output}')
print(f'Ceilings: CUDA FP32={cuda_compute_gflops:.0f} GF/s, Tensor TF32={tensor_compute_gflops:.0f} GF/s, L1={l1_gbs:.0f} GB/s, L2={l2_gbs:.0f} GB/s, HBM={hbm_gbs:.0f} GB/s')
