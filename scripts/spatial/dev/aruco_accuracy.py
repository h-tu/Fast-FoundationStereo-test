"""Stereo vs marker depth accuracy, raw and after a disparity-offset correction (plan doc §13.3).

Pools the aruco_reads.csv of the given clips. The correction is one disparity offset for the whole session
(median of d_model - d_marker over all reads), i.e. d_inf_true = d_used + offset; it is also cross-validated
leave-one-clip-out so the corrected error is not fitted on the same reads it is measured on.

python scripts/spatial/dev/aruco_accuracy.py 3354 3346 3347
"""
import csv, sys
import numpy as np

BINS = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 12)]
reads = {}
for vid in sys.argv[1:]:
    rows = list(csv.DictReader(open(f'work/{vid}_full/aruco/aruco_reads.csv')))
    reads[vid] = {k: np.array([float(r[k]) for r in rows]) for k in ('z_marker', 'd_marker', 'd_model', 'side_px', 'reproj_px')}
fxb = 1554.372464 * 0.019255


def report(label, zm, zs):
    rel = zs / zm - 1; ab = np.abs(zs - zm)
    print(f'{label:<34} n {len(zm):5d}  median {np.median(rel) * 100:+5.1f} %  |err| median {np.median(np.abs(rel)) * 100:4.1f} % / {np.median(ab) * 100:4.0f} cm'
          f'  p90 {np.percentile(np.abs(rel), 90) * 100:5.1f} %  within 5 %: {np.mean(np.abs(rel) < 0.05) * 100:3.0f} %  within 10 %: {np.mean(np.abs(rel) < 0.10) * 100:3.0f} %')
    for lo, hi in BINS:
        m = (zm >= lo) & (zm < hi)
        if m.sum() >= 10:
            print(f'    {lo}-{hi} m: n {m.sum():5d}  median {np.median(rel[m]) * 100:+5.1f} %  |err| median {np.median(np.abs(rel[m])) * 100:4.1f} % / {np.median(ab[m]) * 100:4.0f} cm  p90 {np.percentile(np.abs(rel[m]), 90) * 100:5.1f} %')


allz = np.concatenate([r['z_marker'] for r in reads.values()])
alld = np.concatenate([r['d_model'] for r in reads.values()]); alldm = np.concatenate([r['d_marker'] for r in reads.values()])
report('raw (d_inf -1.0 px)', allz, fxb / alld)
off = float(np.median(alld - alldm))
print(f'\nsession offset (all clips): {off:+.2f} px  ->  d_inf_true = {-1 + off:+.2f} px')
report('corrected, one session offset', allz, fxb / np.maximum(alld - off, 1e-3))
# leave-one-clip-out
zs_cv = []
for vid, r in reads.items():
    others = [v for v in reads if v != vid]
    if not others:
        continue
    o = float(np.median(np.concatenate([reads[v]['d_model'] - reads[v]['d_marker'] for v in others])))
    zs_cv.append(fxb / np.maximum(r['d_model'] - o, 1e-3))
if len(zs_cv) == len(reads):
    report('corrected, leave-one-clip-out', allz, np.concatenate(zs_cv))
# disparity residual in px after correction (what limits accuracy)
res = (alld - off) - alldm
print(f'\ndisparity residual after correction: median {np.median(res):+.3f} px, MAD {np.median(np.abs(res - np.median(res))):.3f} px, p90 |res| {np.percentile(np.abs(res), 90):.2f} px')
for vid, r in reads.items():
    dd = r['d_model'] - r['d_marker']
    print(f'  {vid}: per-clip offset {np.median(dd):+.2f} px (d_inf {-1 + np.median(dd):+.2f}), MAD {np.median(np.abs(dd - np.median(dd))):.2f} px, n {len(dd)}')

# Which parameter error explains the data?  Each leaves a different fingerprint in d_model vs d_marker = fx*b/z_marker:
#   wrong fx (hfov)            : cancels (both depths scale with fx)          -> no effect on the comparison
#   wrong baseline or marker size: multiplicative, d_model = g * d_marker      -> constant % depth error
#   disparity offset (d_inf)   : additive, d_model = d_marker + c              -> error grows ~ z
print('\nerror models fitted to d_model vs d_marker (robust: median absolute residual, px):')
x, y = alldm, alld
g_only = float(np.median(y / x)); r_g = y - g_only * x
c_only = float(np.median(y - x)); r_c = y - (x + c_only)
A = np.stack([x, np.ones_like(x)], 1); gc = np.linalg.lstsq(A, y, rcond=None)[0]
for _ in range(5):
    r = y - A @ gc; m = np.abs(r - np.median(r)) < 3 * 1.4826 * np.median(np.abs(r - np.median(r)))
    gc = np.linalg.lstsq(A[m], y[m], rcond=None)[0]
r_gc = y - A @ gc
for name, r in (('gain only (baseline / marker size)', r_g), ('offset only (d_inf)', r_c), (f'gain {gc[0]:.3f} + offset {gc[1]:+.2f}', r_gc)):
    print(f'  {name:<36} median |res| {np.median(np.abs(r)):.3f} px   trend of residual vs d_marker: {np.polyfit(x, r, 1)[0]:+.3f} px/px')
print(f'  gain-only fit: g = {g_only:.3f} (would mean baseline {g_only * 19.255:.2f} mm or marker {88.9 / g_only:.1f} mm)')
# marker corner error needed to fake the offset: d_marker = b * p / S  ->  dp = c * S / b
print(f'  marker side error needed to fake a {c_only:+.2f} px offset: {abs(c_only) * 0.0889 / 0.019255:.1f} px on a {np.median(np.concatenate([r["side_px"] for r in reads.values()])):.0f} px marker')
