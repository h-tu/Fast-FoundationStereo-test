"""Where does the slow stereo-vs-marker disagreement come from?  Residual disparity vs image position (plan doc §13.3).

residual = d_model - d_marker - session offset (px).  If the two eyes' horizontal alignment varies across the image
(a horizontal scale / rotation mismatch, not a constant d_inf), the residual depends on the marker's image column u
or row v.  Also checks per-marker (clip, id) segments: a residual constant within a segment but different between
segments is a per-marker effect (print/bending of that code or its viewing angle).

python scripts/spatial/dev/aruco_residual_map.py 3354 3346 3347 3348
"""
import csv, sys
from collections import defaultdict
import numpy as np

K = np.array([[1554.372464, 0, 959.5], [0, 1554.372464, 539.5], [0, 0, 1]])
R = []
for vid in sys.argv[1:]:
    for r in csv.DictReader(open(f'work/{vid}_full/aruco/aruco_reads.csv')):
        z = float(r['z_marker'])
        R.append(dict(vid=vid, id=int(r['id']), frame=int(float(r['frame'])), u=K[0, 0] * float(r['x_marker']) / z + K[0, 2],
                      v=K[1, 1] * float(r['y_marker']) / z + K[1, 2], z=z, px=float(r['side_px']), dd=float(r['d_model']) - float(r['d_marker'])))
dd = np.array([r['dd'] for r in R]); off = np.median(dd); res = dd - off
u = np.array([r['u'] for r in R]); v = np.array([r['v'] for r in R]); z = np.array([r['z'] for r in R]); px = np.array([r['px'] for r in R])
mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
print(f'{len(R)} reads, offset {off:+.2f} px, residual robust sigma {mad(res):.3f} px')


def binned(name, x, edges):
    print(f'\nresidual vs {name}:')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 15:
            print(f'   {lo:6.0f}-{hi:<6.0f} n {m.sum():5d}  median residual {np.median(res[m]):+.2f} px   sigma {mad(res[m]):.2f}')
    A = np.stack([x, np.ones_like(x)], 1); s = np.linalg.lstsq(A, res, rcond=None)[0]
    print(f'   linear slope {s[0] * 1000:+.2f} px per 1000 units')


binned('image column u (px)', u, [0, 320, 640, 960, 1280, 1600, 1920])
binned('image row v (px)', v, [0, 180, 360, 540, 720, 900, 1080])
binned('marker side (px)', px, [0, 30, 35, 40, 45, 55, 120])
# 2D fit res = a*u + b*v + c  and how much variance it removes
A = np.stack([u - 960, v - 540, np.ones_like(u)], 1); s = np.linalg.lstsq(A, res, rcond=None)[0]
r2 = res - A @ s
print(f'\nplane fit residual = {s[0] * 1000:+.2f}/1000 px * (u-960) {s[1] * 1000:+.2f}/1000 px * (v-540) {s[2]:+.2f}:  sigma {mad(res):.3f} -> {mad(r2):.3f} px')
# per-marker segments
seg = defaultdict(list)
for r, x in zip(R, res):
    seg[(r['vid'], r['id'])].append(x)
seg = {k: np.array(x) for k, x in seg.items() if len(x) >= 15}
means = np.array([np.median(x) for x in seg.values()]); within = np.median([mad(x) for x in seg.values()])
print(f'\nper marker (clip, id) with >= 15 reads: {len(seg)} markers; spread of their median residuals {mad(means):.2f} px '
      f'(range {means.min():+.2f} .. {means.max():+.2f}); typical within-marker sigma {within:.2f} px')
for k, x in sorted(seg.items(), key=lambda kv: np.median(kv[1])):
    print(f'   {k[0]} id {k[1]:2d}: n {len(x):4d}  median {np.median(x):+.2f} px  sigma {mad(x):.2f}')
