"""Break an aruco_reads.csv down by worker / side / marker size, to find where marker and stereo disagree.

python scripts/spatial/dev/aruco_breakdown.py work/3354_full/aruco/aruco_reads.csv
"""
import csv, sys
from collections import defaultdict
import numpy as np

rows = list(csv.DictReader(open(sys.argv[1])))
num = ('frame', 'side_px', 'reproj_px', 'z_marker', 'z_stereo', 'd_marker', 'd_model', 'valid', 'x_marker', 'y_marker')
for r in rows:
    for k in num:
        r[k] = float(r[k])
    r['rel'] = r['z_stereo'] / r['z_marker'] - 1; r['dd'] = r['d_model'] - r['d_marker']
print(f'{len(rows)} reads')


def table(key, label):
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    print(f'\n{label:>14}    n  z_marker z_stereo   rel    d_model-d_marker  side_px reproj valid')
    for k in sorted(g):
        v = g[k]; m = lambda f: np.median([x[f] for x in v])
        print(f'{str(k):>14} {len(v):4d}  {m("z_marker"):7.2f}  {m("z_stereo"):7.2f}  {m("rel") * 100:+6.1f}%   {m("dd"):+6.2f} px      {m("side_px"):5.0f}  {m("reproj_px"):5.2f}  {m("valid"):4.2f}')


table(lambda r: (r['worker'], r['side']), 'worker/side')
bins = [0, 25, 35, 45, 60, 200]
table(lambda r: f"{bins[np.searchsorted(bins, r['side_px']) - 1]}-{bins[np.searchsorted(bins, r['side_px'])]} px", 'marker side')
by = defaultdict(list)
for r in rows:
    by[(r['frame'], r['worker'])].append(r)
pairs = [v for v in by.values() if len(v) > 1]
if pairs:
    zm = [max(x['z_marker'] for x in v) - min(x['z_marker'] for x in v) for v in pairs]
    zs = [max(x['z_stereo'] for x in v) - min(x['z_stereo'] for x in v) for v in pairs]
    print(f'\nsame helmet, two codes in one frame: {len(pairs)} pairs; median spread marker {np.median(zm):.2f} m, stereo {np.median(zs):.2f} m')
