"""Which is noisier on the helmets, stereo or the ArUco PnP?  Self-consistency of each method (plan doc §13.3).

Two internal checks per method, on the same reads, after the session disparity offset is removed from stereo:
  1. temporal jitter: for one marker id read in 3 consecutive analysed frames, |z_t - median(z_{t-1}, z_t, z_{t+1})|
     (helmet motion over 1/30 s is smooth, so this is mostly measurement noise; common to both methods)
  2. two codes of one helmet read in the same frame: they sit ~10 cm apart on the hat, so their depth difference
     has a real part (<= ~0.1 m, same for both methods) plus each method's noise
Plus the disagreement itself, and what share of it each method's noise explains (independent noises add in quadrature).

python scripts/spatial/dev/aruco_noise.py 3354 3346 3347 3348
"""
import csv, sys
from collections import defaultdict
import numpy as np

fxb = 1554.372464 * 0.019255
reads = []
for vid in sys.argv[1:]:
    for r in csv.DictReader(open(f'work/{vid}_full/aruco/aruco_reads.csv')):
        reads.append(dict(vid=vid, frame=int(float(r['frame'])), id=int(r['id']), worker=r['worker'], zm=float(r['z_marker']),
                          dmod=float(r['d_model']), dmk=float(r['d_marker']), px=float(r['side_px'])))
off = np.median([r['dmod'] - r['dmk'] for r in reads])
for r in reads:
    r['zs'] = fxb / max(r['dmod'] - off, 1e-3)
print(f'{len(reads)} reads, session offset {off:+.2f} px removed from stereo')

# 1. temporal jitter
by = defaultdict(dict)
for r in reads:
    by[(r['vid'], r['id'])][r['frame']] = r
jm, js, zz = [], [], []
for seq in by.values():
    for f, r in seq.items():
        if f - 1 in seq and f + 1 in seq:
            a, c = seq[f - 1], seq[f + 1]
            # second difference: cancels steady motion, keeps noise (sigma * sqrt(1.5) for iid noise)
            jm.append((r['zm'] - 0.5 * (a['zm'] + c['zm'])) / r['zm'])
            js.append((r['zs'] - 0.5 * (a['zs'] + c['zs'])) / r['zs'])
            zz.append(r['zm'])
jm, js, zz = map(np.array, (jm, js, zz))
rs = lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(1.5)     # robust per-read noise sigma
print(f'\n1. frame-to-frame noise (second difference over 3 consecutive frames, robust sigma per read), {len(jm)} triplets:')
print(f'   marker PnP: {rs(jm) * 100:.2f} %      stereo: {rs(js) * 100:.2f} %')
for lo, hi in ((2, 3), (3, 4), (4, 5), (5, 7)):
    m = (zz >= lo) & (zz < hi)
    if m.sum() >= 20:
        print(f'   {lo}-{hi} m (n {m.sum():4d}): marker {rs(jm[m]) * 100:.2f} %   stereo {rs(js[m]) * 100:.2f} %')

# 2. two codes of one helmet in one frame
byf = defaultdict(list)
for r in reads:
    byf[(r['vid'], r['frame'], r['worker'])].append(r)
pm, ps = [], []
for v in byf.values():
    if len(v) == 2:
        pm.append(abs(v[0]['zm'] - v[1]['zm'])); ps.append(abs(v[0]['zs'] - v[1]['zs']))
if pm:
    print(f'\n2. two codes of one helmet, same frame ({len(pm)} pairs): |depth difference| median marker {np.median(pm) * 100:.1f} cm, stereo {np.median(ps) * 100:.1f} cm'
          f'   (p90 {np.percentile(pm, 90) * 100:.1f} / {np.percentile(ps, 90) * 100:.1f} cm)')

# 3. disagreement and its split (robust sigma = 1.4826 * MAD; jitter-based noise from the 3-frame statistic)
rel = np.array([r['zs'] / r['zm'] - 1 for r in reads])
s_dis = 1.4826 * np.median(np.abs(rel - np.median(rel)))
s_m, s_s = rs(jm), rs(js)
print(f'\n3. disagreement stereo vs marker: robust sigma {s_dis * 100:.1f} %  (bias {np.median(rel) * 100:+.1f} %)')
print(f'   noise sigma from jitter: marker {s_m * 100:.1f} %, stereo {s_s * 100:.1f} %;  sqrt(sum of squares) {np.hypot(s_m, s_s) * 100:.1f} %')
print(f'   share of the disagreement variance: marker {s_m ** 2 / s_dis ** 2 * 100:.0f} %, stereo {s_s ** 2 / s_dis ** 2 * 100:.0f} %, '
      f'rest (slow / systematic, not frame-to-frame) {max(0, 1 - (s_m ** 2 + s_s ** 2) / s_dis ** 2) * 100:.0f} %')
