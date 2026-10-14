#!/usr/bin/env python3
"""Far-horizon block-matching estimate of d_inf on work/3143 (plan doc §10 step 4b).

Record of the exact analysis that produced d_inf = -1.0 px; paths are hard-coded
(run from the repo root after extract_views.py + check_rectification.py --json
work/3143/rect.json).  Needs the horizon band to be visible: y < 330, x < 900.
"""
import numpy as np, sys, cv2, json
sys.path.insert(0, 'scripts/spatial')
from check_rectification import match_patches
from run_sequence import fix_right

r = np.load('work/3143/horizon_patches.npy'); far = r[(r[:, 2] < 0.5)]
print('far cluster (d<0.5): n', len(far), 'median %.2f mean %.2f std %.2f  y range %d-%d' % (np.median(far[:, 2]), far[:, 2].mean(), far[:, 2].std(), far[:, 1].min(), far[:, 1].max()))
a, b = np.polyfit(far[:, 0], far[:, 2], 1); print('d vs x slope: %.3f px/1000px, intercept %.2f' % (a * 1000, b))
a, b = np.polyfit(far[:, 1], far[:, 2], 1); print('d vs y slope: %.3f px/1000px, intercept %.2f' % (a * 1000, b))
rect = json.load(open('work/3143/rect.json')); meds = []; meds48 = []; allq = []
for n in range(28):
    L = cv2.imread(f'work/3143/left/{n:06d}.png', 0); R = cv2.imread(f'work/3143/right/{n:06d}.png', 0)
    s = rect[n]; R = fix_right(R, s['dy_vs_y_slope_px_per_1000px'] / 1000, s['dy_vs_y_intercept'], 0.0)
    L = L.astype(np.float32); R = R.astype(np.float32)
    q = match_patches(L, R, P=32, step=16, DY=4, DXMAX=16, min_ncc=0.9); q = q[(q[:, 1] < 330) & (q[:, 0] < 900) & (q[:, 2] < 0.5)]
    q48 = match_patches(L, R, P=48, step=24, DY=4, DXMAX=16, min_ncc=0.9); q48 = q48[(q48[:, 1] < 330) & (q48[:, 0] < 900) & (q48[:, 2] < 0.5)]
    if len(q) >= 5: meds.append(np.median(q[:, 2])); allq.append(q)
    if len(q48) >= 5: meds48.append(np.median(q48[:, 2]))
    m1 = np.median(q[:, 2]) if len(q) else float('nan'); m2 = np.median(q48[:, 2]) if len(q48) else float('nan')
    print(f'{n:02d}: P32 n={len(q):3d} med {m1:+.2f} | P48 n={len(q48):3d} med {m2:+.2f}')
print('P32 (affine-fixed right): frames', len(meds), 'median of medians %.2f  IQR %.2f..%.2f' % (np.median(meds), *np.percentile(meds, [25, 75])))
print('P48 (affine-fixed right): frames', len(meds48), 'median of medians %.2f  IQR %.2f..%.2f' % (np.median(meds48), *np.percentile(meds48, [25, 75])))
allq = np.concatenate(allq); h, e = np.histogram(allq[:, 2], bins=np.arange(-2.5, 0.75, 0.25)); print('hist -2.5..0.5 step .25 (P32 fixed):', ' '.join(str(v) for v in h))
json.dump(dict(p32_median=float(np.median(meds)), p32_iqr=[float(x) for x in np.percentile(meds, [25, 75])], p48_median=float(np.median(meds48)), per_frame_p32=[float(x) for x in meds]), open('work/3143/dinf_blockmatch.json', 'w'), indent=2)
