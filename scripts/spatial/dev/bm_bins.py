"""Model disparity vs independent ZNCC block matching, binned by disparity (plan doc §12.3).

Run from the repo root after the 2026-10-14 batch (needs work/<id>_full/{left,right}/*.png and out/disp/*.npy).
The model saw the right eye shifted by -dinf = +1 px, so block matching on the raw pair is compared as d_bm + 1.
"""
import sys
import cv2
import numpy as np

sys.path.insert(0, 'scripts/spatial')
from check_rectification import match_patches  # noqa: E402

rows = []
for v, n in [('3143', 818), ('3144', 1065), ('3145', 931), ('3142', 1869)]:
    for i in range(0, n, 60):
        L = cv2.imread(f'work/{v}_full/left/{i:06d}.png', 0).astype(np.float32)
        R = cv2.imread(f'work/{v}_full/right/{i:06d}.png', 0).astype(np.float32)
        r = match_patches(L, R, P=32, step=32, DY=6, DXMAX=48, min_ncc=0.9)
        d = np.load(f'work/{v}_full/out/disp/{i:06d}.npy')
        for x, y, dbm, dy, ncc in r:
            x, y = int(x), int(y)
            rows.append((dbm + 1.0, np.median(d[y - 4:y + 5, x - 4:x + 5])))
rows = np.array(rows)
print('patches', len(rows))
for lo, hi in [(0.5, 2), (2, 3), (3, 4), (4, 6), (6, 10), (10, 20), (20, 50)]:
    m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
    if m.sum() > 20:
        diff = rows[m, 1] - rows[m, 0]
        print(f'd_bm in [{lo},{hi}): n {m.sum():5d}  model - bm: median {np.median(diff):+.2f} px  MAD {np.median(np.abs(diff - np.median(diff))):.2f}')
