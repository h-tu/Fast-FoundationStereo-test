#!/usr/bin/env python3
"""Check how well a left/right pair satisfies the rectified-stereo assumptions
that Fast-FoundationStereo relies on, using dense block matching (ZNCC) rather
than sparse keypoints (sparse SIFT is unreliable on repetitive texture).

For every textured PxP patch in the LEFT image we search the RIGHT image over
x in [-DXMAX, +2] (disparity = xL - xR) and y in [-DY, +DY], keep matches with
ZNCC >= --min_ncc, refine to sub-pixel with a parabola, and report:

  * vertical offset  dy = yR - yL     (should be ~0 everywhere)
      - global median / std / fraction within 0.5, 1, 2 px
      - linear fit dy = a*y + b  (a != 0 => vertical scale mismatch between eyes,
                                   b != 0 => constant vertical shift)
      - 3x3 regional medians
  * disparity        d  = xL - xR      (should be >= 0; far content -> 0)
      - fraction of negative disparity, and the "infinity offset" d_inf
        estimated from the 10th percentile of low-texture-far candidates
        (only meaningful if the frame contains truly far content)
  * suggested correction for the RIGHT image:  y' = a*y + b  (affine, vertical
    only) so that horizontal geometry / disparity is untouched.

Usage:
  python scripts/spatial/check_rectification.py L.png R.png [--json out.json] [--plot out.png]
  python scripts/spatial/check_rectification.py --dir work/3143        # uses left/*.png right/*.png
"""
import argparse, glob, json, os, sys
import cv2
import numpy as np


def match_patches(L, R, P=48, step=64, DY=8, DXMAX=96, min_ncc=0.85, min_std=10.0):
    """Return array of rows (xL, yL, disparity, dy, ncc) for textured patches."""
    H, W = L.shape
    rows = []
    for y in range(DY + P, H - DY - P, step):
        for x in range(DXMAX + P, W - P - 2, step):
            t = L[y:y + P, x:x + P]
            if t.std() < min_std:
                continue
            win = R[y - DY:y + P + DY, x - DXMAX:x + P + 2]
            res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, (px, py) = cv2.minMaxLoc(res)
            if mx < min_ncc:
                continue
            # parabolic sub-pixel refinement (skip if on the border of the search window)
            if 0 < px < res.shape[1] - 1:
                a, b, c = res[py, px - 1], res[py, px], res[py, px + 1]
                sx = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            else:
                continue
            if 0 < py < res.shape[0] - 1:
                a, b, c = res[py - 1, px], res[py, px], res[py + 1, px]
                sy = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
            else:
                continue
            d = DXMAX - (px + sx)       # disparity  xL - xR
            dy = (py + sy) - DY         # vertical offset yR - yL
            rows.append((x + P / 2, y + P / 2, d, dy, mx))
    return np.array(rows, dtype=np.float64).reshape(-1, 5)


def summarize(r, W, H):
    if len(r) < 10:
        return {'n_patches': int(len(r)), 'error': 'too few matches'}
    x, y, d, dy = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    a, b = np.polyfit(y, dy, 1)
    ax, bx = np.polyfit(x, dy, 1)
    gy = np.digitize(y, [H / 3, 2 * H / 3]); gx = np.digitize(x, [W / 3, 2 * W / 3])
    grid_dy = [[float(np.median(dy[(gy == i) & (gx == j)])) if ((gy == i) & (gx == j)).sum() > 2 else None
                for j in range(3)] for i in range(3)]
    grid_d = [[float(np.median(d[(gy == i) & (gx == j)])) if ((gy == i) & (gx == j)).sum() > 2 else None
               for j in range(3)] for i in range(3)]
    return {
        'n_patches': int(len(r)),
        'dy_median': float(np.median(dy)), 'dy_mean': float(dy.mean()), 'dy_std': float(dy.std()),
        'dy_abs_lt_0.5': float((np.abs(dy) < 0.5).mean()),
        'dy_abs_lt_1': float((np.abs(dy) < 1).mean()),
        'dy_abs_lt_2': float((np.abs(dy) < 2).mean()),
        'dy_vs_y_slope_px_per_1000px': float(a * 1000), 'dy_vs_y_intercept': float(b),
        'dy_vs_x_slope_px_per_1000px': float(ax * 1000),
        'residual_std_after_affine': float((dy - (a * y + b)).std()),
        'disp_median': float(np.median(d)), 'disp_p5': float(np.percentile(d, 5)),
        'disp_p95': float(np.percentile(d, 95)), 'disp_min': float(d.min()), 'disp_max': float(d.max()),
        'disp_frac_neg': float((d < 0).mean()), 'disp_frac_lt_-1': float((d < -1).mean()),
        'disp_p10_as_infinity_offset_estimate': float(np.percentile(d, 10)),
        'grid_dy_median_3x3': grid_dy, 'grid_disp_median_3x3': grid_d,
        'suggested_right_vertical_affine': {'a': float(a), 'b': float(b),
                                            'note': "right eye row y_src = (1+a)*y_dst + b; run_sequence.py --right_affine a b applies this"},
    }


def check_pair(lp, rp, **kw):
    L = cv2.imread(lp, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    R = cv2.imread(rp, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    assert L.shape == R.shape, (L.shape, R.shape)
    r = match_patches(L, R, **kw)
    s = summarize(r, L.shape[1], L.shape[0])
    s.update(left=lp, right=rp)
    return s, r


def print_summary(s):
    print(f"\n== {os.path.basename(s['left'])}  vs  {os.path.basename(s['right'])}   patches={s['n_patches']}")
    if 'error' in s:
        print('  ', s['error']); return
    print(f"  vertical dy=yR-yL : median {s['dy_median']:+.2f}  std {s['dy_std']:.2f}  "
          f"|dy|<0.5: {s['dy_abs_lt_0.5']*100:.0f}%  <1: {s['dy_abs_lt_1']*100:.0f}%  <2: {s['dy_abs_lt_2']*100:.0f}%")
    print(f"  dy = a*y + b      : a={s['dy_vs_y_slope_px_per_1000px']:+.2f} px/1000px  b={s['dy_vs_y_intercept']:+.2f}  "
          f"residual std {s['residual_std_after_affine']:.2f}   (dy vs x slope {s['dy_vs_x_slope_px_per_1000px']:+.2f})")
    print(f"  disparity xL-xR   : median {s['disp_median']:+.2f}  p5 {s['disp_p5']:+.2f}  p95 {s['disp_p95']:+.2f}  "
          f"max {s['disp_max']:.1f}  neg {s['disp_frac_neg']*100:.0f}%  < -1px {s['disp_frac_lt_-1']*100:.0f}%")
    g = s['grid_dy_median_3x3']
    print('  dy 3x3 medians    : ' + ' | '.join(' '.join(f'{v:+5.2f}' if v is not None else '  n/a' for v in row) for row in g))
    g = s['grid_disp_median_3x3']
    print('  disp 3x3 medians  : ' + ' | '.join(' '.join(f'{v:+5.1f}' if v is not None else '  n/a' for v in row) for row in g))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('left', nargs='?'); ap.add_argument('right', nargs='?')
    ap.add_argument('--dir', help='directory with left/*.png and right/*.png (checks every k-th pair)')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--json', default=None)
    ap.add_argument('--min_ncc', type=float, default=0.85)
    a = ap.parse_args()
    pairs = []
    if a.dir:
        ls = sorted(glob.glob(f'{a.dir}/left/*.png'))[::a.every]
        # os.path-based (glob returns backslashes on Windows, so a '/left/' string replace would be a no-op)
        pairs = [(l, os.path.join(os.path.dirname(os.path.dirname(l)), 'right', os.path.basename(l))) for l in ls]
    elif a.left and a.right:
        pairs = [(a.left, a.right)]
    else:
        ap.error('give L R or --dir')
    out = []
    for lp, rp in pairs:
        s, _ = check_pair(lp, rp, min_ncc=a.min_ncc)
        print_summary(s); out.append(s)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, indent=2)
