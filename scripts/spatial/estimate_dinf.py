#!/usr/bin/env python3
"""Estimate the disparity at infinity (d_inf) of an iPhone spatial-video pair
from Fast-FoundationStereo output ("far-background method", §4.2 / §7 step 4).

Why not just read disp at the sky?  run_sequence.py clips disparity at 0 and the
cost volume only covers [0, max_disp), so a *negative* d_inf cannot be observed
directly.  Trick: run the sequence with a known right-eye shift, e.g.
`--dinf -3` (right eye moved so every disparity grows by +3 px).  Infinity then
lands at 3 + d_inf > 0, well inside the volume, and
    d_inf = d_far_model - 3 .
This script reads `run_meta.json` of such a run to know the shift that was used.

Far pixels are picked per frame as the dominant low-disparity mode (0.1 px bins)
inside a region of interest (default: top --top_frac of the image, where sky /
horizon / far trees live).  Sky is large and flat, so the histogram mode is much
more robust than a percentile.  Optionally give an explicit --roi x0 y0 x1 y1.

Outputs per-frame mode + the median / IQR over frames, plus the block-matching
estimate from check_rectification (if a rect.json exists) for cross-checking.

Usage
  python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out_shift3 \
      --dinf -3 --auto_affine 1 --keep_negative ...
  python scripts/spatial/estimate_dinf.py --out_dir work/3143/out_shift3 [--top_frac 0.4] [--rect_json work/3143/rect.json]
"""
import argparse, glob, json, os
import numpy as np


def far_mode(disp, roi, shift, bin_w=0.1, search=6.0):
    x0, y0, x1, y1 = roi
    d = disp[y0:y1, x0:x1].ravel()
    d = d[np.isfinite(d)]
    lo, hi = shift - search, shift + search
    d = d[(d > lo) & (d < hi)]
    if len(d) < 1000:
        return None, 0
    hist, edges = np.histogram(d, bins=np.arange(lo, hi + bin_w, bin_w))
    k = hist.argmax()
    # refine: mean of the pixels within +-0.3 px of the peak bin
    c = 0.5 * (edges[k] + edges[k + 1])
    sel = d[np.abs(d - c) <= 0.3]
    return float(sel.mean()), int(len(sel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True, help='run_sequence.py output with disp/*.npy and run_meta.json')
    ap.add_argument('--top_frac', type=float, default=0.4, help='ROI = top fraction of the image (if no --roi)')
    ap.add_argument('--roi', type=int, nargs=4, default=None, metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--rect_json', default=None, help='check_rectification.py JSON for the block-matching cross-check')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    meta = json.load(open(f'{args.out_dir}/run_meta.json'))
    shift = -float(meta.get('dinf', 0.0))          # --dinf -3  ->  all disparities + 3
    scale = float(meta.get('scale', 1.0))
    files = sorted(glob.glob(f'{args.out_dir}/disp/*.npy'))[::args.every]
    assert files, 'no disp files'
    d0 = np.load(files[0]); H, W = d0.shape
    roi = args.roi or (0, 0, W, int(H * args.top_frac))
    rows = []
    for f in files:
        disp = np.load(f) / scale                 # back to full-res pixel units
        m, n = far_mode(disp, roi, shift)
        p = disp[roi[1]:roi[3], roi[0]:roi[2]]
        rows.append(dict(frame=os.path.basename(f), far_mode_disp=m, n_pix=n,
                         roi_p5=float(np.percentile(p, 5)), roi_p10=float(np.percentile(p, 10))))
        print(f"{rows[-1]['frame']}: far mode {m if m is None else f'{m:+.2f}'} px (n={n})  roi p5 {rows[-1]['roi_p5']:+.2f} p10 {rows[-1]['roi_p10']:+.2f}")
    modes = np.array([r['far_mode_disp'] for r in rows if r['far_mode_disp'] is not None])
    res = dict(out_dir=args.out_dir, shift_px=shift, roi=list(roi), n_frames=len(rows), n_frames_with_far=int(len(modes)))
    if len(modes):
        med = float(np.median(modes)); q1, q3 = np.percentile(modes, [25, 75])
        res.update(far_mode_median_px=med, far_mode_iqr_px=[float(q1), float(q3)],
                   d_inf_px=med - shift, d_inf_iqr_px=[float(q1 - shift), float(q3 - shift)])
        print(f'\nfar-mode disparity: median {med:+.2f} px (IQR {q1:+.2f}..{q3:+.2f}) over {len(modes)} frames, shift {shift:+.2f}')
        print(f'==> d_inf = {med - shift:+.2f} px   (IQR {q1 - shift:+.2f} .. {q3 - shift:+.2f})')
    if args.rect_json and os.path.exists(args.rect_json):
        r = json.load(open(args.rect_json))
        p5 = np.array([s['disp_p5'] for s in r if 'disp_p5' in s]); p10 = np.array([s['disp_p10_as_infinity_offset_estimate'] for s in r if 'disp_p5' in s])
        res.update(block_match_p5_median=float(np.median(p5)), block_match_p10_median=float(np.median(p10)))
        print(f'block matching (check_rectification): p5 median {np.median(p5):+.2f}, p10 median {np.median(p10):+.2f} px')
    res['per_frame'] = rows
    if args.json:
        json.dump(res, open(args.json, 'w'), indent=2)


if __name__ == '__main__':
    main()
