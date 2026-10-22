#!/usr/bin/env python3
"""Re-derive metric depth from saved disparities with a corrected disparity offset (no new inference).

run_sequence.py shifted the right eye by -dinf_used before inference, so the saved disparity is
d_model = d_geom + d_inf_true - dinf_used.  Fast-FoundationStereo is shift-equivariant for d >= 0.5 px (plan doc
§10 step 4), so a better d_inf can be applied afterwards:  d_geom = d_model - correction,  correction = d_inf_true - dinf_used,
e.g. from the helmet markers (aruco_compare.py: median of d_model - d_marker).

Writes out_dir/depth/%06d.npy (float32 m, 0 = invalid; same remove-invisible rule as run_sequence.py) and records
dinf_correction_px / dinf_effective in run_meta.json (dinf itself stays the value used for inference).

  python scripts/spatial/apply_dinf.py --out_dir work/3348_full/out --correction_px -2.08
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from run_sequence import disp_to_depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--correction_px', type=float, required=True, help='d_inf_true - dinf_used (negative: depths get shorter)')
    ap.add_argument('--min_disp', type=float, default=0.5, help='disparities below this stay "far" (0 depth); the model outputs 0 for far content')
    a = ap.parse_args()
    meta_p = f'{a.out_dir}/run_meta.json'; meta = json.load(open(meta_p))
    K = np.array(meta['K']); fx = K[0, 0] * meta.get('scale', 1.0); b = meta['baseline_m']
    os.makedirs(f'{a.out_dir}/depth', exist_ok=True)
    files = sorted(glob.glob(f'{a.out_dir}/disp/*.npy'))
    for p in files:
        d = np.load(p)
        dc = np.where(d >= a.min_disp, d - a.correction_px, 0).astype(np.float32)   # far class (d < min_disp) stays far
        np.save(f'{a.out_dir}/depth/{os.path.basename(p)}', disp_to_depth(dc, fx, b).astype(np.float32))
    meta['dinf_correction_px'] = a.correction_px
    meta['dinf_effective'] = meta.get('dinf', 0.0) + a.correction_px
    json.dump(meta, open(meta_p, 'w'), indent=2)
    print(f'{len(files)} depth maps rewritten with correction {a.correction_px:+.2f} px (d_inf effective {meta["dinf_effective"]:+.2f} px)')


if __name__ == '__main__':
    main()
