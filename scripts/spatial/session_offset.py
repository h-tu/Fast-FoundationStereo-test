#!/usr/bin/env python3
"""Session disparity offset (d_inf correction) from the helmet-marker reads of many clips (plan doc §14.3).

For every clip, work/<id>_full/aruco/aruco_reads.csv holds d_model (stereo, as inferred with run_sequence --dinf)
and d_marker = fx*b/z_marker for each read.  The correction is the median of d_model - d_marker: per clip and pooled
(every read weighted equally).  The per-clip spread shows whether one value fits the session.

  python scripts/spatial/session_offset.py --ids 3345 3346 ... --json work/session_offset.json
prints the pooled correction; batch_process.py --stage recal --dinf_correction <value> applies it.
"""
import argparse, csv, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='+', required=True); ap.add_argument('--min_reads', type=int, default=30); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    per, pooled = {}, []
    for vid in a.ids:
        p = f'work/{vid}_full/aruco/aruco_reads.csv'
        if not os.path.exists(p):
            print(f'{vid}: no reads file'); continue
        rows = list(csv.DictReader(open(p)))
        dd = np.array([float(r['d_model']) - float(r['d_marker']) for r in rows])
        dd = dd[np.isfinite(dd)]
        if len(dd) < a.min_reads:
            print(f'{vid}: only {len(dd)} reads, not used'); continue
        dinf_used = json.load(open(f'work/{vid}_full/out/run_meta.json')).get('dinf', 0.0)
        per[vid] = dict(n=int(len(dd)), offset_px=float(np.median(dd)), mad_px=float(np.median(np.abs(dd - np.median(dd)))), dinf_used=dinf_used,
                        d_inf_true=float(np.median(dd) + dinf_used))
        pooled.append(dd)
        print(f'{vid}: n {len(dd):5d}  offset {np.median(dd):+.2f} px  (d_inf {np.median(dd) + dinf_used:+.2f})  MAD {per[vid]["mad_px"]:.2f}')
    allv = np.concatenate(pooled)
    offs = np.array([v['offset_px'] for v in per.values()])
    res = dict(clips=per, pooled_offset_px=float(np.median(allv)), n_reads=int(len(allv)), clip_offset_median=float(np.median(offs)),
               clip_offset_range=[float(offs.min()), float(offs.max())], clip_offset_std=float(offs.std()))
    print(f'\npooled offset {res["pooled_offset_px"]:+.2f} px over {len(allv)} reads; per clip median {res["clip_offset_median"]:+.2f}, '
          f'range {offs.min():+.2f} .. {offs.max():+.2f}, std {offs.std():.2f} px')
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=2)


if __name__ == '__main__':
    main()
