#!/usr/bin/env python3
"""Collect the per-clip results of batch_process.py into one table (Markdown + JSON).

Reads, per clip: work/probe_<id>.json, work/<id>_full/{meta.json, out/run_meta.json, video/video_meta.json, eval.json,
pose/IMG_<id>_left.pose_depth.json, pose/dinf_bodies.json, .gpu_done, .post_done}.

Usage
  python scripts/spatial/batch_summary.py --ids 3345 3346 ... --json work/batch_summary.json --md work/batch_summary.md
"""
import argparse, json, os
import numpy as np


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='+', required=True); ap.add_argument('--json', default=None); ap.add_argument('--md', default=None)
    a = ap.parse_args()
    rows = []
    for vid in a.ids:
        d = f'work/{vid}_full'
        pr, meta, rm, vm = load(f'work/probe_{vid}.json'), load(f'{d}/meta.json'), load(f'{d}/out/run_meta.json'), load(f'{d}/video/video_meta.json')
        ev, pdp, db = load(f'{d}/eval.json'), load(f'{d}/pose/IMG_{vid}_left.pose_depth.json'), load(f'{d}/pose/dinf_bodies.json')
        g, p = load(f'{d}/.gpu_done') or {}, load(f'{d}/.post_done') or {}
        r = dict(id=vid, duration_s=pr and pr['duration_s'], src_frames=pr and pr['nb_frames'], stride=meta and meta['stride'], frames=meta and meta['n_frames_extracted'])
        if rm:
            aff = rm['affine_per_frame']; av = np.array([x[0] for x in aff.values()]) * 1000; bv = np.array([x[1] for x in aff.values()]); res = np.array([x[3] for x in aff.values()])
            r.update(model_ms=rm['mean_ms_after_warmup'], affine_a_permille=[float(av.mean()), float(av.min()), float(av.max())],
                     affine_b_px=[float(bv.mean()), float(bv.min()), float(bv.max())], affine_resid_px=float(res.mean()), dinf=rm['dinf'])
        if vm:
            r.update(fps_out=vm['fps_out'], files_mb={k: round(v / 1e6, 1) for k, v in vm['files'].items()})
        if ev:
            r.update(coverage=ev['coverage']['mean'], flicker_p90_px=ev['flicker']['p90_px'], flicker_p90_rel=ev['flicker']['p90_rel_depth'],
                     plane_rms_cm=ev['plane'].get('rms_cm_median'), cam_height_m=ev['plane'].get('height_m_median'),
                     bm_bias_px=ev['blockmatch'].get('bias_px'), bm_mad_px=ev['blockmatch'].get('mad_px'),
                     lr_lt1px=ev.get('lr', {}).get('frac_lt_1px_mean'))
        if pdp:
            c = pdp.get('comparison', {}); near = [t for t in pdp.get('tracks', []) if t['near_frames'] >= 60 and t['pelvis_valid_frac'] >= 0.8]
            r.update(tracks_pkg=pdp.get('n_tracks_in_package'), tracks_used=pdp.get('n_tracks'), tracks_near=len(near),
                     vp3d_vs_stereo_cm=c.get('vp3d_scaled_vs_stereo', {}).get('median_3d_cm'), adjusted_vs_stereo_cm=c.get('fused_vs_stereo', {}).get('median_3d_cm'),
                     flip_rate=c.get('depth_flip_rate'), limb_ratio=pdp.get('anthropometric_scale_ratio', {}).get('limbs_median'))
        if db:
            near = [t for t in db['tracks'] if t['pelvis_z_m'] <= 4.5 and t['pelvis_valid'] >= 0.8 and t['n_joint_samples'] >= 200]
            far = [t for t in db['tracks'] if t['pelvis_z_m'] > 5.0 and t['pelvis_valid'] >= 0.8 and t['n_joint_samples'] >= 200]
            r.update(bodies_dinf_near=float(np.median([t['d_inf_est'] for t in near])) if near else None, bodies_near_n=len(near),
                     bodies_dinf_far=float(np.median([t['d_inf_est'] for t in far])) if far else None, bodies_far_n=len(far))
        ar = load(f'{d}/aruco/aruco_compare.json')
        if ar and ar.get('stats'):
            st, fit = ar['stats'], ar.get('fit', {})
            r.update(aruco_reads=ar['reads_used'], aruco_workers=len(ar['workers']), aruco_median_rel=st['median_rel_err'], aruco_p90_abs_rel=st['p90_abs_rel_err'],
                     aruco_gain=fit.get('gain'), aruco_dinf_true=fit.get('d_inf_true'), aruco_dinf_if_gain1=fit.get('d_inf_true_if_gain_1'),
                     aruco_by_distance=st.get('by_distance'))
        r.update(timing={**g, **p})
        rows.append(r)
    f = lambda x, fmt: (fmt.format(x) if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else '—')
    lines = ['| Clip | length / stride → frames | model ms | affine a ‰ / b px (mean [min, max]) | coverage 0.3–8 m | flicker p90 px (rel) | road RMS cm / cam height m | model−BM bias / MAD px | LR < 1 px | pose tracks pkg/used/near | VP3D→adjusted vs stereo cm | stereo limb / typical | bodies d_inf near (n) / far (n) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        aa, bb = r.get('affine_a_permille'), r.get('affine_b_px')
        lines.append(f"| {r['id']} | {f(r['duration_s'], '{:.0f}')} s / {r['stride']} → {r['frames']} | {f(r.get('model_ms'), '{:.0f}')} | "
                     f"{(f'{aa[0]:.2f} [{aa[1]:.2f}, {aa[2]:.2f}] / {bb[0]:.2f} [{bb[1]:.2f}, {bb[2]:.2f}]') if aa else '—'} | {f(r.get('coverage'), '{:.0%}')} | "
                     f"{f(r.get('flicker_p90_px'), '{:.2f}')} ({f(r.get('flicker_p90_rel'), '{:.1%}')}) | {f(r.get('plane_rms_cm'), '{:.1f}')} / {f(r.get('cam_height_m'), '{:.2f}')} | "
                     f"{f(r.get('bm_bias_px'), '{:+.3f}')} / {f(r.get('bm_mad_px'), '{:.2f}')} | {f(r.get('lr_lt1px'), '{:.1%}')} | "
                     f"{r.get('tracks_pkg', '—')}/{r.get('tracks_used', '—')}/{r.get('tracks_near', '—')} | {f(r.get('vp3d_vs_stereo_cm'), '{:.1f}')} → {f(r.get('adjusted_vs_stereo_cm'), '{:.1f}')} | "
                     f"{f(r.get('limb_ratio'), '{:.2f}')} | {f(r.get('bodies_dinf_near'), '{:+.1f}')} ({r.get('bodies_near_n', 0)}) / {f(r.get('bodies_dinf_far'), '{:+.1f}')} ({r.get('bodies_far_n', 0)}) |")
    lines += ['', '| Clip | ArUco reads (workers) | stereo vs marker depth, median (abs p90) | disparity fit gain | true d_inf (fit) | true d_inf (gain 1) |',
              '|---|---|---|---|---|---|']
    for r in rows:
        if 'aruco_reads' in r:
            lines.append(f"| {r['id']} | {r['aruco_reads']} ({r['aruco_workers']}) | {f(r['aruco_median_rel'], '{:+.1%}')} ({f(r['aruco_p90_abs_rel'], '{:.1%}')}) | "
                         f"{f(r.get('aruco_gain'), '{:.3f}')} | {f(r.get('aruco_dinf_true'), '{:+.2f}')} px | {f(r.get('aruco_dinf_if_gain1'), '{:+.2f}')} px |")
    md = '\n'.join(lines)
    print(md)
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=2)
    if a.md:
        open(a.md, 'w', encoding='utf-8').write(md + '\n')


if __name__ == '__main__':
    main()
