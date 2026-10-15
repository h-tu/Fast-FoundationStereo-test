#!/usr/bin/env python3
"""Estimate d_inf using people as calibration objects (plan doc §12).

Idea: the lateral (x, y) size of a person measured by stereo is
    x = (u - cx) * b / (d - d_inf)
which depends on the baseline and the disparity offset only, not on fx.  VideoPose3D
(trained on Human3.6M, metric) gives the same person's root-relative joints in metres.
For every track the ratio r = stereo size / VideoPose3D size (least squares on the image-plane
axes x, y over all valid joints) must be 1 if d_inf is right; with the disparity d_used that
run_sequence applied (--dinf), the offset that makes r = 1 is
    d_inf_true = d_used - (r - 1) * (d_model - 0) ,   d_model = fx*b / z_stereo_pelvis
(the model's disparity already includes the --dinf shift).  Reported per track and pooled.

Also prints the stereo bone lengths against typical adult values (independent of VideoPose3D).

Usage
  python scripts/spatial/dinf_from_bodies.py --tracks work/3143_full/pose/IMG_3143_left.tracks.npz --seq_dir work/3143_full --out_dir work/3143_full/out
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from pose_depth import load_tracks, coco_to_h36m, sample_depth, backproject, bone_lengths, H36M_BONES, H36M_BONE_NAMES, TYPICAL_BONE_M  # noqa: E402
from run_sequence import read_k  # noqa: E402


def stereo_joints(tr, idx, depth_path, K, stride, radius=3):
    S = np.full((len(idx), 17, 3), np.nan, np.float32); V = np.zeros((len(idx), 17), bool)
    for k, i in enumerate(idx):
        f = int(tr['frames'][i]); p = depth_path(f // stride)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        for j in range(17):
            u, v = tr['pose_2d'][i, j]
            if np.isfinite(u) and tr['conf_2d'][i, j] >= 0.3:
                z, frac, mad = sample_depth(d, u, v, radius)
                if np.isfinite(z) and frac >= 0.3 and mad <= 0.25 and 0.3 <= z <= 20:
                    S[k, j] = backproject(u, v, z, K); V[k, j] = True
    return S, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True); ap.add_argument('--seq_dir', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--every', type=int, default=5); ap.add_argument('--min_frames', type=int, default=60); ap.add_argument('--json', default=None)
    ap.add_argument('--body_z', type=int, default=1, help='1: one body depth per frame (see code); 0: per-joint depth samples')
    args = ap.parse_args()
    meta_seq = json.load(open(f'{args.seq_dir}/meta.json')); stride = int(meta_seq.get('stride', 1))
    run_meta = json.load(open(f'{args.out_dir}/run_meta.json')); d_used = float(run_meta.get('dinf', 0.0))
    K, b = read_k(f'{args.seq_dir}/K.txt'); fxb = K[0, 0] * b
    tracks, _, _ = load_tracks(args.tracks)
    rows = []
    for tr in tracks:
        if tr['pose_3d'] is None:
            continue
        keep = np.where(tr['frames'] % stride == 0)[0][::args.every]
        if len(keep) < args.min_frames // args.every:
            continue
        S, V = stereo_joints(tr, keep, lambda i: f'{args.out_dir}/depth/{i:06d}.npy', K, stride)
        SH, VH = coco_to_h36m(S, V); P = tr['pose_3d'][keep]
        if args.body_z:
            # one depth per frame for the whole body (25th percentile of the valid torso/hip/shoulder joint depths,
            # i.e. the body surface, not the background behind thin limbs) applied to every 2D joint:
            # lateral size then comes from the precise 2D keypoints and a single z, not from per-joint depth samples.
            torso = [5, 6, 11, 12]
            zb = np.array([np.nanpercentile(S[k, torso, 2][V[k, torso]], 25) if V[k, torso].sum() >= 2 else np.nan for k in range(len(keep))])
            SH = np.full_like(SH, np.nan); VH = np.zeros_like(VH)
            for k in range(len(keep)):
                if not np.isfinite(zb[k]):
                    continue
                S2 = np.full((17, 3), np.nan, np.float32); V2 = np.zeros(17, bool)
                for j in range(17):
                    u, v = tr['pose_2d'][keep[k], j]
                    if np.isfinite(u) and tr['conf_2d'][keep[k], j] >= 0.3:
                        S2[j] = backproject(u, v, zb[k], K); V2[j] = True
                sh, vh = coco_to_h36m(S2[None], V2[None]); SH[k], VH[k] = sh[0], vh[0]
        rel = SH - SH[:, :1]; m = VH & VH[:, :1]; m[:, [7, 10]] = False
        a = rel[..., :2][m]; p = P[..., :2][m]; ok = np.isfinite(a).all(1) & np.isfinite(p).all(1)
        a, p = a[ok], p[ok]
        if len(a) < 50 or VH[:, 0].mean() < 0.5:
            continue
        r = float((a * p).sum() / (p * p).sum())                     # LS scale on x,y jointly
        z_pel = float(np.nanmedian(SH[:, 0, 2])); d_model = fxb / z_pel
        d_inf = d_used - (r - 1.0) * d_model
        Ls = bone_lengths(SH, VH, H36M_BONES)
        limbs = {H36M_BONE_NAMES[k]: float(np.nanmedian(Ls[:, k])) for k in range(len(H36M_BONES)) if H36M_BONE_NAMES[k] in TYPICAL_BONE_M and np.isfinite(Ls[:, k]).sum() > 10}
        anth = float(np.median([limbs[k] / TYPICAL_BONE_M[k] for k in limbs])) if limbs else np.nan
        rows.append(dict(track=tr['id'], frames=int(len(keep)) * args.every, pelvis_z_m=z_pel, d_model_px=d_model, scale_xy=r, d_inf_est=d_inf,
                         anthropometric_ratio=anth, pelvis_valid=float(VH[:, 0].mean()), n_joint_samples=int(len(a))))
        print(f"track {tr['id']:3d}: z {z_pel:5.2f} m  d_model {d_model:5.2f} px  stereo/VP3D xy scale {r:.3f}  -> d_inf {d_inf:+.2f} px | stereo limb / typical {anth:.2f}  (pelvis valid {VH[:, 0].mean()*100:.0f}%, {len(a)} joint samples)")
    if rows:
        good = [x for x in rows if x['pelvis_valid'] >= 0.8 and x['n_joint_samples'] >= 200]
        est = np.array([x['d_inf_est'] for x in (good or rows)])
        w = np.array([x['d_model_px'] for x in (good or rows)])   # nearer tracks (larger disparity) constrain d_inf better
        print(f"\n{len(good or rows)} tracks: d_inf median {np.median(est):+.2f} px, weighted mean {np.average(est, weights=w):+.2f}, range {est.min():+.2f}..{est.max():+.2f}  (d_used {d_used:+.2f})")
        res = dict(tracks=rows, d_used=d_used, d_inf_median=float(np.median(est)), d_inf_weighted=float(np.average(est, weights=w)), n_good=len(good))
        if args.json:
            json.dump(res, open(args.json, 'w'), indent=2)


if __name__ == '__main__':
    main()
