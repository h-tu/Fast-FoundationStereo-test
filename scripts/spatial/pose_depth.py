#!/usr/bin/env python3
"""Combine field-pose-annotator 2D/3D skeletons with Fast-FoundationStereo metric depth.

Inputs
  --tracks   <video>.tracks.npz from field-pose-annotator (scripts/preprocess_video.py, rtmlib backend):
             per track COCO-17 2D keypoints (pixels, left eye) and VideoPose3D H36M-17 3D
             (root-relative, "model units" ~ metres, camera axes x right / y down / z depth).
  --seq_dir  extract_views.py output (K.txt, meta.json with stride; left/*.png)
  --out_dir  run_sequence.py output (depth/%06d.npy, metres, 0 = invalid)

What it does, per track and frame that has a depth map (pose frame f -> depth frame f/stride):
  1. STEREO 3D: depth at every 2D joint (median of a (2r+1)^2 window, invalid pixels ignored),
     back-projected with K -> metric joint position; a joint is valid when the 2D confidence,
     the share of valid depth pixels and the window's depth spread pass thresholds.
  2. COMPARISON (validation): the stereo COCO joints are mapped to H36M-17 (pelvis = mid-hips,
     thorax = mid-shoulders, nose, head skipped); VideoPose3D is anchored at the stereo pelvis and
     scaled by one factor per track (median ratio of stereo / VideoPose3D bone lengths); the z sign
     of VideoPose3D is checked against stereo.  Per-joint disagreement (3D and z-only) and the
     bone-length statistics of both sources are reported.
  3. ADJUSTED VideoPose3D ("fused"): scale + pelvis anchoring (always), optional depth-direction
     flip fix per limb (--fix_flips 1): when stereo says a limb points towards the camera and
     VideoPose3D says away (confident disagreement in the sign of the bone's dz), the limb's
     subtree is mirrored about its parent; optional per-joint blending with stereo (--fuse_weight w).

Outputs (in --video_dir, default <tracks dir>)
  <id>.pose_depth.npz / .json   per track: stereo COCO 3D + validity, stereo H36M 3D, VideoPose3D
                                scaled+anchored H36M 3D, fused H36M 3D, scale, flip counts; metrics
  pose_compare.mp4              left image with the 2D skeleton coloured by stereo depth | depth map with
                                skeleton | bird's-eye (x-z, metres): stereo joints (filled) vs scaled
                                VideoPose3D (hollow) with the per-frame disagreement
  pose_fused.mp4                the adjusted VideoPose3D skeleton reprojected onto the left image |
                                front view (x-y) | top view (x-z) in metres
  <id>_fused.tracks.npz/.json   (--write_fpa_package 1) copy of the tracks package with pose_3d replaced
                                by the fused metric skeleton (units "metres", origin "camera")

Needs ffmpeg on PATH, numpy, opencv.  Example
  python scripts/spatial/pose_depth.py --tracks work/3143_full/pose/IMG_3143_left.tracks.npz --seq_dir work/3143_full --out_dir work/3143_full/out
"""
import argparse, hashlib, json, os, sys, time
import cv2
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir)
from make_depth_video import colour_bar, depth_to_colour, ffmpeg_writer   # noqa: E402
from run_sequence import read_k                                            # noqa: E402

COCO = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist',
        'L_hip', 'R_hip', 'L_knee', 'R_knee', 'L_ankle', 'R_ankle']
COCO_BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]
H36M = ['Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Spine', 'Thorax', 'Neck/Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist']
H36M_BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]
H36M_BONE_NAMES = ['pelvis-R_hip', 'R_thigh', 'R_shin', 'pelvis-L_hip', 'L_thigh', 'L_shin', 'pelvis-spine', 'spine-thorax', 'thorax-nose', 'nose-head',
                   'thorax-L_sh', 'L_upper_arm', 'L_forearm', 'thorax-R_sh', 'R_upper_arm', 'R_forearm']
# H36M joint <- COCO joint(s) (pelvis/thorax/spine are midpoints; head is not observable in COCO -> excluded)
H36M_FROM_COCO = {0: (11, 12), 1: (12,), 2: (14,), 3: (16,), 4: (11,), 5: (13,), 6: (15,), 8: (5, 6), 9: (0,), 11: (5,), 12: (7,), 13: (9,), 14: (6,), 15: (8,), 16: (10,)}
COMPARE_JOINTS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16]        # H36M joints with a stereo counterpart
# limb subtrees for the depth-flip fix: bone (parent, child) -> joints that move with the child
FLIP_BONES = {(1, 2): [2, 3], (2, 3): [3], (4, 5): [5, 6], (5, 6): [6], (11, 12): [12, 13], (12, 13): [13], (14, 15): [15, 16], (15, 16): [16]}
TYPICAL_BONE_M = {'R_thigh': 0.42, 'L_thigh': 0.42, 'R_shin': 0.42, 'L_shin': 0.42, 'L_upper_arm': 0.30, 'R_upper_arm': 0.30,
                  'L_forearm': 0.26, 'R_forearm': 0.26, 'shoulder_width': 0.39, 'hip_width': 0.20}
ZMIN, ZMAX = 0.3, 8.0


def load_tracks(npz_path):
    meta = json.load(open(os.path.splitext(os.path.splitext(npz_path)[0])[0] + '.tracks.json'))
    d = np.load(npz_path, allow_pickle=False)
    tracks = []
    spurious = set(meta.get('spurious_track_ids', []))
    for tid in d['track_ids'].tolist():
        p = f't{tid}_'
        tracks.append(dict(id=int(tid), frames=d[p + 'frames'], bbox=d[p + 'bbox'], pose_2d=d[p + 'pose_2d'], conf_2d=d[p + 'confidence_2d'],
                           pose_3d=d[p + 'pose_3d'] if p + 'pose_3d' in d.files else None, spurious=int(tid) in spurious))
    return tracks, meta, {k: d[k] for k in d.files}


def sample_depth(depth, u, v, r=3, pct=30.0):
    """Depth at a 2D joint: the `pct` percentile of the valid pixels in a (2r+1)^2 window.
    A low percentile (default 30) prefers the body surface over the background behind thin limbs
    and silhouette joints (shoulders, hips); the median of the window would drift to the background
    for small, distant people.  Returns (z, valid fraction, MAD of the window)."""
    H, W = depth.shape
    x0, x1 = max(0, int(round(u)) - r), min(W, int(round(u)) + r + 1)
    y0, y1 = max(0, int(round(v)) - r), min(H, int(round(v)) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan, 0.0, np.nan
    w = depth[y0:y1, x0:x1].ravel()
    ok = (w > 0) & (w < 60)
    if ok.sum() == 0:
        return np.nan, 0.0, np.nan
    z = w[ok]
    med = float(np.median(z))
    return float(np.percentile(z, pct)), float(ok.mean()), float(np.median(np.abs(z - med)))


def backproject(u, v, z, K):
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], dtype=np.float32)


def project(P, K):
    z = np.maximum(P[..., 2], 1e-3)
    return np.stack([K[0, 0] * P[..., 0] / z + K[0, 2], K[1, 1] * P[..., 1] / z + K[1, 2]], -1)


def coco_to_h36m(S, V):
    """S (n,17,3) stereo COCO joints, V (n,17) validity -> H36M (n,17,3), (n,17) validity (head invalid)."""
    n = S.shape[0]
    out = np.full((n, 17, 3), np.nan, np.float32); ok = np.zeros((n, 17), bool)
    for h, src in H36M_FROM_COCO.items():
        v = np.all(V[:, list(src)], axis=1)
        out[v, h] = S[v][:, list(src)].mean(axis=1)
        ok[:, h] = v
    # spine = midpoint of pelvis and thorax when both exist
    v = ok[:, 0] & ok[:, 8]
    out[v, 7] = 0.5 * (out[v, 0] + out[v, 8]); ok[:, 7] = v
    return out, ok


def bone_lengths(P, ok, bones):
    L = np.full((P.shape[0], len(bones)), np.nan, np.float32)
    for k, (a, b) in enumerate(bones):
        v = ok[:, a] & ok[:, b]
        L[v, k] = np.linalg.norm(P[v, a] - P[v, b], axis=1)
    return L


def interp_fill(X, ok):
    """Fill invalid rows of X (n,3) by linear interpolation in time (edge-hold)."""
    X = X.copy(); idx = np.arange(len(X))
    if ok.sum() == 0:
        return X, ok
    for c in range(X.shape[1]):
        X[~ok, c] = np.interp(idx[~ok], idx[ok], X[ok, c])
    return X, np.ones_like(ok)


def median_filter_rows(X, k=5):
    if len(X) < k:
        return X
    pad = k // 2
    Xp = np.pad(X, ((pad, pad), (0, 0)), mode='edge')
    return np.stack([np.median(Xp[i:i + k], axis=0) for i in range(len(X))])


def process_track(tr, depth_of, K, stride, args):
    """Return per-frame results for the frames of this track that have a depth map, or None."""
    fr = tr['frames']; keep = (fr % stride == 0)
    if keep.sum() < args.min_frames or tr['pose_3d'] is None:
        return None
    idx = np.where(keep)[0]
    n = len(idx)
    S = np.full((n, 17, 3), np.nan, np.float32); V = np.zeros((n, 17), bool); Zinfo = np.zeros((n, 17, 3), np.float32)
    for k, i in enumerate(idx):
        depth = depth_of(int(fr[i]) // stride)
        if depth is None:
            continue
        for j in range(17):
            u, v = tr['pose_2d'][i, j]
            if not np.isfinite(u) or tr['conf_2d'][i, j] < args.min_conf:
                continue
            z, frac, mad = sample_depth(depth, u, v, args.radius, args.depth_pct)
            if args.disp_offset and np.isfinite(z) and z > 0:      # re-map with an extra disparity offset (calibration experiments)
                z = args.fxb / (args.fxb / z + args.disp_offset)
            Zinfo[k, j] = (z, frac, mad)
            if np.isfinite(z) and frac >= args.min_valid_frac and mad <= args.max_mad and ZMIN <= z <= 20:
                S[k, j] = backproject(u, v, z, K); V[k, j] = True
    SH, VH = coco_to_h36m(S, V)
    P = tr['pose_3d'][idx].astype(np.float32)                 # VideoPose3D root-relative (x right, y down, z forward, metres)
    Ls = bone_lengths(SH, VH, H36M_BONES); Lp = bone_lengths(P, np.ones_like(VH), H36M_BONES)
    # ---- scale: per frame, least squares on the image-plane axes (x, y) of all valid joints relative to the pelvis,
    #      using ONE body depth per frame (25th percentile of the valid torso-joint depths) so the lateral size comes
    #      from the precise 2D keypoints and is not inflated by per-joint depth samples on the background.
    torso = [5, 6, 11, 12]
    ratios = []
    for k in range(n):
        if V[k, torso].sum() < 2:
            continue
        zb = float(np.percentile(S[k, torso, 2][V[k, torso]], 25))
        p2 = tr['pose_2d'][idx[k]]; okj = np.isfinite(p2).all(1) & (tr['conf_2d'][idx[k]] >= args.min_conf)
        if not (okj[11] and okj[12]):
            continue
        xy = np.stack([(p2[:, 0] - K[0, 2]) * zb / K[0, 0], (p2[:, 1] - K[1, 2]) * zb / K[1, 1]], 1)
        hx, hv = coco_to_h36m(np.concatenate([xy, np.full((17, 1), zb, np.float32)], 1)[None], okj[None])
        rel = (hx[0] - hx[0, 0])[:, :2]; m = hv[0] & np.isfinite(rel).all(1); m[[0, 7, 10]] = False
        if m.sum() < 6:
            continue
        a, p = rel[m].ravel(), P[k, m, :2].ravel()
        ratios.append(float((a * p).sum() / max((p * p).sum(), 1e-9)))
    if len(ratios) < 10:
        return None
    scale = float(np.median(ratios))
    zsign = 1.0          # VideoPose3D camera axes match ours (checked: y correlation +0.98, z positive); not re-estimated per track
    Pm = P * scale
    # ---- anchor at the stereo pelvis (median-filtered, interpolated over invalid frames)
    pelvis, pel_ok = interp_fill(np.nan_to_num(SH[:, 0]), VH[:, 0])
    pelvis = median_filter_rows(pelvis, args.pelvis_smooth)
    A = Pm + pelvis[:, None, :]                                # scaled + anchored VideoPose3D, absolute metres
    # ---- comparison metrics (before fix)
    diff3 = np.linalg.norm(A - SH, axis=2); diffz = (A - SH)[..., 2]
    cmp_mask = VH.copy(); cmp_mask[:, [7, 10]] = False
    # ---- adjusted VideoPose3D: depth-direction flip fix per limb, only where stereo can resolve it:
    #      pelvis closer than --near_max and |dz_stereo| above both --flip_min_dz and 2 sigma of the stereo depth
    #      noise at that distance (sigma_z = z^2 * 0.15 px / (fx*b)).
    F = A.copy(); flips = np.zeros(len(H36M_BONES), int); flip_checked = np.zeros(len(H36M_BONES), int)
    near = np.isfinite(pelvis[:, 2]) & (pelvis[:, 2] <= args.near_max) & VH[:, 0]
    if args.fix_flips:
        for k, (a, b) in enumerate(H36M_BONES):
            if (a, b) not in FLIP_BONES:
                continue
            sub = FLIP_BONES[(a, b)]
            for t in range(n):
                if not (near[t] and VH[t, a] and VH[t, b]):
                    continue
                dz_s = SH[t, b, 2] - SH[t, a, 2]; dz_p = F[t, b, 2] - F[t, a, 2]
                sig = 2.0 * (pelvis[t, 2] ** 2) * 0.15 / args.fxb
                flip_checked[k] += 1
                if abs(dz_s) > max(args.flip_min_dz, sig) and abs(dz_p) > args.flip_min_dz and np.sign(dz_s) != np.sign(dz_p):
                    delta = -2.0 * dz_p
                    F[t, sub, 2] += delta; flips[k] += 1
    if args.fuse_weight > 0:
        w = args.fuse_weight
        for j in range(17):
            v = VH[:, j]
            F[v, j] = (1 - w) * F[v, j] + w * SH[v, j]
    diff3_f = np.linalg.norm(F - SH, axis=2); diffz_f = (F - SH)[..., 2]
    return dict(id=tr['id'], frames=fr[idx], dframes=fr[idx] // stride, pose_2d=tr['pose_2d'][idx], conf_2d=tr['conf_2d'][idx],
                stereo_coco=S, stereo_valid=V, zinfo=Zinfo, stereo_h36m=SH, stereo_h36m_valid=VH, vp3d_scaled=A, fused=F,
                scale=scale, zsign=zsign, pelvis_valid_frac=float(VH[:, 0].mean()), spurious=tr['spurious'], near=near,
                median_z=float(np.nanmedian(pelvis[:, 2])),
                diff3=diff3, diffz=diffz, diff3_f=diff3_f, diffz_f=diffz_f, cmp_mask=cmp_mask, flips=flips, flip_checked=flip_checked,
                bone_len_stereo=Ls, bone_len_vp3d=Lp * scale)


def summarize(results, near_max=5.0):
    out = dict(n_tracks=len(results), tracks=[])
    all_d3, all_dz, all_d3f, all_dzf, all_adz = [], [], [], [], []
    far_d3, far_d3f = [], []
    per_joint = {j: [] for j in COMPARE_JOINTS}; per_joint_f = {j: [] for j in COMPARE_JOINTS}
    valid_rate = np.zeros(17); nval = 0
    bl_s = {nm: [] for nm in H36M_BONE_NAMES}; bl_p = {nm: [] for nm in H36M_BONE_NAMES}
    flips = np.zeros(len(H36M_BONES), int); checked = np.zeros(len(H36M_BONES), int)
    shoulder_w, hip_w = [], []
    for r in results:
        m = r['cmp_mask'] & r['near'][:, None]           # the comparison is only meaningful inside the near range
        mf = r['cmp_mask'] & ~r['near'][:, None]
        far_d3.append(r['diff3'][mf]); far_d3f.append(r['diff3_f'][mf])
        all_d3.append(r['diff3'][m]); all_dz.append(r['diffz'][m]); all_adz.append(np.abs(r['diffz'][m]))
        all_d3f.append(r['diff3_f'][m]); all_dzf.append(r['diffz_f'][m])
        for j in COMPARE_JOINTS:
            per_joint[j].append(r['diff3'][m[:, j], j]); per_joint_f[j].append(r['diff3_f'][m[:, j], j])
        valid_rate += r['stereo_valid'].sum(0); nval += len(r['stereo_valid'])
        for k, nm in enumerate(H36M_BONE_NAMES):
            v = np.isfinite(r['bone_len_stereo'][:, k]); bl_s[nm].append(r['bone_len_stereo'][v, k]); bl_p[nm].append(r['bone_len_vp3d'][v, k])
        flips += r['flips']; checked += r['flip_checked']
        S, V = r['stereo_coco'], r['stereo_valid']
        v = V[:, 5] & V[:, 6]; shoulder_w.append(np.linalg.norm(S[v, 5] - S[v, 6], axis=1))
        v = V[:, 11] & V[:, 12]; hip_w.append(np.linalg.norm(S[v, 11] - S[v, 12], axis=1))
        out['tracks'].append(dict(id=r['id'], frames=int(len(r['frames'])), near_frames=int(r['near'].sum()), spurious=r['spurious'], scale=r['scale'],
                                  pelvis_valid_frac=r['pelvis_valid_frac'], median_z_m=r['median_z'],
                                  joint_diff3_median_cm=float(np.median(r['diff3'][m]) * 100) if m.any() else None,
                                  joint_diff3_fused_median_cm=float(np.median(r['diff3_f'][m]) * 100) if m.any() else None,
                                  flips=int(r['flips'].sum()), flip_checked=int(r['flip_checked'].sum())))
    cat = lambda L: np.concatenate(L) if L and sum(len(x) for x in L) else np.array([np.nan])
    d3, dz, d3f, dzf, adz = cat(all_d3), cat(all_dz), cat(all_d3f), cat(all_dzf), cat(all_adz)
    fd3, fd3f = cat(far_d3), cat(far_d3f)
    out['comparison_far_tracks'] = dict(n_joint_frames=int(np.isfinite(fd3).sum()), median_3d_cm=float(np.nanmedian(fd3) * 100), fused_median_3d_cm=float(np.nanmedian(fd3f) * 100),
                                        note='pelvis beyond near_max: stereo joint noise >= 0.3 m, numbers given for completeness only')
    out['comparison'] = dict(n_joint_frames=int(np.isfinite(d3).sum()), near_max_m=near_max,
                             vp3d_scaled_vs_stereo=dict(median_3d_cm=float(np.nanmedian(d3) * 100), p90_3d_cm=float(np.nanpercentile(d3, 90) * 100),
                                                        median_abs_z_cm=float(np.nanmedian(adz) * 100), mean_signed_z_cm=float(np.nanmean(dz) * 100)),
                             fused_vs_stereo=dict(median_3d_cm=float(np.nanmedian(d3f) * 100), p90_3d_cm=float(np.nanpercentile(d3f, 90) * 100),
                                                  median_abs_z_cm=float(np.nanmedian(np.abs(dzf)) * 100)),
                             per_joint_median_cm={H36M[j]: [float(np.nanmedian(cat(per_joint[j])) * 100), float(np.nanmedian(cat(per_joint_f[j])) * 100)] for j in COMPARE_JOINTS},
                             depth_flips={H36M_BONE_NAMES[k]: [int(flips[k]), int(checked[k])] for k in range(len(H36M_BONES)) if checked[k]},
                             depth_flip_rate=float(flips.sum() / max(checked.sum(), 1)))
    out['stereo_joint_valid_rate'] = {COCO[j]: float(valid_rate[j] / max(nval, 1)) for j in range(17)}
    bl = {}
    for nm in H36M_BONE_NAMES:
        s = cat(bl_s[nm]); p = cat(bl_p[nm])
        if np.isfinite(s).sum() > 20:
            bl[nm] = dict(stereo_median_m=float(np.nanmedian(s)), stereo_iqr_m=[float(np.nanpercentile(s, 25)), float(np.nanpercentile(s, 75))],
                          vp3d_scaled_median_m=float(np.nanmedian(p)), typical_m=TYPICAL_BONE_M.get(nm))
    sw, hw = cat(shoulder_w), cat(hip_w)
    bl['shoulder_width'] = dict(stereo_median_m=float(np.nanmedian(sw)), stereo_iqr_m=[float(np.nanpercentile(sw, 25)), float(np.nanpercentile(sw, 75))], typical_m=0.39)
    bl['hip_width'] = dict(stereo_median_m=float(np.nanmedian(hw)), stereo_iqr_m=[float(np.nanpercentile(hw, 25)), float(np.nanpercentile(hw, 75))], typical_m=0.20)
    out['bone_lengths'] = bl
    limbs = [v['stereo_median_m'] / v['typical_m'] for k, v in bl.items() if v.get('typical_m') and 'arm' in k or 'thigh' in k or 'shin' in k]
    out['anthropometric_scale_ratio'] = dict(limbs_median=float(np.median(limbs)) if limbs else None,
                                             shoulder=float(bl['shoulder_width']['stereo_median_m'] / 0.39),
                                             note='stereo bone length / typical adult length; 1.0 = scale consistent with d_inf; +-10% is normal human variation')
    return out


# ----------------------------------------------------------------------------- rendering
def turbo_z(z):
    if not np.isfinite(z) or z <= 0:
        return (90, 90, 90)
    t = 1.0 - (np.log(np.clip(z, ZMIN, ZMAX)) - np.log(ZMIN)) / (np.log(ZMAX) - np.log(ZMIN))
    c = cv2.applyColorMap(np.array([[int(t * 255)]], np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def draw_skeleton_2d(img, pts, zs, ok, bones, label=None, thick=2):
    for a, b in bones:
        if ok[a] and ok[b]:
            za = zs[a] if np.isfinite(zs[a]) else zs[b]
            cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), turbo_z(za), thick, cv2.LINE_AA)
    for j in range(len(pts)):
        if ok[j]:
            cv2.circle(img, tuple(np.round(pts[j]).astype(int)), 4, turbo_z(zs[j]), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(np.round(pts[j]).astype(int)), 4, (255, 255, 255), 1, cv2.LINE_AA)
    if label and ok.any():
        p = pts[ok].min(axis=0)
        cv2.putText(img, label, (int(p[0]), max(int(p[1]) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (int(p[0]), max(int(p[1]) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


class MetricView:
    """Orthographic metric canvas: top view (x right, z up on screen) or front view (x right, y down)."""

    def __init__(self, size, xr, yr, kind):
        self.w, self.h = size; self.xr, self.yr, self.kind = xr, yr, kind

    def to_px(self, P):
        x = (P[..., 0] - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w
        if self.kind == 'top':
            y = self.h - (P[..., 2] - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h
        else:
            y = (P[..., 1] - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h
        return np.stack([x, y], -1)

    def canvas(self, title):
        img = np.full((self.h, self.w, 3), 28, np.uint8)
        step = 1.0
        for gx in np.arange(np.ceil(self.xr[0]), self.xr[1] + 1e-6, step):
            x = int((gx - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w)
            cv2.line(img, (x, 0), (x, self.h), (60, 60, 60), 1)
            cv2.putText(img, f'{gx:g}', (x + 2, self.h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        for gy in np.arange(np.ceil(self.yr[0]), self.yr[1] + 1e-6, step):
            y = int(self.h - (gy - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h) if self.kind == 'top' else int((gy - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h)
            cv2.line(img, (0, y), (self.w, y), (60, 60, 60), 1)
            cv2.putText(img, f'{gy:g}m', (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        if self.kind == 'top':
            cx = int((0 - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w)
            cv2.drawMarker(img, (cx, self.h - 1), (0, 200, 255), cv2.MARKER_TRIANGLE_UP, 14, 2)   # camera
        cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def draw(self, img, P, ok, bones, colour, hollow=False, thick=2):
        pts = self.to_px(P)
        for a, b in bones:
            if ok[a] and ok[b]:
                cv2.line(img, tuple(np.round(pts[a]).astype(int)), tuple(np.round(pts[b]).astype(int)), colour, thick, cv2.LINE_AA)
        for j in range(len(P)):
            if ok[j]:
                cv2.circle(img, tuple(np.round(pts[j]).astype(int)), 4, colour, 1 if hollow else -1, cv2.LINE_AA)


def render_videos(results, args, K, stride, fps, names_depth, vdir, left_path_of):
    # frame -> list of (result, row)
    by_frame = {}
    for r in results:
        for row, f in enumerate(r['frames']):
            by_frame.setdefault(int(f), []).append((r, row))
    frames = sorted(by_frame)
    if not frames:
        return
    f0, f1 = frames[0], frames[-1]
    all_frames = list(range(f0, f1 + 1, stride))
    s = args.preview_scale; pw, ph = int(1920 * s), int(1080 * s); ph -= ph % 2
    bar = colour_bar(ph, ZMIN, ZMAX)[:ph]
    top = MetricView((ph, ph), (-4, 4), (0, 8), 'top'); front = MetricView((ph, ph), (-3, 3), (-2, 2), 'front')
    wc = pw * 2 + ph + bar.shape[1]; wc -= wc % 2
    wf = pw + ph * 2 + bar.shape[1]; wf -= wf % 2
    w_cmp = ffmpeg_writer(f'{vdir}/pose_compare.mp4', wc, ph, fps, 'bgr24', ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart'], args.ffmpeg)
    w_fus = ffmpeg_writer(f'{vdir}/pose_fused.mp4', wf, ph, fps, 'bgr24', ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart'], args.ffmpeg)
    t0 = time.perf_counter()
    for n, f in enumerate(all_frames):
        di = f // stride
        left = cv2.imread(left_path_of(di))
        depth = np.load(names_depth(di)) if os.path.exists(names_depth(di)) else None
        if left is None or depth is None:
            continue
        left_s = cv2.resize(left, (pw, ph), interpolation=cv2.INTER_AREA)
        dep_s = cv2.resize(depth_to_colour(depth, ZMIN, ZMAX), (pw, ph), interpolation=cv2.INTER_NEAREST)
        topc = top.canvas('top view (m): stereo filled, VideoPose3D scaled hollow'); frontc = front.canvas('front view (m): adjusted VideoPose3D')
        topf = top.canvas('top view (m): adjusted VideoPose3D')
        fus_img = left_s.copy()
        texts = []
        for r, row in by_frame.get(f, []):
            p2 = r['pose_2d'][row] * s; ok2 = np.isfinite(p2).all(1) & (r['conf_2d'][row] >= args.min_conf)
            zs = np.where(r['stereo_valid'][row], r['stereo_coco'][row, :, 2], np.nan)
            lbl = f"T{r['id']} {r['stereo_h36m'][row, 0, 2]:.2f}m" if r['stereo_h36m_valid'][row, 0] else f"T{r['id']}"
            draw_skeleton_2d(left_s, p2, zs, ok2, COCO_BONES, lbl)
            draw_skeleton_2d(dep_s, p2, zs, ok2, COCO_BONES)
            SH, VH, A, F = r['stereo_h36m'][row], r['stereo_h36m_valid'][row], r['vp3d_scaled'][row], r['fused'][row]
            okA = np.isfinite(A).all(1)
            top.draw(topc, A, okA, H36M_BONES, (200, 200, 200), hollow=True, thick=1)
            top.draw(topc, SH, VH, H36M_BONES, turbo_z(SH[0, 2]) if VH[0] else (255, 255, 255))
            m = r['cmp_mask'][row]
            if m.any():
                texts.append(f"T{r['id']}: |VP3D-stereo| med {np.median(r['diff3'][row][m])*100:.0f} cm, adj {np.median(r['diff3_f'][row][m])*100:.0f} cm")
            # fused video: reprojection + views
            okF = np.isfinite(F).all(1)
            pf = project(F, K) * s
            zf = F[:, 2]
            draw_skeleton_2d(fus_img, pf, zf, okF, H36M_BONES, f"T{r['id']} {F[0, 2]:.2f}m")
            front.draw(frontc, F, okF, H36M_BONES, turbo_z(F[0, 2]))
            top.draw(topf, F, okF, H36M_BONES, turbo_z(F[0, 2]))
        for k, tx in enumerate(texts[:6]):
            cv2.putText(topc, tx, (6, 34 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(left_s, f'frame {f}  t={f / fps:6.2f}s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(fus_img, f'frame {f}  adjusted VideoPose3D (scale+anchor{"+flip fix" if args.fix_flips else ""})', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.concatenate([left_s, dep_s, topc, bar], axis=1)[:, :wc]
        w_cmp.stdin.write(np.ascontiguousarray(panel).tobytes())
        panel2 = np.concatenate([fus_img, frontc, topf, bar], axis=1)[:, :wf]
        w_fus.stdin.write(np.ascontiguousarray(panel2).tobytes())
        if (n + 1) % 100 == 0:
            print(f'  render {n + 1}/{len(all_frames)} ({(n + 1) / (time.perf_counter() - t0):.1f} fps)', file=sys.stderr, flush=True)
    for w in (w_cmp, w_fus):
        w.stdin.close(); w.wait(); assert w.returncode == 0, 'ffmpeg failed'


def write_fpa_package(results, tracks_npz, arrays, meta, vdir, vid):
    """Copy of the tracks package with pose_3d = fused metric H36M skeleton (frames without depth: interpolated)."""
    arrays = dict(arrays)
    for r in results:
        p = f"t{r['id']}_"
        full_frames = arrays[p + 'frames']; n = len(full_frames)
        out = np.zeros((n, 17, 3), np.float32)
        pos = np.searchsorted(full_frames, r['frames'])
        for j in range(17):
            for c in range(3):
                src = r['fused'][:, j, c]; ok = np.isfinite(src)
                out[:, j, c] = np.interp(np.arange(n), pos[ok], src[ok]) if ok.sum() else 0
        arrays[p + 'pose_3d'] = out
    npz_path = f'{vdir}/{vid}_fused.tracks.npz'
    np.savez_compressed(npz_path, **arrays)
    meta = dict(meta)
    meta['package_sha256'] = hashlib.sha256(open(npz_path, 'rb').read()).hexdigest()
    meta['package_file_name'] = os.path.basename(npz_path)
    meta['pose_3d_origin'] = 'camera'; meta['pose_3d_units'] = 'metres'
    meta['pose_3d_coordinate_description'] = ('VideoPose3D scaled by stereo bone lengths, anchored at the stereo pelvis, limb depth flips corrected '
                                              'with Fast-FoundationStereo depth (iPhone spatial video); left-eye camera frame, x right, y down, z forward, metres. '
                                              'Tracks without a usable stereo pelvis keep zeros.')
    meta['display_transform'] = [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    json.dump(meta, open(f'{vdir}/{vid}_fused.tracks.json', 'w'), indent=2)
    return npz_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True); ap.add_argument('--seq_dir', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--video_dir', default=None)
    ap.add_argument('--radius', type=int, default=3, help='depth sampling window radius (px)')
    ap.add_argument('--depth_pct', type=float, default=30.0, help='percentile of the window used as the joint depth (see sample_depth)')
    ap.add_argument('--disp_offset', type=float, default=0.0, help='extra disparity offset (px) applied to every sampled depth: z = fxb/(fxb/z + off)')
    ap.add_argument('--min_conf', type=float, default=0.3); ap.add_argument('--min_valid_frac', type=float, default=0.3)
    ap.add_argument('--max_mad', type=float, default=0.25, help='max depth MAD inside the window (m); larger = joint on a depth edge')
    ap.add_argument('--min_frames', type=int, default=30, help='ignore tracks with fewer depth frames')
    ap.add_argument('--pelvis_smooth', type=int, default=5)
    ap.add_argument('--fix_flips', type=int, default=1); ap.add_argument('--flip_min_dz', type=float, default=0.08)
    ap.add_argument('--near_max', type=float, default=5.0, help='pelvis depth (m) up to which stereo joints are trusted for comparison / flip fix')
    ap.add_argument('--fuse_weight', type=float, default=0.0, help='0..1 blend of stereo joints into the adjusted VideoPose3D')
    ap.add_argument('--preview_scale', type=float, default=0.5); ap.add_argument('--crf', type=int, default=20); ap.add_argument('--ffmpeg', default='ffmpeg')
    ap.add_argument('--no_video', action='store_true'); ap.add_argument('--write_fpa_package', type=int, default=1)
    args = ap.parse_args()

    meta_seq = json.load(open(f'{args.seq_dir}/meta.json')); stride = int(meta_seq.get('stride', 1)); fps = meta_seq['fps']
    K, baseline = read_k(f'{args.seq_dir}/K.txt'); args.fxb = float(K[0, 0] * baseline)
    tracks, tmeta, arrays = load_tracks(args.tracks)
    vdir = args.video_dir or os.path.dirname(args.tracks); os.makedirs(vdir, exist_ok=True)
    vid = os.path.basename(args.tracks)[:-len('.tracks.npz')]
    names_depth = lambda i: f'{args.out_dir}/depth/{i:06d}.npy'
    left_path_of = lambda i: f'{args.seq_dir}/left/{i:06d}.png'
    cache = {}

    def depth_of(i):
        if i not in cache:
            cache.clear()
            cache[i] = np.load(names_depth(i)) if os.path.exists(names_depth(i)) else None
        return cache[i]

    t0 = time.perf_counter(); results = []
    # process tracks in frame order so the depth cache is reused
    for tr in sorted(tracks, key=lambda t: t['frames'][0]):
        r = process_track(tr, depth_of, K, stride, args)
        if r is not None:
            results.append(r)
            mm = r['cmp_mask'] & r['near'][:, None]
            print(f"track {r['id']}: {len(r['frames'])} frames ({r['near'].sum()} within {args.near_max} m, median z {r['median_z']:.1f} m), scale {r['scale']:.3f}, pelvis valid {r['pelvis_valid_frac']*100:.0f}%, "
                  f"near |VP3D-stereo| median {np.nanmedian(r['diff3'][mm])*100 if mm.any() else float('nan'):.1f} cm -> adjusted {np.nanmedian(r['diff3_f'][mm])*100 if mm.any() else float('nan'):.1f} cm, flips {r['flips'].sum()}/{r['flip_checked'].sum()}", flush=True)
    summary = summarize(results, args.near_max)
    summary.update(video=vid, stride=stride, n_tracks_in_package=len(tracks), args=vars(args), seconds_processing=time.perf_counter() - t0)
    print(json.dumps({k: v for k, v in summary.items() if k in ('comparison', 'bone_lengths', 'anthropometric_scale_ratio', 'stereo_joint_valid_rate')}, indent=1))
    np.savez_compressed(f'{vdir}/{vid}.pose_depth.npz', **{f"t{r['id']}_{k}": r[k] for r in results for k in
                        ('frames', 'dframes', 'stereo_coco', 'stereo_valid', 'stereo_h36m', 'stereo_h36m_valid', 'vp3d_scaled', 'fused', 'zinfo')},
                        track_ids=np.array([r['id'] for r in results]))
    if args.write_fpa_package and results:
        write_fpa_package(results, args.tracks, arrays, tmeta, vdir, vid)
    if not args.no_video and results:
        render_videos(results, args, K, stride, fps, names_depth, vdir, left_path_of)
    summary['seconds_total'] = time.perf_counter() - t0
    json.dump(summary, open(f'{vdir}/{vid}.pose_depth.json', 'w'), indent=2)
    print(f'done -> {vdir}')


if __name__ == '__main__':
    main()
