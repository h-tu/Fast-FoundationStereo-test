#!/usr/bin/env python3
"""Ground-truth-free evaluation of a run_sequence.py output (plan doc §11).

Tests (all without external truth):
  coverage   fraction of pixels with valid depth in [zmin, zmax]
  flicker    temporal noise of disparity: |d_t - median(d_{t-1}, d_t, d_{t+1})| on consecutive
             full-rate frames (pixels with d > 2 px); reported in px and as relative depth error
  plane      RANSAC plane fit to the road ROI (bottom rows, central columns): RMS residual of the
             inliers (metric flatness), camera height above the road and tilt. A hand-held phone
             should give a plausible height (~1.2-1.7 m); the height scales 1:1 with the d_inf choice.
  blockmatch model disparity vs. independent ZNCC block matching on textured near patches
             (NCC >= 0.9, 3 <= d_bm <= 40 px): bias and spread of (d_model - (d_bm - dinf))
  lr         (--lr_frames N, needs CUDA) left-right consistency: the model is run again on the
             horizontally flipped (right, left) pair to get the right-image disparity, then
             |d_L(x) - d_R(x - d_L(x))| is measured; fraction < 1 px is a per-pixel reliability score.

Usage
  python scripts/spatial/eval_depth.py --seq_dir work/3143_full --out_dir work/3143_full/out --json work/3143_full/eval.json [--lr_frames 10]
"""
import argparse, glob, json, os, sys, time
import cv2
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir); sys.path.append(f'{code_dir}/../../')
from check_rectification import match_patches          # noqa: E402
from run_sequence import fix_right, read_k             # noqa: E402


def load_run(out_dir):
    meta = json.load(open(f'{out_dir}/run_meta.json'))
    names = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(f'{out_dir}/disp/*.npy'))]
    return meta, names


def affine_for(meta, name):
    """(a, b) that run_sequence used for this frame: the latest auto_affine fit at or before it."""
    fits = meta.get('affine_per_frame', {})
    a, b = meta.get('right_affine', (0.0, 0.0))
    for k in sorted(fits):
        if k <= name:
            a, b = fits[k][0], fits[k][1]
    return a, b


def coverage(out_dir, names, zmin, zmax, every):
    fr = []
    for n in names[::every]:
        z = np.load(f'{out_dir}/depth/{n}.npy')
        fr.append(((z >= zmin) & (z <= zmax)).mean())
    return dict(frames=len(fr), mean=float(np.mean(fr)), min=float(np.min(fr)), max=float(np.max(fr)))


def flicker(out_dir, names, fxb, n_windows=60):
    if len(names) < 3:
        return dict(error='need >= 3 consecutive frames')
    if len(names) < 5:
        return dict(error='need >= 5 consecutive frames')
    idx = np.linspace(2, len(names) - 3, min(n_windows, len(names) - 4)).astype(int)
    px, rel, px_f = [], [], []
    for i in idx:
        d = [np.load(f'{out_dir}/disp/{names[j]}.npy') for j in range(i - 2, i + 3)]   # 5 consecutive frames
        med = np.median(np.stack(d[1:4]), axis=0)                                       # med(t-1, t, t+1)
        m = (d[2] > 2) & (med > 2)
        dev = np.abs(d[2] - med)[m]
        px.append(dev); rel.append(dev / med[m])
        # same statistic after a 3-frame running median (what --temporal_median 3 would output)
        f = [np.median(np.stack(d[k:k + 3]), axis=0) for k in range(3)]                # filtered t-1, t, t+1
        medf = np.median(np.stack(f), axis=0)
        px_f.append(np.abs(f[1] - medf)[m])
    px = np.concatenate(px); rel = np.concatenate(rel); px_f = np.concatenate(px_f)
    return dict(windows=int(len(idx)), mean_px=float(px.mean()), p90_px=float(np.percentile(px, 90)),
                p99_px=float(np.percentile(px, 99)), frac_gt_0p5px=float((px > 0.5).mean()),
                p90_rel_depth=float(np.percentile(rel, 90)), p99_rel_depth=float(np.percentile(rel, 99)),
                after_median3=dict(mean_px=float(px_f.mean()), p90_px=float(np.percentile(px_f, 90)),
                                   p99_px=float(np.percentile(px_f, 99)), frac_gt_0p5px=float((px_f > 0.5).mean())))


def ransac_plane(P, thr=0.05, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    best = (None, 0)
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn; c = -n @ s[0]
        inl = np.abs(P @ n + c) < thr
        if inl.sum() > best[1]:
            best = ((n, c), inl.sum())
    (n, c), _ = best
    inl = np.abs(P @ n + c) < thr
    # least-squares refit on inliers
    Q = P[inl]; cen = Q.mean(0); u, s_, vt = np.linalg.svd(Q - cen, full_matrices=False)
    n = vt[-1]; n = n / np.linalg.norm(n); c = -n @ cen
    if n[1] < 0:            # make the normal point "up" in camera coords (y down) -> n_y negative; keep sign convention: height positive
        n, c = -n, -c
    res = (P @ n + c)
    inl = np.abs(res) < thr
    return n, c, inl, res


def plane_fit(out_dir, seq_dir, names, K, every, zmin=1.0, zmax=8.0):
    H = W = None; rows = []
    for n in names[::every]:
        z = np.load(f'{out_dir}/depth/{n}.npy'); H, W = z.shape
        ys, xs = np.mgrid[int(0.72 * H):H, int(0.2 * W):int(0.8 * W)]
        zz = z[ys, xs]; m = (zz > zmin) & (zz < zmax)
        if m.sum() < 5000:
            continue
        xs, ys, zz = xs[m][::7], ys[m][::7], zz[m][::7]
        X = (xs - K[0, 2]) * zz / K[0, 0]; Y = (ys - K[1, 2]) * zz / K[1, 1]
        P = np.stack([X, Y, zz], 1).astype(np.float64)
        nrm, c, inl, res = ransac_plane(P)
        height = abs(c)                                   # distance from camera centre to the plane
        tilt = float(np.degrees(np.arccos(abs(nrm[1]))))  # angle between plane normal and camera y axis
        rows.append(dict(frame=n, n_pts=int(len(P)), inlier_frac=float(inl.mean()),
                         rms_cm=float(np.sqrt((res[inl] ** 2).mean()) * 100), height_m=float(height), tilt_deg=tilt,
                         z_range=[float(zz[inl].min()), float(zz[inl].max())]))
    if not rows:
        return dict(error='no road ROI with enough valid depth')
    h = np.array([r['height_m'] for r in rows]); rms = np.array([r['rms_cm'] for r in rows]); inl = np.array([r['inlier_frac'] for r in rows])
    return dict(frames=len(rows), height_m_median=float(np.median(h)), height_m_iqr=[float(x) for x in np.percentile(h, [25, 75])],
                rms_cm_median=float(np.median(rms)), rms_cm_p90=float(np.percentile(rms, 90)), inlier_frac_median=float(np.median(inl)),
                tilt_deg_median=float(np.median([r['tilt_deg'] for r in rows])), per_frame=rows)


def blockmatch_agreement(out_dir, seq_dir, meta, names, every):
    dinf = float(meta.get('dinf', 0.0)); diffs = []; per = []
    for n in names[::every]:
        L = cv2.imread(f'{seq_dir}/left/{n}.png', 0); R = cv2.imread(f'{seq_dir}/right/{n}.png', 0)
        if L is None or R is None:
            continue
        L = L.astype(np.float32); R = R.astype(np.float32)
        r = match_patches(L, R, P=32, step=32, DY=6, DXMAX=48, min_ncc=0.9)
        r = r[(r[:, 2] >= 3) & (r[:, 2] <= 40)]
        if len(r) < 20:
            continue
        d = np.load(f'{out_dir}/disp/{n}.npy')
        xs, ys = r[:, 0].astype(int), r[:, 1].astype(int)
        dm = np.array([np.median(d[y - 4:y + 5, x - 4:x + 5]) for x, y in zip(xs, ys)])
        diff = dm - (r[:, 2] - dinf)
        diffs.append(diff); per.append(dict(frame=n, n=int(len(r)), median=float(np.median(diff)), mad=float(np.median(np.abs(diff - np.median(diff))))))
    if not diffs:
        return dict(error='no textured near patches')
    diff = np.concatenate(diffs)
    return dict(frames=len(per), n_patches=int(len(diff)), bias_px=float(np.median(diff)), mean_px=float(diff.mean()),
                mad_px=float(np.median(np.abs(diff - np.median(diff)))), p90_abs_px=float(np.percentile(np.abs(diff), 90)),
                frac_abs_gt_1px=float((np.abs(diff) > 1).mean()), per_frame=per)


def lr_consistency(out_dir, seq_dir, meta, names, n_frames):
    import torch
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE
    torch.autograd.set_grad_enabled(False)
    model = torch.load(meta['model'], map_location='cpu', weights_only=False)
    model.args.valid_iters = meta['valid_iters']; model.args.max_disp = meta['max_disp']; model.cuda().eval()
    dinf = float(meta.get('dinf', 0.0)); vol = meta.get('volume', 'pytorch1')
    sel = [names[i] for i in np.linspace(0, len(names) - 1, n_frames).astype(int)]
    rows = []
    for n in sel:
        a, b = affine_for(meta, n)
        img0 = cv2.cvtColor(cv2.imread(f'{seq_dir}/left/{n}.png'), cv2.COLOR_BGR2RGB)
        img1 = fix_right(cv2.cvtColor(cv2.imread(f'{seq_dir}/right/{n}.png'), cv2.COLOR_BGR2RGB), a, b, dinf)
        # right-image disparity: run on the flipped pair (right, left)
        fl = np.ascontiguousarray(img1[:, ::-1]); fr = np.ascontiguousarray(img0[:, ::-1])
        t_l = torch.as_tensor(fl).cuda().float()[None].permute(0, 3, 1, 2); t_r = torch.as_tensor(fr).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t_l.shape, divis_by=32, force_square=False); t_l, t_r = padder.pad(t_l, t_r)
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            out = model.forward(t_l, t_r, iters=meta['valid_iters'], test_mode=True, optimize_build_volume=vol)
        dR = padder.unpad(out.float()).cpu().numpy().reshape(img0.shape[0], img0.shape[1])[:, ::-1].clip(0, None)
        dL = np.load(f'{out_dir}/disp/{n}.npy')
        H, W = dL.shape; xs = np.arange(W)[None, :].repeat(H, 0).astype(np.float32); ys = np.arange(H)[:, None].repeat(W, 1).astype(np.float32)
        xr = xs - dL; valid = (xr >= 0) & (dL > 2)
        dRw = cv2.remap(dR.astype(np.float32), xr, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        err = np.abs(dL - dRw); m = valid & (dRw >= 0)
        e = err[m]
        near = m & (dL > 6)
        rows.append(dict(frame=n, valid_frac=float(m.mean()), median_px=float(np.median(e)), p90_px=float(np.percentile(e, 90)),
                         frac_lt_1px=float((e < 1).mean()), frac_lt_0p5px=float((e < 0.5).mean()),
                         near_frac_lt_1px=float((err[near] < 1).mean()) if near.any() else None))
        print(f'  lr {n}: median {rows[-1]["median_px"]:.2f} px, <1px {rows[-1]["frac_lt_1px"]*100:.1f}%  (near <1px {100*(rows[-1]["near_frac_lt_1px"] or 0):.1f}%)', flush=True)
    return dict(frames=len(rows), median_px=float(np.median([r['median_px'] for r in rows])),
                frac_lt_1px_mean=float(np.mean([r['frac_lt_1px'] for r in rows])),
                frac_lt_0p5px_mean=float(np.mean([r['frac_lt_0p5px'] for r in rows])),
                near_frac_lt_1px_mean=float(np.mean([r['near_frac_lt_1px'] for r in rows if r['near_frac_lt_1px'] is not None])), per_frame=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_dir', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--every', type=int, default=30, help='frame step for coverage / plane / block matching')
    ap.add_argument('--zmin', type=float, default=0.3); ap.add_argument('--zmax', type=float, default=8.0)
    ap.add_argument('--lr_frames', type=int, default=0, help='number of frames for the GPU left-right check (0 = skip)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    meta, names = load_run(args.out_dir)
    K, baseline = read_k(f'{args.seq_dir}/K.txt'); fxb = K[0, 0] * baseline
    t0 = time.perf_counter()
    res = dict(seq_dir=args.seq_dir, out_dir=args.out_dir, n_frames=len(names), dinf=meta.get('dinf', 0.0))
    res['coverage'] = coverage(args.out_dir, names, args.zmin, args.zmax, args.every); print('coverage', res['coverage'])
    res['flicker'] = flicker(args.out_dir, names, fxb); print('flicker', {k: v for k, v in res['flicker'].items()})
    res['plane'] = plane_fit(args.out_dir, args.seq_dir, names, K, args.every); print('plane', {k: v for k, v in res['plane'].items() if k != 'per_frame'})
    res['blockmatch'] = blockmatch_agreement(args.out_dir, args.seq_dir, meta, names, args.every); print('blockmatch', {k: v for k, v in res['blockmatch'].items() if k != 'per_frame'})
    if args.lr_frames:
        res['lr'] = lr_consistency(args.out_dir, args.seq_dir, meta, names, args.lr_frames); print('lr', {k: v for k, v in res['lr'].items() if k != 'per_frame'})
    res['seconds'] = time.perf_counter() - t0
    if args.json:
        json.dump(res, open(args.json, 'w'), indent=2)


if __name__ == '__main__':
    main()
