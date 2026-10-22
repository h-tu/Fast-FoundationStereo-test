#!/usr/bin/env python3
"""Helmet ArUco markers as ground truth for the stereo depth (plan doc §13.3).

The workers wear field-pose-annotator helmet markers (ArUco 4x4_50, printed 88.9 mm; util/aruco in that repo).
For every marker read in the left eye, OpenCV solvePnP (IPPE square) with the same K as the stereo depth gives the
marker centre's depth z_marker.  The stereo depth is read on the marker's own surface (median disparity inside the
marker quad shrunk to --inner of its size), z_stereo = fx*b / d_model.

Why this is a clean test: z_marker ~ fx (fixed pixel size) and z_stereo ~ fx, so their ratio does not depend on the
field of view; it tests the baseline and the disparity offset only.  The disparity the marker predicts is
d_marker = fx*b / z_marker, and the model disparity follows  d_model = g * d_marker + c  where
g = b_true / b_metadata (disparity gain) and c = d_inf_true - d_used (d_used = run_sequence --dinf), fitted robustly.

Outputs (in --video_dir, default <seq_dir>/aruco):
  aruco_compare.mp4   left eye with each read outlined and labelled "W01 marker 3.42 m | stereo 3.55 m +3.8 %" |
                      stereo depth map with the same outlines | live scatter z_stereo vs z_marker with running stats
  aruco_reads.csv     every read: frame, time, id, worker, side, px size, marker x/y/z, stereo z, disparities, valid share
  aruco_compare.json  error statistics overall and per distance bin, gain/offset fit, settings

Needs: ffmpeg (MV-HEVC) on PATH, OpenCV with aruco (opencv-contrib), numpy; field-pose-annotator for the marker scheme.
Example:
  python scripts/spatial/aruco_compare.py --video vid/IMG_3354.MOV --seq_dir work/3354_full --out_dir work/3354_full/out
"""
import argparse, csv, json, math, os, subprocess, sys, time
import cv2
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir)
from make_depth_video import colour_bar, depth_to_colour, ffmpeg_writer  # noqa: E402
from run_sequence import read_k  # noqa: E402

ZMIN, ZMAX = 0.3, 8.0
BINS = [(0.5, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 12)]


def marker_pose(corners, K, size_m):
    """Translation of the marker centre (camera frame) from its 4 corners; lowest-reprojection IPPE solution."""
    h = size_m / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)   # ArUco corner order TL, TR, BR, BL
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners.astype(np.float32), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        n = 0
    if not n:
        ok, rv, tv = cv2.solvePnP(obj, corners.astype(np.float32), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        return (tv.reshape(3), 0.0) if ok else (None, None)
    k = int(np.argmin(np.asarray(errs).ravel()))
    return np.asarray(tvecs[k]).reshape(3), float(np.asarray(errs).ravel()[k])


def stereo_on_marker(disp, corners, inner):
    c = corners.mean(axis=0)
    quad = (c + inner * (corners - c)).astype(np.int32)
    x0, y0 = np.maximum(quad.min(0), 0); x1, y1 = np.minimum(quad.max(0) + 1, [disp.shape[1], disp.shape[0]])
    if x1 <= x0 or y1 <= y0:
        return np.nan, 0.0, 0
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(mask, quad - [x0, y0], 1)
    vals = disp[y0:y1, x0:x1][mask.astype(bool)]
    if vals.size == 0:
        return np.nan, 0.0, 0
    ok = vals > 0.3
    return (float(np.median(vals[ok])) if ok.sum() else np.nan), float(ok.mean()), int(vals.size)


def robust_line(x, y, iters=5, k=3.0):
    """y = g x + c with iterative MAD outlier rejection; returns g, c, inlier mask."""
    m = np.isfinite(x) & np.isfinite(y)
    for _ in range(iters):
        if m.sum() < 5:
            return np.nan, np.nan, m
        g, c = np.polyfit(x[m], y[m], 1)
        r = y - (g * x + c); s = 1.4826 * np.median(np.abs(r[m] - np.median(r[m]))) + 1e-6
        m = np.isfinite(r) & (np.abs(r) < k * s)
    g, c = np.polyfit(x[m], y[m], 1)
    return float(g), float(c), m


def stats_of(rows):
    if not rows:
        return {}
    zm = np.array([r['z_marker'] for r in rows]); zs = np.array([r['z_stereo'] for r in rows])
    rel = zs / zm - 1.0
    out = dict(n=len(rows), median_rel_err=float(np.median(rel)), mad_rel_err=float(np.median(np.abs(rel - np.median(rel)))),
               median_abs_rel_err=float(np.median(np.abs(rel))), p90_abs_rel_err=float(np.percentile(np.abs(rel), 90)),
               median_abs_err_m=float(np.median(np.abs(zs - zm))))
    out['by_distance'] = {}
    for lo, hi in BINS:
        sel = (zm >= lo) & (zm < hi)
        if sel.sum() >= 5:
            out['by_distance'][f'{lo}-{hi} m'] = dict(n=int(sel.sum()), median_rel_err=float(np.median(rel[sel])),
                                                       median_abs_rel_err=float(np.median(np.abs(rel[sel]))),
                                                       median_abs_err_m=float(np.median(np.abs(zs[sel] - zm[sel]))))
    return out


class Scatter:
    def __init__(self, size, zr=(0, 8)):
        self.w = self.h = size; self.zr = zr

    def px(self, zm, zs):
        f = lambda v: (v - self.zr[0]) / (self.zr[1] - self.zr[0])
        return int(40 + f(zm) * (self.w - 50)), int(self.h - 30 - f(zs) * (self.h - 50))

    def draw(self, pts, cur, text):
        img = np.full((self.h, self.w, 3), 28, np.uint8)
        for z in range(int(self.zr[0]), int(self.zr[1]) + 1):
            x, _ = self.px(z, 0); _, y = self.px(0, z)
            cv2.line(img, (x, 20), (x, self.h - 30), (55, 55, 55), 1); cv2.line(img, (40, y), (self.w - 10, y), (55, 55, 55), 1)
            cv2.putText(img, f'{z}', (x - 4, self.h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(img, f'{z}', (18, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.line(img, self.px(0, 0), self.px(8, 8), (0, 200, 255), 1, cv2.LINE_AA)
        for lo in (0.9, 1.1):
            cv2.line(img, self.px(0, 0), self.px(8, 8 * lo), (0, 120, 160), 1, cv2.LINE_AA)
        for zm, zs in pts:
            cv2.circle(img, self.px(zm, zs), 2, (170, 170, 170), -1)
        for zm, zs, col in cur:
            cv2.circle(img, self.px(zm, zs), 6, col, 2, cv2.LINE_AA)
        cv2.putText(img, 'stereo z (m) vs marker z (m); lines: 1:1 and +-10 %', (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        for k, t in enumerate(text):
            cv2.putText(img, t, (48, 34 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return img


def err_colour(rel):
    a = min(abs(rel) / 0.2, 1.0)            # green at 0, red at >= 20 %
    return (0, int(255 * (1 - a)), int(255 * a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--seq_dir', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--video_dir', default=None)
    ap.add_argument('--size_mm', type=float, default=88.9, help='printed marker side (field-pose-annotator make_sheets default)')
    ap.add_argument('--max_id', type=int, default=48, help='ids >= this are misreads (12 workers x 4 codes)')
    ap.add_argument('--inner', type=float, default=0.6, help='share of the marker quad used for the stereo sample')
    ap.add_argument('--min_valid', type=float, default=0.6)
    ap.add_argument('--fpa_aruco', default=os.path.expanduser('~/Documents/GitHub/field-pose-annotator/util/aruco'))
    ap.add_argument('--preview_scale', type=float, default=0.5); ap.add_argument('--crf', type=int, default=21)
    ap.add_argument('--no_video', action='store_true'); ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    sys.path.insert(0, a.fpa_aruco)
    import markers as mk   # field-pose-annotator marker scheme (detector settings, id -> worker/side)
    fam = mk.FAMILIES['aruco4x4']; det = mk.detector(fam, (1920, 1080))
    meta = json.load(open(f'{a.seq_dir}/meta.json')); stride = int(meta.get('stride', 1)); fps = meta['fps']
    rm = json.load(open(f'{a.out_dir}/run_meta.json')); d_used = float(rm.get('dinf', 0.0))
    K, b = read_k(f'{a.seq_dir}/K.txt'); fxb = float(K[0, 0] * b); size_m = a.size_mm / 1000.0
    vdir = a.video_dir or f'{a.seq_dir}/aruco'; os.makedirs(vdir, exist_ok=True)
    W, H = int(meta['width']), int(meta['height'])
    vf = ['-vf', f'select=not(mod(n\\,{stride}))', '-fps_mode', 'vfr'] if stride > 1 else []
    dec = subprocess.Popen([a.ffmpeg, '-v', 'error', '-i', a.video, '-map', '0:v:0:vpos:left', *vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    s = a.preview_scale; pw, ph = int(W * s), int(H * s); ph -= ph % 2
    bar = colour_bar(ph, ZMIN, ZMAX)[:ph]; sc = Scatter(ph)
    wv = pw * 2 + ph + bar.shape[1]; wv -= wv % 2
    writer = None if a.no_video else ffmpeg_writer(f'{vdir}/aruco_compare.mp4', wv, ph, fps / stride, 'bgr24',
                                                   ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(a.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart'], a.ffmpeg)
    rows, pts = [], []
    n_frames = int(meta['n_frames_extracted']); t0 = time.perf_counter(); i = 0
    while i < n_frames:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        dp = f'{a.out_dir}/disp/{i:06d}.npy'
        disp = np.load(dp) if os.path.exists(dp) else None
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        cur = []; frame_reads = []
        if ids is not None and disp is not None:
            for cc, mid in zip(corners, ids.ravel()):
                if mid >= a.max_id:
                    continue
                c4 = cc.reshape(4, 2)
                side_px = float(np.mean(np.linalg.norm(c4 - np.roll(c4, 1, 0), axis=1)))
                if side_px < mk.MIN_SIDE_PX:
                    continue
                t, reproj = marker_pose(c4, K, size_m)
                if t is None or not np.isfinite(t).all() or t[2] <= 0.2:
                    continue
                d_s, valid, npx = stereo_on_marker(disp, c4, a.inner)
                worker, side = mk.decode(int(mid))
                r = dict(frame=i, src_frame=i * stride, t_s=i * stride / fps, id=int(mid), worker=mk.worker_label(worker), side=side,
                         side_px=side_px, reproj_px=reproj, x_marker=float(t[0]), y_marker=float(t[1]), z_marker=float(t[2]),
                         d_marker=fxb / float(t[2]), d_model=d_s, valid=valid, n_px=npx,
                         z_stereo=(fxb / d_s) if np.isfinite(d_s) and d_s > 0 else np.nan)
                frame_reads.append((r, c4))
                if np.isfinite(r['z_stereo']) and valid >= a.min_valid:
                    rows.append(r); pts.append((r['z_marker'], r['z_stereo']))
        if writer is not None:
            left_s = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            dep_s = cv2.resize(depth_to_colour(np.where(disp > 0, fxb / np.maximum(disp, 1e-6), 0) if disp is not None else np.zeros((H, W)), ZMIN, ZMAX),
                               (pw, ph), interpolation=cv2.INTER_NEAREST)
            for r, c4 in frame_reads:
                good = np.isfinite(r['z_stereo']) and r['valid'] >= a.min_valid
                rel = r['z_stereo'] / r['z_marker'] - 1 if good else np.nan
                col = err_colour(rel) if good else (160, 160, 160)
                q = (c4 * s).astype(np.int32)
                for im_ in (left_s, dep_s):
                    cv2.polylines(im_, [q], True, col, 2, cv2.LINE_AA)
                txt = f"{r['worker']} {r['side'][0]}  marker {r['z_marker']:.2f} m | stereo " + (f"{r['z_stereo']:.2f} m {rel * 100:+.1f}%" if good else 'n/a')
                p = (int(q[:, 0].min()), max(int(q[:, 1].min()) - 8, 12))
                for im_ in (left_s, dep_s):
                    cv2.putText(im_, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(im_, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                if good:
                    cur.append((r['z_marker'], r['z_stereo'], col))
            cv2.putText(left_s, f'frame {i * stride}  t={i * stride / fps:6.2f}s  ArUco {a.size_mm:g} mm, PnP with stereo K', (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            text = []
            if len(rows) >= 3:
                st = stats_of(rows); text.append(f"reads {st['n']}: median error {st['median_rel_err'] * 100:+.1f} %, |err| p90 {st['p90_abs_rel_err'] * 100:.1f} %")
                if len(rows) >= 20:
                    g, c, _ = robust_line(np.array([x['d_marker'] for x in rows]), np.array([x['d_model'] for x in rows]))
                    text.append(f'fit d_model = {g:.3f} d_marker {c:+.2f} px')
            panel = np.concatenate([left_s, dep_s, sc.draw(pts[-4000:], cur, text), bar], axis=1)[:, :wv]
            writer.stdin.write(np.ascontiguousarray(panel).tobytes())
        i += 1
        if i % 200 == 0:
            print(f'  {i}/{n_frames} frames, {len(rows)} reads, {i / (time.perf_counter() - t0):.1f} fps', file=sys.stderr, flush=True)
    dec.stdout.close(); dec.wait()
    if writer is not None:
        writer.stdin.close(); writer.wait()
    with open(f'{vdir}/aruco_reads.csv', 'w', newline='') as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    res = dict(video=a.video, frames=i, stride=stride, size_mm=a.size_mm, fx=float(K[0, 0]), baseline_m=b, d_used=d_used, inner=a.inner, min_valid=a.min_valid,
               reads_used=len(rows), workers=sorted({r['worker'] for r in rows}), stats=stats_of(rows), seconds=time.perf_counter() - t0)
    if len(rows) >= 20:
        dm = np.array([r['d_marker'] for r in rows]); dmod = np.array([r['d_model'] for r in rows])
        g, c, m = robust_line(dm, dmod)
        res['fit'] = dict(gain=g, offset_px=c, inliers=int(m.sum()), d_inf_true_if_gain_1=float(np.median(dmod - dm) + d_used),
                          implied_baseline_m=g * b, d_inf_true=c + d_used,
                          note='d_model = gain * fx*b/z_marker + offset; offset = d_inf_true - d_used; gain = b_true / b_metadata')
    json.dump(res, open(f'{vdir}/aruco_compare.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k in ('reads_used', 'workers', 'stats', 'fit')}, indent=1))


if __name__ == '__main__':
    main()
