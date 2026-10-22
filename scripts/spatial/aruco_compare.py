#!/usr/bin/env python3
"""Helmet ArUco markers as ground truth for the stereo depth (plan doc §6.10, §13.3).

The workers wear field-pose-annotator helmet markers (ArUco 4x4_50, printed 88.9 mm; util/aruco in that repo).
For every marker read in the left eye, OpenCV solvePnP (IPPE square) with the same K as the stereo depth gives the
marker centre position; the stereo disparity is the median inside the marker quad shrunk to --inner of its size.

Why this is a clean test: marker depth ~ fx and stereo depth = fx*b/d ~ fx, so their ratio does not depend on the
field of view; it tests the baseline and the disparity offset only.  The marker predicts the disparity
d_marker = fx*b / z_marker, and the model disparity follows  d_model = g * d_marker + c  (g = b_true / b_metadata,
c = d_inf_true - d_used).

Stereo correction (--stereo_offset): the stereo disparity is corrected by an offset c before converting to depth,
z_stereo = fx*b / (d_model - c).  'auto' fits c as the median of d_model - d_marker over this clip's own reads
(a first pass over the video); a number uses that offset (e.g. a session value); 0 shows the stereo as produced.

Outputs (in --video_dir, default <seq_dir>/aruco):
  aruco_compare.mp4   full-resolution left eye: every read boxed, labelled with the worker id and three colour-coded
                      numbers (marker distance, stereo distance, stereo - marker), plus a side panel with the camera
                      model, the colour legend, the workers in view, running statistics and a top view (rings per metre)
  aruco_reads.csv     every read: frame, time, id, worker, side, px size, marker x/y/z and range, stereo z and range
                      (raw and corrected), disparities, valid share
  aruco_compare.json  error statistics (raw and corrected; overall and per distance bin), gain/offset fit, settings

Needs: ffmpeg (MV-HEVC) on PATH, OpenCV with aruco, numpy, Pillow; field-pose-annotator for the marker scheme.
Example:
  python scripts/spatial/aruco_compare.py --video vid/IMG_3354.MOV --seq_dir work/3354_full --out_dir work/3354_full/out
"""
import argparse, csv, json, os, subprocess, sys, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, code_dir)
from make_depth_video import ffmpeg_writer  # noqa: E402
from run_sequence import read_k  # noqa: E402

BINS = [(0.5, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 12)]
# The three numbers, fixed colours (RGB), and per-worker colours for the ids (as in the field-pose-annotator demo)
C_MARKER, C_STEREO, C_DIFF = (0, 229, 255), (255, 150, 30), (255, 80, 205)
WORKER_RGB = [(255, 196, 0), (30, 144, 255), (60, 210, 60), (255, 99, 71), (190, 110, 255), (0, 206, 209),
              (255, 140, 0), (235, 235, 235), (255, 105, 180), (170, 220, 60), (120, 160, 255), (210, 180, 140)]
PANEL_W = 600
BG = (30, 33, 38)


def font(size, bold=False):
    for name in (('arialbd.ttf', 'DejaVuSans-Bold.ttf') if bold else ('arial.ttf', 'DejaVuSans.ttf')):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_LABEL, F_LABEL_B, F_PANEL, F_PANEL_B, F_HEAD, F_SMALL = font(20), font(21, True), font(19), font(19, True), font(22, True), font(16)


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


def stats_of(rows, zs_key='z_stereo'):
    if not rows:
        return {}
    zm = np.array([r['z_marker'] for r in rows]); zs = np.array([r[zs_key] for r in rows])
    rel = zs / zm - 1.0
    out = dict(n=len(rows), median_rel_err=float(np.median(rel)), mad_rel_err=float(np.median(np.abs(rel - np.median(rel)))),
               median_abs_rel_err=float(np.median(np.abs(rel))), p90_abs_rel_err=float(np.percentile(np.abs(rel), 90)),
               median_abs_err_m=float(np.median(np.abs(zs - zm))), within_5pct=float(np.mean(np.abs(rel) < 0.05)), within_10pct=float(np.mean(np.abs(rel) < 0.10)))
    out['by_distance'] = {}
    for lo, hi in BINS:
        sel = (zm >= lo) & (zm < hi)
        if sel.sum() >= 5:
            out['by_distance'][f'{lo}-{hi} m'] = dict(n=int(sel.sum()), median_rel_err=float(np.median(rel[sel])),
                                                       median_abs_rel_err=float(np.median(np.abs(rel[sel]))),
                                                       median_abs_err_m=float(np.median(np.abs(zs[sel] - zm[sel]))))
    return out


# ----------------------------------------------------------------------------- reading
def tuned_detector(fam):
    """ArUco 4x4 reader tuned for small helmet codes (scripts/spatial/dev/aruco_detect_tuning.py: on IMG_3348 it reads a
    code in 60 % of frames against 39 % for field-pose-annotator's default settings): finer adaptive-threshold windows,
    looser polygon fit, more bit error correction (misreads are filtered by min_id_reads), 10 px minimum side."""
    p = cv2.aruco.DetectorParameters()
    p.minMarkerPerimeterRate = 4.0 * 10 / 1920
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 33, 4
    p.polygonalApproxAccuracyRate = 0.05
    p.errorCorrectionRate = 0.8
    p.minCornerDistanceRate = 0.02
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(fam.dictionary), p)


LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def track_corners(prev_g, cur_g, c4, fb_max=0.5):
    """Follow a marker's 4 corners from the previous frame (pyramidal Lucas-Kanade, forward-backward check)."""
    p0 = c4.astype(np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, p0, None, **LK)
    if p1 is None or not st.all():
        return None
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(cur_g, prev_g, p1, None, **LK)
    if pb is None or not stb.all() or np.linalg.norm((pb - p0).reshape(-1, 2), axis=1).max() > fb_max:
        return None
    q = p1.reshape(4, 2)
    s0 = np.linalg.norm(c4 - np.roll(c4, 1, 0), axis=1); s1 = np.linalg.norm(q - np.roll(q, 1, 0), axis=1)
    if np.abs(s1 / np.maximum(s0, 1e-3) - 1).max() > 0.2:      # shape changed too much: lost
        return None
    return q


def measure(c4, mid, i, a, mk, K, size_m, fxb, disp, meta, stride, tracked=0):
    side_px = float(np.mean(np.linalg.norm(c4 - np.roll(c4, 1, 0), axis=1)))
    if side_px < mk.MIN_SIDE_PX * (0.8 if tracked else 1.0):
        return None
    t, reproj = marker_pose(c4, K, size_m)
    if t is None or not np.isfinite(t).all() or t[2] <= 0.2:
        return None
    d_s, valid, npx = stereo_on_marker(disp, c4, a.inner)
    worker, side = mk.decode(int(mid))
    return dict(frame=i, src_frame=i * stride, t_s=i * stride / meta['fps'], id=int(mid), worker=mk.worker_label(worker), worker_no=worker,
                side=side, side_px=side_px, reproj_px=reproj, x_marker=float(t[0]), y_marker=float(t[1]), z_marker=float(t[2]),
                range_marker=float(np.linalg.norm(t)), d_marker=fxb / float(t[2]), d_model=d_s, valid=valid, n_px=npx,
                u=float(c4[:, 0].mean()), v=float(c4[:, 1].mean()), corners=c4, tracked=tracked)


def read_pass(a, mk, det, K, size_m, fxb, meta, stride, W, H, keep_frames):
    """Decode the left eye, detect every marker and follow it between detections; returns per-frame reads.
    A read has tracked = 0 when detected in that frame, or k >= 1 when its corners were followed k frames since the last
    detection (at most --track_s seconds).  Ids detected fewer than --min_id_reads times in the clip are dropped (misreads)."""
    vf = ['-vf', f'select=not(mod(n\\,{stride}))', '-fps_mode', 'vfr'] if stride > 1 else []
    dec = subprocess.Popen([a.ffmpeg, '-v', 'error', '-i', a.video, '-map', '0:v:0:vpos:left', *vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                           stdout=subprocess.PIPE)
    n_frames = int(meta['n_frames_extracted']); per_frame = []; t0 = time.perf_counter(); i = 0
    max_track = int(round(a.track_s * meta['fps'] / stride)); active = {}; prev_g = None
    while i < n_frames:
        buf = dec.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dp = f'{a.out_dir}/disp/{i:06d}.npy'
        disp = np.load(dp) if os.path.exists(dp) else None
        corners, ids, _ = det.detectMarkers(g)
        reads = []; seen = set()
        if disp is not None:
            if ids is not None:
                for cc, mid in zip(corners, ids.ravel()):
                    if mid >= a.max_id or int(mid) in seen:
                        continue
                    r = measure(cc.reshape(4, 2), mid, i, a, mk, K, size_m, fxb, disp, meta, stride)
                    if r is not None:
                        reads.append(r); seen.add(int(mid)); active[int(mid)] = (r['corners'], 0)
            if prev_g is not None and max_track > 0:
                for mid, (c4, age) in list(active.items()):
                    if mid in seen:
                        continue
                    q = track_corners(prev_g, g, c4) if age < max_track else None
                    r = measure(q, mid, i, a, mk, K, size_m, fxb, disp, meta, stride, tracked=age + 1) if q is not None else None
                    if r is None:
                        del active[mid]; continue
                    reads.append(r); active[mid] = (q, age + 1)
        per_frame.append((reads, img.copy() if keep_frames else None))
        prev_g = g; i += 1
        if i % 300 == 0:
            print(f'  read {i}/{n_frames} frames ({i / (time.perf_counter() - t0):.1f} fps)', file=sys.stderr, flush=True)
    dec.stdout.close(); dec.wait()
    counts = {}
    for reads, _ in per_frame:
        for r in reads:
            if not r['tracked']:
                counts[r['id']] = counts.get(r['id'], 0) + 1
    keep = {m for m, n in counts.items() if n >= a.min_id_reads}
    dropped = sorted(set(counts) - keep)
    if dropped:
        print(f'  dropped ids read fewer than {a.min_id_reads} times (misreads): {dropped}', flush=True)
    return [([r for r in reads if r['id'] in keep], img) for reads, img in per_frame]


def finish_read(r, K, fxb, offset, min_valid):
    """Stereo depth/range at the marker, raw (offset 0) and corrected (disparity - offset)."""
    ok = np.isfinite(r['d_model']) and r['valid'] >= min_valid
    ray = np.linalg.norm([(r['u'] - K[0, 2]) / K[0, 0], (r['v'] - K[1, 2]) / K[1, 1], 1.0])
    r['z_stereo_raw'] = fxb / r['d_model'] if ok and r['d_model'] > 0 else np.nan
    dc = r['d_model'] - offset if ok else np.nan
    r['z_stereo'] = fxb / dc if ok and dc > 0.05 else np.nan
    r['range_stereo'] = r['z_stereo'] * ray if np.isfinite(r['z_stereo']) else np.nan
    r['offset_px'] = offset
    r['good'] = bool(np.isfinite(r['z_stereo']))


# ----------------------------------------------------------------------------- drawing
def rounded(draw, box, fill, r=8):
    draw.rounded_rectangle(box, radius=r, fill=fill)


def _overlap(a, b, gap=6):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def place_label(bb, tw, th, taken, W, H):
    """Label rectangle next to box bb = (x0, y0, x1, y1): right, left, above, below, then further out, avoiding `taken`."""
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    cands = []
    for step in range(0, 6):
        dy = step * (th + 8)
        cands += [(bb[2] + 12, cy - th / 2 + dy), (bb[0] - 12 - tw, cy - th / 2 + dy), (bb[2] + 12, cy - th / 2 - dy), (bb[0] - 12 - tw, cy - th / 2 - dy),
                  (cx - tw / 2, bb[1] - 12 - th - dy), (cx - tw / 2, bb[3] + 12 + dy)]
    for x, y in cands:
        x = min(max(5, x), W - tw - 5); y = min(max(5, y), H - th - 44)
        rect = (x, y, x + tw, y + th)
        if not any(_overlap(rect, t) for t in taken):
            return rect
    x = min(max(5, bb[2] + 12), W - tw - 5); y = min(max(5, cy - th / 2), H - th - 44)
    return (x, y, x + tw, y + th)


QUAD_MARGIN = 1.5      # outline = the marker's own quad scaled about its centre (follows its edges and perspective)
QUAD_MIN_SIDE = 70     # px: small codes get a larger factor so the outline is at least this big


def outline_quad(c4):
    c = c4.mean(0); side = float(np.mean(np.linalg.norm(c4 - np.roll(c4, 1, 0), axis=1)))
    return c + max(QUAD_MARGIN, QUAD_MIN_SIDE / max(side, 1.0)) * (c4 - c)


def draw_frame_labels(img, reads, W, fps_out=30.0):
    """Outlines and labels on the full-resolution frame (PIL RGBA overlay for translucent label backgrounds).
    Each outline is the marker's quad, enlarged, so it follows the code's edges; r['tracked'] = k > 0 marks a marker
    whose corners were followed k frames since its last detection (thinner outline, 'tracked' in the label)."""
    base = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGBA')
    over = Image.new('RGBA', base.size, (0, 0, 0, 0)); d = ImageDraw.Draw(over)
    H = img.shape[0]
    quads = [outline_quad(r['corners']) for r in reads]
    boxes = [(q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()) for q in quads]
    taken = list(boxes)                        # labels must not cover any marker outline
    for r, q, bb in sorted(zip(reads, quads, boxes), key=lambda t: (t[0]['tracked'] > 0, t[2][0])):
        wc = WORKER_RGB[r['worker_no'] % len(WORKER_RGB)]
        tracked = r['tracked']
        poly = [tuple(p) for p in q] + [tuple(q[0])]
        d.line(poly, fill=(0, 0, 0, 230), width=9 if not tracked else 7, joint='curve')     # black edge: visible on any helmet
        d.line(poly, fill=wc + (255,), width=5 if not tracked else 3, joint='curve')
        head = f"{r['worker']}  {r['side'].lower()}" + ('   tracked' if tracked else '')
        lines = [(head, wc, F_LABEL_B)]
        lines.append((f"marker  {r['range_marker']:.2f} m", C_MARKER, F_LABEL))
        if r['good']:
            diff = r['range_stereo'] - r['range_marker']
            lines.append((f"stereo  {r['range_stereo']:.2f} m", C_STEREO, F_LABEL))
            lines.append((f"diff   {diff:+.2f} m ({diff / r['range_marker'] * 100:+.1f} %)", C_DIFF, F_LABEL))
        else:
            lines.append(('stereo  n/a', C_STEREO, F_LABEL))
        tw = max(d.textlength(t, font=f) for t, _, f in lines) + 20; th = 26 * len(lines) + 12
        rect = place_label(bb, tw, th, taken, W, H); taken.append(rect)
        x, y = rect[0], rect[1]
        # leader line from the box to the label when the label is not right next to it
        lx = rect[0] if rect[0] > bb[2] else (rect[2] if rect[2] < bb[0] else (rect[0] + rect[2]) / 2)
        ly = min(max((bb[1] + bb[3]) / 2, rect[1] + 8), rect[3] - 8)
        bx = bb[2] if lx > bb[2] else (bb[0] if lx < bb[0] else (bb[0] + bb[2]) / 2)
        by = min(max(ly, bb[1]), bb[3])
        d.line([(bx, by), (lx, ly)], fill=wc + (230,), width=2)
        rounded(d, (x, y, x + tw, y + th), (15, 18, 22, 185 if tracked else 215))
        d.line([(x, y + 6), (x, y + th - 6)], fill=wc + (255,), width=4)
        for k, (t, col, f) in enumerate(lines):
            d.text((x + 12, y + 6 + 26 * k), t, font=f, fill=col + ((215,) if tracked else (255,)))
    return Image.alpha_composite(base, over).convert('RGB')


def draw_panel(H, info, legend_note, in_view, run, top_pts, clip, t_s):
    p = Image.new('RGB', (PANEL_W, H), BG); d = ImageDraw.Draw(p); x0 = 20; y = 16
    d.text((x0, y), 'STEREO DEPTH vs HELMET MARKERS', font=F_HEAD, fill=(255, 255, 255)); y += 32
    d.text((x0, y), f'{clip}    t = {t_s:6.2f} s', font=F_PANEL, fill=(190, 195, 200)); y += 36
    # camera block
    rounded(d, (10, y, PANEL_W - 10, y + 26 * len(info) + 44), (44, 48, 55))
    d.text((x0, y + 8), 'CAMERA MODEL', font=F_PANEL_B, fill=(255, 255, 255)); yy = y + 38
    for k, v in info:
        d.text((x0, yy), k, font=F_PANEL, fill=(160, 165, 172)); d.text((x0 + 150, yy), v, font=F_PANEL, fill=(235, 235, 235)); yy += 26
    y = yy + 16
    # legend
    rounded(d, (10, y, PANEL_W - 10, y + 176), (44, 48, 55))
    d.text((x0, y + 8), 'LEGEND (next to each box)', font=F_PANEL_B, fill=(255, 255, 255)); yy = y + 38
    for col, name, desc in ((C_MARKER, 'marker', 'distance from ArUco PnP (88.9 mm code)'),
                            (C_STEREO, 'stereo', f'distance from Fast-FoundationStereo ({legend_note})'),
                            (C_DIFF, 'diff', 'stereo minus marker, m and %')):
        d.rectangle((x0, yy + 4, x0 + 16, yy + 20), fill=col); d.text((x0 + 26, yy), name, font=F_PANEL_B, fill=col)
        d.text((x0 + 100, yy), desc, font=F_SMALL, fill=(220, 220, 220)); yy += 28
    d.text((x0, yy + 2), 'distances: lens to marker centre, 0.01 m resolution', font=F_SMALL, fill=(160, 165, 172))
    d.text((x0, yy + 22), 'thick outline = detected now, thin = tracked since detection (< 0.5 s)', font=F_SMALL, fill=(160, 165, 172))
    y += 192
    # in view
    rows_h = 26 * max(1, len(in_view)) + 66
    rounded(d, (10, y, PANEL_W - 10, y + rows_h), (44, 48, 55))
    d.text((x0, y + 8), 'IN VIEW', font=F_PANEL_B, fill=(255, 255, 255)); yy = y + 36
    for cx, t, col in ((x0, 'worker', (160, 165, 172)), (x0 + 140, 'marker', C_MARKER), (x0 + 270, 'stereo', C_STEREO), (x0 + 400, 'diff', C_DIFF)):
        d.text((cx, yy), t, font=F_SMALL, fill=col)
    yy += 24
    if not in_view:
        d.text((x0, yy), '(no marker read in this frame)', font=F_SMALL, fill=(140, 140, 140))
    for wlabel, wc, side, rm, rs in in_view:
        d.ellipse((x0, yy + 6, x0 + 12, yy + 18), fill=wc); d.text((x0 + 20, yy), f'{wlabel} {side[0]}', font=F_PANEL_B, fill=wc)
        d.text((x0 + 140, yy), f'{rm:.2f} m', font=F_PANEL, fill=C_MARKER)
        if np.isfinite(rs):
            d.text((x0 + 270, yy), f'{rs:.2f} m', font=F_PANEL, fill=C_STEREO)
            d.text((x0 + 400, yy), f'{rs - rm:+.2f} m', font=F_PANEL, fill=C_DIFF)
        yy += 26
    y += rows_h + 16
    # running statistics
    rounded(d, (10, y, PANEL_W - 10, y + 92), (44, 48, 55))
    d.text((x0, y + 8), 'SO FAR IN THIS CLIP', font=F_PANEL_B, fill=(255, 255, 255))
    for k, t in enumerate(run[:2]):
        d.text((x0, y + 36 + 24 * k), t, font=F_SMALL, fill=(220, 220, 220))
    y += 108
    # top view, rings per metre: a half disc that fits the remaining box (width PANEL_W - 20, height H - y - 10)
    bw, bh = PANEL_W - 20, H - y - 10
    if bh > 120:
        rounded(d, (10, y, 10 + bw, y + bh), (22, 24, 28))
        cxp, cyp = 10 + bw / 2, y + bh - 16
        R = min(bw / 2 - 12, bh - 50); scale = R / 7.0
        for m in range(1, 8):
            rr = m * scale
            d.arc((cxp - rr, cyp - rr, cxp + rr, cyp + rr), 180, 360, fill=(70, 74, 80), width=1)
            d.text((cxp + 4, cyp - rr - 1), f'{m} m', font=F_SMALL, fill=(110, 114, 120))
        d.line([(cxp - R, cyp), (cxp + R, cyp)], fill=(70, 74, 80), width=1)
        d.polygon([(cxp, cyp - 9), (cxp - 8, cyp + 6), (cxp + 8, cyp + 6)], outline=(255, 255, 255))
        d.text((20, y + 6), 'top view: marker (ring) vs stereo (dot)', font=F_SMALL, fill=(200, 200, 200))
        for xm, zm, xs, zs, wc in top_pts:
            pm = (cxp + xm * scale, cyp - zm * scale)
            if np.isfinite(zs):
                ps = (cxp + xs * scale, cyp - zs * scale)
                d.line([pm, ps], fill=(200, 200, 200), width=1)
                d.ellipse((ps[0] - 5, ps[1] - 5, ps[0] + 5, ps[1] + 5), fill=C_STEREO)
            d.ellipse((pm[0] - 8, pm[1] - 8, pm[0] + 8, pm[1] + 8), outline=C_MARKER, width=3)
            d.ellipse((pm[0] - 3, pm[1] - 3, pm[0] + 3, pm[1] + 3), fill=wc)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--seq_dir', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--video_dir', default=None)
    ap.add_argument('--size_mm', type=float, default=88.9, help='printed marker side (field-pose-annotator make_sheets default)')
    ap.add_argument('--max_id', type=int, default=48, help='ids >= this are misreads (12 workers x 4 codes)')
    ap.add_argument('--inner', type=float, default=0.6, help='share of the marker quad used for the stereo sample')
    ap.add_argument('--min_valid', type=float, default=0.6)
    ap.add_argument('--stereo_offset', default='auto', help="'auto' (fit on this clip's markers), a number in px, or 0 (stereo as produced)")
    ap.add_argument('--fpa_aruco', default=os.path.expanduser('~/Documents/GitHub/field-pose-annotator/util/aruco'))
    ap.add_argument('--camera_name', default='iPhone 17 Pro Max, main camera (1x, 24 mm equiv.) = left eye')
    ap.add_argument('--stereo_model', default='Fast-FoundationStereo 23-36-37')
    ap.add_argument('--track_s', type=float, default=0.5, help='follow a marker this long after its last detection (optical flow on its corners)')
    ap.add_argument('--min_id_reads', type=int, default=3, help='ids detected fewer times in the clip are treated as misreads')
    ap.add_argument('--detector', default='fpa', choices=['fpa', 'tuned'],
                    help="'fpa' = field-pose-annotator settings (default); 'tuned' reads ~2x more codes but its extra reads are poor "
                         "(15 %% median error on IMG_3348, dev/aruco_detect_tuning.py), so it is not used for measurements")
    ap.add_argument('--crf', type=int, default=21); ap.add_argument('--no_video', action='store_true'); ap.add_argument('--ffmpeg', default='ffmpeg')
    a = ap.parse_args()
    sys.path.insert(0, a.fpa_aruco)
    import markers as mk   # field-pose-annotator marker scheme (detector settings, id -> worker/side)
    fam = mk.FAMILIES['aruco4x4']; det = tuned_detector(fam) if a.detector == 'tuned' else mk.detector(fam, (1920, 1080))
    meta = json.load(open(f'{a.seq_dir}/meta.json')); stride = int(meta.get('stride', 1)); fps = meta['fps']
    rm = json.load(open(f'{a.out_dir}/run_meta.json')); d_used = float(rm.get('dinf', 0.0))
    K, b = read_k(f'{a.seq_dir}/K.txt'); fxb = float(K[0, 0] * b); size_m = a.size_mm / 1000.0
    vdir = a.video_dir or f'{a.seq_dir}/aruco'; os.makedirs(vdir, exist_ok=True)
    W, H = int(meta['width']), int(meta['height'])
    t0 = time.perf_counter()

    per_frame = read_pass(a, mk, det, K, size_m, fxb, meta, stride, W, H, keep_frames=False)
    raw = [r for reads, _ in per_frame for r in reads if not r['tracked'] and np.isfinite(r['d_model']) and r['valid'] >= a.min_valid]
    if a.stereo_offset == 'auto':
        offset = float(np.median([r['d_model'] - r['d_marker'] for r in raw])) if len(raw) >= 20 else 0.0
        how = f'auto: median over {len(raw)} reads of this clip'
    else:
        offset, how = float(a.stereo_offset), 'given'
    for reads, _ in per_frame:
        for r in reads:
            finish_read(r, K, fxb, offset, a.min_valid)
    rows = [r for reads, _ in per_frame for r in reads if r['good'] and not r['tracked']]          # statistics: detections only
    tracked_rows = [r for reads, _ in per_frame for r in reads if r['good'] and r['tracked']]
    n_shown = sum(1 for reads, _ in per_frame if reads); n_det = sum(1 for reads, _ in per_frame if any(not r['tracked'] for r in reads))
    print(f'{len(rows)} detected reads ({len(tracked_rows)} more tracked); a marker shown in {n_shown}/{len(per_frame)} frames '
          f'({n_det} with a detection); stereo offset {offset:+.2f} px ({how}) -> d_inf {d_used + offset:+.2f} px', flush=True)

    if not a.no_video:
        hfov = float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
        info = [('camera', a.camera_name.split(',')[0]), ('lens', a.camera_name.split(',', 1)[1].strip() if ',' in a.camera_name else ''),
                ('video', f'{W}x{H} @ {fps:.2f} fps, spatial (MV-HEVC)'), ('field of view', f'{hfov:.1f} deg (file metadata)'),
                ('fx, cx, cy', f'{K[0, 0]:.1f}, {K[0, 2]:.1f}, {K[1, 2]:.1f} px'), ('baseline', f'{b * 1000:.3f} mm (file metadata)'),
                ('stereo', a.stereo_model), ('d_inf', f'{d_used + offset:+.2f} px (marker-calibrated)' if offset else f'{d_used:+.2f} px (as produced)'),
                ('markers', f'ArUco 4x4_50, {a.size_mm:g} mm, PnP IPPE')]
        legend_note = f'd_inf {d_used + offset:+.2f} px' if offset else 'as produced'
        wv = W + PANEL_W
        writer = ffmpeg_writer(f'{vdir}/aruco_compare.mp4', wv, H, fps / stride, 'rgb24',
                               ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(a.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart'], a.ffmpeg)
        vf = ['-vf', f'select=not(mod(n\\,{stride}))', '-fps_mode', 'vfr'] if stride > 1 else []
        dec = subprocess.Popen([a.ffmpeg, '-v', 'error', '-i', a.video, '-map', '0:v:0:vpos:left', *vf, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                               stdout=subprocess.PIPE)
        seen = []; last_pos = {}
        for i, (reads, _) in enumerate(per_frame):
            buf = dec.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            shown = reads
            frame = draw_frame_labels(img, shown, W, fps / stride)
            fd = ImageDraw.Draw(frame)
            status = f'{W}x{H}, {a.camera_name.split(",")[0]} main (1x): {np.degrees(2 * np.arctan(W / 2 / K[0, 0])):.1f} deg (spatial-video metadata), {a.size_mm:g} mm codes, ArUco 4x4_50'
            fd.rounded_rectangle((8, H - 36, 18 + fd.textlength(status, font=F_SMALL), H - 8), radius=4, fill=(15, 18, 22))
            fd.text((14, H - 32), status, font=F_SMALL, fill=(230, 230, 230))
            seen += [r for r in reads if r['good'] and not r['tracked']]
            for r in reads:   # top view keeps each worker's last position for one second
                last_pos[r['worker']] = (i, r)
            in_view = []
            per_worker = {}
            for r in sorted(shown, key=lambda r: r['tracked']):     # a detection wins over a tracked marker
                per_worker.setdefault(r['worker'], r)
            for wl, r in sorted(per_worker.items()):
                in_view.append((wl, WORKER_RGB[r['worker_no'] % len(WORKER_RGB)], r['side'], r['range_marker'], r['range_stereo'] if r['good'] else np.nan))
            top = []
            for wl, (fi, r) in last_pos.items():
                if i - fi <= fps / stride:
                    xs = r['x_marker'] / r['z_marker'] * r['z_stereo'] if r['good'] else np.nan
                    top.append((r['x_marker'], r['z_marker'], xs, r['z_stereo'], WORKER_RGB[r['worker_no'] % len(WORKER_RGB)]))
            run = []
            if len(seen) >= 3:
                rel = np.array([x['range_stereo'] / x['range_marker'] - 1 for x in seen])
                dm = np.array([x['range_stereo'] - x['range_marker'] for x in seen])
                run.append(f'{len(seen)} reads: median diff {np.median(dm):+.2f} m ({np.median(rel) * 100:+.1f} %), median |diff| {np.median(np.abs(dm)):.2f} m')
                run.append(f'|diff| within 5 %: {np.mean(np.abs(rel) < 0.05) * 100:.0f} % of reads, within 10 %: {np.mean(np.abs(rel) < 0.10) * 100:.0f} %')
            panel = draw_panel(H, info, legend_note, in_view, run, top, os.path.basename(a.video), i * stride / fps)
            out = np.concatenate([np.asarray(frame), np.asarray(panel)], axis=1)
            writer.stdin.write(np.ascontiguousarray(out).tobytes())
            if (i + 1) % 300 == 0:
                print(f'  render {i + 1}/{len(per_frame)}', file=sys.stderr, flush=True)
        dec.stdout.close(); dec.wait(); writer.stdin.close(); writer.wait()

    keys = ['frame', 'src_frame', 't_s', 'id', 'worker', 'side', 'side_px', 'reproj_px', 'x_marker', 'y_marker', 'z_marker', 'range_marker',
            'd_marker', 'd_model', 'offset_px', 'z_stereo_raw', 'z_stereo', 'range_stereo', 'valid', 'n_px', 'u', 'v']
    with open(f'{vdir}/aruco_reads.csv', 'w', newline='') as f:          # detections only (tracked frames are display aids)
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    res = dict(renderer='v2: corner tracking between detections, outlines on the marker quad', video=a.video, frames=len(per_frame), stride=stride, size_mm=a.size_mm,
               fx=float(K[0, 0]), baseline_m=b, d_used=d_used, inner=a.inner, detector=a.detector, track_s=a.track_s,
               min_valid=a.min_valid, stereo_offset_px=offset, stereo_offset_how=how, d_inf_display=d_used + offset,
               frames_with_marker_shown=n_shown, frames_with_detection=n_det, tracked_reads=len(tracked_rows),
               reads_used=len(rows), workers=sorted({r['worker'] for r in rows}),
               stats=stats_of(rows, 'z_stereo_raw'), stats_corrected=stats_of(rows, 'z_stereo'),
               stats_corrected_tracked=stats_of(tracked_rows, 'z_stereo'), seconds=time.perf_counter() - t0)
    if len(rows) >= 20:
        dm = np.array([r['d_marker'] for r in rows]); dmod = np.array([r['d_model'] for r in rows])
        g, c, m = robust_line(dm, dmod)
        res['fit'] = dict(gain=g, offset_px=c, inliers=int(m.sum()), d_inf_true_if_gain_1=float(np.median(dmod - dm) + d_used),
                          implied_baseline_m=g * b, d_inf_true=c + d_used,
                          note='d_model = gain * fx*b/z_marker + offset; offset = d_inf_true - d_used; gain = b_true / b_metadata')
    json.dump(res, open(f'{vdir}/aruco_compare.json', 'w'), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k in ('reads_used', 'workers', 'stereo_offset_px', 'd_inf_display', 'fit')}, indent=1))
    for name, sc in (('corrected', res['stats_corrected']), ('tracked (display only)', res['stats_corrected_tracked'])):
        if sc:
            print(f"{name}: median {sc['median_rel_err'] * 100:+.1f} %, |err| median {sc['median_abs_rel_err'] * 100:.1f} % ({sc['median_abs_err_m'] * 100:.0f} cm), "
                  f"within 5 % {sc['within_5pct'] * 100:.0f} %, within 10 % {sc['within_10pct'] * 100:.0f} %")


if __name__ == '__main__':
    main()
