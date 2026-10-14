#!/usr/bin/env python3
"""Turn the per-frame output of run_sequence.py into videos "with depth".

Inputs
  --out_dir   run_sequence.py output: depth/%06d.npy (float32 metres, 0 = invalid)
  --seq_dir   extract_views.py output: left/%06d.png, meta.json (fps, stride)

Outputs (written next to the depth folder unless --video_dir is given)
  preview.mp4    left | depth false-colour (turbo, fixed range --zmin..--zmax, log scale,
                 colour bar with metre ticks, invalid / out-of-range = black).
                 fps = source fps / stride, libx264 yuv420p, --preview_scale (default 0.5).
  depth_mm.mkv   16-bit single-channel depth in millimetres (uint16, 0 = invalid, > 65.5 m -> 0),
                 FFV1 lossless, gray16le, full resolution.  Decode with
                   ffmpeg -i depth_mm.mkv -fps_mode passthrough -f rawvideo -pix_fmt gray16le - | np.frombuffer(..., '<u2')
                 (`-fps_mode passthrough` matters: without it ffmpeg duplicates frames to a guessed
                 constant rate when fps < 1, e.g. stride-30 previews).  cv2.VideoCapture may not
                 honour 16 bit; ffmpeg is safer.  Round-trip against depth/*.npy is bit-exact.
  overlay.mp4    (--overlay 1) left image with the depth colours alpha-blended on top.

Optional --temporal_median N (odd, default 0 = off): running median over N consecutive
*disparity* frames before colouring / writing; reduces flicker, blurs fast motion.

Needs ffmpeg (>= 4 is enough here) on PATH, numpy, opencv.

Example
  python scripts/spatial/make_depth_video.py --seq_dir work/3143 --out_dir work/3143/out_full
"""
import argparse, glob, json, os, subprocess, sys, time
import cv2
import numpy as np


def colour_bar(h, zmin, zmax, width=70, ticks=(0.3, 0.5, 1, 2, 3, 5, 8)):
    """Vertical turbo bar, near (zmin, red) at the top, far (zmax, blue) at the bottom."""
    bar_w, pad = 22, 6
    t = np.linspace(1.0, 0.0, h, dtype=np.float32)        # 1 = near (red) at top
    strip = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, :]
    img = np.zeros((h, width, 3), np.uint8)
    img[:, pad:pad + bar_w] = strip[:, None, :]
    for z in ticks:
        if z < zmin or z > zmax:
            continue
        y = int(round((np.log(z) - np.log(zmin)) / (np.log(zmax) - np.log(zmin)) * (h - 1)))
        cv2.line(img, (pad + bar_w, y), (pad + bar_w + 4, y), (255, 255, 255), 1)
        label = f'{z:g}m'
        fs = max(0.35, min(0.5, h / 1080 * 0.5))
        ty = min(max(y + 4, 10), h - 2)
        cv2.putText(img, label, (pad + bar_w + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def depth_to_colour(depth, zmin, zmax):
    """depth (m) -> BGR turbo; 0 / out of range -> black.  Log scale, near = red."""
    valid = (depth >= zmin) & (depth <= zmax)
    z = np.clip(depth, zmin, zmax)
    t = 1.0 - (np.log(z) - np.log(zmin)) / (np.log(zmax) - np.log(zmin))
    col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~valid] = 0
    return col


def ffmpeg_writer(path, w, h, fps, pix_fmt_in, codec_args, ffmpeg='ffmpeg'):
    from fractions import Fraction
    fr = Fraction(fps).limit_denominator(100000)          # 29.97002997 -> 30000/1001
    cmd = [ffmpeg, '-hide_banner', '-v', 'error', '-y',
           '-f', 'rawvideo', '-pix_fmt', pix_fmt_in, '-s', f'{w}x{h}', '-framerate', f'{fr.numerator}/{fr.denominator}',
           '-i', '-', *codec_args, path]
    print(' '.join(cmd), file=sys.stderr)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq_dir', required=True)
    ap.add_argument('--out_dir', required=True, help='run_sequence.py --out_dir (contains depth/, disp/)')
    ap.add_argument('--video_dir', default=None, help='where to write the videos (default: out_dir)')
    ap.add_argument('--zmin', type=float, default=0.3)
    ap.add_argument('--zmax', type=float, default=8.0)
    ap.add_argument('--preview_scale', type=float, default=0.5)
    ap.add_argument('--overlay', type=int, default=0)
    ap.add_argument('--overlay_alpha', type=float, default=0.55)
    ap.add_argument('--temporal_median', type=int, default=0, help='odd window over disparity frames, 0 = off')
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
    ap.add_argument('--ffmpeg', default='ffmpeg')
    args = ap.parse_args()

    with open(f'{args.seq_dir}/meta.json') as f:
        meta = json.load(f)
    fps = meta['fps'] / meta.get('stride', 1)
    with open(f'{args.out_dir}/run_meta.json') as f:
        run_meta = json.load(f)
    K = np.array(run_meta['K']); fx = K[0, 0] * run_meta.get('scale', 1.0); baseline = run_meta['baseline_m']

    depths = sorted(glob.glob(f'{args.out_dir}/depth/*.npy'))[args.start:args.end]
    assert depths, f'no depth frames in {args.out_dir}/depth'
    names = [os.path.splitext(os.path.basename(p))[0] for p in depths]
    vdir = args.video_dir or args.out_dir
    os.makedirs(vdir, exist_ok=True)

    d0 = np.load(depths[0]); H, W = d0.shape
    s = args.preview_scale
    ph, pw = int(round(H * s)), int(round(W * s))
    bar = colour_bar(ph, args.zmin, args.zmax)
    prev_w = pw * 2 + bar.shape[1]
    prev_w -= prev_w % 2; ph -= ph % 2                       # yuv420p needs even dims
    bar = bar[:ph]

    x264 = ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    w_prev = ffmpeg_writer(f'{vdir}/preview.mp4', prev_w, ph, fps, 'bgr24', x264, args.ffmpeg)
    w_mm = ffmpeg_writer(f'{vdir}/depth_mm.mkv', W, H, fps, 'gray16le', ['-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gray16le'], args.ffmpeg)
    w_ov = ffmpeg_writer(f'{vdir}/overlay.mp4', W - W % 2, H - H % 2, fps, 'bgr24', x264, args.ffmpeg) if args.overlay else None

    # temporal median over disparity (depth is 1/disp so median in disp space is the natural choice)
    win = args.temporal_median
    if win and win % 2 == 0:
        raise SystemExit('--temporal_median must be odd')
    half = win // 2 if win else 0
    disp_cache = {}

    def disp_at(j):
        j = min(max(j, 0), len(names) - 1)
        if j not in disp_cache:
            disp_cache[j] = np.load(f'{args.out_dir}/disp/{names[j]}.npy')
        return disp_cache[j]

    t0 = time.perf_counter()
    for i, dp in enumerate(depths):
        if win:
            stack = np.stack([disp_at(i + k) for k in range(-half, half + 1)])
            disp = np.median(stack, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                depth = np.where(disp > 0, fx * baseline / disp, 0).astype(np.float32)
            depth[~np.isfinite(depth)] = 0
            for k in list(disp_cache):
                if k < i - half:
                    del disp_cache[k]
        else:
            depth = np.load(dp)
        left = cv2.imread(f'{args.seq_dir}/left/{names[i]}.png')
        assert left is not None, f'missing left frame for {names[i]}'

        mm = np.where((depth > 0) & (depth * 1000 < 65535), np.round(depth * 1000), 0).astype('<u2')
        w_mm.stdin.write(mm.tobytes())

        col = depth_to_colour(depth, args.zmin, args.zmax)
        lp = cv2.resize(left, (pw, ph), interpolation=cv2.INTER_AREA)
        cp = cv2.resize(col, (pw, ph), interpolation=cv2.INTER_NEAREST)
        panel = np.concatenate([lp, cp, bar], axis=1)[:, :prev_w]
        t_s = meta['frame_time_s'][i] if i < len(meta.get('frame_time_s', [])) else i / fps
        cv2.putText(panel, f'{names[i]}  t={t_s:6.2f}s', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        w_prev.stdin.write(panel.tobytes())

        if w_ov is not None:
            valid = (col.sum(axis=2) > 0)[..., None]
            ov = np.where(valid, (left * (1 - args.overlay_alpha) + col * args.overlay_alpha), left).astype(np.uint8)
            w_ov.stdin.write(ov[:H - H % 2, :W - W % 2].tobytes())
        if (i + 1) % 50 == 0 or i + 1 == len(depths):
            print(f'  [{i+1}/{len(depths)}] {(i+1)/(time.perf_counter()-t0):.1f} fps', file=sys.stderr, flush=True)

    for w in (w_prev, w_mm, w_ov):
        if w is not None:
            w.stdin.close(); w.wait()
            assert w.returncode == 0, f'ffmpeg failed ({w.returncode})'
    dt = time.perf_counter() - t0
    info = dict(n_frames=len(depths), fps_out=fps, zmin=args.zmin, zmax=args.zmax, temporal_median=win,
                encode_seconds=dt, encode_fps=len(depths) / dt,
                files={k: os.path.getsize(f'{vdir}/{k}') for k in ('preview.mp4', 'depth_mm.mkv', 'overlay.mp4') if os.path.exists(f'{vdir}/{k}')})
    with open(f'{vdir}/video_meta.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
