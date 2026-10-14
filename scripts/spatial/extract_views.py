#!/usr/bin/env python3
"""Split an Apple spatial video (MV-HEVC) into left/right PNG frame sequences.

Uses ffmpeg's view specifiers (ffmpeg >= 7.1):  -map 0:v:0:vpos:left / vpos:right
so the eye assignment comes from the file's own SEI, not from a guessed view id.

Output layout (lossless PNG, RGB, original 1920x1080; the model pads to /32 itself):
  <out_dir>/left/000000.png, 000001.png, ...
  <out_dir>/right/000000.png, ...
  <out_dir>/K.txt            (via probe_metadata.py)
  <out_dir>/meta.json

Frame index i corresponds to source time  start + i * stride / fps.

Usage:
  python scripts/spatial/extract_views.py vid/IMG_3143.MOV work/3143 --start 0 --duration 10 --stride 30
"""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_metadata import probe, write_k  # noqa: E402


def run(cmd):
    print(' '.join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def extract(video, out_dir, start=0.0, duration=None, stride=1, max_frames=None, fmt='png'):
    info = probe(video)
    os.makedirs(f'{out_dir}/left', exist_ok=True)
    os.makedirs(f'{out_dir}/right', exist_ok=True)
    write_k(info, f'{out_dir}/K.txt')

    vf = f'select=not(mod(n\\,{stride}))' if stride > 1 else None
    base = ['ffmpeg', '-hide_banner', '-v', 'error', '-y']
    if start:
        base += ['-ss', str(start)]
    base += ['-i', video]
    if duration:
        base += ['-t', str(duration)]
    for eye in ('left', 'right'):
        cmd = base + ['-map', f'0:v:0:vpos:{eye}']
        if vf:
            cmd += ['-vf', vf, '-fps_mode', 'vfr']
        if max_frames:
            cmd += ['-frames:v', str(max_frames)]
        cmd += ['-start_number', '0', f'{out_dir}/{eye}/%06d.{fmt}']
        run(cmd)

    nL = len(os.listdir(f'{out_dir}/left')); nR = len(os.listdir(f'{out_dir}/right'))
    assert nL == nR, f'left/right frame count mismatch: {nL} vs {nR}'
    meta = dict(info, start_s=start, duration_s_extracted=duration, stride=stride, n_frames_extracted=nL,
                frame_time_s=[start + i * stride / info['fps'] for i in range(nL)])
    with open(f'{out_dir}/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'{nL} frame pairs -> {out_dir}', file=sys.stderr)
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('out_dir')
    ap.add_argument('--start', type=float, default=0.0, help='start time (s)')
    ap.add_argument('--duration', type=float, default=None, help='length to extract (s)')
    ap.add_argument('--stride', type=int, default=1, help='keep every N-th frame')
    ap.add_argument('--max_frames', type=int, default=None)
    a = ap.parse_args()
    extract(a.video, a.out_dir, a.start, a.duration, a.stride, a.max_frames)
