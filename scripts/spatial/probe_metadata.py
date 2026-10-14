#!/usr/bin/env python3
"""Read the stereo metadata of an Apple spatial video (MV-HEVC .MOV) and
derive the pinhole intrinsics + baseline that Fast-FoundationStereo needs.

Requires ffprobe >= 7.1 (MV-HEVC support).  Tested with ffmpeg 9.0.1.

Outputs (stdout, JSON):
  device, software, width, height, fps, nb_frames, duration_s
  baseline_m            from Stereo3D side data `baseline` (micrometres -> m)
  hfov_deg              from `horizontal_field_of_view`
  disparity_adjust      from `horizontal_disparity_adjustment` (playback-only!)
  left_view_id          which MV-HEVC view id ffmpeg labels as the left eye
  fx, fy, cx, cy        fx = W / (2 tan(hfov/2)); fy = fx; principal point = centre
  has_gps               True if the container carries a GPS location tag

Optionally writes a K.txt in the demo_data/K.txt layout:
  line 1: 9 floats of K row-major, line 2: baseline in metres.

Usage:
  python scripts/spatial/probe_metadata.py vid/IMG_3143.MOV [--k_out work/3143/K.txt]
"""
import argparse, json, subprocess, math, os, sys
from fractions import Fraction


def ffprobe_json(args):
    out = subprocess.run(['ffprobe', '-hide_banner', '-v', 'error', '-of', 'json', *args],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def probe(path):
    meta = ffprobe_json(['-show_format', '-show_streams', path])
    fmt_tags = meta['format'].get('tags', {})
    v = next(s for s in meta['streams'] if s['codec_type'] == 'video')
    stereo = next((sd for sd in v.get('side_data_list', []) if sd.get('side_data_type') == 'Stereo 3D'), None)
    if stereo is None:
        raise SystemExit(f'{path}: no Stereo 3D side data. Not a spatial video, or ffprobe is too old.')

    def frac(s):
        return float(Fraction(str(s)))

    W, H = int(v['width']), int(v['height'])
    baseline_m = float(stereo['baseline']) * 1e-6          # micrometres -> metres
    hfov_deg = frac(stereo['horizontal_field_of_view'])     # e.g. "63400/1000"
    dadj = frac(stereo.get('horizontal_disparity_adjustment', '0/1'))
    fx = W / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))

    # Which view id is the left eye?  ffmpeg resolves 'vpos:left' from the
    # 3D reference-display SEI.  Decode one frame per eye and compare view ids.
    left_id = None
    try:
        fr = ffprobe_json(['-view_ids', '-1', '-select_streams', 'v:0', '-read_intervals', '%+#2',
                           '-show_entries', 'frame=pts_time:frame_side_data_list', path])
        ids = [sd['view_id'] for f in fr['frames'] for sd in f.get('side_data_list', [])
               if sd.get('side_data_type') == 'View ID']
        # Empirically (iPhone 15 Pro .. 17 Pro Max, ffmpeg 9.0.1): vpos:left == view id 1.
        # We cannot read the eye label from ffprobe JSON, so record the ids present and
        # let extract_views.py always use the 'vpos:left'/'vpos:right' specifiers.
        left_id = {'view_ids_present': sorted(set(ids)), 'note': 'use -map 0:v:0:vpos:left / vpos:right'}
    except Exception as e:  # noqa: BLE001
        left_id = {'error': str(e)}

    return {
        'file': path,
        'device': fmt_tags.get('com.apple.quicktime.model'),
        'software': fmt_tags.get('com.apple.quicktime.software'),
        'spatial_format_version': fmt_tags.get('com.apple.quicktime.spatial.format-version'),
        'has_gps': any(k.startswith('com.apple.quicktime.location') for k in fmt_tags),
        'width': W, 'height': H,
        'fps': frac(v['r_frame_rate']),
        'nb_frames': int(v.get('nb_frames', 0)),
        'duration_s': float(meta['format']['duration']),
        'codec': v['codec_name'], 'pix_fmt': v.get('pix_fmt'), 'color_transfer': v.get('color_transfer'),
        'baseline_m': baseline_m,
        'hfov_deg': hfov_deg,
        'disparity_adjust_frac_of_width': dadj,
        'disparity_adjust_px': dadj * W,
        'projection': next((sd.get('projection') for sd in v.get('side_data_list', [])
                            if sd.get('side_data_type') == 'Spherical Mapping'), None),
        'fx': fx, 'fy': fx, 'cx': (W - 1) / 2.0, 'cy': (H - 1) / 2.0,
        'left_view': left_id,
        # disparity (px) at 1 m: fx * baseline
        'disparity_px_at_1m': fx * baseline_m,
    }


def write_k(info, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    K = [info['fx'], 0.0, info['cx'], 0.0, info['fy'], info['cy'], 0.0, 0.0, 1.0]
    with open(path, 'w') as f:
        f.write(' '.join(f'{k:.6f}' for k in K) + '\n')
        f.write(f"{info['baseline_m']:.6f}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--k_out', default=None, help='write K.txt (demo_data/K.txt layout) here')
    a = ap.parse_args()
    info = probe(a.video)
    print(json.dumps(info, indent=2))
    if a.k_out:
        write_k(info, a.k_out)
        print(f'wrote {a.k_out}', file=sys.stderr)
