#!/usr/bin/env python3
"""Run the whole iPhone spatial-video pipeline over many clips (plan doc §13).

Two cooperating workers so the CPU stages overlap the GPU stages:

  --stage gpu   per clip: extract_views -> run_sequence (depth) -> left-eye MP4 -> field-pose-annotator
                preprocess_video (2D pose + VideoPose3D); writes work/<id>_full/.gpu_done
  --stage post  per clip (after .gpu_done): make_depth_video (preview / depth_mm / overlay) -> pose_depth
                (pose_compare / pose_fused / fused tracks package) -> dinf_from_bodies -> eval_depth
                (+ 8-frame left-right check) -> stills for the docs -> copy deliverables -> cleanup
                (PNG frames and depth npy removed; depth stays in depth_mm.mkv, disparity npy kept);
                writes work/<id>_full/.post_done

Every step is skipped when its output already exists, so a stage can be restarted.
The gpu stage waits while two clips are done on the GPU but not yet post-processed (disk bound).

Usage (two terminals / background jobs, same arguments):
  python scripts/spatial/batch_process.py --stage gpu  --ids 3345 3346 ... --deliver_dir "%USERPROFILE%/Desktop/spatial_depth_results_2026-10-21"
  python scripts/spatial/batch_process.py --stage post --ids 3345 3346 ... --deliver_dir ...
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
SP = os.path.join(ROOT, 'scripts', 'spatial')


def log(msg, fh):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    fh.write(line + '\n'); fh.flush()


def run(cmd, fh, cwd=ROOT):
    log('$ ' + ' '.join(str(c) for c in cmd), fh)
    t0 = time.perf_counter()
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    tail = '\n'.join(p.stdout.strip().splitlines()[-6:])
    fh.write(p.stdout + '\n'); fh.flush()
    log(f'  -> exit {p.returncode} in {time.perf_counter() - t0:.0f} s' + (f'\n{tail}' if p.returncode else ''), fh)
    if p.returncode:
        raise RuntimeError(f'step failed: {cmd[1] if len(cmd) > 1 else cmd}')
    return time.perf_counter() - t0


def stage_gpu(a, vid, fh, timing):
    d = f'work/{vid}_full'; mov = f'vid/IMG_{vid}.MOV'
    stride = a.stride_long if json.load(open(f'work/probe_{vid}.json'))['duration_s'] > a.long_s else 1
    if not os.path.exists(f'{d}/meta.json'):
        timing['extract_s'] = run([a.python, f'{SP}/extract_views.py', mov, d, '--stride', stride], fh)
    if not os.path.exists(f'{d}/out/run_meta.json'):
        timing['depth_s'] = run([a.python, f'{SP}/run_sequence.py', '--seq_dir', d, '--out_dir', f'{d}/out', '--valid_iters', 8, '--max_disp', 96,
                                 '--volume', 'triton', '--auto_affine', 10, '--dinf', a.dinf, '--pc_every', 0, '--zfar', 8, '--save_vis', 0], fh)
    os.makedirs(f'{d}/pose', exist_ok=True)
    left = f'{d}/pose/IMG_{vid}_left.mp4'
    if not os.path.exists(left):
        timing['left_mp4_s'] = run(['ffmpeg', '-hide_banner', '-v', 'error', '-y', '-i', mov, '-map', '0:v:0:vpos:left', '-an', '-fps_mode', 'passthrough',
                                    '-c:v', 'libx264', '-preset', 'fast', '-crf', 12, '-pix_fmt', 'yuv420p', left], fh)
    if not os.path.exists(f'{d}/pose/IMG_{vid}_left.tracks.json'):
        timing['pose_s'] = run([a.pose_python, f'{a.fpa_dir}/scripts/preprocess_video.py', '--video', left, '--output-dir', f'{d}/pose',
                                '--models-dir', 'work/fpa_models', '--backend', 'rtmlib', '--tier', 'full', '--device', 'cuda:0'], fh)
    open(f'{d}/.gpu_done', 'w').write(json.dumps(dict(timing, stride=stride)))


def stage_post(a, vid, fh, timing):
    d = f'work/{vid}_full'; pose = f'{d}/pose'; stem = f'IMG_{vid}_left'
    if not os.path.exists(f'{d}/video/video_meta.json'):
        timing['videos_s'] = run([a.python, f'{SP}/make_depth_video.py', '--seq_dir', d, '--out_dir', f'{d}/out', '--video_dir', f'{d}/video', '--overlay', 1], fh)
    if not os.path.exists(f'{pose}/{stem}.pose_depth.json'):
        timing['pose_depth_s'] = run([a.python, f'{SP}/pose_depth.py', '--tracks', f'{pose}/{stem}.tracks.npz', '--seq_dir', d, '--out_dir', f'{d}/out'], fh)
    if not os.path.exists(f'{pose}/dinf_bodies.json'):
        try:
            run([a.python, f'{SP}/dinf_from_bodies.py', '--tracks', f'{pose}/{stem}.tracks.npz', '--seq_dir', d, '--out_dir', f'{d}/out', '--json', f'{pose}/dinf_bodies.json'], fh)
        except RuntimeError as e:
            log(f'  dinf_from_bodies skipped: {e}', fh)
    if not os.path.exists(f'{d}/eval.json'):
        timing['eval_s'] = run([a.python, f'{SP}/eval_depth.py', '--seq_dir', d, '--out_dir', f'{d}/out', '--lr_frames', 8, '--json', f'{d}/eval.json'], fh)
    # stills for the docs (middle of the clip)
    os.makedirs(f'docs/results/{vid}', exist_ok=True)
    dur = json.load(open(f'work/probe_{vid}.json'))['duration_s']
    for src, name in ((f'{d}/video/preview.mp4', 'preview_mid.jpg'), (f'{pose}/pose_compare.mp4', 'pose_compare_mid.jpg'), (f'{pose}/pose_fused.mp4', 'pose_fused_mid.jpg')):
        if os.path.exists(src) and not os.path.exists(f'docs/results/{vid}/{name}'):
            run(['ffmpeg', '-hide_banner', '-v', 'error', '-y', '-ss', f'{dur / 2:.2f}', '-i', src, '-frames:v', 1, '-q:v', 3, f'docs/results/{vid}/{name}'], fh)
    # deliverables
    out = os.path.join(a.deliver_dir, f'IMG_{vid}'); os.makedirs(out, exist_ok=True)
    files = [f'{d}/video/preview.mp4', f'{d}/video/overlay.mp4', f'{d}/video/depth_mm.mkv', f'{pose}/pose_compare.mp4', f'{pose}/pose_fused.mp4',
             f'{pose}/{stem}_fused.tracks.npz', f'{pose}/{stem}_fused.tracks.json', f'{pose}/{stem}.pose_depth.json', f'{d}/eval.json', f'{d}/K.txt']
    for f in files:
        if os.path.exists(f):
            dst = os.path.join(out, os.path.basename(f))
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(f):
                shutil.copy2(f, dst)
        else:
            log(f'  missing deliverable {f}', fh)
    # cleanup: frames and depth npy (depth is in depth_mm.mkv, and is fx*b/disp of the kept disparity)
    freed = 0
    for pat in (f'{d}/left/*.png', f'{d}/right/*.png', f'{d}/out/depth/*.npy'):
        for p in glob.glob(pat):
            freed += os.path.getsize(p); os.remove(p)
    log(f'  cleanup freed {freed / 1e9:.1f} GB', fh)
    open(f'{d}/.post_done', 'w').write(json.dumps(timing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['gpu', 'post'])
    ap.add_argument('--ids', nargs='+', required=True)
    ap.add_argument('--deliver_dir', required=True)
    ap.add_argument('--dinf', type=float, default=-1.0)
    ap.add_argument('--long_s', type=float, default=90.0, help='clips longer than this use --stride_long')
    ap.add_argument('--stride_long', type=int, default=2)
    ap.add_argument('--max_ahead', type=int, default=2)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--pose_python', default=os.path.expanduser('~/.venvs/field-pose-annotator-pose/Scripts/python.exe'))
    ap.add_argument('--fpa_dir', default=os.path.expanduser('~/Documents/GitHub/field-pose-annotator'))
    a = ap.parse_args()
    os.chdir(ROOT)
    tools = glob.glob(os.path.join(ROOT, 'tools', 'ffmpeg-*', 'bin'))
    if tools:
        os.environ['PATH'] = tools[0] + os.pathsep + os.environ['PATH']
    fh = open(f'work/batch_{a.stage}.log', 'a', encoding='utf-8')
    log(f'=== stage {a.stage}: {len(a.ids)} clips', fh)
    for vid in a.ids:
        d = f'work/{vid}_full'
        done_flag = f'{d}/.gpu_done' if a.stage == 'gpu' else f'{d}/.post_done'
        if os.path.exists(done_flag):
            log(f'{vid}: already done', fh); continue
        if a.stage == 'gpu':
            while sum(os.path.exists(f'work/{v}_full/.gpu_done') and not os.path.exists(f'work/{v}_full/.post_done') for v in a.ids) >= a.max_ahead:
                time.sleep(20)
        else:
            while not os.path.exists(f'{d}/.gpu_done') and not os.path.exists(f'{d}/.gpu_failed'):
                time.sleep(20)
            if os.path.exists(f'{d}/.gpu_failed'):
                log(f'{vid}: GPU stage failed, skipped', fh); continue
        t0 = time.perf_counter(); timing = {}
        log(f'--- {vid} start', fh)
        try:
            (stage_gpu if a.stage == 'gpu' else stage_post)(a, vid, fh, timing)
            log(f'--- {vid} {a.stage} done in {time.perf_counter() - t0:.0f} s  {timing}', fh)
        except Exception as e:  # noqa: BLE001
            log(f'!!! {vid} FAILED: {e}', fh)
            if a.stage == 'gpu':
                open(f'{d}/.gpu_failed', 'w').write(str(e))
    log(f'=== stage {a.stage} finished', fh)


if __name__ == '__main__':
    main()
