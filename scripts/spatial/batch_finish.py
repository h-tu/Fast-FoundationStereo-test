#!/usr/bin/env python3
"""Finish a batch_process.py run: retry failed clips, compute the session d_inf correction, recalibrate everything.

Phases (current phase in work/batch_finish_status.json, shown by batch_progress.py):
  1. wait   until the gpu stage has finished (its log says 'stage gpu finished')
  2. retry  every clip with .gpu_failed: clear its partial output, then gpu -> post -> aruco for that clip
            (up to --retries times; a clip that still fails is left out of the session and reported)
  3. wait   until every remaining clip has .post_done and .aruco_done
  4. offset session_offset.py over all clips -> pooled correction (work/session_offset_<tag>.json)
  5. recal  batch_process.py --stage recal with that correction, --workers parallel processes
Restart-safe: every phase checks the marker files, so running it again continues where it stopped.

  python scripts/spatial/batch_finish.py --ids 3354 3346 ... --deliver_dir <folder> --tag 2026-10-21 [--workers 3]
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
SP = os.path.join(ROOT, 'scripts', 'spatial')
STATUS = 'work/batch_finish_status.json'


def status(phase, **kw):
    s = dict(phase=phase, time=time.strftime('%H:%M:%S'), **kw)
    json.dump(s, open(STATUS, 'w'), indent=2)
    print(f'[{s["time"]}] {phase} {kw if kw else ""}', flush=True)


def has(vid, flag):
    return os.path.exists(f'work/{vid}_full/.{flag}')


def bp(a, stage, ids, extra=()):
    cmd = [a.python, f'{SP}/batch_process.py', '--stage', stage, '--deliver_dir', a.deliver_dir, *extra, '--ids', *ids]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='+', required=True); ap.add_argument('--deliver_dir', required=True)
    ap.add_argument('--tag', default=time.strftime('%Y-%m-%d')); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--retries', type=int, default=1); ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args()
    os.chdir(ROOT)
    tries = {}

    status('wait: gpu stage running')
    while not (os.path.exists('work/batch_gpu.log') and 'stage gpu finished' in open('work/batch_gpu.log', encoding='utf-8', errors='replace').read()):
        time.sleep(30)

    while True:
        failed = [v for v in a.ids if has(v, 'gpu_failed') and tries.get(v, 0) < a.retries]
        if not failed:
            break
        for v in failed:
            tries[v] = tries.get(v, 0) + 1
            status('retry', clip=v, attempt=tries[v])
            os.remove(f'work/{v}_full/.gpu_failed')
            shutil.rmtree(f'work/{v}_full/out', ignore_errors=True)
            bp(a, 'gpu', [v])
            if has(v, 'gpu_done'):
                bp(a, 'post', [v]); bp(a, 'aruco', [v])
    dropped = [v for v in a.ids if has(v, 'gpu_failed')]
    ids = [v for v in a.ids if v not in dropped]

    status('wait: post-processing and marker comparison', dropped=dropped)
    while not all(has(v, 'post_done') and has(v, 'aruco_done') for v in ids):
        time.sleep(30)

    status('offset: session d_inf from all marker reads', dropped=dropped)
    out = f'work/session_offset_{a.tag}.json'
    subprocess.run([a.python, f'{SP}/session_offset.py', '--ids', *ids, '--json', out], cwd=ROOT, check=True)
    c = round(json.load(open(out))['pooled_offset_px'], 2)

    todo = [v for v in ids if not (has(v, 'recal_done') and json.load(open(f'work/{v}_full/.recal_done')).get('dinf_correction') == c)]
    for v in todo:                      # a clip recalibrated with another value is redone
        if has(v, 'recal_done'):
            os.remove(f'work/{v}_full/.recal_done')
    groups = [todo[k::a.workers] for k in range(a.workers)]
    status('recal', correction_px=c, clips=len(todo), dropped=dropped)
    procs = [subprocess.Popen([a.python, f'{SP}/batch_process.py', '--stage', 'recal', '--dinf_correction', str(c), '--deliver_dir', a.deliver_dir, '--ids', *g],
                              cwd=ROOT) for g in groups if g]
    for p in procs:
        p.wait()
    not_done = [v for v in ids if not has(v, 'recal_done')]
    status('finished', correction_px=c, recalibrated=len(ids) - len(not_done), not_recalibrated=not_done, dropped=dropped)


if __name__ == '__main__':
    main()
