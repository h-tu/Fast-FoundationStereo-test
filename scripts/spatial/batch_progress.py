#!/usr/bin/env python3
"""Live progress of a batch_process.py run.

Per clip: stage (waiting / extract / depth n/N / pose / post / done / FAILED), frames, and an overall ETA from
the measured GPU rate.  `--watch S` refreshes every S seconds (Ctrl+C to stop).

  python scripts/spatial/batch_progress.py --ids 3354 3346 ... [--watch 10]
"""
import argparse, glob, json, os, time


def status(vid):
    d = f'work/{vid}_full'
    pr = json.load(open(f'work/probe_{vid}.json')) if os.path.exists(f'work/probe_{vid}.json') else {}
    stride = 2 if pr.get('duration_s', 0) > 90 else 1
    total = (pr.get('nb_frames', 0) + stride - 1) // stride
    if os.path.exists(f'{d}/.gpu_failed'):
        return 'GPU stage failed (batch_finish.py retries it after the GPU stage)', total, 0
    if os.path.exists(f'{d}/.post_done'):
        return 'done', total, total
    if os.path.exists(f'{d}/.gpu_done'):
        return 'post-processing (videos, pose fusion, eval)', total, total
    if os.path.exists(f'{d}/out/run_meta.json'):
        return 'pose (2D + VideoPose3D)', total, total
    n = len(glob.glob(f'{d}/out/disp/*.npy'))
    if n:
        return f'depth {n}/{total}', total, n
    if os.path.exists(f'{d}/left'):
        return 'extracting frames', total, 0
    return 'waiting', total, 0


def show(ids, t_start):
    rows = [(v, *status(v)) for v in ids]
    tot = sum(r[2] for r in rows); done = sum(r[3] for r in rows)
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f'batch progress  {time.strftime("%H:%M:%S")}\n')
    for v, st, n, k in rows:
        bar = '#' * int(20 * k / max(n, 1))
        ar = ('aruco done' if os.path.exists(f'work/{v}_full/.aruco_done') else '') + ('  RECALIBRATED' if os.path.exists(f'work/{v}_full/.recal_done') else '')
        print(f'  IMG_{v}  {n:5d} frames  [{bar:<20}]  {st:<46} {ar}')
    # GPU rate from finished clips' timings
    rates = []
    for v in ids:
        p = f'work/{v}_full/.gpu_done'
        if os.path.exists(p):
            t = json.load(open(p)); fr = status(v)[1]
            s = sum(x for k, x in t.items() if k.endswith('_s'))
            if s > 0:
                rates.append(fr / s)
    eta = f'{(tot - done) / (sum(rates) / len(rates)) / 3600:.1f} h (GPU stage; post adds ~5 min after the last clip)' if rates else 'n/a yet'
    print(f'\n  frames through depth inference: {done}/{tot} ({100 * done / max(tot, 1):.0f} %),  ETA {eta}')
    if os.path.exists('work/batch_finish_status.json'):
        try:
            s = json.load(open('work/batch_finish_status.json'))
            extra = ', '.join(f'{k} {v}' for k, v in s.items() if k not in ('phase', 'time'))
            print(f'  follow-up (batch_finish.py): {s["phase"]} since {s["time"]}' + (f'  [{extra}]' if extra else ''))
        except (OSError, ValueError):
            pass
    for name in ('work/batch_gpu.log', 'work/batch_post.log', 'work/batch_aruco.log', 'work/batch_recal.log'):
        if os.path.exists(name):
            last = [l for l in open(name, encoding='utf-8', errors='replace').read().splitlines() if l.startswith('[')][-1:]
            print(f'  {os.path.basename(name)}: {last[0][:150] if last else ""}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', nargs='+', required=True); ap.add_argument('--watch', type=float, default=0)
    a = ap.parse_args()
    t0 = time.time()
    while True:
        show(a.ids, t0)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == '__main__':
    main()
