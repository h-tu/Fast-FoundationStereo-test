"""Quick scan: how many ArUco 4x4_50 reads per clip (left eye, every 15th frame), sizes in px."""
import sys, subprocess, numpy as np, cv2
sys.path.insert(0, r'C:\Users\Admin\Documents\GitHub\field-pose-annotator\util\aruco')
import markers as mk
fam = mk.FAMILIES['aruco4x4']; det = mk.detector(fam, (1920, 1080))
for vid in sys.argv[1:]:
    cmd = ['ffmpeg', '-v', 'error', '-i', f'vid/IMG_{vid}.MOV', '-map', '0:v:0:vpos:left', '-vf', 'select=not(mod(n\\,15))', '-fps_mode', 'vfr',
           '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = reads = 0; ids = {}; sides = []
    while True:
        buf = p.stdout.read(1920 * 1080)
        if len(buf) < 1920 * 1080:
            break
        g = np.frombuffer(buf, np.uint8).reshape(1080, 1920)
        c, i, _ = det.detectMarkers(g)
        n += 1
        if i is not None:
            for cc, ii in zip(c, i.ravel()):
                if ii < 48:
                    reads += 1; ids[int(ii)] = ids.get(int(ii), 0) + 1
                    sides.append(float(np.mean(np.linalg.norm(cc[0] - np.roll(cc[0], 1, 0), axis=1))))
    s = f'median side {np.median(sides):.0f} px [{np.min(sides):.0f}, {np.max(sides):.0f}]' if sides else ''
    print(f'{vid}: {n} frames sampled, {reads} reads, frames-with-read share n/a, ids {dict(sorted(ids.items()))} {s}', flush=True)
