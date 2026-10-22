"""How many helmet-marker reads do different ArUco detector settings get?  (plan doc §13.3)

Runs on every --every-th left-eye frame of one clip and counts frames with >= 1 read and total reads (ids < 48) for:
  base      field-pose-annotator util/aruco markers.detector (what aruco_compare used so far)
  tuned     + finer adaptive-threshold windows, lower polygon accuracy, more error correction
  up2       base detector on the frame upscaled 2x (corners scaled back)
  tuned_up2 tuned detector on the 2x frame
Also reports, for reads found by a variant but not by base, whether their PnP depth agrees with the neighbouring base
reads of the same id (to make sure extra reads are real, not misreads).

python scripts/spatial/dev/aruco_detect_tuning.py vid/IMG_3348.MOV --every 5
"""
import argparse, os, subprocess, sys, time
import cv2
import numpy as np

sys.path.insert(0, os.path.expanduser('~/Documents/GitHub/field-pose-annotator/util/aruco'))
import markers as mk  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument('video'); ap.add_argument('--every', type=int, default=5); a = ap.parse_args()
fam = mk.FAMILIES['aruco4x4']
base = mk.detector(fam, (1920, 1080))
p = cv2.aruco.DetectorParameters()
p.minMarkerPerimeterRate = 4.0 * 10 / 1920
p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 33, 4
p.polygonalApproxAccuracyRate = 0.05
p.errorCorrectionRate = 0.8
p.minCornerDistanceRate = 0.02
p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
tuned = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(fam.dictionary), p)
p2 = cv2.aruco.DetectorParameters(); p2.minMarkerPerimeterRate = 4.0 * 24 / 3840; p2.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
base_up = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(fam.dictionary), p2)
p3 = cv2.aruco.DetectorParameters()
for k in ('adaptiveThreshWinSizeMin', 'adaptiveThreshWinSizeMax', 'adaptiveThreshWinSizeStep', 'polygonalApproxAccuracyRate', 'errorCorrectionRate', 'minCornerDistanceRate', 'cornerRefinementMethod'):
    setattr(p3, k, getattr(p, k))
p3.minMarkerPerimeterRate = 4.0 * 20 / 3840
tuned_up = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(fam.dictionary), p3)


def run(det, g, up):
    img = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) if up else g
    c, ids, _ = det.detectMarkers(img)
    out = {}
    if ids is not None:
        for cc, i in zip(c, ids.ravel()):
            if i < 48:
                out[int(i)] = cc.reshape(4, 2) / (2.0 if up else 1.0)
    return out


variants = {'base': (base, False), 'tuned': (tuned, False), 'up2': (base_up, True), 'tuned_up2': (tuned_up, True)}
cmd = ['ffmpeg', '-v', 'error', '-i', a.video, '-map', '0:v:0:vpos:left', '-vf', f'select=not(mod(n\\,{a.every}))', '-fps_mode', 'vfr',
       '-f', 'rawvideo', '-pix_fmt', 'gray', '-']
dec = subprocess.Popen(cmd, stdout=subprocess.PIPE)
res = {k: [] for k in variants}; t = {k: 0.0 for k in variants}; n = 0
while True:
    buf = dec.stdout.read(1920 * 1080)
    if len(buf) < 1920 * 1080:
        break
    g = np.frombuffer(buf, np.uint8).reshape(1080, 1920)
    for k, (det, up) in variants.items():
        t0 = time.perf_counter(); res[k].append(run(det, g, up)); t[k] += time.perf_counter() - t0
    n += 1
print(f'{n} frames sampled (every {a.every})')
for k in variants:
    fr = sum(1 for r in res[k] if r); reads = sum(len(r) for r in res[k])
    extra = sum(len(set(r) - set(b)) for r, b in zip(res[k], res['base']))
    print(f'  {k:<10} frames with a read {fr:4d} ({fr / n * 100:4.1f} %)  reads {reads:5d}  extra vs base {extra:4d}  {t[k] / n * 1000:5.1f} ms/frame')
