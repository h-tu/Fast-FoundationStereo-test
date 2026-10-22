"""How much does a marker bent over a helmet bias the flat-square PnP?  (plan doc §14.6)

An 88.9 mm ArUco sticker is wrapped (arc length preserved) on a curved patch with horizontal radius Rh and vertical
radius Rv (Rv = inf: cylinder bent left-right only).  Its four corners are projected with the stereo K at a distance,
the helmet turned by a yaw angle, plus 0.15 px corner noise; then solvePnP IPPE_SQUARE with the flat 88.9 mm model
(what aruco_compare.py does) is solved and compared with the true depth of the sticker's surface centre.

Reported per case: depth error of the PnP centre vs the surface centre (mm and %), and the equivalent disparity error
d_marker - d_true in px (the quantity that enters the stereo offset fit).

python scripts/spatial/dev/curved_marker_sim.py
"""
import cv2
import numpy as np

K = np.array([[1554.372464, 0, 959.5], [0, 1554.372464, 539.5], [0, 0, 1]])
fxb = 1554.372464 * 0.019255
S = 0.0889; h = S / 2


def wrapped_corners(Rh, Rv):
    """Corners of a square sticker of side S wrapped on a patch with radii Rh (about the vertical axis) and Rv.
    Surface centre at the origin, normal +z towards the camera (z of the corners is negative = recede)."""
    pts = []
    for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1)):           # ArUco order TL, TR, BR, BL (y up)
        ax, ay = sx * h, sy * h                                   # arc lengths from the centre
        x = Rh * np.sin(ax / Rh) if np.isfinite(Rh) else ax
        y = Rv * np.sin(ay / Rv) if np.isfinite(Rv) else ay
        z = -((Rh * (1 - np.cos(ax / Rh)) if np.isfinite(Rh) else 0) + (Rv * (1 - np.cos(ay / Rv)) if np.isfinite(Rv) else 0))
        pts.append((x, y, z))
    return np.array(pts)


def run(Rh, Rv, dist, yaw_deg, noise=0.15, trials=300, rng=np.random.default_rng(0)):
    P = wrapped_corners(Rh, Rv)
    # camera frame: x right, y down, z forward; the sticker faces the camera (normal -z in camera), turned by yaw about y
    a = np.radians(yaw_deg)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    flip = np.diag([1, -1, -1])                                   # sticker y up -> camera y down, normal towards camera
    Xc = (Ry @ flip @ P.T).T + np.array([0.0, 0.0, dist])
    uv = (K @ Xc.T).T; uv = uv[:, :2] / uv[:, 2:]
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    errs_z = []
    for _ in range(trials):
        img = (uv + rng.normal(0, noise, uv.shape)).astype(np.float32)
        n, rv, tv, e = cv2.solvePnPGeneric(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        t = np.asarray(tv[int(np.argmin(np.asarray(e).ravel()))]).ravel()
        errs_z.append(t[2] - dist)                                # true surface-centre depth = dist
    ez = float(np.median(errs_z))
    dd = fxb / (dist + ez) - fxb / dist
    side_px = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, 0), axis=1)))
    return ez, dd, side_px


print('flat-square PnP on a bent 88.9 mm sticker: depth error of the PnP centre vs the true surface centre')
print(f'{"shape":<28}{"dist":>6}{"yaw":>6}{"side px":>9}{"dz mm":>9}{"dz %":>8}{"d_marker err px":>17}')
for name, Rh, Rv in (('flat', np.inf, np.inf), ('cylinder R 120 mm', 0.12, np.inf), ('cylinder R 100 mm', 0.10, np.inf),
                     ('dome Rh 120, Rv 180 mm', 0.12, 0.18), ('dome Rh 100, Rv 150 mm', 0.10, 0.15), ('sphere R 110 mm', 0.11, 0.11)):
    for dist in (2.0, 3.5, 5.0):
        for yaw in (0, 30, 50):
            ez, dd, px = run(Rh, Rv, dist, yaw)
            print(f'{name:<28}{dist:6.1f}{yaw:6d}{px:9.1f}{ez * 1000:9.1f}{ez / dist * 100:8.2f}{dd:17.3f}')
