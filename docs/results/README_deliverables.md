# iPhone 17 Pro Max spatial video → Fast-FoundationStereo depth + skeletons (2026-10-14)

One folder per clip (IMG_3142 at frame stride 2 = 14.985 fps; the others at full 29.97 fps). Full method, numbers and caveats:
`docs/iphone_spatial_video_plan.md` in the Fast-FoundationStereo-test repository (sections 10–12).

| File | What it is |
|---|---|
| `preview.mp4` | left eye \| depth false colour (turbo, fixed 0.3–8 m, log scale, near = red) \| colour bar; black = invalid or > 8 m |
| `overlay.mp4` | left eye with the depth colours blended on top, full resolution |
| `depth_mm.mkv` | **the data**: uint16 depth in millimetres per pixel, 0 = invalid, FFV1 lossless, gray16le, 1920×1080, aligned with the left eye. Decode: `ffmpeg -i depth_mm.mkv -fps_mode passthrough -f rawvideo -pix_fmt gray16le - \| np.frombuffer(..., '<u2').reshape(-1, 1080, 1920)` |
| `K.txt` | line 1: 3×3 intrinsics row-major (fx = fy = 1554.37, cx = 959.5, cy = 539.5); line 2: baseline 0.019255 m. Back-projection: X = (u − cx)·Z/fx, Y = (v − cy)·Z/fy |
| `pose_compare.mp4` | validation video: 2D skeleton coloured by stereo depth with each worker's distance \| skeleton on the depth map \| top view (metres): stereo skeleton filled, scaled VideoPose3D hollow, per-worker disagreement |
| `pose_fused.mp4` | adjusted VideoPose3D (scaled per person, anchored at the stereo pelvis, limb depth flips corrected) reprojected on the image \| front view \| top view, metres |
| `IMG_xxxx_left_fused.tracks.npz/.json` | field-pose-annotator tracks package whose `pose_3d` is the adjusted metric skeleton (H36M-17, camera frame x right / y down / z forward, metres). Loads in the annotator |
| `IMG_xxxx_left.pose_depth.json` | per-track metrics of the stereo-vs-VideoPose3D comparison, bone lengths, scale factors |
| `eval.json` | depth-quality tests of the clip (coverage, flicker, road-plane fit, block-matching agreement, left-right consistency) |

Caveats that matter when using the numbers:
- Depth is reliable inside about 5 m. Metric scale is right to ±10 % inside 4.5 m with the current d_inf = −1 px; beyond 5 m depths are 30–70 % too large (effective offset −2.5…−4 px). A calibration recording (A4 sheet / tape at 1, 2, 3, 5, 8 m) is needed to fix the far range.
- Pixels the model classifies as far (sky, tree line, field) are 0 in `depth_mm.mkv`; d < 0.5 px is a class, not a measurement.
- Joint depths are window samples, so workers standing in front of the corn field get inflated limb lengths (3142, 3143).
- VideoPose3D is root-relative. The stereo-adjusted package adds per-person metric scale and an absolute pelvis position in the camera coordinate system, but the current results do not demonstrate a meaningful improvement in relative joint accuracy: the median change from the limb-depth correction is only 0.2–0.5 cm, and the reported runs used `--fuse_weight 0`. Stereo is also the anchor and the comparison reference, so this is not an independent absolute-pose accuracy test. See §12.6 of `docs/iphone_spatial_video_plan.md` for the required before/after evaluation.
- Videos contain recognisable people and the source files carry GPS; keep them within the project.
