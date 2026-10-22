# iPhone 17 Pro Max spatial video → Fast-FoundationStereo depth + skeletons (recorded 2026-10-21, processed 2026-10-22)

One folder per clip (IMG_3345, IMG_3352 and IMG_3378 at frame stride 2 = 14.985 fps; the other twelve at full 29.97 fps). Same settings as the 2026-10-14 batch (folder spatial_depth_results_2026-10-14). Full method, numbers and caveats:
`docs/iphone_spatial_video_plan.md` in the Fast-FoundationStereo-test repository (sections 10–13; batch summary of these clips in section 13).

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
| `aruco_compare.mp4` | ground-truth check: every helmet ArUco marker outlined and labelled with its PnP depth vs the stereo depth on the marker (colour = error) \| stereo depth map \| live scatter of stereo vs marker depth with the running error and disparity fit |
| `aruco_compare.json`, `aruco_reads.csv` | per-read marker and stereo positions, error statistics by distance, fitted disparity gain and true d_inf for the clip |

Caveats that matter when using the numbers:
- Depth is reliable inside about 5 m. Metric scale is right to ±10 % inside 4.5 m with the current d_inf = −1 px; beyond 5 m depths are 30–70 % too large (effective offset −2.5…−4 px). A calibration recording (A4 sheet / tape at 1, 2, 3, 5, 8 m) is needed to fix the far range.
- Pixels the model classifies as far (sky, tree line, field) are 0 in `depth_mm.mkv`; d < 0.5 px is a class, not a measurement.
- Joint depths are window samples, so people in front of textured background or partly hidden behind another person get wrong joint depths and inflated limb lengths.
- Videos contain recognisable people and the source files carry GPS; keep them within the project.


