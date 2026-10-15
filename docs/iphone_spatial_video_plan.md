# iPhone Spatial Video → Fast-FoundationStereo: feasibility, measurements and execution plan

> Status (2026-10-14): format parsing, geometry checks and tooling were done on a laptop (macOS, M1 Pro, no CUDA);
> model inference, metric-scale estimation and the depth videos were done on the CUDA box (Windows 11, RTX 3090) — see §8.1 and §10.
> Current result: all four videos converted to depth videos (`work/<id>_full/video/`, 10 s clips in `docs/results/<id>/`), d_inf = −1.0 px from the far-background method only
> (pending verification with a known-distance object), 1.27 frames/s end to end at 1920×1080. Evaluation and recommendations in §11.
> Skeletons (field-pose-annotator) combined with the depth, VideoPose3D validated and adjusted, per-clip videos in `work/<id>_full/pose/`: §12.
> §12.3 shows the metric scale is right inside ~4.5 m and drifts beyond 5 m (effective offset −2.5…−4 px at 6–9 m); a calibration target is still needed.
> The source videos live in `vid/` and are excluded by `.gitignore` (`vid/`, `*.MOV`, `work/`). **Do not** change that rule:
> the files carry GPS coordinates (`com.apple.quicktime.location.ISO6709`) and recognisable faces.

---

## 1. Goal

Use spatial video (MV-HEVC stereo) recorded on an iPhone 17 Pro Max as input to Fast-FoundationStereo and obtain a **metric depth map / point cloud per frame**.
The objects of interest are within **5 m**.

Three questions and their answers:

| Question | Answer |
|---|---|
| Can the left/right eyes be split out and fed to the model? | Yes. ffmpeg ≥ 7.1 decodes MV-HEVC natively; verified with 9.0.1 on the laptop and 9.0.2 on the CUDA box |
| Does different hardware make the model inaccurate? | The model is hardware-agnostic; it only requires **rectified input with correct left/right**. Physical parameters enter only in the disparity→depth step |
| Do we need our own calibration? | Not up front. Baseline and field of view are in the file metadata (Apple factory calibration). Measurements show the right eye has a **row-dependent vertical offset (about ±2.5 px)**, fixed with one affine warp; the metric scale still needs one check against an object at known distance |

---

## 2. Model-side facts (from the paper and the code)

Paper: Wen, Dewan, Birchfield, *Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching*, CVPR 2026, arXiv:2512.11130v2.

- Architecture: distilled EdgeNeXt-small backbone → GWC (8 groups) + concat cost volume → NAS-searched 3D hourglass (with 4 Disparity Transformer layers) → softmax regression of the initial disparity → single-layer SelectiveConvGRU refinement → convex upsampling. 14.6 M parameters.
- **Interface** (`core/foundation_stereo.py::FastFoundationStereo.forward`):
  - Input: `image1, image2`, `(B,3,H,W)` float, RGB, **0–255**; H and W must be multiples of 32 (replicate-pad with `core/utils/utils.py::InputPadder`). ImageNet normalisation is done inside.
  - Keywords: `iters` (GRU iterations, README uses 8), `test_mode=True` (return only the final disparity), `optimize_build_volume='pytorch1'|'triton'`.
  - Output: `(B,1,H,W)` disparity in **left-image pixels**; positive means the match in the right image is further left (d = x_L − x_R).
  - The network **does not take** K or the baseline; only the post-processing does `depth = fx * baseline / disp`.
- Training data: synthetic data + 1.4 M Stereo4D internet VR180 pairs with pseudo-labels. The paper says it also works on monochrome / IR (RealSense), i.e. some tolerance to photometric differences between eyes.
- Input assumptions: horizontal epipolar lines, no distortion, do not swap left/right. `max_disp` defaults to 192 (48 at 1/4 resolution).
- Zero-shot accuracy (paper Table 1): Middlebury-H BP-2 2.20 (teacher 1.10, RT-IGEV 7.82). About 49 ms at 640×480 on a 3090, 21 ms with TensorRT.
- The code is inference / export / profiling only, no training; the weights are a whole pickled module loaded with `torch.load`, so the architecture is defined by the checkpoint. Everything hard-codes `.cuda()`, `autocast('cuda')`, `torch.compile` and Triton.

---

## 3. iPhone spatial video format (verified on the four local videos)

Four videos `vid/IMG_3142..3145.MOV`, all iPhone 17 Pro Max, iOS 27.2, road-works scene (people, truck, corn field).

| Item | Value | Source |
|---|---|---|
| Container / codec | QuickTime, HEVC Main, MV-HEVC with two views | ffprobe |
| Resolution / rate | 1920×1080 @ 29.97 fps, yuv420p, BT.709 SDR | ffprobe |
| Bitrate | ≈ 25 Mbps (both eyes; the second layer stores only the residual) | ffprobe |
| Baseline `blin` | **19 255 µm = 19.255 mm** | Stereo3D side data `baseline` |
| Horizontal FOV `hfov` | **63.4°** | `horizontal_field_of_view = 63400/1000` |
| Comfort disparity adjustment `dadj` | 200/10000 = 2 % of width = 38.4 px | `horizontal_disparity_adjustment`; **playback only, not baked into pixels, nothing to undo** |
| Projection | rectilinear | Spherical Mapping side data |
| Left eye | ffmpeg `vpos:left` ↔ **view_id 1**; `vpos:right` ↔ view_id 0 | pixel-wise comparison |
| Hero eye | main (Wide) camera; the right eye is the ultra-wide, cropped, upscaled and aligned in post | Apple docs |
| Timestamps | identical for both eyes (two frames per pts) | ffprobe `-view_ids -1` |

Derived pinhole parameters (principal point at the image centre):

```
fx = fy = W / (2·tan(hfov/2)) = 1920 / (2·tan(31.7°)) = 1554.37 px
cx = 959.5, cy = 539.5
baseline = 0.019255 m
fx·b = 29.93 px·m   →  disparity ≈ 30 px at 1 m, ≈ 6 px at 5 m
```

---

## 4. Geometry check results (ZNCC block matching, `scripts/spatial/check_rectification.py`)

48×48 block matching on 9 sampled frame pairs (4 videos, different times), statistics of dy = y_R − y_L and d = x_L − x_R:

### 4.1 Vertical offset: a systematic **row-dependent** offset exists

| Frame | dy median | dy std | \|dy\|<1 px | fit dy = a·y + b | residual after affine |
|---|---|---|---|---|---|
| 3142 t=10 | +0.87 | 0.90 | 57 % | a=−2.73‰, b=+2.29 | 0.45 |
| 3142 t=60 | +0.93 | 1.25 | 46 % | a=−4.52‰, b=+3.40 | 0.32 |
| 3142 t=110 | +0.88 | 1.48 | 45 % | a=−5.16‰, b=+4.20 | 0.31 |
| 3143 t=5 | +1.04 | 1.31 | 43 % | a=−3.74‰, b=+3.01 | — |
| 3143 t=20 | +0.95 | 1.72 | 36 % | a=−4.85‰, b=+3.47 | 0.83 |
| 3144 t=5 | +0.70 | 1.22 | 53 % | a=−4.02‰, b=+2.93 | 0.38 |
| 3144 t=30 | +0.32 | 1.56 | 45 % | a=−5.32‰, b=+3.47 | 0.49 |
| 3145 t=5 | +0.62 | 1.38 | 51 % | a=−4.23‰, b=+2.98 | 0.75 |
| 3145 t=25 | +0.61 | 1.27 | 50 % | a=−4.64‰, b=+3.49 | 0.33 |

Interpretation: the right eye has a vertical **scale difference of about 0.4–0.5 %** relative to the left, with the zero crossing near y≈700; the top of the frame is off by +2–3 px, the bottom by −1 px.
dy hardly depends on x (slope < 0.8‰), so this is not a rotation but a vertical scale / principal-point residual from the ultra-wide crop-and-upscale, plus the per-frame crop of the stabiliser (a, b jitter by ±1‰ between frames).

**Fix**: a vertical-only affine on the right eye, `y_src = (1+a)·y_dst + b` (horizontal untouched, so disparity is unaffected). Verified on 3143 t=20:
dy std 1.72 → 0.60, fraction with |dy|<1 px 36 % → **94 %**, disparity median 7.42 → 7.45 (unchanged).
`run_sequence.py --auto_affine N` re-fits it every N frames.

The model samples the cost volume at 1/4 resolution, so a 2 px vertical error is 0.5 px there; it would most likely work without the fix, but the fix is cheap (one warpAffine) and recommended.

**Re-check on the CUDA box (2026-10-14, IMG_3143, one pair per second over the whole clip, 28 frames, `check_rectification.py --dir work/3143`)**:
a = −4.37 ‰ (range −5.38 … −3.72, std 0.42), b = +3.27 px (range 2.81 … 4.54, std 0.35);
mean dy median +0.91 px, dy std 1.44 → mean residual after affine 0.61 px (max 1.02). Consistent with the two 3143 rows above (a −3.74/−4.85‰, b 3.01/3.47);
the parameters are stable over the clip, only the last frame (t≈27 s) jumps to b = 4.54. The ±0.4‰ / ±0.35 px frame-to-frame jitter justifies periodic re-fitting with `--auto_affine`.

### 4.2 Horizontal: disparity at infinity is slightly negative

Far blocks (sky edge, distant corn field) have disparities of **−1.3 … −0.1 px**, p5 about −0.3 … −1.3 px. So there is a convergence offset of about **d_inf ≈ −0.5 ± 0.5 px** between the eyes, or a slight horizontal scale mismatch (the laptop data cannot separate the two).

Impact: metric depth is z = fx·b/(d − d_inf). At 3 m, d≈10 px, so a 0.5 px bias is a **5 % depth error**; at 1 m only 1.7 %.
Hence **d_inf must be calibrated once with an object at known distance** (§7 step 4), then `--dinf` shifts the whole right eye so that infinity returns to 0, which also avoids feeding negative disparities to a cost volume that only covers [0, max_disp).

**Re-check on the CUDA box (2026-10-14, §10 step 4)**: on the 28 frames of IMG_3143, using only the horizon tree band (y < 330, x < 900) with 32×32 / 48×48 ZNCC blocks (NCC ≥ 0.9, right eye vertically affine-corrected first),
the far cluster has a disparity median of **−0.95 px (P32, median over 13 frames, IQR −1.10…−0.86) / −1.02 px (P48)**; the histogram has a clear peak at −1.5…−0.75 px, and the mass near 0 is the nearer top of the corn field.
Since the far trees still have 0.1–0.3 px of true disparity, d_inf may be slightly below −1.0. **Currently used: d_inf = −1.0 px (far-background method only, pending verification with a real object).**
Also: the model itself outputs exactly 0 for "far" content regardless of input shift (a learned prior), so d_inf cannot be estimated from the model output — see §10 step 4.

### 4.3 Accuracy budget (1080p, fx=1554, b=19.255 mm)

| Distance | Disparity | Δz @ 0.25 px | Δz @ 0.5 px |
|---|---|---|---|
| 1 m | 29.9 px | 0.8 cm | 1.7 cm |
| 2 m | 15.0 px | 3.3 cm | 6.7 cm |
| 3 m | 10.0 px | 7.5 cm | 15 cm |
| 5 m | 6.0 px | 21 cm | 42 cm |

Δz = z²·Δd/(fx·b). Conclusion: reliable within 3 m, usable at 3–5 m with tens of centimetres of uncertainty.
**Do not use the README's `--scale 0.5`** (only 3 px of disparity left at 5 m); run at 1920×1088 (after padding).

---

## 5. Other risks (not quantified)

- The right eye is an ultra-wide crop upscaled about 2× and encoded as a low-bitrate residual: softer, different noise and tone. The model tolerates this to a degree, but low-texture regions (asphalt, sky) may be less stable.
- Stabilisation (`cinematicExtendedEnhanced`) crops per frame → the hfov metadata is nominal and the real fx may vary slightly between frames (< 1 %); whether the metadata hfov is the post-crop value is unknown, so the scale check in §7 step 4 is the safety net.
- Rolling shutter: two different sensors; during fast motion the row timing difference may break the epipolar assumption.
- HEVC compression artefacts; the README recommends lossless PNG, which we cannot have here.
- Beyond about 20 m the disparity is < 1.5 px and depth is essentially meaningless; truncate the point cloud with `--zfar`.

---

## 6. Tools and interfaces (`scripts/spatial/`)

Everything depends only on ffmpeg ≥ 7.1, numpy and opencv; the non-dry-run path of `run_sequence.py` additionally needs the repo environment (torch + CUDA, timm, …).

### 6.1 `probe_metadata.py` — read metadata, write K.txt
```
python scripts/spatial/probe_metadata.py vid/IMG_3143.MOV [--k_out work/3143/K.txt]
```
stdout is JSON: `device, width, height, fps, baseline_m, hfov_deg, disparity_adjust_px, fx, fy, cx, cy, has_gps, …`.
`K.txt` has the `demo_data/K.txt` layout: line 1 = 9 floats (K row-major), line 2 = baseline in metres.

### 6.2 `extract_views.py` — split left/right eyes into PNG sequences
```
python scripts/spatial/extract_views.py vid/IMG_3143.MOV work/3143 --start 5 --duration 2 --stride 15
```
Output:
```
work/3143/left/000000.png    # 1920x1080 RGB, lossless
work/3143/right/000000.png
work/3143/K.txt
work/3143/meta.json          # probe result + frame_time_s[] (source time of frame i)
```
The eye is selected with ffmpeg `-map 0:v:0:vpos:left|right` (from the file's own SEI), not a guessed view id.
A full 30 s clip is about 900 pairs × 2 × ~2.7 MB ≈ 5 GB; use `--stride` first.

### 6.3 `check_rectification.py` — rectification quality check
```
python scripts/spatial/check_rectification.py L.png R.png [--json out.json]
python scripts/spatial/check_rectification.py --dir work/3143 --every 10
```
Prints all statistics of §4; `suggested_right_vertical_affine.{a,b}` in the JSON can be passed directly to `run_sequence.py --right_affine a b`.
Python API: `match_patches(L, R) -> (N,5)[xL, yL, d, dy, ncc]`, `summarize(rows, W, H) -> dict`.

### 6.4 `run_sequence.py` — batch inference (CUDA)
```
python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out \
    --model weights/23-36-37/model_best_bp2_serialize.pth \
    --valid_iters 8 --max_disp 96 --volume triton \
    --auto_affine 10 --dinf -1.0 --pc_every 30 --zfar 8
```
Per frame: read left/right → one `warpAffine` on the right eye (vertical affine + horizontal shift of −dinf, bilinear, replicated border) → optional `--scale` → pad to /32 → `model.forward(test_mode=True)` → unpad → disparity (left-image px) → depth (m, `remove_invisible` logic as in `run_demo.py`).
Output:
```
out/disp/000000.npy      float32, disparity px (output resolution = input resolution × scale)
out/depth/000000.npy     float32, metres; invalid = 0
out/vis/000000.png       left | right (corrected) | disparity turbo
out/cloud/000000.ply     binary PLY (x y z r g b), one every pc_every frames, z ≤ zfar
out/run_meta.json        all parameters, K, warp matrix, per-frame timings, (a, b) of every auto_affine fit
```
`--dry_run` does not import torch and walks the whole I/O + geometry path with zero disparity.
`--max_disp 96` corresponds to a nearest distance of 0.31 m; go back to 192 if nearer objects matter.

### 6.5 `make_depth_video.py` — turn run_sequence output into videos (added on the CUDA box)
```
python scripts/spatial/make_depth_video.py --seq_dir work/3143_full --out_dir work/3143_full/out --video_dir work/3143_full/video --overlay 1
```
Output (frame rate = source fps / stride, written as the exact rational 30000/1001):
- `preview.mp4`: left image | depth false colour | colour bar. Turbo, **fixed range 0.3–8 m on a log scale**, near = red, far = blue; pixels with depth 0 or > 8 m are black;
  the bar has ticks at 0.3/0.5/1/2/3/5/8 m; frame index and source time in the top-left. `--preview_scale 0.5` → 1990×540, libx264 crf 20 yuv420p.
- `depth_mm.mkv`: uint16 depth in millimetres, 0 = invalid (including > 65.535 m), FFV1 level 3 lossless, gray16le, full resolution; bit-exact round trip against `depth/*.npy`.
  Decode: `ffmpeg -i depth_mm.mkv -fps_mode passthrough -f rawvideo -pix_fmt gray16le - | np.frombuffer(..., '<u2').reshape(-1, H, W)`
  (without `-fps_mode passthrough`, ffmpeg duplicates frames of a < 1 fps stride preview to a guessed constant rate).
- `overlay.mp4` (`--overlay 1`): left image with the depth colours alpha-blended on top (alpha 0.55), full resolution.
- `--temporal_median N` (odd, off by default): running median over N frames in the **disparity** domain before colouring/writing, to reduce flicker; `video_meta.json` records parameters, time and file sizes.

### 6.6 `eval_depth.py` — ground-truth-free evaluation of a run (added on the CUDA box)
```
python scripts/spatial/eval_depth.py --seq_dir work/3143_full --out_dir work/3143_full/out --lr_frames 8 --json work/3143_full/eval.json
```
Computes (see §11 for results and interpretation): valid-depth coverage in [zmin, zmax]; temporal flicker of disparity (deviation from the 3-frame temporal median, raw and after a 3-frame median filter); RANSAC plane fit of the road ROI (RMS, inlier fraction, camera height, tilt); agreement between the model and independent ZNCC block matching on textured near patches; and, with `--lr_frames N` (CUDA), a left-right consistency check obtained by running the model on the flipped (right, left) pair.

### 6.8 `pose_depth.py` and `dinf_from_bodies.py` — skeletons + depth (added on the CUDA box, §12)
```
python scripts/spatial/pose_depth.py --tracks work/3143_full/pose/IMG_3143_left.tracks.npz --seq_dir work/3143_full --out_dir work/3143_full/out
python scripts/spatial/dinf_from_bodies.py --tracks ... --seq_dir ... --out_dir ...
```
`pose_depth.py` takes a field-pose-annotator tracks package (COCO-17 2D keypoints + VideoPose3D H36M-17 3D per track) made on the left-eye video, samples the metric depth at every 2D joint (30th percentile of a 7×7 window, validity by 2D confidence, valid-pixel share and depth spread), back-projects with K (stereo 3D skeleton), maps it to H36M, scales and anchors VideoPose3D to it (one scale per track from the image-plane joint extents with one body depth per frame; pelvis from stereo, median-filtered), optionally mirrors a limb when stereo and VideoPose3D confidently disagree on its depth direction (`--fix_flips 1`, only for pelvis depth ≤ `--near_max` 5 m), optionally blends (`--fuse_weight`). Writes `<id>.pose_depth.npz/.json` (all arrays, metrics), `pose_compare.mp4` (left image with the depth-coloured 2D skeleton | depth map | top view with stereo filled and scaled VideoPose3D hollow), `pose_fused.mp4` (adjusted skeleton reprojected | front view | top view) and `<id>_fused.tracks.npz/.json`, a copy of the tracks package whose `pose_3d` is the adjusted metric skeleton (loads with the annotator's `load_tracks_package`). `--disp_offset` re-maps all sampled depths with an extra disparity offset for calibration experiments.
`dinf_from_bodies.py` uses people as calibration objects: lateral stereo size = px · b / (d − d_inf) is independent of fx, and VideoPose3D gives the same person in metres, so the ratio per track yields the d_inf that would make them agree (§12.3).

### 6.7 `estimate_dinf.py`, `dinf_blockmatch.py`, `build_trt_engine.py`
`estimate_dinf.py` reads a run made with a known `--dinf` shift and takes the low-disparity histogram mode of the top of the frame (it showed that the model's far output is a constant 0, §10 step 4a). `dinf_blockmatch.py` is the record of the far-horizon block-matching estimate that produced d_inf = −1.0 px. `build_trt_engine.py` builds a TensorRT engine from the single ONNX with the Python API (no `trtexec` needed), see §10 step 7.

---

## 7. Execution plan on the CUDA machine (original plan; status in §8.1, log in §10)

0. Environment: Docker from the README (`docker/dockerfile`) or pip (torch 2.6 + cu124, `requirements.txt`).
   Check `ffmpeg -version` ≥ 7.1 (the apt ffmpeg inside the image may be too old; download a static build if needed).
   Download the research checkpoint into `weights/23-36-37/` (Google Drive link in the README; `cfg.yaml` must sit next to the `.pth`).
1. Run `scripts/run_demo.py` on `demo_data/` once to confirm environment and weights (output should match `output_docker/disp_vis.png`).
2. Extraction: `--stride 30` first (one pair per second), then full rate.
3. Inference: the command in §6.4 with `--auto_affine 10`. Check `vis/` for sanity, in particular sky, asphalt and far background.
4. **Metric scale check (mandatory)**:
   - find a frame with an object of known size / distance (truck wheel hub, lane-line width, or place a tape measure / A4 sheet at 1, 2, 3 m in the next recording);
   - compare `depth/*.npy` with the truth and solve d_inf: `d_inf = d_model − fx·b/z_true`;
   - cross-check with the mean disparity of sky / far trees (should be ≈ d_inf, not 0);
   - rerun with `--dinf` set to that value. If the scale error is still > 3 %, suspect fx (stabiliser crop) and solve fx from the same object.
5. Comparison experiments (a few dozen frames each): `valid_iters 4 vs 8`, `scale 1.0 vs 0.75`, `max_disp 96 vs 192`, `--auto_affine` on/off; record accuracy and time (`run_meta.json`).
6. If real time is needed: export the single-file ONNX → TensorRT per README (`scripts/make_single_onnx.py --height 1088 --width 1920`); the input must be pre-normalised.
7. If the affine fix of §4.1 is insufficient (residual still > 1 px), the fallback is OpenCV `stereoCalibrate + stereoRectify` with a checkerboard on the **extracted pairs**; that calibrates Apple's virtual cameras and also yields fx and d_inf.

---

## 8. Verified facts

- [x] ffmpeg 9.0.1 `-view_ids` / `vpos:left|right` work; `-vsync` was removed, the scripts use `-fps_mode vfr`
- [x] Metadata identical across the four videos: baseline 19.255 mm, hfov 63.4°, dadj 2 %, 1080p29.97
- [x] vpos:left == view_id 1 (pixel-wise identical); geometrically view 1 content sits further right, matching the definition of "left eye"
- [x] The vertical affine fix of the right eye works (§4.1)
- [x] `extract_views.py`, `check_rectification.py`, `run_sequence.py --dry_run` run end to end
- [x] Model inference (CUDA) — see §10 from step 1 on
- [ ] Metric check of d_inf and fx (needs a ground-truth object)

### 8.1 Progress on the CUDA box (step numbers from prompt.txt, details in §10)

- [x] Step 0 environment and data self-check (Windows 11 + RTX 3090 + venv, static ffmpeg 9.0.2 in `tools/`)
- [x] Step 1 weights downloaded, `run_demo.py --headless 1` matches the reference output, 640×480 baseline timing
- [x] Step 2 extraction (stride 30) and geometry check (28 pairs; a = −4.37 ± 0.42 ‰, b = +3.27 ± 0.35 px)
- [x] Step 3 first real inference + comparison experiments (1920×1088, iters 8, max_disp 96, triton: 243 ms/frame)
- [x] Step 4 d_inf by the far-background method: **d_inf = −1.0 px (far block matching only, pending real-object check)**; the model output carries no information about far content (see §10)
- [x] Step 5 `make_depth_video.py`, IMG_3143 full rate (818 frames) depth video; 1.27 frames/s end to end, `docs/results/3143/preview_10s.mp4`
- [x] Step 6 remaining three videos + summary table (§10 step 6; all four videos have `work/<id>_full/video/{preview.mp4,depth_mm.mkv,overlay.mp4}`)
- [x] Step 8 (added 2026-10-14) 2D/3D skeletons from field-pose-annotator combined with the stereo depth for all four videos: validation of VideoPose3D against stereo, adjusted (scaled, anchored, limb-flip-corrected) VideoPose3D, two videos per clip, annotator-compatible package (§12)
- [x] Step 7 (optional) TensorRT path works end to end (ONNX 1088×1920 → engine → `run_sequence.py --backend trt`), numerically identical, but fp32-only under TensorRT 11 and therefore slower (330 vs 243 ms); temporal median available as `make_depth_video.py --temporal_median 3` (§10 step 7)

---

## 9. References

- Paper arXiv:2512.11130v2; repo README / `model_card.md`
- Apple WWDC24 *Build compelling spatial photo and video experiences* (builtInDualWideCamera, hero eye, "should be stereo rectified … not enforced", dadj semantics)
- Apple *Stereo Video ISOBMFF Extensions* (`vexu` / `blin` / `dadj` / `hfov` box definitions)
- SVD: Spatial Video Dataset, arXiv:2506.06037 (iPhone 16 Pro: 19.2 mm, 63.4°, ultra-wide cropped and aligned in post)
- Apple iPhone 17 Pro Max Tech Specs (main 24 mm f/1.78, ultra-wide 13 mm f/2.2 120°, spatial video 1080p30)
- FFmpeg MV-HEVC decoding (7.1+, `view_ids`, view specifiers)

---

## 10. Execution log (CUDA machine)

Machine: Windows 11 Home 10.0.26200, NVIDIA GeForce RTX 3090 24 GB (WDDM), driver 610.60 (`nvidia-smi` reports CUDA UMD 13.3).
**Neither Docker nor conda**: the system only has Python 3.13.15 and no conda / docker / git on PATH, so a venv is used:

```
python -m venv %USERPROFILE%\venvs\ffs
%USERPROFILE%\venvs\ffs\Scripts\python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
%USERPROFILE%\venvs\ffs\Scripts\python -m pip install -r requirements.txt "triton-windows<3.3"
```

This gives torch 2.6.0+cu124, triton 3.2.0 (`triton-windows` 3.2.0.post21), timm 1.0.30, opencv 5.0.0, open3d 0.20.0.
`xformers` was not installed (nothing in the code references it). Every `python` below means this venv's python.
git is the one bundled with GitHub Desktop: `%LOCALAPPDATA%\GitHubDesktop\app-3.6.7\resources\app\git\cmd\git.exe`.

### Step 0 — environment and data self-check (2026-10-14)

- `nvidia-smi`: RTX 3090, driver 610.60. `torch.cuda.is_available()` = True, `torch.version.cuda` = 12.4. `import timm, omegaconf, triton` all succeed.
- ffmpeg: not on PATH. Downloaded gyan.dev `ffmpeg-release-essentials.zip` and unpacked into `tools/ffmpeg-9.0.2-essentials_build/bin/` (gitignored).
  `ffmpeg -version` = 9.0.2; `ffmpeg -h decoder=hevc` lists `-view_ids` / `-view_ids_available` / `-view_pos_available`.
  All later commands are prefixed with `$env:PATH = "$PWD\tools\ffmpeg-9.0.2-essentials_build\bin;$env:PATH"`; the scripts are unchanged.
- `vid/IMG_3142..3145.MOV` are present (403 / 88 / 115 / 101 MB).
- `probe_metadata.py` on the four files (command `python scripts/spatial/probe_metadata.py vid/IMG_xxxx.MOV`, JSON saved as `work/probe_xxxx.json`):

  | File | device / iOS | resolution @ fps | nb_frames / duration | baseline_m | hfov | dadj px | fx | GPS |
  |---|---|---|---|---|---|---|---|---|
  | 3142 | iPhone 17 Pro Max / 27.2 | 1920×1080 @ 29.97 | 3738 / 124.68 s | 0.019255 | 63.4° | 38.4 | 1554.37 | yes |
  | 3143 | same | same | 818 / 27.28 s | same | same | same | same | yes |
  | 3144 | same | same | 1065 / 35.52 s | same | same | same | same | yes |
  | 3145 | same | same | 931 / 31.05 s | same | same | same | same | yes |

  Identical to §3, no document change needed. view_ids_present = [0, 1].
- Problem met: PowerShell adds a UTF-8 BOM when piping python stdout, so `python probe.py | python -c json.load` failed; fixed by calling `probe()` inside one python process and writing files.

### Step 1 — weights and headless demo (2026-10-14)

- Weights: `python -m gdown --folder <Google Drive link from the README> -O weights` (gdown 5.x in the system python; `--remaining-ok` no longer exists).
  This yields `weights/{15-44-51,20-26-39,20-30-48,23-36-37}/{cfg.yaml,model_best_bp2_serialize.pth}` plus `weights/onnx/…` (about 1 GB in total, all gitignored).
  `23-36-37/model_best_bp2_serialize.pth` is 71.1 MB, `cfg.yaml` sits next to it.
- Changes to `scripts/run_demo.py` (default behaviour unchanged):
  - new `--headless 1`: skips `cv2.imshow/waitKey` and the open3d window, files are still written;
  - new `--volume pytorch1|triton` (default pytorch1, same as the original code);
  - `os.system('rm -rf … && mkdir -p …')` does not work on Windows, replaced with `shutil.rmtree + os.makedirs` (same effect).
- Command:
  ```
  python scripts/run_demo.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --left_file demo_data/left.png --right_file demo_data/right.png --intrinsic_file demo_data/K.txt --out_dir work/demo_out --remove_invisible 0 --denoise_cloud 1 --scale 1 --get_pc 1 --valid_iters 8 --max_disp 192 --zfar 100 --headless 1
  ```
  The first forward takes about 10 s (torch.compile / Triton compilation); outputs `disp_vis.png, depth_meter.npy, cloud.ply, cloud_denoise.ply`.
- Comparison with `output_docker/disp_vis.png` (same size 2880×540): the left panel is pixel-identical; the disparity colour panel differs by a mean |Δ| of 0.35 / 255, p99 = 4 / 255, only 0.14 % of pixels differ by > 16 — fp16 / compilation-level noise, visually identical. Depth median 0.497 m.
- Baseline timing (`python scripts/profile_speed.py --valid_iters 8 --max_disp 192 --warmup 15 --total 45`, 640×480, `optimize_build_volume='triton'`, fp16 autocast):
  **52.9 ms / frame** (mean of 30; the first iteration is 10.6 s of compilation). The README's 3090 number is 49.4 ms; Windows/WDDM is about 7 % slower, which is plausible.
- Conclusion: the Triton kernel and torch.compile both work under Windows + venv + triton-windows; Docker is not needed.
- `git push` fails: this machine's git has no GitHub credentials usable non-interactively (`could not read Username … terminal prompts disabled`).
  All commits are on the local master; **the user needs to push with GitHub Desktop or their own credentials.**

### Step 2 — extraction and geometry check (2026-10-14)

- Extraction: `python scripts/spatial/extract_views.py vid/IMG_3143.MOV work/3143 --stride 30` → 28 pairs of 1920×1080 PNG (about 2.5 MB each); `K.txt` line 1 fx = 1554.372464, cx = 959.5, cy = 539.5, line 2 = 0.019255.
- **Windows bug found and fixed**: `check_rectification.py --dir` and `run_sequence.py` located the right eye with `path.replace('/left/', '/right/')`;
  glob returns backslash paths on Windows, so the replace was a no-op and the right eye was read as the left (the first check gave dy and disparity of exactly 0 everywhere, obvious at a glance).
  Changed to `os.path.join(dirname(dirname(lp)), 'right', basename(lp))`; Linux/macOS behaviour unchanged.
- `python scripts/spatial/check_rectification.py --dir work/3143 --json work/3143/rect.json` (28 frames, 190–260 blocks each):
  a mean −4.37 ‰ (−5.38 … −3.72), b mean +3.27 px (2.81 … 4.54), residual 0.61 px; see the end of §4.1.
  Block-matching far disparity: p5 median −0.65 px, p10 median +0.58 px (p10 ranges −0.96 … +2.82 across frames because not every frame has enough far content).
- `python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out_dry --dry_run --auto_affine 1`: 28 frames of I/O fine,
  the per-frame auto_affine fits match the JSON above (e.g. 000027: a=−0.00522, b=+4.54).

### Step 3 — first real inference and comparison experiments (2026-10-14)

Command (28 frames, one pair per second):
```
python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out_s30 --valid_iters 8 --max_disp 96 --volume triton --auto_affine 1 --pc_every 5 --zfar 8
```
- The Triton cost volume works directly on Windows, no fallback to pytorch1. First frame 12.5 s (compilation), then **mean_ms_after_warmup = 243.2 ms** (median 241.7);
  the timing covers upload + forward + sync only (1920×1088 after padding), not the CPU block matching of auto_affine (about 1 s per frame, the dominant cost with `--auto_affine 1`).
- Visual check of `vis/` (`docs/results/3143/vis_s30_{000000,000009,000016}.jpg`, left | disparity, 960 px per panel):
  the two workers, the truck and the road are cleanly layered, road disparity varies smoothly with distance, sky and far corn field ≈ 0.
  Disparity median 5.5–7.7 px (≈ 4–5 m), per-frame p99.9 11–16 px (nearest worker ≈ 2 m); `--max_disp 96` (0.31 m) is more than enough.
  Sky (top-left 900×150 region): disparity median 0.00, 98–100 % of pixels < 0.5 px (note: the output is clipped to ≥ 0, so a negative d_inf is invisible here; see step 4).
- **Problem 1: repetitive-texture mismatches.** In frame 000010 the corn field at the left edge (x < 311, y 177–554, 3.4 % of the frame) was matched as a near object at 40–152 px (0.2–0.7 m).
  Cause: periodic corn-stalk texture plus the left edge having no counterpart in the right eye. Frames 000014 / 000021 / 000024 also have sporadic outliers > 30 px (< 0.1 % of pixels).
  `vis_disparity` auto-scales to the per-frame maximum, so those frames look dark overall; it is not a global failure. The depth video uses a fixed range (step 5).
- **Problem 2 (note)**: low-texture asphalt shows no obvious problem; the freshly laid black asphalt also has continuous depth.

Comparison experiments (same first 10 frames `--end 10`, triton unless noted; |Δd| per pixel against the baseline; "near" = baseline d > 6 px, i.e. < 5 m):

| Configuration | ms/frame (mean of last 7) | mean \|Δd\| px | p95 \|Δd\| px | pixels with \|Δd\| > 1 px | near mean \|Δd\| |
|---|---|---|---|---|---|
| baseline iters 8, max_disp 96, auto_affine 1 | 243.7 | 0 | 0 | 0 | 0 |
| iters 4 | **185.5** | 0.016 | 0.055 | 0.02 % | 0.023 |
| max_disp 192 | 314.7 | 0.011 | 0.037 | 0.04 % | 0.016 |
| auto_affine 0 (right eye uncorrected) | 243.0 | 0.022 | 0.075 | 0.06 % | 0.029 |
| volume pytorch1 | 250.4 | 0.001 | 0.002 | 0 | 0.001 |

Conclusions:
- `max_disp 96` and 192 give practically the same result (0.01 px) and 96 is 23 % faster; keep 96.
- `valid_iters 4` is 24 % faster than 8 with a 0.016 px difference (far below the 0.25 px budget of §4.3); fine for full-rate videos, the main results here still use 8.
- Without the vertical affine fix the output changes by only 0.022 px (p95 0.075) — the model really is robust to ±2.5 px of vertical misalignment (the §4.1 prediction holds).
  The fix is kept (one warpAffine), but `--auto_affine` can be relaxed to every 10–30 frames to save CPU.
- triton and pytorch1 agree numerically (0.001 px); triton is 3 % faster.

### Step 4 — metric scale calibration: d_inf (2026-10-14)

**4a. Far-background method on the model output → fails, but reveals an important property.** `run_sequence.py` clips disparity to ≥ 0 and the cost volume only covers [0, max_disp),
so a negative d_inf is invisible. Hence a shift trick: `--dinf -3 --keep_negative` (right eye shifted right by 3 px, all disparities +3, infinity should land at 3 + d_inf > 0),
then the new script `scripts/spatial/estimate_dinf.py` takes the mode of the low-disparity histogram in the top 40 % of the frame:

```
python scripts/spatial/run_sequence.py --seq_dir work/3143 --out_dir work/3143/out_shift3 --valid_iters 8 --max_disp 96 --volume triton --auto_affine 1 --dinf -3 --keep_negative
python scripts/spatial/estimate_dinf.py --out_dir work/3143/out_shift3 --rect_json work/3143/rect.json --json work/3143/dinf_shift3.json
```

Result: the sky / tree-line mode is **+0.01 px in all 28 frames** (not +3), while near pixels did move by +3.00 (binned by baseline disparity:
d ∈ [0.5, 1.5) → +2.94, [1.5, 3) → +3.02, [3, 20) → +3.00 ± 0.03). In other words:
**for pixels it classifies as "far / sky" the model outputs exactly 0 regardless of the true input shift (a prior learned from the Stereo4D pseudo-labels); for pixels with d ≥ 0.5 px it is precisely shift-equivariant.**
Consequences: (1) d_inf cannot be estimated from the model's sky / far-tree output; (2) d ≈ 0 regions in the output mean "infinity class", not a measurement; (3) within 5 m (d ≥ 6 px) the output is a real measurement, and a d_inf bias propagates unchanged into depth.

**4b. Far block matching (adopted).** In the horizon tree band (y < 330, x < 900, avoiding the truck) with `check_rectification.match_patches` (P=32/48, step 16/24, DY 4, DXMAX 16, NCC ≥ 0.9),
right eye first vertically affine-corrected with the per-frame (a, b), take the far cluster d < 0.5 (script `scripts/spatial/dinf_blockmatch.py`, paths hard-coded to work/3143, result `work/3143/dinf_blockmatch.json`):

| Method | valid frames | far-cluster disparity median | IQR |
|---|---|---|---|
| P=32 | 13 (later frames have the tree line hidden by the truck / people) | **−0.95 px** | −1.10 … −0.86 |
| P=48 | 11 | **−1.02 px** | −1.10 … −0.97 |
| original §4.2 estimate (48 px blocks, whole-frame p5) | 9 | −0.5 ± 0.5 | — |

Histogram (P32, 0.25 px bins from −1.5): 180, 234, 126, 51 | 59, 104, 95, 16 — −1.5…−0.75 are the far trees, −0.25…+0.25 is the top of the corn field at 20–30 m.
The far trees are about 50–150 m away and still carry +0.2…0.6 px of true disparity, so the true d_inf may be between −1.0 and −1.4.
The far cluster's d has a slope of +1.06 px/1000px in x (intercept −1.44), but the band only covers x < 900 and depth changes along x, so a horizontal scale difference cannot be separated from the scene; treated as a constant for now.

**4c. Ground-truth object: none.** The videos contain no object of reliably known size (truck hub model unknown, lane-line width uncertain, hard hats too far and occluded), therefore:
**d_inf = −1.0 px is a far-background-only result, pending verification with a real object.** Suggestion for the next recording: place a tape measure or an A4 sheet (210 × 297 mm) at 1 / 2 / 3 m, facing the camera, hold 2 s each;
then solve d_inf = d_model − fx·b / z_true and verify fx.

**4d. Rerun of the 28 frames with `--dinf -1.0`** (`work/3143/out_s30_dinf`, other parameters as in step 3, 239.5 ms/frame):

| Frame | near disparity change | nearest worker p99.5 disparity → depth (base → dinf) | median valid depth | sky fraction with disp = 0 |
|---|---|---|---|---|
| 000000 | +1.00 px | 12.43 → 13.41 px; 2.41 → 2.23 m | 4.39 → 3.83 m | 1.00 → 1.00 |
| 000009 | +1.00 px | 14.00 → 14.94 px; 2.14 → 2.00 m | 4.75 → 4.14 m | 1.00 → 0.99 |
| 000016 | +1.00 px | 11.49 → 12.50 px; 2.60 → 2.39 m | 4.37 → 3.83 m | 0.99 → 0.99 |
| 000024 | +0.98 px | 11.02 → 12.04 px; 2.72 → 2.49 m | 4.99 → 4.40 m | 1.00 → 1.00 |

So d_inf = −1 pulls depth at 2–2.5 m closer by 7–8 % and at 4–5 m by 12–13 %, consistent with the §4.2 error budget — exactly the magnitude that needs a real-object check.
The sky still has 0 disparity; pixels with a tiny d (0.001–0.1 px) in `depth/*.npy` get finite depths of hundreds to thousands of metres, **not** 0;
the point cloud relies on `--zfar`, the depth video on `--zmax` (> 8 m → black) and the uint16 limit (> 65.5 m → 0) to cut them.

### Step 5 — video with depth: IMG_3143 at full rate (2026-10-14)

New `scripts/spatial/make_depth_video.py` (interface in §6.5). First verified on the 28-frame `out_s30`: preview 1990×540 @ 0.999 fps, `depth_mm.mkv` round-trips bit-exactly against the npy files, overlay fine.

Full-rate pipeline and timing (818 pairs, 1920×1080, 29.97 fps, 27.3 s):
```
python scripts/spatial/extract_views.py vid/IMG_3143.MOV work/3143_full --stride 1                      # 4.2 GB PNG
python scripts/spatial/run_sequence.py --seq_dir work/3143_full --out_dir work/3143_full/out --valid_iters 8 --max_disp 96 --volume triton --auto_affine 10 --dinf -1.0 --pc_every 0 --zfar 8
python scripts/spatial/make_depth_video.py --seq_dir work/3143_full --out_dir work/3143_full/out --video_dir work/3143_full/video --overlay 1
ffmpeg -ss 5 -t 10 -i work/3143_full/video/preview.mp4 -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart docs/results/3143/preview_10s.mp4
```

| Stage | wall time | throughput | notes |
|---|---|---|---|
| extract_views (ffmpeg MV-HEVC → PNG×2) | ≈ 25 s | ≈ 40 pairs/s | the 1065 pairs of 3144 took 24 s |
| run_sequence total (PNG read, auto_affine every 10 frames, npy + 5760×1080 vis PNG writes) | 473 s (14:55:53 → 15:03:46) | **1.73 frames/s** | model alone: mean_ms_after_warmup = **244.0 ms** (median 243.1, p95 247.5) = 4.1 frames/s, 206 s in total; the rest is PNG I/O and block matching |
| make_depth_video (three ffmpeg encoders) | 150 s | 5.5 frames/s | CPU |
| **end to end** | ≈ 650 s | **≈ 1.27 frames/s (about 0.79 s/frame)** | to speed up: skip the vis PNGs, `--valid_iters 4` (−24 %), TensorRT (step 7) |

- auto_affine, 82 fits: a ∈ [−5.33, −3.63] ‰ (mean −4.39), b ∈ [2.48, 4.54] px (mean 3.31), residual after affine mean 0.59 px, max 1.16 px.
- Disparity: p99.9 median 13.4 px over sampled frames (nearest about 2.0 m), max 15.7 px; no pixel > 30 px (< 1 m) in the sampled frames — the corn-field mismatch of step 3 did **not** show up in the sample (it was a single frame near t = 10 s; at stride 1 that frame can still have it, the sample just missed it).
- Outputs: `work/3143_full/video/preview.mp4` (21.8 MB, 27.29 s), `depth_mm.mkv` (527 MB), `overlay.mp4` (41.3 MB);
  `work/3143_full/out/` totals 19.9 GB (disp + depth npy at 8 MB per frame each, plus vis PNGs).
- Committed: `docs/results/3143/preview_10s.mp4` (t = 5–15 s, 300 frames, 4.5 MB).
- Visual: at t = 13.5 s (frame 000405) the foreground worker is ≈ 2 m (cyan), the truck 1–2 m (green), the road runs smoothly from 2 m to 8 m (blue → purple), sky and corn field black (> 8 m).
  There is mild frame-to-frame flicker (mostly on asphalt and the truck side); `--temporal_median 3` reduces it but is off by default.

### Step 6 — the other three videos and the summary (2026-10-14)

Same pipeline as step 5 for IMG_3144, IMG_3145 (stride 1) and IMG_3142 (stride 2, because it is 125 s long; its videos play at 14.985 fps):
```
python scripts/spatial/extract_views.py vid/IMG_<id>.MOV work/<id>_full --stride <1|2>
python scripts/spatial/run_sequence.py --seq_dir work/<id>_full --out_dir work/<id>_full/out --valid_iters 8 --max_disp 96 --volume triton --auto_affine 10 --dinf -1.0 --pc_every 0 --zfar 8
python scripts/spatial/make_depth_video.py --seq_dir work/<id>_full --out_dir work/<id>_full/out --video_dir work/<id>_full/video --overlay 1
ffmpeg -ss 5 -t 10 -i work/<id>_full/video/preview.mp4 -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -movflags +faststart docs/results/<id>/preview_10s.mp4
```
Extraction of all three took 110 s in total (14:58:24 → 15:00:13); the GPU runs were chained (3144 15:06:19 → 15:16:22, 3145 → 15:25:11, 3142 → 15:42) with the video encodes overlapping on the CPU.

| Video | Source frames / length | stride → frames | Model ms/frame (mean after warm-up) | GPU wall time | auto_affine a (‰) / b (px): mean [min, max] | residual px | d_inf | p99.9 disparity (median over sampled frames) | Encode s (fps) | preview / depth_mm / overlay MB | Doc clip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IMG_3142 | 3738 / 124.7 s | 2 → 1869 | 242.0 | 17.0 min | −4.25 [−5.73, −1.85] / 3.22 [1.52, 5.22] (187 fits) | 0.39 | −1.0 | 11.1 px (≈ 2.7 m) | 342 (5.5) | 100.5 / 1392.7 / 200.4 | `docs/results/3142/preview_10s.mp4` (3.6 MB) |
| IMG_3143 | 818 / 27.3 s | 1 → 818 | 244.0 | 7.9 min | −4.39 [−5.33, −3.63] / 3.31 [2.48, 4.54] (82 fits) | 0.59 | −1.0 | 13.4 px (≈ 2.2 m) | 150 (5.5) | 21.8 / 527.3 / 41.3 | `docs/results/3143/preview_10s.mp4` (4.5 MB) |
| IMG_3144 | 1065 / 35.5 s | 1 → 1065 | 243.5 | 10.1 min | −4.61 [−6.48, −3.73] / 3.43 [2.41, 4.94] (107 fits) | 0.44 | −1.0 | 13.0 px (≈ 2.3 m) | 201 (5.3) | 33.9 / 875.5 / 56.7 | `docs/results/3144/preview_10s.mp4` (4.1 MB) |
| IMG_3145 | 931 / 31.1 s | 1 → 931 | 245.4 | 8.8 min | −4.19 [−6.02, −2.89] / 2.93 [0.01, 5.24] (94 fits) | 0.46 | −1.0 | 17.2 px (≈ 1.7 m; max 27.8 px ≈ 1.1 m) | 182 (5.1) | 36.5 / 736.1 / 66.9 | `docs/results/3145/preview_10s.mp4` (6.5 MB) |

- The vertical affine is the same for all four clips (a ≈ −4.2…−4.6 ‰, b ≈ 3 px), i.e. a property of the camera/pipeline, not of the scene. 3145 has one fit with b = 0.01 (a frame with few far matches); because `--auto_affine 10` refits every 10 frames, such an outlier only affects 10 frames and the model is robust to it anyway (§10 step 3).
- The model time is 242–245 ms/frame on all clips (1920×1088, iters 8, max_disp 96, triton); the GPU wall time above includes PNG I/O and the vis PNG writes (≈ 1.75–1.85 frames/s).
- No clip has pixels closer than 1 m except 3145 (a worker's back at ≈ 1.1 m); the fraction of pixels with d > 30 px is ≤ 0.03 % everywhere, so `--max_disp 96` is safe.
- Sample frames: `docs/results/<id>/vis_full_<frame>.jpg` (left | disparity, 960 px per panel): 3142 frames 000400 / 001400, 3143 000405, 3144 000300 / 000800, 3145 000200 / 000700.
- Disk: `work/` now holds about 90 GB (PNG frames ≈ 24 GB, npy + vis ≈ 60 GB, videos ≈ 4 GB). The PNG frames and `out/vis` can be deleted; `out/disp`, `out/depth` and `video/` are the results.

### Step 7 (optional) — TensorRT and temporal median (2026-10-14)

**Temporal median**: implemented as `make_depth_video.py --temporal_median 3` (off by default); its effect is quantified in §11.1 ("flicker after 3-frame median").

**TensorRT** (`pip install onnx tensorrt-cu12` → onnx 1.23.2, tensorrt 11.4.0.106; no `trtexec` on a pip install, hence `scripts/spatial/build_trt_engine.py`):
```
python scripts/make_single_onnx.py --model_dir weights/23-36-37/model_best_bp2_serialize.pth --save_path work/onnx_1088x1920 --height 1088 --width 1920 --valid_iters 8 --max_disp 96 --global_pool_reduce 1
python scripts/spatial/build_trt_engine.py --onnx work/onnx_1088x1920/fast_foundationstereo.onnx          # 226 s, 2.3 GB engine
python scripts/spatial/run_sequence.py --seq_dir work/3143_full --out_dir work/3143_full/cmp_trt --end 20 --auto_affine 10 --dinf -1.0 --backend trt --engine work/onnx_1088x1920/fast_foundationstereo.engine
```
- Two export fixes were needed at this resolution, both behind the new `--global_pool_reduce 1` switch of `scripts/make_single_onnx.py` (default behaviour unchanged):
  the channel attention's `AdaptiveMax/AvgPool2d(1)` becomes a global MaxPool with a 136×240 window that TensorRT's ONNX parser rejects (`addPoolingNd … maxKernelDimsProduct`) → exported as ReduceMax / ReduceMean;
  and the `[1,2]` average pools of the correlation pyramid (`core/geometry.py`, batch of 130 560 tiny maps) made the builder fail (`Could not find any implementation for node /AveragePool_1`) → exported as reshape + mean. Both are mathematically identical.
- TensorRT 11 has no `BuilderFlag.FP16` any more (strongly-typed networks: precision follows the ONNX dtypes), so the engine is **fp32**. An fp16 engine needs a half-precision ONNX export, which was not attempted (the model mixes fp16 and forced-fp32 sections; it needs care).
- `run_sequence.py --backend trt --engine …` (new switch; default `torch` unchanged) runs the engine with ImageNet-normalised inputs through `SingleEngineTrtRunner` from `scripts/run_demo_single_trt.py`.
- Result on 20 frames of 3143: **329.6 ms/frame (fp32 TensorRT) vs 243 ms (torch fp16 + Triton)**; |Δdisparity| mean 0.0008 px, p99.9 0.017 px, no pixel > 1 px — numerically the same model.
  So the TensorRT path is validated but brings no speed-up until an fp16 export is done; the README's 2× TensorRT gain (fp16, `trtexec --fp16`, TensorRT ≤ 10) is expected to carry over then.

---

## 11. Evaluation: tests, depth quality and recommendations (2026-10-14)

All tests are ground-truth-free (`scripts/spatial/eval_depth.py`, §6.6); results in `work/<id>_full/eval.json`.
Sampling: coverage / plane / block matching every 30th frame, flicker on 60 windows of 5 consecutive frames, left-right check on 8 frames per clip.

### 11.1 Results

| Test | 3142 (stride 2) | 3143 | 3144 | 3145 | What it measures |
|---|---|---|---|---|---|
| Coverage of valid depth in 0.3–8 m | 56 % [37, 73] | 69 % [61, 75] | 62 % [44, 73] | 52 % [32, 69] | the rest is sky / field beyond 8 m and the left occlusion band |
| Flicker, raw: p90 / p99 px; share > 0.5 px | 0.48 / 1.16; 9.3 % | 0.18 / 0.41; 0.5 % | 0.20 / 0.51; 1.0 % | 0.33 / 1.31; 6.2 % | \|d_t − median(d_{t−1..t+1})\| for d > 2 px; includes real motion |
| Flicker as relative depth, p90 / p99 | 8.1 % / 21 % | 2.4 % / 7 % | 3.2 % / 11 % | 5.7 % / 28 % | Δz/z = Δd/d |
| Flicker after 3-frame median: p99 px; share > 0.5 px | 0.19; 0.02 % | 0.14; 0.04 % | 0.11; 0.03 % | 0.03; 0.04 % | what `--temporal_median 3` leaves (spikes removed) |
| Road plane: RMS of inliers / inlier share | 2.6 cm / 65 % | 2.5 cm / 56 % | 2.5 cm / 75 % | 2.4 cm / 59 % | metric flatness over 1–8 m, 5 cm inlier band; non-inliers are people, truck, asphalt pile in the ROI |
| Camera height above road / tilt | 1.30 m / 14° | 1.27 m / 16° | 1.20 m / 18° | 1.32 m / 17° | plausibility only (hand-held at chest height); **insensitive to d_inf**, see below |
| Model vs ZNCC block matching (near, 3–40 px): bias / MAD / share > 1 px | −0.001 / 0.12 / 5.8 % (23 567 patches) | −0.014 / 0.15 / 8.2 % (14 765) | −0.007 / 0.12 / 9.5 % (22 138) | +0.003 / 0.12 / 9.3 % (15 696) | independent measurement of the same images |
| Left-right consistency: share < 1 px (all / near < 5 m); median px | 99.6 % / 99.8 %; 0.022 | 99.2 % / 99.6 %; 0.027 | 99.4 % / 99.7 %; 0.020 | 98.7 % / 99.2 %; 0.023 | self-consistency on the non-occluded 64–74 % of pixels |

### 11.2 Depth quality verdict

- **Relative accuracy in the working range (0.5–5 m) is good.** The model agrees with independent block matching on the same frames to within 0.01 px bias and 0.12–0.15 px MAD; the left-right check is below 1 px for more than 99 % of the non-occluded pixels; the road is flat to 2.5 cm RMS over 1–8 m. 0.15 px at 3 m is about 4.5 cm (§4.3).
- **Temporal noise is 2–3 % of depth at 30 fps (p90) in the calmer clips (3143, 3144) and 6–8 % in the clips with more camera motion (3145, and 3142 which has twice the motion per frame at stride 2).** The p99 tail (0.4–1.3 px) is single-frame spikes on asphalt and truck sides; a 3-frame temporal median removes them almost completely (share > 0.5 px drops from 0.5–9 % to < 0.05 %) at the price of one frame of latency and smeared depth on fast-moving limbs. Use it for display and for static scenes, not for measuring moving objects.
- **Absolute scale is the open issue: uncertain by about ±5–10 % at 2–5 m.** d_inf = −1.0 px comes from the far-horizon block-matching cluster only (§10 step 4). The road-plane test cannot help: a plane's disparity is affine in image coordinates with or without a constant offset, so d_inf is absorbed into the fit (camera height 1.257 m with d_inf = 0 vs 1.267 m with −1.0 on the same frames, only the tilt changes 15.1° → 16.2°). The heights of 1.20–1.32 m are plausible for a phone held at chest height, nothing more. Only an object of known size or distance settles this.
- **Known failure modes.** (1) Far content (sky, tree line, field beyond ~20 m) is output as a hard 0 — a learned class, not a measurement; treat d < 0.5 px as "far", never as a depth. (2) Repetitive texture at the frame edge (corn stalks, §10 step 3 frame 000010) can produce spurious near depth (0.2–0.7 m) in a few percent of the frame; rare at full rate (≤ 0.03 % of pixels have d > 30 px in the sampled frames) but it does happen. (3) The left ~d px of every row have no counterpart in the right eye and are invalid (`remove_invisible`). (4) Thin structures (shovel handles, chains, cables) are partly resolved and sometimes bleed into the background. (5) 8–9 % of near textured patches differ by > 1 px from block matching; visually these are depth discontinuities and the asphalt pile, where block matching is at least as wrong as the model.
- **Overall:** for the stated goal (metric depth within 5 m) the output is usable now for relative measurements, segmentation by distance, and visualisation; for absolute distances it is usable with a ±10 % caveat until d_inf (and possibly fx) is verified.

### 11.3 Recommendations for future use

Capture:
- Record a **calibration shot** at the start of each session: an A4 sheet or a tape measure facing the camera at 1, 2 and 3 m (2 s each), plus a checkerboard if a full `stereoCalibrate` of Apple's virtual cameras is wanted (§7 step 7). This resolves d_inf (and fx) once per device/iOS version.
- Keep subjects within 0.5–4 m; beyond 5 m the disparity is < 6 px and 0.25 px of noise is already 20 cm (§4.3).
- Avoid filling the frame edges with periodic texture (corn, fences, gratings); keep the same stabilisation mode (the right-eye affine depends on it); avoid fast pans (rolling shutter, two sensors).
- 30 fps full-rate recording is fine; for long clips stride 2 halves the work and the temporal median still works.

Processing:
- Defaults that worked: `--valid_iters 8 --max_disp 96 --volume triton --auto_affine 10 --dinf -1.0`, 1920×1088, no `--scale`. `--valid_iters 4` is 24 % faster at a 0.016 px cost (§10 step 3); `--max_disp 192` only if objects come closer than 0.3 m.
- Use `depth_mm.mkv` (uint16 mm, lossless) with `K.txt` for anything metric; use `preview.mp4` only for looking. Mask d < 0.5 px as far, and use the left-right check of `eval_depth.py` as a per-pixel reliability mask when it matters.
- `--temporal_median 3` for display and for static scenes.
- Speed: the model runs at 4.1 frames/s (244 ms) but the pipeline at 1.3 frames/s, because of PNG I/O (4.2 GB per 818 frames) and the 5760×1080 vis PNGs. Feeding frames straight from an ffmpeg pipe, skipping the vis PNGs and using `--valid_iters 4` would roughly triple the throughput before touching TensorRT (§10 step 7).
- If Apple's LiDAR depth or ARKit depth is available in a future capture, use it to calibrate d_inf / fx instead of a target.

---

## 12. Skeletons + depth: validating and adjusting VideoPose3D with stereo (2026-10-14)

### 12.1 Setup

- 2D/3D pose from the lab's `field-pose-annotator` (`C:\Users\Admin\Documents\GitHub\field-pose-annotator`, its own venv `~/.venvs/field-pose-annotator-pose`: torch 2.6+cu124, rtmlib 1.26, onnxruntime-gpu CUDA, ultralytics 8.4). Pipeline `scripts/preprocess_video.py --backend rtmlib --tier full`: YOLOv8m-pose person detector → RTMPose-m COCO-17 keypoints → tracker with identity linking → VideoPose3D (H36M-17, root-relative, metres, camera axes x right / y down / z forward). Nothing in that repository was changed.
- Input to it: a left-eye-only MP4 per clip (`ffmpeg -map 0:v:0:vpos:left -fps_mode passthrough -c:v libx264 -crf 12`), frame counts identical to the MOV (818 / 1065 / 931 / 3738), so pose frame i is depth frame i (i/2 for 3142 at stride 2). Pose run times: 63 s, 67 s, 62 s, 178 s; packages in `work/<id>_full/pose/IMG_<id>_left.tracks.npz`.
- Combination, validation, adjustment and rendering: `scripts/spatial/pose_depth.py` (§6.8), 131 / 143 / 156 / 302 s per clip including both videos.
- Checked first: VideoPose3D's raw output is in metres with the same axes as ours (thigh 0.41–0.44 m, shin 0.41–0.43 m, y correlation with stereo +0.98 on the nearest track), so only a scale and the pelvis position are missing from it.

### 12.2 Validation: scaled + anchored VideoPose3D against stereo joints

Only joints of tracks whose pelvis is within 5 m count (beyond that, 0.15 px of disparity noise is ≥ 0.3 m per joint and the comparison says nothing). "Adjusted" = scaled + anchored + limb depth-flip fix.

| Clip | tracks in package / used / with ≥ 60 near frames | near joint-frames | VideoPose3D vs stereo, median / p90 (cm) | adjusted vs stereo, median (cm) | limb flips applied | far tracks (> 5 m) median (cm) | stereo limb length / typical adult |
|---|---|---|---|---|---|---|---|
| 3143 | 5 / 5 / 1 | 13 435 | 19.1 / 48 | 18.7 | 7.7 % | 63 | 1.43 (one worker at 3.4 m in front of the corn field) |
| 3144 | 10 / 9 / 4 | 16 322 | 21.5 / 96 | 21.3 | 2.7 % | 30 | 1.02 |
| 3145 | 20 / 10 / 5 | 22 113 | 20.6 / 115 | 20.3 | 6.5 % | 39 | 1.01 |
| 3142 | 13 / 7 / 3 | 23 319 | 21.6 / 67 | 21.2 | 6.8 % | 58 | 1.42 |

Per joint (median over the four clips, VideoPose3D → adjusted, cm): pelvis 2–7 (anchored), hips 9–15, knees 15–26, ankles 21–33, thorax 17–24, shoulders 18–30, elbows 23–36, wrists 22–40, nose 27–41. Signed z: VideoPose3D joints sit 10–21 cm nearer the camera than the stereo samples on average.

Reading:
- **Agreement of about 20 cm per joint at 2–5 m is the sum of three things**: stereo joint noise (5–13 cm at 3–5 m for 0.15 px), the fact that stereo samples the body surface while VideoPose3D predicts joint centres, and VideoPose3D's own error (its published in-the-wild error is of this order). The pelvis agrees by construction; error grows along the kinematic chain, as expected from a root-relative model.
- **The stereo skeleton is independent evidence**: the 2D keypoints are exact enough that the lateral (x, y) extents of the stereo skeleton match VideoPose3D after one scale, and stereo limb lengths of the well-measured workers (3144, 3145) are within 1–2 % of typical adult values. In 3143 and 3142 the near workers stand directly in front of the corn field and their stereo limbs come out 40 % too long: the depth window at silhouette joints catches the field behind the legs. Joint depth needs a person mask, not a window, to go further.
- **Adjusting VideoPose3D with the depth-flip test changes little (0.2–0.5 cm)** because inside 5 m VideoPose3D rarely contradicts stereo on a limb's depth direction confidently (3–8 % of limb-frames) and the test is deliberately conservative (|Δz| above 2 σ of the stereo noise). What the adjustment does deliver is the part VideoPose3D cannot: absolute position (pelvis in metres) and absolute scale per person.
- **Beyond 5 m the comparison breaks down** (30–63 cm medians) and the fused skeleton inherits the far-range stereo bias of §12.3.

Videos (full in `work/<id>_full/pose/`, 10 s clips in `docs/results/<id>/`):
- `pose_compare.mp4` (validation): left image with the 2D skeleton coloured by stereo depth and each worker's pelvis distance | depth map with the skeleton | top view in metres, stereo skeleton filled and scaled VideoPose3D hollow, with the per-worker joint disagreement. `docs/results/3143/pose_compare_frame000405.jpg`.
- `pose_fused.mp4` (adjusted VideoPose3D): the adjusted 3D skeleton reprojected onto the left image | front view | top view, metres, 1 m grid. `docs/results/3143/pose_fused_frame000405.jpg`.
- `IMG_<id>_left_fused.tracks.npz/.json`: the tracks package with `pose_3d` replaced by the adjusted metric skeleton (`pose_3d_units = "metres"`, origin camera); verified to load with the annotator's `load_tracks_package`.

### 12.3 Using people and a truck wheel as calibration objects: the metric scale is right near, wrong far

`dinf_from_bodies.py` (§6.8): the lateral size of a person from stereo is px · b / (d − d_inf), independent of fx; VideoPose3D gives the same person in metres; the ratio r per track implies d_inf = d_used − (r − 1) · d_model. On the long, fully visible tracks:

| Track (clip, pelvis depth) | stereo / VideoPose3D size | implied d_inf (px) |
|---|---|---|
| 3145 T1, 1.8 m | 0.90 | +0.6 |
| 3144 T9, 3.1 m | 0.87 | +0.2 |
| 3143 T1, 3.2 m | 1.15 | −2.4 |
| 3144 T1, 3.9 m | 1.11 | −1.8 |
| 3142 T6, 4.2 m | 0.95 | −0.7 |
| 3142 T1, 4.1 m | 1.27 | −3.0 |
| 3145 T10, 5.2 m | 1.14 | −1.8 |
| 3142 T10, 6.4 m | 1.63 | −3.9 |
| 3143 T2, 7.3 m | 1.68 | −3.8 |
| 3142 T14, 8.2 m | 1.50 | −2.9 |
| 3142 T11, 12.9 m | 2.55 | −4.6 |

A rigid object confirms the far end: the rear wheel of the dump truck in 3144 frame 000300 (`docs/results/3144/wheel_frame000300_crop.jpg`) has a 150 px rim and a 270 px tyre at a disparity of 3.5 px (8.5 m); a standard 22.5-inch truck rim (0.572 m) and an 11R22.5 tyre (1.05 m) both give a stereo/true ratio of 1.44, i.e. d_inf ≈ −2.5 px at that distance (any standard rim from 19.5 to 24.5 inch gives −3.3 … −2.1).

Cross-check that this is not the model: binned against independent block matching on 54 447 patches of all four clips, the model's disparity agrees to within 0.03 px in every bin from 0.5 px upwards (§11 method), so the drift is in the image pair's geometry (or in VideoPose3D for the far bodies, but not for the wheel).

Conclusion: **with d_inf = −1.0 px the metric scale is right to about ±10 % inside 4.5 m** (people at 1.8–4.2 m: ratios 0.87–1.15, d_inf +0.6 … −1.8, VideoPose3D's own ±10–15 % included). **Beyond 5 m stereo sizes and depths are 30–70 % too large**; the effective offset there is −2.5 … −4 px. A single constant d_inf cannot fit both ends, so the raw disparity is not simply offset; a disparity gain (d_raw ≈ 1.3 · d_geom − 4 px fits the near bodies, the 1.8 m worker and the wheel together) is one explanation, i.e. an effective baseline of about 25 mm rather than the 19.255 mm of the metadata, or a depth-dependent step in Apple's eye alignment. The data here cannot decide; a calibration recording (A4 sheet or tape at 1, 2, 3, 5 and 8 m, plus a checkerboard) will, and `pose_depth.py --disp_offset` / `run_sequence.py --dinf` can then be re-run from the saved disparities without new inference. Until then: use the depth and the skeletons inside 5 m, treat anything farther as relative.

### 12.4 How to use this

- For annotation: `IMG_<id>_left_fused.tracks.npz` drops into the annotator in place of the original package (same tracks, ids, 2D; `pose_3d` now metric and absolute). Frames of a track without depth (stride 2 in 3142) are interpolated and the metadata says so.
- For measurements inside 5 m: distance between workers, worker-to-truck, hand height above the road: from `pose_depth.npz` (`t<id>_fused` H36M joints in metres, `t<id>_stereo_coco` + `_stereo_valid` for the raw stereo joints).
- Next improvements, in order of value: a person mask for the joint depth (removes the corn-field contamination that inflates limbs), the calibration recording (fixes the far-range scale), per-joint blending (`--fuse_weight 0.3`) once the mask exists.
