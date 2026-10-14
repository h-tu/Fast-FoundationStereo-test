# iPhone Spatial Video → Fast-FoundationStereo: feasibility, measurements and execution plan

> Status (2026-10-14): format parsing, geometry checks and tooling were done on a laptop (macOS, M1 Pro, no CUDA);
> model inference, metric-scale estimation and the depth videos were done on the CUDA box (Windows 11, RTX 3090) — see §8.1 and §10.
> Current result: all four videos converted to depth videos (`work/<id>_full/video/`), d_inf = −1.0 px from the far-background method only
> (pending verification with a known-distance object), 1.27 frames/s end to end at 1920×1080.
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
- [ ] Step 6 remaining three videos + summary table
- [ ] Step 7 (optional) TensorRT / temporal median

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
