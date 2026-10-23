# Metric depth from iPhone 17 Pro Max spatial video with Fast-FoundationStereo — findings and results

*Status 2026-10-22. Compiled from the work logged in `docs/iphone_spatial_video_plan.md` (sections 3–4, 10–12, 14), which holds the full execution log, commands and per-step details. This report is self-contained and written as source material for a presentation: every number below was measured on the data described, and the section and script that produced it are named so it can be checked.*

---

## 0. At a glance

| Question | Answer |
|---|---|
| Can iPhone spatial video (MV-HEVC stereo) be turned into metric depth video? | Yes. Both eyes decode losslessly with ffmpeg ≥ 7.1; Fast-FoundationStereo gives dense disparity at full 1920×1080; depth = fx·b / disparity with the baseline (19.255 mm) and field of view (63.4°) stored in the file. |
| How accurate is it, against an independent ground truth? | Against 12 427 helmet-ArUco-marker readings in 15 clips: **no bias (−0.3 %) and 5.8 % median absolute error (≈ 20 cm) at 2–7 m**; 44 % of readings within 5 %, 70 % within 10 %. |
| What has to be calibrated? | One number per recording session: the stereo pair's disparity offset at infinity, **d_inf**. It was −2.98 px for the 2026-10-21 session. Without it, every depth is 10–36 % too far. It is not a camera-spec value; the markers measure it automatically. |
| How fast? | 243 ms per 1920×1088 frame on an RTX 3090 (4.1 frames/s for the model); about 1.3 frames/s for the whole pipeline including video encoding. |
| What else came out of it? | The depth also validates and adjusts VideoPose3D skeletons from the lab's field-pose-annotator: absolute position and scale per worker, ≈ 20 cm joint agreement inside 5 m. |
| Main open issue | The ground truth itself: markers sit on curved helmets, which biases the flat-marker PnP by an estimated 1–3 % (simulated). A curvature-aware comparison is designed (§9) but not yet run. |

---

## 1. Goal and setup

**Goal.** Use spatial videos recorded on an iPhone 17 Pro Max (stereo, MV-HEVC) as input to a stereo depth network and obtain a per-frame metric depth map for workers on construction sites. Objects of interest are within 5 m.

**Model.** NVIDIA Fast-FoundationStereo (Wen, Dewan, Birchfield, CVPR 2026, arXiv:2512.11130), research checkpoint `23-36-37`: distilled EdgeNeXt backbone, group-wise correlation + concat cost volume, NAS-searched 3D hourglass, one GRU refinement layer; 14.6 M parameters. Input: rectified RGB pair; output: disparity in left-image pixels. It takes no camera parameters; metric depth comes from `depth = fx · baseline / disparity`.

**Machine.** Windows 11, NVIDIA RTX 3090 24 GB, driver 610.60; Python venv with torch 2.6.0+cu124, triton-windows 3.2.0; static ffmpeg 9.0.2. No Docker or conda needed (the official instructions assume Linux).

**Data.**

| Batch | Clips | Recorded | Content | Ground truth |
|---|---|---|---|---|
| A | 4 (IMG_3142–3145), 218.5 s, 6 552 frames | 2026-10-13 | road works: workers, dump truck, corn field | none (internal consistency checks only) |
| B | 15 (IMG_3345–3385), 853.5 s, 25 590 frames | 2026-10-21, 09:44–13:35 | road-side works, 2–4 workers | helmet ArUco markers (4 per helmet, 88.9 mm) |

All clips: iPhone 17 Pro Max, iOS 27.2, 1920×1080 at 29.97 fps, identical stereo metadata (§2).

---

## 2. The iPhone spatial video format (measured)

| Item | Value | How it was established |
|---|---|---|
| Codec | HEVC Main with two MV-HEVC views, ≈ 25 Mbps total | ffprobe |
| Eyes | ffmpeg `vpos:left` = view id 1 = main (Wide) camera, the "hero" eye; right eye = ultra-wide camera, cropped and aligned by Apple | pixel comparison; Apple documentation |
| Baseline | 19.255 mm | Stereo3D side data `baseline` |
| Horizontal field of view | 63.4° → fx = fy = 1554.37 px, principal point at the image centre | `horizontal_field_of_view` |
| Disparity comfort adjustment | 2 % of width (38.4 px) — playback only, not in the pixels | `horizontal_disparity_adjustment` |
| Disparity scale | fx·b = 29.93 px·m: 30 px at 1 m, 15 px at 2 m, 10 px at 3 m, 6 px at 5 m | derived |

**Field-of-view note.** The metadata value (63.4°) is the field of view of this exact video mode. The 24 mm-equivalent spec of the main camera corresponds to about 72° over the full sensor width, and field-pose-annotator's `iphone-main` preset uses 69°; spatial video crops more (to match the ultra-wide eye and leave room for stabilisation). For the stereo-vs-marker comparison this does not matter (fx cancels, §6.1); for absolute scale of both it does, and one tape-measure shot would confirm 63.4°.

**Accuracy budget implied by the geometry** (Δz = z²·Δd / (fx·b)):

| Distance | Disparity | Δz for 0.25 px | Δz for 0.5 px |
|---|---|---|---|
| 1 m | 29.9 px | 0.8 cm | 1.7 cm |
| 2 m | 15.0 px | 3.3 cm | 6.7 cm |
| 3 m | 10.0 px | 7.5 cm | 15 cm |
| 5 m | 6.0 px | 21 cm | 42 cm |

The 19 mm baseline is the fundamental limit: reliable inside 3 m, usable to 5 m, relative only beyond. Running at full resolution (not the README's half scale) is essential.

---

## 3. Pipeline

```
MOV --ffmpeg vpos:left/right--> PNG pairs --right-eye vertical affine--> Fast-FoundationStereo (1920x1088, 8 GRU iterations,
max_disp 96, Triton cost volume) --> disparity (px) --(- d_inf correction)--> metric depth --> videos / skeletons / marker comparison
```

All code is in `scripts/spatial/` (overview: `scripts/spatial/README.md`): `probe_metadata.py`, `extract_views.py`, `check_rectification.py`, `run_sequence.py`, `make_depth_video.py`, `eval_depth.py`, `pose_depth.py`, `dinf_from_bodies.py`, `aruco_compare.py`, `session_offset.py`, `apply_dinf.py`, the batch tools `batch_process.py` / `batch_finish.py` / `batch_progress.py` / `batch_summary.py`, and one-off analyses in `scripts/spatial/dev/`.

**Right-eye vertical correction.** Block matching shows the right eye has a row-dependent vertical offset of up to ±2.5 px (a vertical scale difference of 0.4–0.5 %), removed with one affine warp per frame, refitted every 10 frames. Parameters: a ≈ −4.4 ‰, b ≈ +3.3 px in batch A; a median −0.58 ‰, b +2.20 px in batch B — Apple's eye alignment differs between recording sessions. The model is robust to it anyway (switching the correction off changes the output by only 0.022 px).

**Speed (RTX 3090, 1920×1088).**

| Setting | ms / frame | Output difference vs default |
|---|---|---|
| default: 8 iterations, max_disp 96, Triton | 243 | — |
| 4 iterations | 185 | 0.016 px |
| max_disp 192 | 315 | 0.011 px |
| PyTorch cost volume | 250 | 0.001 px |
| TensorRT engine (fp32; TensorRT 11 has no fp16 flag) | 330 | 0.0008 px |
| official 640×480 benchmark | 52.9 | (README: 49.4) |

End to end (decode, inference, three encoded videos): 1.27 frames/s; the model alone is 4.1 frames/s, the rest is PNG I/O and encoding. Batch B's 20 225 processed frames need 1.4 h of pure inference at 0.244 s/frame; with frame extraction, disk I/O and an interrupted session it took about 3.5 h of wall time on one GPU.

---

## 4. The key physical finding: the disparity offset d_inf

Depth needs the disparity of an infinitely far point to be zero. For Apple's two eyes it is not; it is a small negative number, **d_inf**, that has to be subtracted. Getting it wrong by 1 px changes depth at 3 m by about 10 %.

**4.1 How it was found, step by step**

1. *Far tree line (batch A).* Block matching on the horizon trees gave −0.95 to −1.02 px. Used as d_inf = −1.0 px.
2. *The model cannot measure it.* Shifting the right eye by a known 3 px moved all near disparities by exactly +3.00 px (±0.03), but the sky and far trees stayed at exactly 0: the network treats far content as a learned "infinity" class. Consequence: d_inf must come from outside the model, and d < 0.5 px in the output means "far", not a measurement. Above 0.5 px the model is exactly shift-equivariant, so d_inf can be corrected after inference without re-running it.
3. *People and a truck wheel as calibration objects (batch A).* VideoPose3D gives each person's body in metres; the stereo size of the same person depends only on b / (d − d_inf), so their ratio implies d_inf. People at 1.8–4.2 m gave +0.6 to −3.0 px, people at 5–13 m −1.8 to −4.6 px, and a standard 22.5″ truck rim at 8.5 m about −2.5 px. Inconclusive between "offset" and "gain" (VideoPose3D's own size error is ±10–15 %); the model itself agreed with independent block matching to within 0.03 px at every disparity (54 447 patches).
4. *Helmet markers (batch B), the decisive test.* §6.

**4.2 Result (batch B, 12 427 marker readings).** d_inf = **−2.98 px** for the 2026-10-21 session (correction −1.98 px relative to the −1.0 px used for inference). Per clip: −2.27 to −1.22 px, standard deviation 0.33 px; morning median −2.03, afternoon −1.87 (no clear drift over the day). The −1.0 px value of batch A was therefore probably too small as well.

---

## 5. Depth quality without ground truth (batch A)

Ground-truth-free tests on 4 clips (`eval_depth.py`; plan doc §11):

| Test | Result |
|---|---|
| Valid depth in 0.3–8 m | 52–69 % of pixels (rest is sky and far field) |
| Frame-to-frame flicker (p90) | 0.18–0.48 px; 2–3 % of depth in calm clips, 6–8 % with camera motion; a 3-frame temporal median removes the spikes |
| Road plane fit | 2.4–2.6 cm RMS over 1–8 m |
| Model vs independent block matching | bias ≤ 0.014 px, MAD 0.12–0.15 px |
| Left-right consistency | > 98.7 % of non-occluded pixels within 1 px |

Known failure modes: far content is a 0-disparity class; repetitive texture at the frame edge (corn stalks) can produce spurious near depth; the left image border has no right-eye counterpart; thin objects are partly resolved.

---

## 6. Ground truth from helmet ArUco markers (batch B)

**6.1 Method.** Every worker's hard hat carries four field-pose-annotator markers (ArUco 4×4_50, printed 88.9 mm edge to edge, one per side, id = worker × 4 + side). For each marker read in the left eye:

- **Marker depth:** OpenCV `solvePnP` (IPPE square) on the four sub-pixel corners, modelled as a flat 88.9 mm square, with the *same* camera intrinsics as the stereo → 3D position of the square's centre.
- **Stereo depth:** median model disparity inside the marker quad shrunk to 60 % (away from edge bleed), at least 60 % valid pixels, then fx·b / (disparity − correction).

Both depths are proportional to fx (marker: fx · size / pixels; stereo: fx · b / disparity), so **their ratio does not depend on the field of view**; it tests the baseline and the disparity offset only.

**6.2 Which error is it? Fitting three models** to `d_model` against `d_marker = fx·b / z_marker` (first 4 clips, 2 586 readings):

| Error model | Residual (median, px) | Remaining trend | Implies |
|---|---|---|---|
| gain only (baseline or marker size wrong) | 0.64 | +0.24 px/px | a 14.5 mm baseline or a 118 mm marker |
| **offset only (d_inf)** | **0.46** | −0.01 px/px | constant −2.1 px |
| gain + offset | 0.47 | 0 | gain 0.98 (≈ 1) |

The disagreement is a constant disparity offset, not a wrong camera parameter. Supporting checks: two codes of one helmet in the same frame agree to 2–3 cm by PnP; to fake the offset, every marker edge would have to be mis-measured by 9.6 px on a 37 px code.

**6.3 Accuracy after the session correction (all 15 clips).**

| Clip | Readings | Error before | After: median | Median abs. | Median abs. (m) | Within 5 % | Within 10 % |
|---|---|---|---|---|---|---|---|
| 3345 | 2067 | +34.2 % | +2.9 % | 4.8 % | 0.16 | 51 % | 74 % |
| 3346 | 567 | +35.7 % | +1.9 % | 5.1 % | 0.18 | 49 % | 66 % |
| 3347 | 670 | +36.4 % | +0.3 % | 6.1 % | 0.24 | 42 % | 71 % |
| 3348 | 1007 | +34.0 % | +0.7 % | 5.6 % | 0.21 | 46 % | 71 % |
| 3349 | 942 | +31.3 % | +2.3 % | 4.0 % | 0.13 | 61 % | 91 % |
| 3350 | 488 | +22.9 % | −4.8 % | 6.7 % | 0.24 | 37 % | 69 % |
| 3352 | 485 | +19.4 % | −2.1 % | 4.7 % | 0.13 | 52 % | 70 % |
| 3353 | 1443 | +29.8 % | −0.8 % | 5.6 % | 0.20 | 46 % | 74 % |
| 3354 | 339 | +25.9 % | +0.8 % | 4.3 % | 0.13 | 55 % | 79 % |
| 3355 | 683 | +13.3 % | −1.5 % | 4.5 % | 0.08 | 57 % | 87 % |
| 3356 | 407 | +9.6 % | −5.4 % | 6.1 % | 0.12 | 42 % | 77 % |
| 3378 | 162 | +16.9 % | +3.2 % | 8.4 % | 0.17 | 31 % | 60 % |
| 3381 | 1131 | +13.5 % | −7.1 % | 12.3 % | 0.35 | 21 % | 40 % |
| 3382 | 1254 | +22.5 % | −2.9 % | 8.9 % | 0.28 | 30 % | 56 % |
| 3385 | 782 | +32.0 % | −0.2 % | 6.4 % | 0.24 | 41 % | 67 % |
| **All** | **12 427** | | **−0.3 %** | **5.8 %** | | **44 %** | **70 %** |

(Exact per-clip shares: `aruco_summary.csv` in the deliverables folder.) IMG_3381 is the outlier: its own best offset is −1.23 px, 0.75 px away from the session value.

**By distance** (first 4 clips, 2 586 readings, offset fitted on all; leave-one-clip-out gives the same numbers):

| Distance | Before correction | After: median abs. error |
|---|---|---|
| 2–3 m | +24 % | 3.7 % (10 cm) |
| 3–4 m | +31 % | 6.7 % (23 cm) |
| 4–5 m | +43 % | 5.4 % (24 cm) |
| 5–7 m | +62 % | 6.3 % (34 cm) |

The error before correction grows with distance, which is the signature of a disparity offset.

**6.4 Which method is worse where they disagree.**

- *Precision:* frame-to-frame noise is 0.3–0.75 % for the marker and 1.2–2.4 % for stereo at 2–7 m — the marker is about 4× steadier.
- *But frame noise explains only 4 % of the disagreement variance* (robust σ 7.9 %); the rest is slow and systematic.
- *Stereo-side systematic:* with the offset removed, the residual is ≈ 0 px in the image centre but +1.4 px at the left edge (columns 0–320) and −0.8 px at the right edge — the eyes' alignment varies across the width (the ultra-wide right eye is digitally aligned). Lens distortion on the marker side could explain at most 0.1 px.
- *Per-marker biases:* most codes within ±0.4 px, a few ±1.3 px for a whole segment (a code bent over the helmet, or stereo on a small textured patch); not attributable with these data.
- *Conclusion:* where markers exist, they are the better depth reference and the ideal automatic calibration for the stereo; stereo's value is that it is dense, everywhere in the frame.

**6.5 Marker detection and the comparison videos.** The field-pose-annotator detector reads a code in 39 % of frames (clip 3348). A more permissive detector reads 60 %, but its extra readings have a 15.5 % median error (misshaped quads, mis-decoded codes at real worker ids), so it is not used. Instead, codes are tracked between detections with optical flow on their corners for up to 0.5 s (forward-backward check): a marker is outlined in 70 % of all frames of the batch (39–98 % per clip), and tracked frames are as accurate as detections (5.1 % vs 5.6 %) but kept out of the statistics.

**6.6 The marker's own limitation: curved helmets.** The markers are flat stickers on rounded hard hats. A simulation (`scripts/spatial/dev/curved_marker_sim.py`) of the flat-square PnP on a wrapped 88.9 mm sticker:

| Sticker shape | PnP depth too far by | Disparity error at 2–5 m |
|---|---|---|
| flat | 0.0–0.4 % | ≈ 0 px |
| cylinder (bent one way, R 100–120 mm) | 0.1–0.5 % | 0.01–0.07 px |
| hard-hat dome (Rh 100–120, Rv 150–180 mm) | 1.2–2.3 % (3–8 cm) | 0.07–0.33 px |
| sphere (R 110 mm) | 2.6–3.6 % (7–15 cm) | 0.15–0.5 px |

Two causes: the corners lie up to ≈ 15 mm behind the sticker's centre, and the wrapped sticker looks smaller than a flat one. If the stickers follow the dome, the true d_inf is 0.1–0.3 px more negative than −2.98 px and corrected stereo is 1–2 % nearer than reported. It does not explain the ±1.3 px per-marker biases or the 5.8 % scatter.

---

## 7. Skeletons combined with depth (batch A)

2D keypoints (RTMPose) and 3D poses (VideoPose3D, root-relative, metres) from the lab's field-pose-annotator were combined with the stereo depth (`pose_depth.py`; plan doc §12): depth sampled at every joint, VideoPose3D scaled per person and anchored at the stereo pelvis, limb depth direction corrected where stereo is confident.

| Result | Value |
|---|---|
| VideoPose3D (scaled, anchored) vs stereo joints inside 5 m | 19–22 cm median per joint (pelvis 2–7 cm, wrists 22–40 cm) |
| Limbs whose depth direction VideoPose3D got wrong (by stereo) | 3–8 % of limb-frames; correcting them changes the median by 0.2–0.5 cm |
| Stereo limb length / typical adult | 1.01–1.02 for well-measured workers; 1.42 for workers in front of the corn field (depth window catches the background) |
| Effect of the marker-derived d_inf on a batch-B clip (3354) | stereo limb ratio 1.29 → 1.00; VideoPose3D vs stereo 24.6 → 18.7 cm; stereo vs marker 26 % → 4.2 % |

Value of depth for pose: absolute position and per-person scale (which VideoPose3D lacks); the adjusted skeletons load into the annotator as a metric tracks package. A literature positioning and a research plan around this combination are in the plan doc §13 (written separately).

---

## 8. Deliverables and where everything is

| What | Where |
|---|---|
| Batch B comparison videos (15), summary, per-read data | `Desktop\spatial_depth_results_2026-10-21\` (1.46 GB): `IMG_<id>_aruco_compare.mp4`, `aruco_summary.csv`, `aruco_data/` |
| Batch A depth and skeleton videos (4 clips) | `Desktop\spatial_depth_results_2026-10-14\` (4.13 GB) |
| Stills for slides | `docs/results/<id>/aruco_compare_mid.jpg` (all 15 batch-B clips; e.g. `docs/results/3353/aruco_compare_mid.jpg`); batch A: `docs/results/3143/vis_s30_000000.jpg`, `docs/results/3143/vis_full_000405.jpg`, `docs/results/3143/pose_compare_frame000405.jpg`, `docs/results/3143/pose_fused_frame000405.jpg`, `docs/results/3144/wheel_frame000300_crop.jpg` |
| Short clips for slides | `docs/results/<id>/preview_10s.mp4`, `pose_compare_10s.mp4`, `pose_fused_10s.mp4` for 3142–3145 |
| Full method and log | `docs/iphone_spatial_video_plan.md` |
| Code | `scripts/spatial/` (README there), analyses in `scripts/spatial/dev/` (README there) |

What the comparison video shows: the full-resolution left eye with every helmet code outlined along its own edges in the worker's colour; beside it the worker id and three colour-coded distances (marker in cyan, stereo in orange, stereo − marker in magenta, metres to the centimetre); a side panel with the camera model, legend, workers in view, running statistics and a top view with 1 m rings. The videos and source files show recognisable people and carry GPS; keep them within the project.

---

## 9. Recommendations and next steps

1. **Record markers in every session** and fit d_inf from them automatically (`session_offset.py`); log the value per phone. Do not reuse a value across sessions until several sessions agree within ≈ 0.2 px.
2. **Make the comparison curvature-aware (designed, not yet run):** model each sticker as wrapped on the helmet surface (radii Rh, Rv measured once on the actual hard hat), solve a non-planar PnP for the sticker's surface centre, fit the stereo disparity over the whole code with the shape that pose predicts and evaluate it at the same centre pixel, re-fit gain and offset (the gain should then be 1). Check with front + side codes of one helmet (same helmet position) and with the residual vs viewing angle. Needed from the field: hard-hat model or two bend measurements at the sticker position, a ruler check of one printed code.
3. **One tape-measure or A4-sheet shot** at 1, 2, 3, 5 m per phone, to pin the field of view (absolute scale of both methods).
4. **Model the column dependence of the offset** (up to ±1.4 px at the image edges) once enough edge readings exist, or avoid measuring at the frame edges.
5. For skeletons: a person mask for joint depth (removes background contamination of limbs); per-joint blending of stereo and VideoPose3D once that exists.
6. Throughput: feed frames from an ffmpeg pipe instead of PNGs, use 4 GRU iterations (−24 % time, 0.016 px change), and an fp16 TensorRT engine (needs a half-precision ONNX export).

---

## 10. Caveats to state in a presentation

- The 5.8 % accuracy is measured on helmet markers at 2–7 m in daylight with mostly static or slowly moving workers; it is the accuracy of stereo **on a 30–50 px patch**, which is harder than on large surfaces (the road plane fits to 2.5 cm RMS).
- The ground truth is itself approximate: flat-marker PnP on curved helmets is estimated to read 1–3 % too far (simulation, not yet measured).
- d_inf varies between sessions (Apple's per-recording alignment) and slightly between clips (±0.33 px); it must be calibrated, ideally per session.
- Absolute scale of both stereo and marker depends on the field of view from the file metadata (63.4°); confirmed indirectly, not yet with a tape measure.
- Beyond ≈ 5 m the 19 mm baseline makes depth noisy (≥ 20 cm per 0.25 px) and far content is a learned class, not a measurement.

---

## 11. Suggested presentation structure

1. Problem and idea: phone spatial video as a cheap metric depth sensor for site ergonomics (§1).
2. What the file contains and the physical limits of a 19 mm baseline (§2).
3. Pipeline and speed (§3), with a still from `docs/results/3143/vis_full_000405.jpg`.
4. The d_inf story: why depth was 25–35 % too far and how markers fixed it (§4, §6.2) — the gain-vs-offset table is the key slide.
5. Accuracy against markers (§6.3) — the per-clip table and the distance table; a frame from a comparison video (`docs/results/3353/aruco_compare_mid.jpg`).
6. Which is better, marker or stereo, and why (§6.4).
7. Skeletons with depth (§7), with `docs/results/3143/pose_compare_frame000405.jpg`.
8. Limitations, including curved helmets (§6.6, §10), and next steps (§9).
