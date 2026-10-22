# scripts/spatial — iPhone spatial video → metric depth and skeletons

Pipeline for Apple MV-HEVC spatial videos (iPhone 15 Pro and later) on top of Fast-FoundationStereo.
Background, measured numbers and caveats: `docs/iphone_spatial_video_plan.md` (§6 interfaces, §10–§13 logs and results).

## Requirements

- CUDA GPU; the repo environment (torch 2.6 + cu124, `requirements.txt`, `triton-windows<3.3` on Windows).
- ffmpeg ≥ 7.1 with MV-HEVC (`ffmpeg -h decoder=hevc` lists `view_ids`). A static build in `tools/ffmpeg-*/bin` is picked up by `batch_process.py`; for the single scripts put it on `PATH`.
- Weights: `weights/23-36-37/model_best_bp2_serialize.pth` with `cfg.yaml` next to it.
- Pose steps only: a checkout of `field-pose-annotator` and its pose venv (`~/.venvs/field-pose-annotator-pose`).
- Never commit `vid/`, `work/`, `weights/`, `tools/` (see `CLAUDE.md`).

## Scripts

| Script | Role | Main output |
|---|---|---|
| `probe_metadata.py` | stereo metadata (baseline, hfov, eye mapping) → pinhole K | JSON, `K.txt` |
| `extract_views.py` | split left/right eyes to PNG (`--stride`) | `left/`, `right/`, `K.txt`, `meta.json` |
| `check_rectification.py` | ZNCC block matching: vertical offset, right-eye affine, far disparity | stats, JSON |
| `run_sequence.py` | Fast-FoundationStereo over a sequence (right-eye affine, `--dinf`, `--backend torch\|trt`, `--save_vis`) | `disp/`, `depth/` npy, `vis/`, `run_meta.json` |
| `make_depth_video.py` | preview, 16-bit lossless depth, overlay (`--temporal_median`) | `preview.mp4`, `depth_mm.mkv`, `overlay.mp4` |
| `eval_depth.py` | ground-truth-free quality tests (coverage, flicker, road plane, block matching, left-right) | `eval.json` |
| `estimate_dinf.py`, `dinf_blockmatch.py` | disparity-at-infinity estimates (shift trick, far horizon) | JSON |
| `pose_depth.py` | field-pose-annotator skeletons + stereo depth: validation, adjusted metric VideoPose3D, videos, annotator package | `pose_compare.mp4`, `pose_fused.mp4`, `*_fused.tracks.npz`, `*.pose_depth.json` |
| `dinf_from_bodies.py` | people (VideoPose3D metric size) as calibration objects for d_inf | JSON |
| `session_offset.py` | d_inf correction for a recording session from all clips' marker reads (per clip and pooled) | console, JSON |
| `apply_dinf.py` | re-derive depth from the saved disparities with a corrected d_inf (no new inference) | `depth/` npy, `run_meta.json` fields |
| `aruco_compare.py` | helmet ArUco markers (field-pose-annotator, 88.9 mm) as ground truth: PnP depth vs stereo depth on the marker, live video, disparity gain/offset fit | `aruco_compare.mp4`, `aruco_reads.csv`, `aruco_compare.json` |
| `build_trt_engine.py` | TensorRT engine from the single ONNX (no `trtexec` needed) | `.engine` |
| `batch_process.py` | everything above for many clips: `gpu`, `post` and `aruco` workers in parallel, restartable, deliverables + cleanup | `work/<id>_full/…`, deliverables folder |
| `batch_summary.py` | one table over all clips of a batch (depth, pose and marker results) | Markdown + JSON |
| `batch_finish.py` | after the gpu stage: retry failed clips, session d_inf from all markers, `recal` in parallel workers; restart-safe, phase in `work/batch_finish_status.json` | marker files, status |
| `batch_progress.py` | live per-clip progress and ETA of a batch (`--watch 10`) | console |
| `dev/` | one-off analysis scripts behind numbers in the plan doc (see `dev/README.md`) | console |

## One clip, step by step

```
python scripts/spatial/extract_views.py vid/IMG_3143.MOV work/3143_full --stride 1
python scripts/spatial/run_sequence.py --seq_dir work/3143_full --out_dir work/3143_full/out --valid_iters 8 --max_disp 96 --volume triton --auto_affine 10 --dinf -1.0 --pc_every 0 --zfar 8
python scripts/spatial/make_depth_video.py --seq_dir work/3143_full --out_dir work/3143_full/out --video_dir work/3143_full/video --overlay 1
python scripts/spatial/eval_depth.py --seq_dir work/3143_full --out_dir work/3143_full/out --lr_frames 8 --json work/3143_full/eval.json
# pose (field-pose-annotator venv) on the left eye, then fusion
ffmpeg -i vid/IMG_3143.MOV -map 0:v:0:vpos:left -an -fps_mode passthrough -c:v libx264 -crf 12 -pix_fmt yuv420p work/3143_full/pose/IMG_3143_left.mp4
<pose venv python> <field-pose-annotator>/scripts/preprocess_video.py --video work/3143_full/pose/IMG_3143_left.mp4 --output-dir work/3143_full/pose --backend rtmlib --tier full --device cuda:0
python scripts/spatial/pose_depth.py --tracks work/3143_full/pose/IMG_3143_left.tracks.npz --seq_dir work/3143_full --out_dir work/3143_full/out
```

## Many clips

```
python scripts/spatial/probe_metadata.py vid/IMG_xxxx.MOV          # check every clip is a spatial video first
python scripts/spatial/batch_process.py --stage gpu  --ids 3345 3346 ... --deliver_dir <folder>
python scripts/spatial/batch_process.py --stage post --ids 3345 3346 ... --deliver_dir <folder>   # second terminal, same ids
python scripts/spatial/batch_process.py --stage aruco --ids 3345 3346 ... --deliver_dir <folder>  # third, if helmets carry markers
python scripts/spatial/batch_progress.py --ids 3345 3346 ... --watch 10                           # live progress
python scripts/spatial/batch_finish.py --ids 3345 3346 ... --deliver_dir <folder> --tag <date>    # fourth: retry failures, session offset, recal
# or by hand:
python scripts/spatial/session_offset.py --ids 3345 3346 ... --json work/session_offset.json      # session d_inf offset from the markers
python scripts/spatial/batch_process.py --stage recal --dinf_correction -2.08 --ids ... --deliver_dir <folder>   # re-derive all outputs with it
python scripts/spatial/batch_summary.py --ids 3345 3346 ... --md work/batch_summary.md --json work/batch_summary.json
```

`batch_process.py` expects `work/probe_<id>.json` (written by probing, see the plan doc §13), uses stride 1 for clips up to 90 s and stride 2 above (`--long_s`, `--stride_long`), d_inf −1.0 px (`--dinf`), skips every step whose output exists (restart-safe, logs in `work/batch_{gpu,post}.log`), keeps the GPU stage at most two clips ahead of the post stage (disk), and after a clip is finished deletes its PNG frames and depth npy (depth stays losslessly in `depth_mm.mkv`; disparity npy is kept, depth = fx·b/disparity). Peak disk is about 25 MB per frame of the two clips in flight.

## Known limits (details in the plan doc §11–§12)

- Depth and skeletons are reliable inside about 5 m; beyond, the metric scale drifts (30–70 % too far at 6–9 m) until a calibration recording fixes d_inf / baseline.
- Far content (sky, distant field) is a learned "0 disparity" class, not a measurement.
- Joint depths are window samples; people in front of textured background (corn) get inflated limb lengths.
