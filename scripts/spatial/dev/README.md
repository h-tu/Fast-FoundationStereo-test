# scripts/spatial/dev — one-off analyses

Small scripts that produced specific numbers in `docs/iphone_spatial_video_plan.md`. They are kept so every number
can be re-derived; they are not part of the pipeline and may hard-code clip ids and paths. Run from the repo root with
the repo venv unless noted.

| Script | Produces | Doc section |
|---|---|---|
| `probe_new.py` | `work/probe_<id>.json` for every `vid/IMG_33*.MOV`, metadata table of the 2026-10-21 clips | §13.1 |
| `scan_markers.py <ids>` | ArUco reads per clip on every 15th left-eye frame, marker sizes in px (field-pose-annotator venv or repo venv) | §13.3 |
| `aruco_breakdown.py <aruco_reads.csv>` | marker-vs-stereo disagreement by worker / side / marker size, two-codes-one-helmet consistency | §13.3 |
| `aruco_accuracy.py <ids>` | stereo vs marker accuracy raw and after one session disparity offset (also leave-one-clip-out), per distance bin; which error model (gain / offset) fits | §13.3 |
| `aruco_noise.py <ids>` | frame-to-frame noise of marker PnP vs stereo (second differences), two-codes-one-helmet spread, share of the disagreement explained | §13.3 |
| `aruco_residual_map.py <ids>` | stereo residual vs image column / row / marker size, and per marker | §13.3 |
| `bm_bins.py` | model disparity vs block matching binned by disparity on the 2026-10-14 clips (needs their PNG frames) | §12.3 |
