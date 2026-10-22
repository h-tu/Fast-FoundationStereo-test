# Stereo depth vs helmet ArUco markers — iPhone 17 Pro Max spatial videos recorded 2026-10-21

One video per clip, `IMG_xxxx_aruco_compare.mp4` (IMG_3345, IMG_3352 and IMG_3378 at 14.985 fps = every second frame; the others at 29.97 fps).

What each video shows:
- The left eye (main camera, 1x) at full 1920x1080. Every helmet marker read in a frame is boxed in its worker's colour;
  corner brackets mark a marker last read less than 0.5 s ago.
- Next to each box: worker id and the side of the helmet, then three distances from the lens to the marker centre, in metres to the centimetre:
  **cyan = marker** (ArUco PnP of the 88.9 mm code), **orange = stereo** (Fast-FoundationStereo depth on the marker),
  **magenta = diff** (stereo minus marker, in m and %).
- Side panel: camera model and settings, the colour legend, the workers in view, running statistics for the clip,
  and a top view (rings per metre) with each helmet's marker position (ring) and stereo position (dot).

The stereo is corrected by one disparity offset for the whole recording session, fitted from all marker reads
(stated in each video's camera panel as d_inf). Both distances use the same camera intrinsics, so the comparison does
not depend on the field-of-view value.

`aruco_summary.csv`: per clip, number of reads, workers, offset used, error before and after the correction, share within 5 % / 10 %.
`aruco_data/`: per clip, every read (`*_aruco_reads.csv`) and the statistics (`*_aruco_compare.json`).

Method, numbers and caveats: `docs/iphone_spatial_video_plan.md` in the Fast-FoundationStereo-test repository (sections 6.10, 14).
The videos show recognisable people; keep them within the project.
