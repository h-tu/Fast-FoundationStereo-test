import sys, json, glob, os
sys.path.insert(0, 'scripts/spatial')
from probe_metadata import probe
os.makedirs('work', exist_ok=True)
tot = 0
for p in sorted(glob.glob('vid/IMG_33*.MOV')):
    n = os.path.basename(p)[4:8]
    try:
        d = probe(p)
    except SystemExit as e:
        print(n, 'NOT SPATIAL:', e); continue
    json.dump(d, open(f'work/probe_{n}.json', 'w'), indent=2)
    tot += d['nb_frames']
    print(f"{n}: {d['device']} iOS {d['software']} {d['width']}x{d['height']} @ {d['fps']:.3f} frames {d['nb_frames']} {d['duration_s']:.1f}s "
          f"baseline {d['baseline_m']:.6f} hfov {d['hfov_deg']} dadj {d['disparity_adjust_px']:.1f} fx {d['fx']:.2f} {d['pix_fmt']} {d['color_transfer']} views {d['left_view'].get('view_ids_present')}")
print('total frames', tot)
