"""Fill the per-clip accuracy table of plan doc §14.3 from a deliverables folder's aruco_summary.csv.

python scripts/spatial/dev/fill_accuracy_table.py "%USERPROFILE%/Desktop/spatial_depth_results_2026-10-21/aruco_summary.csv"
"""
import csv, re, sys

rows = {r['clip'][4:]: r for r in csv.DictReader(open(sys.argv[1]))}
p = sys.argv[2] if len(sys.argv) > 2 else 'docs/iphone_spatial_video_plan.md'   # optional: another document with the same table
raw = open(p, 'rb').read().decode('utf-8'); nl = '\r\n' if '\r\n' in raw else '\n'
lines = [l.rstrip('\r') for l in raw.split('\n')]; n = 0     # also normalises mixed line endings to the file's main one
sgn = lambda x: f'{float(x):+.1f} %'.replace('-', '−')
for i, l in enumerate(lines):
    m = re.match(r'^\| (33\d\d) \| (\d+) \| ([+\-−]\d+\.\d %) \| ', l)
    if m and m.group(1) in rows and l.count('|') == 9:
        r = rows[m.group(1)]
        lines[i] = (f"| {m.group(1)} | {r['reads']} | {sgn(r['raw_median_err_pct'])} | {sgn(r['corrected_median_err_pct'])} | "
                    f"{float(r['corrected_median_abs_err_pct']):.1f} % | {float(r['corrected_median_abs_err_m']):.2f} | {r['within_5pct']} % | {r['within_10pct']} % |")
        n += 1
lines = [l.replace('(per-clip within-5/10 % shares are in `aruco_summary.csv`). ', '') for l in lines]
open(p, 'wb').write(nl.join(lines).encode('utf-8'))
print(n, 'rows rewritten')
