# Designed cells by order: Cpk of r12 under manufacturing tolerances

Comparison of `design_cpk.py` runs that differ only in the design order window (bounce count of the
nominal P_max; --min-bounces / --max-bounces): r12 best (it700), tolerances separation +-1 mm, laser
pitch / yaw +-0.25 deg, laser dx / dy +-0.2 mm, 8 designs (one per separation band) x 512 as-built cells.
Per-run details: b20-40/, b41-60/, b61-80/, design_cpk.md (81-127: the longest designs below the cap).
Possible: as-built cells with P_max >= P_floor. Align: P_final >= 0.9 P_max(as-built) over the possible
cells. Delivered: P_final >= 0.9 P_max(nominal) over all cells; ceiling = cells where P_max(as-built) >=
0.9 P_max(nominal). Cpk = Phi^-1(yield) / 3 [95 %].

## All designs

| design order (bounces) | nominal P_max (m) | possible | align yield | align Cpk | delivered ceiling | delivered yield | delivered Cpk |
|---|---|---|---|---|---|---|---|
| 20-40 | 1.6-14.8 | 90.6 % | 93.0 % | +0.49 [+0.47, +0.51] | 86.1 % | 83.5 % | +0.33 [+0.31, +0.34] |
| 41-60 | 2.3-20.5 | 92.1 % | 90.0 % | +0.43 [+0.41, +0.44] | 71.8 % | 67.7 % | +0.15 [+0.14, +0.17] |
| 61-80 | 2.9-27.7 | 76.9 % | 91.3 % | +0.45 [+0.43, +0.47] | 48.9 % | 45.7 % | -0.04 [-0.05, -0.02] |
| 81-127 (longest) | 3.6-31.3 | 72.0 % | 86.7 % | +0.37 [+0.35, +0.39] | 31.5 % | 24.9 % | -0.23 [-0.24, -0.21] |

## Per separation band: align Cpk / delivered yield

| band (mm) | 20-40 | 41-60 | 61-80 | 81-127 (longest) |
|---|---|---|---|---|
| 20-50 | +0.36 / 93 % (37 b, 1.6 m) | +0.11 / 71 % (57 b, 2.3 m) | +0.26 / 80 % (61 b, 2.9 m) | +0.15 / 44 % (85 b, 3.6 m) |
| 50-100 | +0.81 / 100 % (37 b, 3.4 m) | +0.43 / 71 % (55 b, 5.2 m) | +0.61 / 48 % (73 b, 7.0 m) | +0.55 / 27 % (101 b, 9.1 m) |
| 100-150 | +0.46 / 71 % (39 b, 5.6 m) | +0.44 / 72 % (57 b, 8.4 m) | +0.37 / 49 % (73 b, 10.3 m) | +0.32 / 18 % (123 b, 17.2 m) |
| 150-200 | +0.29 / 48 % (37 b, 7.0 m) | >= +0.81 / 100 % (57 b, 10.8 m) | +0.64 / 68 % (77 b, 14.8 m) | +0.44 / 17 % (115 b, 18.1 m) |
| 200-250 | +0.55 / 99 % (37 b, 9.2 m) | +0.45 / 59 % (59 b, 14.3 m) | +0.52 / 41 % (79 b, 19.5 m) | +0.48 / 44 % (113 b, 23.2 m) |
| 250-300 | +0.62 / 100 % (39 b, 11.4 m) | +0.62 / 70 % (57 b, 16.8 m) | +0.68 / 50 % (75 b, 22.2 m) | +0.29 / 18 % (105 b, 27.7 m) |
| 300-350 | +0.51 / 94 % (37 b, 12.5 m) | +0.55 / 57 % (57 b, 19.7 m) | +0.53 / 19 % (77 b, 26.1 m) | +0.65 / 22 % (103 b, 31.3 m) |
| 350-400 | +0.59 / 63 % (39 b, 14.8 m) | +0.57 / 42 % (57 b, 20.5 m) | +0.49 / 11 % (77 b, 27.7 m) | +0.54 / 10 % (81 b, 29.5 m) |
