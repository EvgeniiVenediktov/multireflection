# Designed cells by order: Cpk of r12 under manufacturing tolerances

Comparison of `design_cpk.py` runs that differ only in the design order window (bounce count of the
nominal P_max; --min-bounces / --max-bounces): r12 best (it700), tolerances separation +-1 mm, laser
pitch / yaw +-0.25 deg, laser dx / dy +-0.2 mm, 8 designs (one per separation band) x 512 as-built cells.
Per-run details: b20-40/, b41-60/, b61-80/, design_cpk.md (81-127: the longest designs below the cap).
Possible: as-built cells with P_max >= P_floor (the rest hold no long path at any M2 tilt: manufacturing
loss). Both specs over the possible cells only: align = P_final >= 0.9 P_max(as-built); delivered =
P_final >= 0.9 P_max(nominal), ceiling = possible cells with P_max(as-built) >= 0.9 P_max(nominal).
Cpk = Phi^-1(yield) / 3 [95 %].

## All designs

| design order (bounces) | nominal P_max (m) | possible | n | align yield | align Cpk | delivered ceiling | delivered yield | delivered Cpk |
|---|---|---|---|---|---|---|---|---|
| 20-40 | 1.6-14.8 | 90.6 % | 3709 | 93.0 % | +0.49 [+0.47, +0.51] | 95.1 % | 92.3 % | +0.47 [+0.45, +0.49] |
| 41-60 | 2.3-20.5 | 92.1 % | 3774 | 90.0 % | +0.43 [+0.41, +0.44] | 78.0 % | 73.5 % | +0.21 [+0.19, +0.22] |
| 61-80 | 2.9-27.7 | 76.9 % | 3150 | 91.3 % | +0.45 [+0.43, +0.47] | 63.6 % | 59.5 % | +0.08 [+0.07, +0.09] |
| 81-127 (longest) | 3.6-31.3 | 72.0 % | 2951 | 86.7 % | +0.37 [+0.35, +0.39] | 43.7 % | 34.6 % | -0.13 [-0.15, -0.12] |

## Per separation band (possible cells): align Cpk / delivered Cpk (design order, nominal P_max)

| band (mm) | 20-40 | 41-60 | 61-80 | 81-127 (longest) |
|---|---|---|---|---|
| 20-50 | +0.36 / +0.50 (37 b, 1.6 m) | +0.11 / +0.19 (57 b, 2.3 m) | +0.26 / +0.29 (61 b, 2.9 m) | +0.15 / -0.05 (85 b, 3.6 m) |
| 50-100 | +0.81 / >= +0.81 (37 b, 3.4 m) | +0.43 / +0.23 (55 b, 5.2 m) | +0.61 / +0.31 (73 b, 7.0 m) | +0.55 / -0.20 (101 b, 9.1 m) |
| 100-150 | +0.46 / +0.19 (39 b, 5.6 m) | +0.44 / +0.19 (57 b, 8.4 m) | +0.37 / -0.01 (73 b, 10.3 m) | +0.32 / -0.18 (123 b, 17.2 m) |
| 150-200 | +0.29 / +0.36 (37 b, 7.0 m) | >= +0.81 / >= +0.81 (57 b, 10.8 m) | +0.64 / +0.27 (77 b, 14.8 m) | +0.44 / -0.21 (115 b, 18.1 m) |
| 200-250 | +0.55 / +0.78 (37 b, 9.2 m) | +0.45 / +0.14 (59 b, 14.3 m) | +0.52 / -0.04 (79 b, 19.5 m) | +0.48 / +0.04 (113 b, 23.2 m) |
| 250-300 | +0.62 / >= +0.81 (39 b, 11.4 m) | +0.62 / +0.20 (57 b, 16.8 m) | +0.68 / +0.21 (75 b, 22.2 m) | +0.29 / -0.21 (105 b, 27.7 m) |
| 300-350 | +0.51 / +0.60 (37 b, 12.5 m) | +0.55 / +0.18 (57 b, 19.7 m) | +0.53 / -0.14 (77 b, 26.1 m) | +0.65 / -0.10 (103 b, 31.3 m) |
| 350-400 | +0.59 / +0.38 (39 b, 14.8 m) | +0.57 / +0.04 (57 b, 20.5 m) | +0.49 / -0.30 (77 b, 27.7 m) | +0.54 / -0.27 (81 b, 29.5 m) |
