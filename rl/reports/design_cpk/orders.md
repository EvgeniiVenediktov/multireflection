# Designed cells by order: Cpk of r12 under manufacturing tolerances

Comparison of `design_cpk.py` runs that differ only in the design order window (bounce count of the
nominal P_max; --min-bounces / --max-bounces): r12 best (it700), tolerances separation +-1 mm, laser
pitch / yaw +-0.25 deg, laser dx / dy +-0.2 mm, 8 designs (one per separation band) x 4096 as-built cells.
Per-run details: b20-40/, b41-60/, b61-80/, design_cpk.md (81-127: the longest designs below the cap).
Possible: as-built cells with P_max >= P_floor (the rest hold no long path at any M2 tilt: manufacturing
loss). Both specs over the possible cells only: align = P_final >= 0.9 P_max(as-built); delivered =
P_final >= 0.9 P_max(nominal), ceiling = possible cells with P_max(as-built) >= 0.9 P_max(nominal).
Cpk = Phi^-1(yield) / 3 [95 %].

## All designs

| design order (bounces) | nominal P_max (m) | possible | n | align yield | align Cpk | delivered ceiling | delivered yield | delivered Cpk |
|---|---|---|---|---|---|---|---|---|
| 20-40 | 1.6-14.8 | 90.5 % | 29666 | 93.5 % | +0.51 [+0.50, +0.51] | 95.3 % | 92.9 % | +0.49 [+0.48, +0.50] |
| 41-60 | 2.3-20.5 | 91.9 % | 30121 | 90.4 % | +0.44 [+0.43, +0.44] | 77.6 % | 74.0 % | +0.21 [+0.21, +0.22] |
| 61-80 | 2.9-27.7 | 76.4 % | 25031 | 91.4 % | +0.46 [+0.45, +0.46] | 63.6 % | 59.5 % | +0.08 [+0.07, +0.09] |
| 81-127 (longest) | 3.6-31.3 | 72.4 % | 23733 | 85.9 % | +0.36 [+0.35, +0.37] | 42.8 % | 33.5 % | -0.14 [-0.15, -0.14] |

## Per separation band (possible cells): align Cpk / delivered Cpk (design order, nominal P_max)

| band (mm) | 20-40 | 41-60 | 61-80 | 81-127 (longest) |
|---|---|---|---|---|
| 20-50 | +0.34 / +0.51 (37 b, 1.6 m) | +0.10 / +0.20 (57 b, 2.3 m) | +0.26 / +0.28 (61 b, 2.9 m) | +0.14 / -0.07 (85 b, 3.6 m) |
| 50-100 | +0.74 / +1.10 (37 b, 3.4 m) | +0.43 / +0.26 (55 b, 5.2 m) | +0.62 / +0.33 (73 b, 7.0 m) | +0.52 / -0.21 (101 b, 9.1 m) |
| 100-150 | +0.50 / +0.22 (39 b, 5.6 m) | +0.53 / +0.19 (57 b, 8.4 m) | +0.41 / +0.01 (73 b, 10.3 m) | +0.31 / -0.16 (123 b, 17.2 m) |
| 150-200 | +0.33 / +0.39 (37 b, 7.0 m) | +1.16 / +1.01 (57 b, 10.8 m) | +0.55 / +0.25 (77 b, 14.8 m) | +0.41 / -0.27 (115 b, 18.1 m) |
| 200-250 | +0.59 / +0.90 (37 b, 9.2 m) | +0.45 / +0.17 (59 b, 14.3 m) | +0.50 / -0.04 (79 b, 19.5 m) | +0.48 / +0.04 (113 b, 23.2 m) |
| 250-300 | +0.63 / >= +1.04 (39 b, 11.4 m) | +0.69 / +0.19 (57 b, 16.8 m) | +0.69 / +0.19 (75 b, 22.2 m) | +0.31 / -0.17 (105 b, 27.7 m) |
| 300-350 | +0.52 / +0.58 (37 b, 12.5 m) | +0.55 / +0.13 (57 b, 19.7 m) | +0.52 / -0.15 (77 b, 26.1 m) | +0.58 / -0.15 (103 b, 31.3 m) |
| 350-400 | +0.60 / +0.39 (39 b, 14.8 m) | +0.57 / +0.06 (57 b, 20.5 m) | +0.52 / -0.30 (77 b, 27.7 m) | +0.46 / -0.25 (81 b, 29.5 m) |
