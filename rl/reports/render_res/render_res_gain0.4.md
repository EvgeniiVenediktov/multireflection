# Trapped states vs shorter-path look-alikes at 64 px

Measured cell, +-2 deg M2 box on a 0.01 deg grid: 160801 states, 163 trapped (beam to the 128-bounce cap). d_short: per trapped state, the RMS difference (8-bit levels, 64 x 64) to the nearest image among the states with path < 0.8 x its path. Camera read noise 0.02 full scale = 5.1 levels. `R -> 64`: rendered at R, response and 8-bit per sensor pixel, block mean to 64 px (env.DownscaleRenderer). Current renderer: 64 direct, sigma 0.5 mm, gain 1.6; this run: gain 0.4.

| pipeline | d_short q10 | d_short median | < 1 | < 2 | < 5 | < 10 | trapped RMS | d_short / RMS | look-alike bounces |
|---|---|---|---|---|---|---|---|---|---|
| 64 direct, sigma 0.5 | 1.72 | 2.40 | 0.0 % | 33.1 % | 100.0 % | 100.0 % | 14.4 | 0.167 | 89 |
| 128 -> 64, sigma 0.5 | 1.69 | 2.35 | 0.0 % | 33.7 % | 100.0 % | 100.0 % | 14.2 | 0.166 | 89 |
| 256 -> 64, sigma 0.5 | 1.69 | 2.36 | 0.0 % | 33.7 % | 100.0 % | 100.0 % | 14.2 | 0.166 | 89 |
| 512 -> 64, sigma 0.5 | 1.70 | 2.36 | 0.0 % | 33.7 % | 100.0 % | 100.0 % | 14.2 | 0.166 | 89 |
| 64 direct, sigma 0.25 | 1.01 | 1.80 | 9.8 % | 85.9 % | 100.0 % | 100.0 % | 6.6 | 0.272 | 89 |
| 128 -> 64, sigma 0.25 | 0.93 | 1.68 | 17.2 % | 96.3 % | 100.0 % | 100.0 % | 6.2 | 0.270 | 89 |
| 256 -> 64, sigma 0.25 | 0.91 | 1.66 | 17.2 % | 98.2 % | 100.0 % | 100.0 % | 6.1 | 0.270 | 89 |
| 512 -> 64, sigma 0.25 | 0.91 | 1.66 | 17.2 % | 98.2 % | 100.0 % | 100.0 % | 6.1 | 0.270 | 89 |
| 64 direct, sigma 0.125 | 0.57 | 1.00 | 50.9 % | 100.0 % | 100.0 % | 100.0 % | 3.2 | 0.317 | 57 |
| 128 -> 64, sigma 0.125 | 0.43 | 0.75 | 95.1 % | 100.0 % | 100.0 % | 100.0 % | 2.5 | 0.301 | 57 |
| 256 -> 64, sigma 0.125 | 0.42 | 0.72 | 96.9 % | 100.0 % | 100.0 % | 100.0 % | 2.4 | 0.301 | 57 |
| 512 -> 64, sigma 0.125 | 0.41 | 0.71 | 97.5 % | 100.0 % | 100.0 % | 100.0 % | 2.4 | 0.298 | 57 |

Figure `render_res_gain0.4.jpg`: the trapped states with the smallest d_short under the current renderer, their nearest shorter-path look-alike and the difference x 8 (title: bounces, tilt).
