# Trapped states vs shorter-path look-alikes at 64 px

Measured cell, +-2 deg M2 box on a 0.01 deg grid: 160801 states, 163 trapped (beam to the 128-bounce cap). d_short: per trapped state, the RMS difference (8-bit levels, 64 x 64) to the nearest image among the states with path < 0.8 x its path. Camera read noise 0.02 full scale = 5.1 levels. `R -> 64`: rendered at R, response and 8-bit per sensor pixel, block mean to 64 px (env.DownscaleRenderer). Current renderer: 64 direct, sigma 0.5 mm, gain 1.6; this run: gain 1.6.

| pipeline | d_short q10 | d_short median | < 1 | < 2 | < 5 | < 10 | trapped RMS | d_short / RMS | look-alike bounces |
|---|---|---|---|---|---|---|---|---|---|
| 64 direct, sigma 0.5 | 4.83 | 6.66 | 0.0 % | 0.0 % | 17.2 % | 99.4 % | 43.0 | 0.155 | 89 |
| 128 -> 64, sigma 0.5 | 4.75 | 6.52 | 0.0 % | 0.0 % | 18.4 % | 99.4 % | 42.5 | 0.153 | 89 |
| 256 -> 64, sigma 0.5 | 4.75 | 6.53 | 0.0 % | 0.0 % | 17.8 % | 99.4 % | 42.4 | 0.154 | 89 |
| 512 -> 64, sigma 0.5 | 4.75 | 6.53 | 0.0 % | 0.0 % | 19.0 % | 99.4 % | 42.4 | 0.154 | 89 |
| 64 direct, sigma 0.25 | 3.19 | 5.48 | 0.0 % | 0.0 % | 42.3 % | 100.0 % | 20.3 | 0.270 | 89 |
| 128 -> 64, sigma 0.25 | 2.97 | 5.06 | 0.0 % | 0.0 % | 48.5 % | 100.0 % | 19.3 | 0.262 | 89 |
| 256 -> 64, sigma 0.25 | 2.91 | 4.93 | 0.0 % | 0.0 % | 52.1 % | 100.0 % | 19.1 | 0.258 | 89 |
| 512 -> 64, sigma 0.25 | 2.91 | 4.97 | 0.0 % | 0.0 % | 51.5 % | 100.0 % | 19.1 | 0.260 | 89 |
| 64 direct, sigma 0.125 | 1.96 | 3.27 | 0.0 % | 12.9 % | 99.4 % | 100.0 % | 9.9 | 0.330 | 57 |
| 128 -> 64, sigma 0.125 | 1.47 | 2.53 | 0.0 % | 25.2 % | 100.0 % | 100.0 % | 8.0 | 0.317 | 57 |
| 256 -> 64, sigma 0.125 | 1.41 | 2.43 | 0.0 % | 28.8 % | 100.0 % | 100.0 % | 7.8 | 0.315 | 89 |
| 512 -> 64, sigma 0.125 | 1.40 | 2.40 | 0.0 % | 29.4 % | 100.0 % | 100.0 % | 7.7 | 0.314 | 89 |

Figure `render_res.jpg`: the trapped states with the smallest d_short under the current renderer, their nearest shorter-path look-alike and the difference x 8 (title: bounces, tilt).
